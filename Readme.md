# Navigating the world of Harry Potter with Knowledge Graph

## Aim
Are you a Harry Potter fan who want to have everything about the Harry Potter universe on your fingertips? Or do you simply want to impress your friends with a cool chart of how the different characters in Harry Potter come together? Look no further than knowledge graphs.

This guide will show you how to get a knowledge graph up in Neo4J with just your laptop and your favourite book.

## What is knowledge graph
According to Wikipedia:

> A knowledge graph is a knowledge base that uses a graph-structured data model or topology to represent and operate on data.

## What do you need
In terms of hardware, all you need is a computer, preferably one with a Nvidia graphics card. To be fully self-sufficient, I will go with a local LLM setup, but one could easily also use an OpenAI API for the same purpose.

## Steps in setting up
You will need the following:
1. Ollama, and your favourite LLM model
2. a python environment
3. Neo4J

### Ollama
As I am coding on Ubuntu 24.04 in WSL2, in order for any GPU workload to passthrough easily, I am using Ollama docker. Running Ollama as a docker container is as simple as first installing the [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation), and then the following:

```
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

If you do not have a Nvidia GPU, you can run a CPU-only Ollama using the following command in CLI:

```
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

Once you are done, you can pull your favourite LLM model into Ollama. The list of models available on Ollama is [here](https://ollama.com/library). For example if I want to pull `qwen2.5`, I can run the following command in CLI:

```
docker exec -it ollama ollama run qwen2.5
```

And you are done with Ollama!

### Python environment

You will first want to create a python virtual environment, so that any packages you install, or any configurations changes you made, are restricted to within the environment, instead of having these applied globally. The following command will create a virtual environment `harry-potter-rag`:

```
python -m venv harry-potter-rag
```

You can then activate the virtual environment using the following command:

```
source tutorial-env/bin/activate
```

Next, use `pip` to install the relevant packages, mainly from `LangChain`:

```
%pip install --upgrade --quiet  langchain langchain-community langchain-openai langchain-experimental neo4j
```

### Setting up Neo4J

We will set up Neo4J as a docker container. For ease of setting up with specific configurations, we use docker compose. You may simply copy the following into a file called `docker-compose.yaml`, and then run `docker-compose up -d` in the same directory to set up Neo4J. 

This setup also ensures data, logs and plugins are persisted in local folders, i.e. `/data`. `/logs` and `plugins`.

```
services:
  neo4j:
    container_name: neo4j
    image: neo4j:5.20
    ports:
      - 7474:7474
      - 7687:7687
    environment:
      - NEO4J_AUTH=none
      - NEO4J_apoc_export_file_enabled=true
      - NEO4J_apoc_import_file_enabled=true
      - NEO4J_apoc_import_file_use__neo4j__config=true
      - NEO4J_dbms_security_procedures_unrestricted=custom.pregel.proc.*,pregel.*,apoc.*,gds.*
      - NEO4J_dbms_security_procedures_allowlist=custom.pregel.proc.*,pregel.*,apoc.*,gds.*
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
    volumes:
      - ./neo4j_db/data:/data
      - ./neo4j_db/logs:/logs
      - ./neo4j_db/import:/var/lib/neo4j/import
      - ./neo4j_db/plugins:/plugins

```


## Building the Knowledge Graph

We can now start building the Knowledge Graph in Jupyter Notebook! We first set up an Ollama LLM instance using the following:

```
llm = OllamaLLM(
    base_url="http://host.docker.internal:11434", # env var
    model="qwen2.5:latest", # env var
    system="you are an expert AI agent",
    verbose=True
    )
```

Next, we connect our LLM to Neo4J:

```
import os

from langchain_community.graphs import Neo4jGraph

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

graph = Neo4jGraph()
```

Now, it is time to grab your favourite Harry Potter text, or any favourite book, and we will use LangChain to split the text into chunks. Chunking is a strategy to break down a long text into parts, and we can then send each part to the LLM to convert them into nodes and edges, and insert each chunk's nodes and edges in Neo4J. Just a quick primer, nodes are circles you see on a graph, and each edge joins two nodes together.

The code also prints the first chunk for a quick preview of how the chunks look like.

```
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load example document
with open("harry_potter.txt") as f:
    book = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([book])
print(texts[0])
```

Now, it is time to let our GPU do the heavy lifting and convert out text into Knowledge Graph! Before we dive deep into the entire book, let us experiment with prompts to better guide the LLM in returning a graph in the way we want.

Prompts are essentially examples of what we expect, or instructions of what we want to appear in the response. In the context of knowledge graphs, we can instruct the LLM to only extract `persons` and `organisations` as nodes, and to only accept certain types of relationships given the entities. For example, we can allow the relationship of `spouse` to only happen between a `person` and another `person`, and not between a `person` and an `organisation`.

We can now employ the `LLMGraphTransformer` on the first chunk of text to see how the graph could turn out. This is a good chance for us to tweak the prompt until the result is to our liking.

The following example expects nodes which could be a `Person` or `Organization`, and the `allowed_relationships` specify the types of relationships that are allowed. In order to allow LLM to capture the variety of the original text, I also set `strict_mode` to False, so that any other relationships or entities which are not defined below can also be captured. If you instead set `strict_mode` to True, entities and relationships that do not comply with what is allowed could be either dropped, or forced into what is allowed (which may be inaccurate).

```
llm_transformer_filtered = LLMGraphTransformer(
   llm=llm,
   allowed_nodes=["Person", "Organization"],
   allowed_relationships=[
       ("Person", "WORKED_WITH", "Person"),
       ("Person", "WORKED_IN", "Organization"),
       ("Person", "FRIEND", "Person"),
       ("Person", "SPOUSE", "Person"),
       ("Person", "OWNED", "Organization")
   ],
   strict_mode=False
)
documents = [texts[0]]
graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(
documents
)
print(f"document:{[texts[0]]}")
print(f"Nodes:{graph_documents_filtered[0].nodes}")
print(f"Relationships:{graph_documents_filtered[0].relationships}")
```

After you are satisfied with fine-tuning your prompt, it is now time to ingest into a Knowledge Graph. Note that the `try` - `except` is to explicitly handle any response that could not be properly inserted into Neo4J -- the code is designed so that any error is logged, but does not block the loop from moving on with converting subsequent chunks into graph.

```
for i in range(len(texts)):
    try:
        llm_transformer_filtered = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Person", "Organization"],
        allowed_relationships=[
            ("Person", "WORKED_WITH", "Person"),
            ("Person", "WORKED_IN", "Organization"),
            ("Person", "FRIEND", "Person"),
            ("Person", "SPOUSE", "Person"),
            ("Person", "OWNED", "Organization")
        ],
        strict_mode=False
        )
        documents = [texts[i]]
        print("before converting")
        graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(
        documents
        )
        print("after converting")
        graph.add_graph_documents(graph_documents_filtered, include_source=True)
        print(f"Nodes:{graph_documents_filtered[0].nodes}")
        print(f"Relationships:{graph_documents_filtered[0].relationships}")
    except Exception as e:
        print("EXCEPTION!")
        print(type(e))
        print(f"document:{[texts[i]]}")
        continue
    else:
        continue
```

The loop above took me about 46 minutes to ingest Harry Potter and the Philosopher's Stone, Harry Potter and the Chamber of Secrets, and Harry Potter and the Prisoner of Azkaban. I end up with 4868 unique nodes! A quick preview is available below. You can see that the graph is really crowded, and and it is hard to distinguish who is related to who else, and in what way.

![Preview of the Knowledge Graph](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/sfo9n72m73mbu5qvrtht.png)

We can now leverage on cypher queries to look at say, Dumbledore!

```
MATCH (person:Person WHERE person.id = 'Dumbledore')
RETURN person
```

![Dumbledore as a node](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/cd9xzubwsrsybaua0esk.png)

Ok now we get just Dumbledore himself. Let's see how he is related to Harry Potter.

```
MATCH (person:Person WHERE person.id = 'Dumbledore')--(character:Person)
MATCH (character WHERE character.id = 'Harry')
RETURN person, character
```

![Dumbledore and Harry's relationships](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/2tzowwwxsi3lac758q05.png)

Ok, now we are interested in what Harry and Dumbledore have spoked.

```
MATCH (person:Person WHERE person.id = 'Dumbledore')-[rel:SPOKE_TO]-(character:Person)
MATCH (character WHERE character.id = 'Harry')
MATCH (person)-[:MENTIONS]-(doc:Document)
RETURN person, character, doc
```

![Text which mentioned Dumbledore and Harry](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/hzf64lmhsaj0lkqlgnd6.png)

We can see that the graph is still really confusing, with many documents to go through to really find what we are looking for. We can see that the modelling of documents as nodes is not ideal, and further work could be done on the `LLMGraphTransformer` to make the graph more intuitive to use.

## Conclusion

You can see how easy it is to set up a Knowledge Graph on your own local computer, without even needing to connect to the internet. 