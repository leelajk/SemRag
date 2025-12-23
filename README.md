# AmbedkarGPT
SemRAG-Inspired Graph RAG System over Dr. B. R. Ambedkar’s Works

AmbedkarGPT is a Retrieval-Augmented Generation (RAG) system inspired by the SemRAG research paper.
It builds a graph-based semantic retrieval pipeline over Dr. B. R. Ambedkar’s writings and answers questions using a local LLM (e.g., Mistral via Ollama) in a grounded, context-aware manner.

All components run fully offline, making the system suitable for controlled demos and interviews.

1. Project Overview

The system closely follows the SemRAG architecture:

PDF → Semantic Chunking → Knowledge Graph
    → Local + Global Graph Retrieval → Local LLM Answer

Semantic Chunking (Algorithm 1 – SemRAG)

Splits the PDF into sentences.

Uses sentence embeddings + cosine similarity to group semantically related sentences.

Applies buffer-based merging to preserve local context.

Enforces token limits by splitting oversized chunks into overlapping sub-chunks.

Knowledge Graph Construction

Extracts named entities from chunks using spaCy NER.

Builds a NetworkX graph:

Nodes → entities

Edges → entity co-occurrence relationships

Applies Louvain community detection to identify thematic clusters.

Retrieval (SemRAG-Style)

Local Graph RAG (Equation 4)
Retrieves entity-linked chunks relevant to the query.

Global Graph RAG (Equation 5)
Retrieves top-K community summaries representing high-level themes.

LLM Integration

Combines local evidence + global summaries into a structured prompt.

Uses a local LLM via Ollama (Mistral / LLaMA3).

Generates answers strictly grounded in retrieved context.

2. Directory Structure
ambedkargpt/
│
├── data/
│   ├── Ambedkar_works.pdf
│   └── processed/
│       ├── chunks.json
│       └── knowledge_graph.pkl
│
├── src/
│   ├── chunking/
│   │   ├── semantic_chunker.py
│   │   └── buffer_merger.py
│   │
│   ├── graph/
│   │   ├── entity_extractor.py
│   │   ├── graph_builder.py
│   │   ├── community_detector.py
│   │   └── summarizer.py
│   │
│   ├── retrieval/
│   │   ├── local_search.py
│   │   ├── global_search.py
│   │   └── ranker.py
│   │
│   ├── llm/
│   │   ├── llm_client.py
│   │   ├── prompt_templates.py
│   │   └── answer_generator.py
│   │
│   └── pipeline/
│       └── ambedkargpt.py
│
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_integration.py
│
├── config.yaml
├── requirements.txt
├── setup.py
└── README.md

3. Installation
3.1 Python Environment

Requirements: Python 3.9+

python -m venv .venv

# Activate
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm


Ensure the dataset is placed at:

data/Ambedkar_works.pdf

3.2 Local LLM via Ollama

Install Ollama from:
https://ollama.com

Pull a model (example: Mistral):

ollama pull mistral


Update config.yaml:

llm:
  model: "mistral"
  max_tokens: 512


You may replace "mistral" with any locally available Ollama model (e.g., "llama3").

3.3 Python Package Setup

Add empty __init__.py files so src is importable:

src/
  __init__.py
  chunking/__init__.py
  graph/__init__.py
  retrieval/__init__.py
  llm/__init__.py
  pipeline/__init__.py

4. How to Run the System
Step 1 – Activate environment & set PYTHONPATH
# Activate venv
.venv\Scripts\Activate.ps1

# Windows (PowerShell)
$env:PYTHONPATH = "src"

Step 2 – Run the pipeline
python src/pipeline/ambedkargpt.py


On the first run, the system will:

Load and semantically chunk the PDF
Saves output to data/processed/chunks.json

Build the knowledge graph
Saves graph + enriched chunks to knowledge_graph.pkl

Detect communities & summarize them
Uses the local LLM to generate thematic summaries

Prepare retrieval indexes
Builds embeddings for chunks, entities, and communities

Start interactive Q&A

You will see:

AmbedkarGPT (SemRAG-style) ready.
Type 'exit' to quit.
Question:

Step 3 – Ask Questions

Example queries:

What does Ambedkar say about caste and social justice?

What are Ambedkar’s views on democracy?

How does Ambedkar discuss the Indian Constitution?

For each query:

Local Graph RAG retrieves entity-linked chunks

Global Graph RAG retrieves relevant communities

The LLM generates a grounded answer using retrieved context

Type exit to stop.

5. Configuration

All hyperparameters are defined in config.yaml:

chunking:
  embedding_model: "all-MiniLM-L6-v2"
  buffer_size: 3
  cosine_threshold: 0.25
  max_tokens_chunk: 1024
  subchunk_tokens: 128

graph:
  use_leiden: false
  min_community_size: 3

retrieval:
  top_k_local: 5
  top_k_global: 3
  entity_sim_threshold: 0.3
  chunk_sim_threshold: 0.25


These parameters control recall vs precision trade-offs as described in the SemRAG paper.

6. Testing (Optional)
pytest tests/


test_chunking.py – validates semantic chunking

test_retrieval.py – checks local/global retrieval

test_integration.py – lightweight end-to-end test
