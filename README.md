AmbedkarGPT

A SemRAG-Inspired Graph RAG System over Dr. B. R. Ambedkar’s Works

AmbedkarGPT is a fully local Retrieval-Augmented Generation system built to answer questions about Dr. B. R. Ambedkar’s writings in a grounded and context-aware manner.

Instead of relying on plain vector search, the system follows ideas from the SemRAG research paper, combining semantic chunking, a knowledge graph, and graph-based retrieval before generating answers using a local LLM (via Ollama).

How AmbedkarGPT Works

At a high level, the system follows this flow:

<img width="381" height="284" alt="image" src="https://github.com/user-attachments/assets/ae8f3c27-5664-4811-bf1c-6056b739fb37" />


# Semantic Chunking

The Ambedkar book is first split into sentences.
Sentence embeddings and cosine similarity are used to merge semantically related sentences into coherent chunks, following Algorithm 1 from SemRAG.

A small context buffer ensures continuity, and oversized chunks are split into overlapping sub-chunks to respect token limits.

# Knowledge Graph

Each semantic chunk is processed using spaCy NER to extract entities such as people, institutions, concepts, and legal terms.

A knowledge graph is built where:

Nodes represent entities

Edges represent co-occurrence within the same chunk

To capture higher-level themes, Louvain community detection is applied, grouping related entities into thematic communities. Each community is summarized using the local LLM.

# Graph-Based Retrieval

When a question is asked, retrieval happens at two levels:

Local Graph Retrieval (SemRAG Equation 4)
Finds entity-linked chunks that are most relevant to the query.

Global Graph Retrieval (SemRAG Equation 5)
Retrieves summaries of the most relevant communities to provide broader thematic context.

This combination ensures both precision and contextual completeness.

# Answer Generation

The retrieved local chunks and global community summaries are combined into a structured prompt.

A local LLM (Mistral / LLaMA3 via Ollama) generates the final answer strictly grounded in retrieved evidence, avoiding hallucination and external knowledge.

# Project Layout
<img width="733" height="502" alt="image" src="https://github.com/user-attachments/assets/8e2b1fc3-17d2-4daa-a76e-41ef46791548" />

Getting It Running
Python setup
python -m venv .venv

# Activate
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m spacy download en_core_web_sm


Place the book at:

data/Ambedkar_works.pdf

Local LLM (Ollama)

Install Ollama from https://ollama.com

Then pull a model:

ollama pull mistral


Set the model in config.yaml:

llm:
  model: "mistral"
  max_tokens: 512

Run AmbedkarGPT
# Make src importable
# Linux/macOS
export PYTHONPATH=src
# Windows (PowerShell)
$env:PYTHONPATH = "src"

python src/pipeline/ambedkargpt.py


You’ll enter an interactive mode:

AmbedkarGPT ready. Type 'exit' to quit.
Question:


Ask questions like:

- What were Ambedkar’s views on caste?
- How did Ambedkar define democracy?
- What role did Ambedkar play in drafting the Constitution?

# Why This Design

- Semantic chunking preserves meaning better than fixed-size splits

- Knowledge graphs capture relationships beyond vector similarity

- Local + global retrieval balances precision and thematic coverage

- Fully local execution ensures reproducibility and privacy

- This mirrors the design motivations discussed in the SemRAG paper.
