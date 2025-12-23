AmbedkarGPT

A SemRAG-Inspired Graph RAG System over Dr. B. R. Ambedkar’s Works

AmbedkarGPT is a fully local Retrieval-Augmented Generation system built to answer questions about Dr. B. R. Ambedkar’s writings in a grounded and context-aware manner.

Instead of relying on plain vector search, the system follows ideas from the SemRAG research paper, combining semantic chunking, a knowledge graph, and graph-based retrieval before generating answers using a local LLM (via Ollama).

How AmbedkarGPT Works

At a high level, the system follows this flow:

<img width="381" height="284" alt="image" src="https://github.com/user-attachments/assets/ae8f3c27-5664-4811-bf1c-6056b739fb37" />


Semantic Chunking

The Ambedkar book is first split into sentences.
Sentence embeddings and cosine similarity are used to merge semantically related sentences into coherent chunks, following Algorithm 1 from SemRAG.

A small context buffer ensures continuity, and oversized chunks are split into overlapping sub-chunks to respect token limits.

Knowledge Graph

Each semantic chunk is processed using spaCy NER to extract entities such as people, institutions, concepts, and legal terms.

A knowledge graph is built where:

Nodes represent entities

Edges represent co-occurrence within the same chunk

To capture higher-level themes, Louvain community detection is applied, grouping related entities into thematic communities. Each community is summarized using the local LLM.

Graph-Based Retrieval

When a question is asked, retrieval happens at two levels:

Local Graph Retrieval (SemRAG Equation 4)
Finds entity-linked chunks that are most relevant to the query.

Global Graph Retrieval (SemRAG Equation 5)
Retrieves summaries of the most relevant communities to provide broader thematic context.

This combination ensures both precision and contextual completeness.

Answer Generation

The retrieved local chunks and global community summaries are combined into a structured prompt.

A local LLM (Mistral / LLaMA3 via Ollama) generates the final answer strictly grounded in retrieved evidence, avoiding hallucination and external knowledge.
