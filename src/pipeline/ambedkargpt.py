import json
import pickle
from pathlib import Path

import yaml
import networkx as nx

from chunking.semantic_chunker import SemanticChunker
from graph.entity_extractor import EntityExtractor
from graph.graph_builder import KnowledgeGraphBuilder
from graph.community_detector import CommunityDetector
from graph.summarizer import CommunitySummarizer
from llm.llm_client import LLMClient
from llm.answer_generator import AnswerGenerator
from retrieval.embeddings_store import EmbeddingStore
from retrieval.local_search import LocalGraphRAGSearcher
from retrieval.global_search import GlobalGraphRAGSearcher


def load_config(path: str = "config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_chunks_if_needed(cfg):
    chunks_path = Path(cfg["paths"]["chunks_path"])
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    chunker = SemanticChunker(
        model_name=cfg["chunking"]["embedding_model"],
        buffer_size=cfg["chunking"]["buffer_size"],
        cosine_threshold=cfg["chunking"]["cosine_threshold"],
        max_tokens_chunk=cfg["chunking"]["max_tokens_chunk"],
        subchunk_tokens=cfg["chunking"]["subchunk_tokens"],
    )
    chunks = chunker.chunk_document(cfg["paths"]["pdf_path"])
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunker.save_chunks(chunks, str(chunks_path))
    return chunks


def build_graph_if_needed(cfg, chunks):
    graph_path = Path(cfg["paths"]["graph_path"])
    if graph_path.exists():
        with open(graph_path, "rb") as f:
            return pickle.load(f)

    extractor = EntityExtractor()
    enriched_chunks = extractor.extract_entities(chunks)

    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(enriched_chunks)

    with open(graph_path, "wb") as f:
        pickle.dump({"graph": graph, "chunks": enriched_chunks}, f)

    return {"graph": graph, "chunks": enriched_chunks}


def main():
    cfg = load_config()

    # 1. Chunking
    chunks = build_chunks_if_needed(cfg)

    # 2. Knowledge graph
    kg_data = build_graph_if_needed(cfg, chunks)
    graph: nx.Graph = kg_data["graph"]
    enriched_chunks = kg_data["chunks"]

    # 3. Community detection + summaries
    detector = CommunityDetector()
    partition = detector.detect(graph)

    llm_client = LLMClient(cfg["llm"]["model"])
    summarizer = CommunitySummarizer(llm_client)
    communities = summarizer.summarize_communities(graph, partition)

    # 4. Build embedding indexes
    emb_store = EmbeddingStore(cfg["chunking"]["embedding_model"])

    entity_labels = list(graph.nodes())
    entity_labels, entity_embs = emb_store.build_entity_index(entity_labels)
    chunks_idx, chunk_embs = emb_store.build_chunk_index(enriched_chunks)
    community_ids, community_embs = emb_store.build_community_index(communities)

    # 5. Prepare searchers
    local_searcher = LocalGraphRAGSearcher(
        entity_labels=entity_labels,
        entity_embeddings=entity_embs,
        chunks=chunks_idx,
        chunk_embeddings=chunk_embs,
        tau_entity=cfg["retrieval"]["entity_sim_threshold"],
        tau_chunk=cfg["retrieval"]["chunk_sim_threshold"],
        top_k=cfg["retrieval"]["top_k_local"],
    )

    global_searcher = GlobalGraphRAGSearcher(
        community_summaries=communities,
        community_ids=community_ids,
        community_embeddings=community_embs,
        top_k=cfg["retrieval"]["top_k_global"],
    )

    answer_gen = AnswerGenerator(
        llm_client=llm_client,
        max_tokens=cfg["llm"]["max_tokens"],
    )

    # 6. Simple CLI loop for demo
    print("AmbedkarGPT (SemRAG-style) ready. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        q_emb = emb_store.embed_texts([question])[0]

        retrieved_local = local_searcher.search(q_emb)
        retrieved_global = global_searcher.search(q_emb)

        answer = answer_gen.generate_answer(
            question, retrieved_local, retrieved_global
        )
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
