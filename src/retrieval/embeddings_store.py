from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingStore:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)

    def build_entity_index(
        self, graph_nodes: List[str]
    ) -> Tuple[List[str], np.ndarray]:
        embeddings = self.embed_texts(graph_nodes)
        return graph_nodes, embeddings

    def build_chunk_index(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_texts(texts)
        return chunks, embeddings

    def build_community_index(
        self, communities: Dict[int, Dict[str, Any]]
    ) -> Tuple[List[int], np.ndarray]:
        texts = [c["summary"] for c in communities.values()]
        ids = list(communities.keys())
        embeddings = self.embed_texts(texts)
        return ids, embeddings
