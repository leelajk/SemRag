from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class LocalGraphRAGSearcher:
    """
    Implements Equation (4) from SemRAG:
    D_retrieved = Top_k({ v ∈ V, g ∈ G | sim(v, Q+H) > τ_e ∧ sim(g, v) > τ_d })
    Here we approximate:
    - v: entity embedding
    - g: chunk embedding
    """

    def __init__(
        self,
        entity_labels: List[str],
        entity_embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
        chunk_embeddings: np.ndarray,
        tau_entity: float,
        tau_chunk: float,
        top_k: int,
    ) -> None:
        self.entity_labels = entity_labels
        self.entity_embeddings = entity_embeddings
        self.chunks = chunks
        self.chunk_embeddings = chunk_embeddings
        self.tau_entity = tau_entity
        self.tau_chunk = tau_chunk
        self.top_k = top_k

    def search(
        self, query_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        query_embedding = query_embedding.reshape(1, -1)

        # similarity between query and entities
        ent_sims = cosine_similarity(query_embedding, self.entity_embeddings)[0]

        candidate_entities = [
            (label, score)
            for label, score in zip(self.entity_labels, ent_sims)
            if score >= self.tau_entity
        ]

        if not candidate_entities:
            return []

        # now find chunks related to those entities
        scores: List[Tuple[int, float]] = []
        for idx, ch in enumerate(self.chunks):
            ents = ch.get("entities", [])
            # simple heuristic: overlap with candidate entity labels
            overlap = any(e in dict(candidate_entities) for e in ents)
            if not overlap:
                continue

            ch_emb = self.chunk_embeddings[idx].reshape(1, -1)
            sim = cosine_similarity(query_embedding, ch_emb)[0][0]
            if sim >= self.tau_chunk:
                scores.append((idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[: self.top_k]
        return [self.chunks[i] for i, _ in top]
