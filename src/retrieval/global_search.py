from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class GlobalGraphRAGSearcher:
    """
    Implements Equation (5) from SemRAG in a simplified way:
    - We treat each community summary as a point.
    - We rank community summaries by similarity to the query and take top-K.
    """

    def __init__(
        self,
        community_summaries: Dict[int, Dict[str, Any]],
        community_ids: List[int],
        community_embeddings: np.ndarray,
        top_k: int,
    ) -> None:
        self.community_summaries = community_summaries
        self.community_ids = community_ids
        self.community_embeddings = community_embeddings
        self.top_k = top_k

    def search(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        query_embedding = query_embedding.reshape(1, -1)
        sims = cosine_similarity(query_embedding, self.community_embeddings)[0]
        scored: List[Tuple[int, float]] = list(zip(self.community_ids, sims))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.top_k]
        return [self.community_summaries[cid] for cid, _ in top]
