from typing import Dict, Any, List
import networkx as nx

from ..llm.llm_client import LLMClient


class CommunitySummarizer:
    """
    Generates LLM-based summaries for each community.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def summarize_communities(
        self,
        graph: nx.Graph,
        partition: Dict[Any, int],
    ) -> Dict[int, Dict[str, Any]]:
        communities: Dict[int, List[str]] = {}
        for node, cid in partition.items():
            communities.setdefault(cid, []).append(node)

        summaries: Dict[int, Dict[str, Any]] = {}
        for cid, nodes in communities.items():
            prompt = (
                "You are summarizing a community of entities extracted from "
                "Dr. B.R. Ambedkar's works.\n\n"
                f"Entities:\n- " + "\n- ".join(nodes) + "\n\n"
                "Provide a concise thematic summary (3-5 sentences)."
            )
            summary_text = self.llm.generate(prompt)
            summaries[cid] = {
                "id": cid,
                "nodes": nodes,
                "summary": summary_text,
            }

        return summaries
