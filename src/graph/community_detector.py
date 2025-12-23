from typing import Dict, Any
import networkx as nx
import community as community_louvain  # python-louvain


class CommunityDetector:
    """
    Applies Louvain community detection on the knowledge graph.
    """

    def detect(self, graph: nx.Graph) -> Dict[Any, int]:
        if graph.number_of_nodes() == 0:
            return {}
        partition = community_louvain.best_partition(graph, weight="weight")
        # partition: node -> community_id
        return partition
