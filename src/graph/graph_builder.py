from typing import List, Dict, Any
import networkx as nx
from itertools import combinations


class KnowledgeGraphBuilder:
    """
    Builds a graph: nodes = entities, edges = relationships based on co-occurrence.
    """

    def __init__(self) -> None:
        self.graph = nx.Graph()

    def build_graph(self, chunks: List[Dict[str, Any]]) -> nx.Graph:
        for ch in chunks:
            entities = ch.get("entities") or []
            # add nodes
            for ent in entities:
                ent_norm = ent.strip()
                if not ent_norm:
                    continue
                if ent_norm not in self.graph:
                    self.graph.add_node(
                        ent_norm,
                        type="entity",
                        mentions=1,
                    )
                else:
                    self.graph.nodes[ent_norm]["mentions"] += 1

            # add co-occurrence edges
            for a, b in combinations(set(entities), 2):
                a_norm, b_norm = a.strip(), b.strip()
                if not a_norm or not b_norm:
                    continue
                if self.graph.has_edge(a_norm, b_norm):
                    self.graph[a_norm][b_norm]["weight"] += 1
                else:
                    self.graph.add_edge(a_norm, b_norm, weight=1)

        return self.graph
