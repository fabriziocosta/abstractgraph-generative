from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from abstractgraph_generative.dataset_selection import select_graphs_via_shortest_paths


class _PositionVectorizer:
    def fit_transform(self, graphs):
        return np.asarray([[float(graph.graph["position"])] for graph in graphs])


def test_shortest_path_selection_removes_used_route_edges():
    graphs = []
    for position in (0.0, 2.0, 1.0):
        graph = nx.Graph()
        graph.add_node(0)
        graph.graph["position"] = position
        graphs.append(graph)

    result = select_graphs_via_shortest_paths(
        graphs,
        vectorizer=_PositionVectorizer(),
        source_idx=0,
        dest_idx=1,
        n_paths=2,
    )

    assert result["paths"] == [[0, 1], [0, 2, 1]]
    assert result["selected_indices"] == [0, 1, 2]
    assert result["distance_matrix"][0, 1] == 2.0


def test_shortest_path_selection_rejects_same_endpoints():
    with pytest.raises(ValueError, match="different"):
        select_graphs_via_shortest_paths(
            [nx.path_graph(1), nx.path_graph(2)],
            vectorizer=_PositionVectorizer(),
            source_idx=0,
            dest_idx=0,
        )
