from __future__ import annotations

import networkx as nx
import numpy as np

from abstractgraph import node as node_operator
import abstractgraph_generative.repair as repair_module
from abstractgraph_generative.repair import RepairGenerator


class _SizeTransformer:
    def transform(self, graphs):
        return np.asarray([[g.number_of_nodes(), g.number_of_edges()] for g in graphs])


def test_repair_preserves_query_when_rewrite_has_no_candidates(monkeypatch):
    query = nx.path_graph(3)
    donor = nx.path_graph(4)
    monkeypatch.setattr(
        repair_module, "graphs_to_abstract_graphs", lambda *args, **kwargs: [object()]
    )
    monkeypatch.setattr(
        repair_module, "graph_to_abstract_graph", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(repair_module, "rewrite", lambda *args, **kwargs: [])

    generator = RepairGenerator(
        decomposition_function=node_operator(),
        nbits=6,
        graph_transformer=_SizeTransformer(),
    ).fit([donor])

    result = generator.repair(query)
    batch_result = generator.repair([query])
    assert nx.utils.graphs_equal(result, query)
    assert len(batch_result) == 1
    assert nx.utils.graphs_equal(batch_result[0], query)
