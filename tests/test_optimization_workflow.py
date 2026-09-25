from __future__ import annotations

import networkx as nx
import numpy as np

from abstractgraph_generative.optimize import GraphOptimizer


def test_optimizer_fit_caches_scores_and_clips_to_contract():
    calls = []

    def score(graphs):
        calls.append(len(graphs))
        return np.linspace(-0.5, 1.5, len(graphs))

    graphs = [nx.path_graph(2), nx.path_graph(3), nx.path_graph(4)]
    optimizer = GraphOptimizer(generator=object(), score_function=score).fit(graphs)

    assert np.array_equal(optimizer.scores, np.asarray([0.0, 0.5, 1.0]))
    assert optimizer.sorted_indices == [2, 1, 0]
    assert np.array_equal(optimizer.score_graphs(graphs), optimizer.scores)
    assert calls == [3]
