from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from abstractgraph_generative.edge_generator import EdgeGenerator


class _RecordingFeasibilityEstimator:
    def __init__(self, label: str):
        self.label = label
        self.fit_sizes = []

    def fit(self, graphs):
        self.fit_sizes.append(len(graphs))
        return self

    def predict(self, graphs):
        return np.ones(len(graphs), dtype=bool)

    def number_of_violations(self, graphs):
        return np.zeros(len(graphs), dtype=int)


class _RecordingGraphEstimator:
    def fit(self, graphs, targets):
        self.fit_size = len(graphs)
        self.targets = np.asarray(targets)
        return self


def test_augment_indices_with_nearest_neighbors_adds_k_per_seed_without_duplicates() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())
    distance_matrix = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 0.0, 1.5, 2.5, 3.5],
            [2.0, 1.5, 0.0, 1.0, 2.0],
            [3.0, 2.5, 1.0, 0.0, 1.0],
            [4.0, 3.5, 2.0, 1.0, 0.0],
        ]
    )

    selected = generator._augment_indices_with_nearest_neighbors(
        distance_matrix,
        [0, 3],
        k=2,
    )

    assert selected == [0, 1, 2, 3, 4]


def test_augment_indices_with_nearest_neighbors_can_be_disabled() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())
    distance_matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]])

    selected = generator._augment_indices_with_nearest_neighbors(
        distance_matrix,
        [1, 1, 0],
        k=0,
    )

    assert selected == [1, 0]


def test_augment_indices_with_nearest_neighbors_rejects_negative_k() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())

    with pytest.raises(ValueError, match="n_neighbors_per_path_graph must be >= 0"):
        generator._augment_indices_with_nearest_neighbors(np.zeros((1, 1)), [0], k=-1)


def test_path_matrix_from_distance_matrix_uses_sparse_mst_knn_graph() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())
    distance_matrix = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )

    path_matrix = generator._path_matrix_from_distance_matrix(distance_matrix, k=1)

    assert np.isinf(path_matrix[0, 3])
    assert path_matrix[0, 1] == 1.0
    assert path_matrix[1, 2] == 1.0
    assert path_matrix[2, 3] == 1.0


def test_select_edges_for_surgical_repair_prioritizes_repeated_violations() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())
    graph = nx.path_graph(4)
    state = generator._make_state(
        graph,
        parent=None,
        score=1.0,
        depth=3,
        edge_order=((0, 1), (1, 2), (2, 3)),
    )
    candidates = [
        {
            "selection_score": 0.9,
            "score": 0.9,
            "violating_edge_sets": [frozenset({(1, 2), (2, 3)})],
        },
        {
            "selection_score": 0.8,
            "score": 0.8,
            "violating_edge_sets": [frozenset({(1, 2)})],
        },
    ]

    removed_edges, repair_score = generator._select_edges_for_surgical_repair(
        state,
        candidates,
        rollback_steps=2,
    )

    assert removed_edges == [(1, 2), (2, 3)]
    assert repair_score > 0.0


def test_make_repaired_state_removes_selected_edges_and_updates_depth() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())
    graph = nx.path_graph(4)
    state = generator._make_state(
        graph,
        parent=None,
        score=1.0,
        depth=3,
        edge_order=((0, 1), (1, 2), (2, 3)),
    )

    repaired_state = generator._make_repaired_state(
        state,
        [(1, 2), (2, 3)],
        score=2.5,
    )

    assert sorted(repaired_state["graph"].edges()) == [(0, 1)]
    assert repaired_state["depth"] == 1
    assert repaired_state["edge_order"] == ((0, 1),)
    assert repaired_state["repair_removed_edges"] == ((1, 2), (2, 3))
    assert repaired_state["parent"] is state


def test_select_edges_for_surgical_repair_requires_violation_evidence() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())
    graph = nx.path_graph(4)
    state = generator._make_state(
        graph,
        parent=None,
        score=1.0,
        depth=3,
        edge_order=((0, 1), (1, 2), (2, 3)),
    )

    removed_edges, repair_score = generator._select_edges_for_surgical_repair(
        state,
        [{"selection_score": 0.9, "score": 0.9, "violating_edge_sets": []}],
        rollback_steps=2,
    )

    assert removed_edges == []
    assert repair_score == 0.0


def test_fit_uses_partial_and_final_feasibility_estimators_on_different_graph_sets() -> None:
    partial_estimator = _RecordingFeasibilityEstimator("partial")
    final_estimator = _RecordingFeasibilityEstimator("final")
    graph_estimator = _RecordingGraphEstimator()
    generator = EdgeGenerator(
        feasibility_estimator=partial_estimator,
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=graph_estimator,
        n_negative_per_positive=1,
        n_replicates=1,
    )
    graph = nx.path_graph(3)

    generator.fit([graph])

    assert partial_estimator.fit_sizes == [2]
    assert final_estimator.fit_sizes == [1]
    assert graph_estimator.fit_size > 0


def test_generate_from_pair_none_none_requires_cached_session() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())

    with pytest.raises(ValueError, match="No cached pair session is available"):
        generator.generate_from_pair(None, None)


def test_generate_from_cached_pair_session_reuses_cached_graphs_and_target(monkeypatch) -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object(), seed=0)
    graph_a = nx.path_graph(3)
    graph_b = nx.path_graph(4)
    generator._cache_pair_session(
        graph_a=graph_a,
        graph_b=graph_b,
        size_of_edge_removal=0.5,
        target=7,
    )

    calls = {"remove_edges": 0, "mix": 0, "generate": 0}

    def fake_remove_edges(graph, *, size):
        calls["remove_edges"] += 1
        return graph.copy(), graph.number_of_edges() + 2

    def fake_mix_connected_components(graph1, graph2, *, seed):
        calls["mix"] += 1
        out = nx.compose(graph1, graph2)
        out.graph["seed"] = seed
        return out

    def fake_generate(graph, n_edges, *, target, target_lambda, return_path, draw_graphs_fn, verbose):
        calls["generate"] += 1
        return {
            "n_edges": n_edges,
            "target": target,
            "target_lambda": target_lambda,
            "return_path": return_path,
            "verbose": verbose,
            "n_graph_edges": graph.number_of_edges(),
        }

    monkeypatch.setattr(
        "abstractgraph_generative.edge_generator.remove_edges",
        fake_remove_edges,
    )
    monkeypatch.setattr(
        "abstractgraph_generative.edge_generator.mix_connected_components",
        fake_mix_connected_components,
    )
    monkeypatch.setattr(generator, "generate", fake_generate)

    result = generator.generate_from_pair(
        None,
        None,
        target_lambda=0.25,
        return_path=False,
        verbose=False,
    )

    assert calls == {"remove_edges": 2, "mix": 1, "generate": 1}
    assert result["n_edges"] == 4
    assert result["target"] == 7
    assert result["target_lambda"] == 0.25
    assert result["return_path"] is False
