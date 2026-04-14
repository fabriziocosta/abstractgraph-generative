from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from abstractgraph_generative.edge_generator import (
    EdgeGenerator,
    _OnlineGraphRegressorAdapter,
    remove_edges,
)


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

    def _transform_raw(self, graphs):
        return np.asarray([[graph.number_of_nodes(), graph.number_of_edges()] for graph in graphs], dtype=float)


class _RecordingRiskEstimator:
    def __init__(self):
        self.fit_calls = []

    def fit(self, graphs, targets):
        self.fit_calls.append((list(graphs), list(targets)))
        return self

    def predict(self, graphs):
        return np.asarray([0.25] * len(graphs), dtype=float)


class _NativePartialFitRiskEstimator:
    def __init__(self):
        self.partial_fit_calls = []

    def partial_fit(self, graphs, targets):
        self.partial_fit_calls.append((list(graphs), list(targets)))
        return self

    def predict(self, graphs):
        return np.asarray([0.5] * len(graphs), dtype=float)


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

    def fake_remove_edges(graph, *, size, rng=None):
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


def test_infer_pair_target_returns_mean_in_regression_mode() -> None:
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
        target_estimator_mode="regression",
    )

    assert generator._infer_pair_target(2, 6) == 4.0


def test_infer_pair_target_samples_endpoint_target_in_classification_mode() -> None:
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
        target_estimator_mode="classification",
        seed=0,
    )

    assert generator._infer_pair_target(0, 1) == 1
    assert generator._infer_pair_target(0, 1) == 1


def test_log_repair_training_context_draws_query_and_neighbors_on_separate_rows() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())
    draw_calls = []

    def fake_draw(graphs, **kwargs):
        draw_calls.append((graphs, kwargs))

    repair_context = {
        "query_index": 3,
        "graph": nx.path_graph(3),
        "fit_graphs": [nx.path_graph(2), nx.path_graph(4)],
        "neighbor_indices": [1, 7],
        "neighbor_distances": [0.1, 0.2],
    }

    generator._log_repair_training_context(
        repair_context,
        draw_graphs_fn=fake_draw,
        verbose=True,
    )

    assert len(draw_calls) == 2
    assert draw_calls[0][1] == {"n_graphs_per_line": 1, "titles": ["query"]}
    assert len(draw_calls[0][0]) == 1
    assert draw_calls[1][1] == {
        "n_graphs_per_line": 2,
        "titles": ["nn:1", "nn:7"],
    }
    assert len(draw_calls[1][0]) == 2


def test_remove_edges_is_deterministic_with_seed() -> None:
    graph = nx.cycle_graph(6)

    pruned_a, target_a = remove_edges(graph, size=0.5, seed=13)
    pruned_b, target_b = remove_edges(graph, size=0.5, seed=13)

    assert target_a == graph.number_of_edges()
    assert target_b == graph.number_of_edges()
    assert sorted(pruned_a.edges()) == sorted(pruned_b.edges())


class _EstimatorWithDirectClasses:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, graphs):
        return np.tile(np.asarray([[0.25, 0.75]]), (len(graphs), 1))


def test_class_probability_supports_estimators_exposing_classes_directly() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())

    probs = generator._class_probability(
        _EstimatorWithDirectClasses(),
        [nx.path_graph(2)],
        target=1,
        estimator_name="graph_estimator",
    )

    assert probs.tolist() == [0.75]


def test_online_graph_regressor_adapter_replays_full_fit_when_partial_fit_is_missing() -> None:
    estimator = _RecordingRiskEstimator()
    adapter = _OnlineGraphRegressorAdapter(estimator)

    graphs_a = [nx.path_graph(2)]
    graphs_b = [nx.path_graph(3)]
    adapter.partial_fit(graphs_a, [0.2])
    adapter.partial_fit(graphs_b, [0.8])

    assert adapter.replay_targets_ == [0.2, 0.8]
    assert adapter.training_set_size() == 2
    assert len(adapter.estimator_.fit_calls) == 1
    _, second_targets = adapter.estimator_.fit_calls[0]
    assert second_targets == [0.2, 0.8]
    assert adapter.predict([nx.path_graph(4)]).tolist() == [0.25]


def test_online_graph_regressor_adapter_uses_native_partial_fit_when_available() -> None:
    estimator = _NativePartialFitRiskEstimator()
    adapter = _OnlineGraphRegressorAdapter(estimator)

    adapter.partial_fit([nx.path_graph(2)], [0.3])

    assert adapter.training_set_size() == 1
    assert len(adapter.estimator_.partial_fit_calls) == 1
    assert adapter.estimator_.partial_fit_calls[0][1] == [0.3]
    assert adapter.last_fit_time_seconds() >= 0.0
    assert adapter.predict([nx.path_graph(3)]).tolist() == [0.5]


def test_rollback_search_without_repair_logs_edge_risk_training_set_size(capsys) -> None:
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
        edge_risk_estimator=_RecordingRiskEstimator(),
        edge_risk_lambda=0.25,
        max_restarts=4,
    )
    generator.edge_risk_model_.n_training_examples_ = 12
    generator.edge_risk_model_.last_fit_time_seconds_ = 0.25
    root = generator._make_state(nx.path_graph(2), parent=None, score=1.0, depth=0)
    search = {
        "beam_history": [[root]],
        "beam": [root],
        "depth": 0,
        "visited": {root["key"]},
        "fallback_index": 1,
        "step_start_time": 0.0,
    }

    generator._rollback_search_without_repair(
        search,
        rollback_steps=3,
        beam_limit=5,
        n_fallbacks=4,
        total_phases=5,
        graph_index=0,
        verbose=True,
    )

    out = capsys.readouterr().out
    assert "edge_risk_training_set_size=12" in out
    assert "edge_risk_fit_time=0m 0.2s" in out


def test_make_edge_risk_graph_pair_is_disjoint_and_preserves_attributes() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())
    parent = nx.Graph()
    parent.add_node(0, role="parent")
    parent.add_node(1, role="parent")
    parent.add_edge(0, 1, label="p")
    child = nx.Graph()
    child.add_node("a", role="child")
    child.add_node("b", role="child")
    child.add_edge("a", "b", label="c")

    pair_graph = generator._make_edge_risk_graph_pair(parent, child)

    assert pair_graph.number_of_nodes() == 4
    assert pair_graph.number_of_edges() == 2
    assert sorted(pair_graph.edges(data="label")) == [(0, 1, "p"), (2, 3, "c")]


def test_close_edge_risk_training_states_uses_infeasible_descendant_ratio() -> None:
    risk_estimator = _RecordingRiskEstimator()
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
        edge_risk_estimator=risk_estimator,
    )
    generator._reset_edge_risk_attempt_trace()
    root = generator._make_state(nx.path_graph(2), parent=None, score=1.0, depth=0)
    decision = generator._make_state(nx.path_graph(3), parent=root, score=0.9, depth=1)
    infeasible = generator._make_state(nx.path_graph(4), parent=decision, score=0.0, depth=2)
    feasible = generator._make_state(nx.path_graph(3), parent=decision, score=0.8, depth=2)

    generator._mark_trace_state_status(decision, "retained")
    generator._mark_trace_state_status(infeasible, "partial_infeasible")
    generator._mark_trace_state_status(feasible, "pruned")
    generator._close_edge_risk_training_states(open_state_ids=set())

    assert len(generator.edge_risk_model_.estimator_.fit_calls) == 1
    fit_graphs, fit_targets = generator.edge_risk_model_.estimator_.fit_calls[0]
    assert len(fit_graphs) == 3
    assert fit_targets == [1.0 / 3.0, 1.0, 0.0]


def test_partition_candidates_by_feasibility_applies_edge_risk_penalty() -> None:
    class _AlwaysFeasibleEstimator:
        def predict(self, graphs):
            return np.ones(len(graphs), dtype=bool)

    generator = EdgeGenerator(
        feasibility_estimator=_AlwaysFeasibleEstimator(),
        partial_feasibility_estimator=_AlwaysFeasibleEstimator(),
        final_feasibility_estimator=_AlwaysFeasibleEstimator(),
        graph_estimator=object(),
        edge_risk_estimator=_RecordingRiskEstimator(),
        edge_risk_lambda=2.0,
    )
    generator._reset_edge_risk_attempt_trace()
    generator.edge_risk_model_.is_fitted_ = True
    generator.edge_risk_model_.predict = lambda graphs: np.asarray([0.4], dtype=float)
    generator._positive_scores = lambda graphs: np.asarray([0.9], dtype=float)
    generator._target_scores = lambda graphs, *, target: np.asarray([0.2], dtype=float)

    root = generator._make_state(nx.path_graph(2), parent=None, score=1.0, depth=0)
    cand = generator._make_state(nx.path_graph(3), parent=root, score=None, depth=1)
    feasible_candidates = []
    infeasible_candidates = []

    generator._partition_candidates_by_feasibility(
        [cand],
        n_edges=5,
        target=7,
        target_lambda=0.5,
        feasible_candidates=feasible_candidates,
        infeasible_candidates=infeasible_candidates,
    )

    assert len(feasible_candidates) == 1
    assert infeasible_candidates == []
    assert feasible_candidates[0]["risk_score"] == pytest.approx(0.4)
    assert feasible_candidates[0]["selection_score"] == pytest.approx(0.2)
