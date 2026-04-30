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
        self.fit_graphs = []

    def fit(self, graphs):
        self.fit_sizes.append(len(graphs))
        self.fit_graphs.append([graph.copy() for graph in graphs])
        return self

    def predict(self, graphs):
        return np.ones(len(graphs), dtype=bool)

    def number_of_violations(self, graphs):
        return np.zeros(len(graphs), dtype=int)


class _RejectEdgesFeasibilityEstimator(_RecordingFeasibilityEstimator):
    def __init__(self, rejected_edges):
        super().__init__("reject_edges")
        self.rejected_edges = {tuple(edge) for edge in rejected_edges}

    def predict(self, graphs):
        return np.asarray(
            [
                not any(graph.has_edge(*edge) for edge in self.rejected_edges)
                for graph in graphs
            ],
            dtype=bool,
        )

    def violating_edge_sets(self, graphs):
        return [
            [
                frozenset(
                    edge for edge in self.rejected_edges if graph.has_edge(*edge)
                )
            ]
            for graph in graphs
        ]


class _RejectNonEmptyFeasibilityEstimator(_RecordingFeasibilityEstimator):
    def __init__(self):
        super().__init__("reject_non_empty")

    def predict(self, graphs):
        return np.asarray([graph.number_of_edges() == 0 for graph in graphs], dtype=bool)


class _RecordingGraphEstimator:
    def fit(self, graphs, targets):
        self.fit_size = len(graphs)
        self.targets = np.asarray(targets)
        return self

    def _transform_raw(self, graphs):
        return np.asarray([[graph.number_of_nodes(), graph.number_of_edges()] for graph in graphs], dtype=float)


class _SimpleGraphTransformer:
    def fit_transform(self, graphs, y=None):
        return self.transform(graphs)

    def transform(self, graphs, y=None):
        return np.asarray(
            [[graph.number_of_nodes(), graph.number_of_edges()] for graph in graphs],
            dtype=float,
        )


class _GraphEstimatorWithTransformer:
    transformer = _SimpleGraphTransformer()


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


def _labeled_edge_graph(node_labels: list[str]) -> nx.Graph:
    graph = nx.Graph()
    for idx, label in enumerate(node_labels):
        graph.add_node(idx, label=label)
    for idx in range(len(node_labels) - 1):
        graph.add_edge(idx, idx + 1, label="single")
    return graph


def _directed_path() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(0, label="A")
    graph.add_node(1, label="B")
    graph.add_node(2, label="C")
    graph.add_edge(0, 1, label="x")
    graph.add_edge(1, 2, label="y")
    return graph


def _reversed_directed_path() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(0, label="A")
    graph.add_node(1, label="B")
    graph.add_node(2, label="C")
    graph.add_edge(1, 0, label="x")
    graph.add_edge(2, 1, label="y")
    return graph


def test_unique_graphs_keeps_directed_edge_orientation_distinct() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())
    graph = _directed_path()
    reversed_graph = _reversed_directed_path()

    unique_graphs = generator._unique_graphs([graph, graph.copy(), reversed_graph])

    assert len(unique_graphs) == 2
    assert all(unique_graph.is_directed() for unique_graph in unique_graphs)
    assert {tuple(sorted(unique_graph.edges())) for unique_graph in unique_graphs} == {
        ((0, 1), (1, 2)),
        ((1, 0), (2, 1)),
    }


def test_unique_graphs_keeps_node_labels_distinct() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())
    graph = _labeled_edge_graph(["C", "O"])
    relabeled_graph = _labeled_edge_graph(["C", "N"])

    unique_graphs = generator._unique_graphs([graph, graph.copy(), relabeled_graph])

    assert len(unique_graphs) == 2
    assert [tuple(label for _, label in unique_graph.nodes(data="label")) for unique_graph in unique_graphs] == [
        ("C", "O"),
        ("C", "N"),
    ]


def test_unique_graphs_is_node_id_permutation_invariant() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())
    graph = _directed_path()
    relabeled_graph = nx.relabel_nodes(graph, {0: "z", 1: "x", 2: "y"}, copy=True)

    unique_graphs = generator._unique_graphs([graph, relabeled_graph])

    assert len(unique_graphs) == 1


def test_store_keeps_reversed_directed_graphs_as_distinct_retrieval_entries() -> None:
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=_GraphEstimatorWithTransformer(),
    )

    generator.store([_directed_path(), _reversed_directed_path()])

    assert len(generator.stored_graph_hash_to_index_) == 2


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


def test_repair_state_until_partial_feasible_removes_partial_violations() -> None:
    partial_estimator = _RejectEdgesFeasibilityEstimator([(1, 2)])
    final_estimator = _RecordingFeasibilityEstimator("final")
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=object(),
    )
    graph = nx.path_graph(4)
    state = generator._make_state(
        graph,
        parent=None,
        score=1.0,
        depth=3,
        edge_order=((0, 1), (1, 2), (2, 3)),
    )
    state["repair_removed_edges"] = ((0, 1),)

    repaired_state = generator._repair_state_until_partial_feasible(state)

    assert repaired_state is not None
    assert bool(partial_estimator.predict([repaired_state["graph"]])[0])
    assert not repaired_state["graph"].has_edge(1, 2)
    assert repaired_state["repair_removed_edges"] == ((0, 1), (1, 2))


def test_repair_state_until_partial_feasible_falls_back_to_edge_order() -> None:
    partial_estimator = _RejectNonEmptyFeasibilityEstimator()
    final_estimator = _RecordingFeasibilityEstimator("final")
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=object(),
    )
    graph = nx.path_graph(3)
    state = generator._make_state(
        graph,
        parent=None,
        score=1.0,
        depth=2,
        edge_order=((0, 1), (1, 2)),
    )

    repaired_state = generator._repair_state_until_partial_feasible(state)

    assert repaired_state is not None
    assert repaired_state["graph"].number_of_edges() == 0
    assert repaired_state["repair_removed_edges"] == ((1, 2), (0, 1))


def test_build_repair_start_states_removes_one_random_edge_when_final_violations_have_no_edges(monkeypatch) -> None:
    partial_estimator = _RecordingFeasibilityEstimator("partial")
    final_estimator = _RecordingFeasibilityEstimator("final")
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=object(),
        max_restarts=2,
        fallback_base_steps=1,
        seed=0,
    )
    graph = nx.path_graph(3)

    monkeypatch.setattr(generator, "_positive_scores", lambda graphs: np.asarray([0.5] * len(graphs)))
    monkeypatch.setattr(generator, "_target_scores", lambda graphs, *, target: np.zeros(len(graphs)))
    monkeypatch.setattr(
        generator,
        "_annotate_infeasible_candidates_with_violating_edge_sets",
        lambda candidates: [candidate.update({"violating_edge_sets": []}) for candidate in candidates],
    )

    repaired_states = generator._build_repair_start_states(
        graph,
        target=None,
        target_lambda=0.5,
    )

    assert repaired_states
    assert all(state["graph"].number_of_edges() == graph.number_of_edges() - 1 for state in repaired_states)
    assert all(len(state["repair_removed_edges"]) == 1 for state in repaired_states)


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


def test_fit_can_skip_feasibility_graph_deduplication(monkeypatch) -> None:
    partial_estimator = _RecordingFeasibilityEstimator("partial")
    final_estimator = _RecordingFeasibilityEstimator("final")
    graph_estimator = _RecordingGraphEstimator()
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=graph_estimator,
        n_negative_per_positive=1,
        n_replicates=1,
    )
    graph = nx.path_graph(3)

    def fail_unique_graphs(graphs):
        raise AssertionError("_unique_graphs should not be called")

    monkeypatch.setattr(generator, "_unique_graphs", fail_unique_graphs)
    generator.fit(
        [graph, graph.copy()],
        deduplicate_feasibility_graphs=False,
    )

    assert final_estimator.fit_sizes == [2]
    assert partial_estimator.fit_sizes[0] > final_estimator.fit_sizes[0]


def test_fit_adds_extra_graphs_only_to_partial_feasibility_estimator() -> None:
    partial_estimator = _RecordingFeasibilityEstimator("partial")
    final_estimator = _RecordingFeasibilityEstimator("final")
    graph_estimator = _RecordingGraphEstimator()
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=graph_estimator,
        n_negative_per_positive=1,
        n_replicates=1,
    )
    graph = nx.path_graph(3)
    extra_graph = nx.empty_graph(3)

    generator.fit([graph], partial_feasibility_extra_graphs=[extra_graph])

    assert partial_estimator.fit_sizes == [3]
    assert final_estimator.fit_sizes == [1]
    assert any(fit_graph.number_of_edges() == 0 for fit_graph in partial_estimator.fit_graphs[0])
    assert all(fit_graph.number_of_edges() > 0 for fit_graph in final_estimator.fit_graphs[0])


def test_repair_partial_feasibility_bootstrap_graphs_use_only_query_node_state() -> None:
    generator = EdgeGenerator(
        partial_feasibility_estimator=_RecordingFeasibilityEstimator("partial"),
        final_feasibility_estimator=_RecordingFeasibilityEstimator("final"),
        graph_estimator=_RecordingGraphEstimator(),
        allow_self_loops=False,
    )
    query = nx.DiGraph()
    query.add_node("a", label="A")
    query.add_node("b", label="B")
    neighbor = nx.DiGraph()
    neighbor.add_node(0, label="A")
    neighbor.add_node(1, label="B")
    neighbor.add_edge(0, 1, label="jump")

    bootstrap_graphs = generator._repair_partial_feasibility_bootstrap_graphs(
        query,
        [neighbor],
    )

    assert len(bootstrap_graphs) == 1
    assert bootstrap_graphs[0].number_of_edges() == 0
    assert set(bootstrap_graphs[0].nodes()) == {"a", "b"}
    assert dict(bootstrap_graphs[0].nodes(data="label")) == {"a": "A", "b": "B"}


def test_repair_returns_none_when_neighbor_labels_do_not_match_input(monkeypatch, capsys) -> None:
    partial_estimator = _RecordingFeasibilityEstimator("partial")
    final_estimator = _RecordingFeasibilityEstimator("final")
    graph_estimator = _RecordingGraphEstimator()
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=graph_estimator,
    )
    input_graph = _labeled_edge_graph(["C", "F"])
    neighbor_graph = _labeled_edge_graph(["C", "N"])
    repair_context = {
        "graph": input_graph.copy(),
        "query_index": None,
        "neighbor_indices": [0],
        "neighbor_distances": [0.0],
        "fit_graphs": [neighbor_graph],
        "fit_targets": None,
    }

    monkeypatch.setattr(generator, "_require_stored_dataset", lambda: None)
    monkeypatch.setattr(
        generator,
        "_prepare_repair_training_context",
        lambda graph, *, n_neighbors: repair_context,
    )

    def fail_fit(*args, **kwargs):
        raise AssertionError("repair should fail before fitting local estimators")

    monkeypatch.setattr(generator, "_fit_pair_training_graphs", fail_fit)

    repaired = generator.repair(
        input_graph,
        n_neighbors=1,
        return_path=False,
        verbose=True,
    )

    assert repaired is None
    assert generator.last_repair_label_set_mismatch_ == {
        "graph_labels": ["C", "F"],
        "neighbor_labels": ["C", "N"],
        "missing_from_neighbors": ["F"],
        "extra_in_neighbors": ["N"],
    }
    out = capsys.readouterr().out
    assert "label-set mismatch between input graph and repair neighborhood" in out


def test_repair_allows_extra_neighbor_labels_when_input_labels_are_covered(monkeypatch) -> None:
    partial_estimator = _RecordingFeasibilityEstimator("partial")
    final_estimator = _RecordingFeasibilityEstimator("final")
    graph_estimator = _RecordingGraphEstimator()
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=graph_estimator,
    )
    input_graph = _labeled_edge_graph(["C", "F"])
    neighbor_graph = _labeled_edge_graph(["C", "F", "Cl"])
    repair_context = {
        "graph": input_graph.copy(),
        "query_index": None,
        "neighbor_indices": [0],
        "neighbor_distances": [0.0],
        "fit_graphs": [neighbor_graph],
        "fit_targets": None,
    }

    monkeypatch.setattr(generator, "_require_stored_dataset", lambda: None)
    monkeypatch.setattr(
        generator,
        "_prepare_repair_training_context",
        lambda graph, *, n_neighbors: repair_context,
    )
    monkeypatch.setattr(generator, "_log_repair_training_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(generator, "_fit_pair_training_graphs", lambda *args, **kwargs: None)
    monkeypatch.setattr(generator.final_feasibility_estimator, "predict", lambda graphs: np.asarray([True]))

    repaired = generator.repair(
        input_graph,
        n_neighbors=1,
        return_path=False,
        verbose=True,
    )

    assert repaired is not None
    assert sorted(repaired.nodes(data="label")) == sorted(input_graph.nodes(data="label"))
    assert sorted(repaired.edges(data="label")) == sorted(input_graph.edges(data="label"))
    assert generator.last_repair_label_set_mismatch_ is None


def test_repair_can_skip_label_set_coverage_check_when_configured(monkeypatch) -> None:
    partial_estimator = _RecordingFeasibilityEstimator("partial")
    final_estimator = _RecordingFeasibilityEstimator("final")
    graph_estimator = _RecordingGraphEstimator()
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=graph_estimator,
        enforce_repair_label_set_coverage=False,
    )
    input_graph = _labeled_edge_graph(["C", "F"])
    neighbor_graph = _labeled_edge_graph(["C", "N"])
    repair_context = {
        "graph": input_graph.copy(),
        "query_index": None,
        "neighbor_indices": [0],
        "neighbor_distances": [0.0],
        "fit_graphs": [neighbor_graph],
        "fit_targets": None,
    }
    fit_called = {"value": False}

    monkeypatch.setattr(generator, "_require_stored_dataset", lambda: None)
    monkeypatch.setattr(
        generator,
        "_prepare_repair_training_context",
        lambda graph, *, n_neighbors: repair_context,
    )
    monkeypatch.setattr(generator, "_log_repair_training_context", lambda *args, **kwargs: None)

    def fake_fit(*args, **kwargs):
        fit_called["value"] = True

    monkeypatch.setattr(generator, "_fit_pair_training_graphs", fake_fit)
    monkeypatch.setattr(generator.final_feasibility_estimator, "predict", lambda graphs: np.asarray([True]))

    repaired = generator.repair(
        input_graph,
        n_neighbors=1,
        return_path=False,
        verbose=True,
    )

    assert fit_called["value"] is True
    assert repaired is not None
    assert generator.last_repair_label_set_mismatch_ is None


def test_log_search_step_reports_backtrack_when_no_feasible_candidates_remain(capsys) -> None:
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
        edge_risk_estimator=_RecordingRiskEstimator(),
        edge_risk_lambda=0.25,
    )

    generator._log_search_step(
        retained=[],
        scored={
            "repulsion_lambda": 0.0,
            "generated": [nx.path_graph(2)],
            "feasible_candidates": [],
        },
        start_graph=nx.path_graph(2),
        n_edges=3,
        next_depth=1,
        target=None,
        target_lambda=0.5,
        graph_index=0,
        total_phases=5,
        fallback_index=0,
        beam_limit=3,
        step_start_time=0.0,
        draw_graphs_fn=None,
        verbose=True,
    )

    out = capsys.readouterr().out
    assert "BACKTRACK no feasible candidates remain" in out


def test_log_search_step_reports_failed_when_no_fallback_phases_remain(capsys) -> None:
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
        edge_risk_estimator=_RecordingRiskEstimator(),
        edge_risk_lambda=0.25,
    )

    generator._log_search_step(
        retained=[],
        scored={
            "repulsion_lambda": 0.0,
            "generated": [nx.path_graph(2)],
            "feasible_candidates": [],
        },
        start_graph=nx.path_graph(2),
        n_edges=3,
        next_depth=1,
        target=None,
        target_lambda=0.5,
        graph_index=0,
        total_phases=2,
        fallback_index=0,
        beam_limit=3,
        step_start_time=0.0,
        draw_graphs_fn=None,
        verbose=True,
    )

    out = capsys.readouterr().out
    assert "FAILED no feasible candidates remain" in out


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
