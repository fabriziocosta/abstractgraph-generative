from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from abstractgraph.hashing import hash_graph

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


class _CountingViolationsFeasibilityEstimator(_RecordingFeasibilityEstimator):
    def __init__(self, violations):
        super().__init__("counting_violations")
        self.violations = violations
        self.number_of_violations_calls = []

    def number_of_violations(self, graphs):
        self.number_of_violations_calls.append([graph.copy() for graph in graphs])
        if callable(self.violations):
            return np.asarray([self.violations(graph) for graph in graphs], dtype=float)
        return np.full(len(graphs), self.violations, dtype=float)


class _NoNumberOfViolationsEstimator:
    def fit(self, graphs):
        return self

    def predict(self, graphs):
        return np.ones(len(graphs), dtype=bool)


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
        lookahead_rejection_model="max_envelope",
    )
    graph = nx.path_graph(3)

    generator.fit([graph])

    assert partial_estimator.fit_sizes == [2]
    assert final_estimator.fit_sizes == [1]
    assert generator.lookahead_pruning_active_ is True
    assert generator.lookahead_violation_thresholds_ == {1: 0.0, 2: 0.0}
    assert graph_estimator.fit_size > 0


def test_fit_logs_lookahead_envelope_when_final_estimator_supports_violations(capsys) -> None:
    partial_estimator = _RecordingFeasibilityEstimator("partial")
    final_estimator = _RecordingFeasibilityEstimator("final")
    graph_estimator = _RecordingGraphEstimator()
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=graph_estimator,
        n_negative_per_positive=1,
        n_replicates=1,
        lookahead_rejection_model="max_envelope",
        verbose=True,
    )

    generator.fit([nx.path_graph(3)])

    fit_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("[fit]")
    ]
    assert "partial_feasibility_graphs=" in fit_lines[0]
    assert "final_feasibility_graphs=" not in fit_lines[0]
    assert "lookahead_envelope_stages=" not in fit_lines[0]
    assert "final_feasibility_graphs=" in fit_lines[1]
    assert "partial_feasibility_graphs=" not in fit_lines[1]
    assert "lookahead_envelope_stages=" not in fit_lines[1]
    assert "lookahead_envelope_stages=" in fit_lines[2]
    assert "lookahead_examples=" in fit_lines[2]
    assert "partial_feasibility_graphs=" not in fit_lines[2]
    assert "final_feasibility_graphs=" not in fit_lines[2]


def test_fit_disables_lookahead_when_final_estimator_has_no_violation_counts() -> None:
    partial_estimator = _RecordingFeasibilityEstimator("partial")
    final_estimator = _NoNumberOfViolationsEstimator()
    graph_estimator = _RecordingGraphEstimator()
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=graph_estimator,
        n_negative_per_positive=1,
        n_replicates=1,
        lookahead_rejection_model="max_envelope",
    )

    generator.fit([nx.path_graph(3)])

    assert generator.lookahead_pruning_active_ is False
    assert generator.lookahead_violation_thresholds_ is None


def test_constructor_validates_lookahead_rejection_options() -> None:
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
    )
    assert generator.lookahead_rejection_model == "lognormal_tail"
    assert generator.lookahead_rejection_quantile == pytest.approx(0.95)
    assert generator.lookahead_rejection_temperature == pytest.approx(1.0)

    with pytest.raises(ValueError, match="lookahead_rejection_model"):
        EdgeGenerator(
            feasibility_estimator=object(),
            graph_estimator=object(),
            lookahead_rejection_model="bad",
        )
    with pytest.raises(ValueError, match="lookahead_rejection_quantile"):
        EdgeGenerator(
            feasibility_estimator=object(),
            graph_estimator=object(),
            lookahead_rejection_quantile=1.0,
        )
    with pytest.raises(ValueError, match="lookahead_rejection_temperature"):
        EdgeGenerator(
            feasibility_estimator=object(),
            graph_estimator=object(),
            lookahead_rejection_temperature=0.0,
        )


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
        lookahead_rejection_model="max_envelope",
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
        lookahead_rejection_model="max_envelope",
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
    monkeypatch.setattr(
        generator,
        "_log_repair_training_context",
        lambda *args, **kwargs: None,
    )
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
    monkeypatch.setattr(
        generator,
        "_log_repair_training_context",
        lambda *args, **kwargs: None,
    )

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


def test_repair_attempt_log_reports_removed_edge_count_and_titles(
    monkeypatch, capsys
) -> None:
    partial_estimator = _RecordingFeasibilityEstimator("partial")
    final_estimator = _RecordingFeasibilityEstimator("final")
    graph_estimator = _RecordingGraphEstimator()
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=graph_estimator,
    )
    input_graph = nx.path_graph(4)
    feasible_graph = input_graph.copy()
    feasible_graph.remove_edges_from([(0, 1), (1, 2)])
    repair_context = {
        "graph": input_graph.copy(),
        "query_index": None,
        "neighbor_indices": [0],
        "neighbor_distances": [0.0],
        "fit_graphs": [input_graph.copy()],
        "fit_targets": None,
    }
    draw_calls = []

    def fake_draw(graphs, **kwargs):
        draw_calls.append((graphs, kwargs))

    monkeypatch.setattr(generator, "_require_stored_dataset", lambda: None)
    monkeypatch.setattr(
        generator,
        "_prepare_repair_training_context",
        lambda graph, *, n_neighbors: repair_context,
    )
    monkeypatch.setattr(generator, "_log_repair_training_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(generator, "_fit_pair_training_graphs", lambda *args, **kwargs: None)
    monkeypatch.setattr(generator.final_feasibility_estimator, "predict", lambda graphs: np.asarray([False]))
    monkeypatch.setattr(
        generator,
        "_build_repair_start_states",
        lambda *args, **kwargs: [
            {
                "graph": feasible_graph,
                "repair_removed_edges": ((0, 1), (1, 2)),
            }
        ],
    )
    monkeypatch.setattr(
        generator,
        "generate",
        lambda *args, **kwargs: [feasible_graph],
    )

    repaired = generator.repair(
        input_graph,
        n_neighbors=1,
        draw_graphs_fn=fake_draw,
        verbose=True,
    )

    out = capsys.readouterr().out
    assert "removed_edges=2" in out
    assert "removed_edges=[(0, 1), (1, 2)]" not in out
    assert len(repaired) == 2
    assert sorted(repaired[0].edges()) == sorted(input_graph.edges())
    assert sorted(repaired[1].edges()) == sorted(feasible_graph.edges())
    assert len(draw_calls) == 1
    assert draw_calls[0][1]["n_graphs_per_line"] == 2
    assert draw_calls[0][1]["titles"] == [
        "original input\nedges=3 target_edges=3",
        "feasible input\nedges=1 removed_edges=2",
    ]


def test_repair_deactivates_inconsistent_lookahead_envelope_after_local_fit(
    monkeypatch, recwarn
) -> None:
    class _InconsistentFinalEstimator(_RejectEdgesFeasibilityEstimator):
        def __init__(self):
            super().__init__([(0, 1)])
            self.calls = 0

        def number_of_violations(self, graphs):
            self.calls += 1
            value = 0 if self.calls == 1 else 99
            return np.full(len(graphs), value, dtype=float)

    partial_estimator = _RecordingFeasibilityEstimator("partial")
    final_estimator = _InconsistentFinalEstimator()
    graph_estimator = _RecordingGraphEstimator()
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=graph_estimator,
        n_negative_per_positive=1,
        n_replicates=1,
        lookahead_rejection_model="max_envelope",
    )
    input_graph = nx.path_graph(4)
    feasible_graph = input_graph.copy()
    feasible_graph.remove_edge(0, 1)
    repair_context = {
        "graph": input_graph.copy(),
        "query_index": None,
        "neighbor_indices": [0],
        "neighbor_distances": [0.0],
        "fit_graphs": [input_graph.copy()],
        "fit_targets": None,
    }

    monkeypatch.setattr(generator, "_require_stored_dataset", lambda: None)
    monkeypatch.setattr(
        generator,
        "_prepare_repair_training_context",
        lambda graph, *, n_neighbors: repair_context,
    )
    monkeypatch.setattr(
        generator,
        "_log_repair_training_context",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        generator,
        "_build_repair_start_states",
        lambda *args, **kwargs: [
            {
                "graph": feasible_graph,
                "repair_removed_edges": ((0, 1),),
            }
        ],
    )

    def fake_generate(*args, **kwargs):
        assert generator.lookahead_pruning_active_ is False
        return [feasible_graph]

    monkeypatch.setattr(generator, "generate", fake_generate)

    repaired = generator.repair(input_graph, n_neighbors=1, return_path=False)

    assert repaired is feasible_graph
    assert generator.lookahead_pruning_active_ is False
    assert generator.last_lookahead_failsafe_ == {
        "checked": 3,
        "false_infeasible": 3,
        "deactivated": True,
    }
    warning = recwarn.pop(RuntimeWarning)
    assert "known repair-positive edge-removal graphs infeasible" in str(warning.message)


def test_repair_keeps_valid_lookahead_envelope_after_local_fit(monkeypatch) -> None:
    partial_estimator = _RecordingFeasibilityEstimator("partial")
    final_estimator = _RejectEdgesFeasibilityEstimator([(0, 1)])
    graph_estimator = _RecordingGraphEstimator()
    generator = EdgeGenerator(
        partial_feasibility_estimator=partial_estimator,
        final_feasibility_estimator=final_estimator,
        graph_estimator=graph_estimator,
        n_negative_per_positive=1,
        n_replicates=1,
        lookahead_rejection_model="max_envelope",
    )
    input_graph = nx.path_graph(4)
    feasible_graph = input_graph.copy()
    feasible_graph.remove_edge(0, 1)
    repair_context = {
        "graph": input_graph.copy(),
        "query_index": None,
        "neighbor_indices": [0],
        "neighbor_distances": [0.0],
        "fit_graphs": [input_graph.copy()],
        "fit_targets": None,
    }

    monkeypatch.setattr(generator, "_require_stored_dataset", lambda: None)
    monkeypatch.setattr(
        generator,
        "_prepare_repair_training_context",
        lambda graph, *, n_neighbors: repair_context,
    )
    monkeypatch.setattr(
        generator,
        "_log_repair_training_context",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        generator,
        "_build_repair_start_states",
        lambda *args, **kwargs: [
            {
                "graph": feasible_graph,
                "repair_removed_edges": ((0, 1),),
            }
        ],
    )
    monkeypatch.setattr(generator, "generate", lambda *args, **kwargs: [feasible_graph])

    repaired = generator.repair(input_graph, n_neighbors=1, return_path=False)

    assert repaired is feasible_graph
    assert generator.lookahead_pruning_active_ is True
    assert generator.last_lookahead_failsafe_ == {
        "checked": 3,
        "false_infeasible": 0,
        "deactivated": False,
    }


def test_prepare_repair_training_context_prioritizes_neighbor_label_set_coverage() -> None:
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
    )
    query = _labeled_edge_graph(["A", "B"])
    incompatible_close = _labeled_edge_graph(["A", "C"])
    compatible_far = _labeled_edge_graph(["A", "B", "D"])
    compatible_near = _labeled_edge_graph(["A", "B", "E"])
    generator.stored_graphs_ = [
        query.copy(),
        incompatible_close,
        compatible_far,
        compatible_near,
    ]
    generator.stored_targets_ = ["query", "incompatible", "far", "near"]
    generator.stored_graph_hash_to_index_ = {hash_graph(query): 0}
    generator.stored_distance_matrix_ = np.asarray(
        [
            [0.0, 0.1, 0.4, 0.2],
            [0.1, 0.0, 0.5, 0.3],
            [0.4, 0.5, 0.0, 0.6],
            [0.2, 0.3, 0.6, 0.0],
        ],
        dtype=float,
    )

    context = generator._prepare_repair_training_context(query, n_neighbors=2)

    assert context["neighbor_indices"] == [1, 3]
    assert context["neighbor_distances"] == [0.1, 0.2]
    assert context["fit_targets"] == ["incompatible", "near"]
    assert generator._repair_label_set_mismatch(context) is None


def test_prepare_repair_training_context_can_cover_query_labels_across_neighbors() -> None:
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
    )
    query = _labeled_edge_graph(["A", "B"])
    covers_a = _labeled_edge_graph(["A", "C"])
    covers_b = _labeled_edge_graph(["B", "D"])
    generator.stored_graphs_ = [query.copy(), covers_a, covers_b]
    generator.stored_targets_ = ["query", "a", "b"]
    generator.stored_graph_hash_to_index_ = {hash_graph(query): 0}
    generator.stored_distance_matrix_ = np.asarray(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.3],
            [0.2, 0.3, 0.0],
        ],
        dtype=float,
    )

    context = generator._prepare_repair_training_context(query, n_neighbors=2)

    assert context["neighbor_indices"] == [1, 2]
    assert context["fit_targets"] == ["a", "b"]
    assert generator._repair_label_set_mismatch(context) is None


def test_prepare_repair_training_context_can_skip_label_compatible_filter() -> None:
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
        enforce_repair_label_set_coverage=False,
    )
    query = _labeled_edge_graph(["A", "B"])
    incompatible_close = _labeled_edge_graph(["A", "C"])
    compatible_far = _labeled_edge_graph(["A", "B", "D"])
    generator.stored_graphs_ = [query.copy(), incompatible_close, compatible_far]
    generator.stored_targets_ = None
    generator.stored_graph_hash_to_index_ = {hash_graph(query): 0}
    generator.stored_distance_matrix_ = np.asarray(
        [
            [0.0, 0.1, 0.4],
            [0.1, 0.0, 0.5],
            [0.4, 0.5, 0.0],
        ],
        dtype=float,
    )

    context = generator._prepare_repair_training_context(query, n_neighbors=1)

    assert context["neighbor_indices"] == [1]


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


def test_log_search_step_reports_feasibility_partition_counts(capsys) -> None:
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
    )
    generator.n_tried_ = 594
    partial_failure = {"feasibility_stage": "partial"}
    lookahead_failure = {"feasibility_stage": "lookahead"}
    completion_failure = {"feasibility_stage": "completion"}
    final_failure = {"feasibility_stage": "final"}

    generator._log_search_step(
        retained=[{"graph": nx.path_graph(4), "score": 0.9}],
        scored={
            "repulsion_lambda": 0.0,
            "generated": [nx.path_graph(2) for _ in range(594)],
            "feasible_candidates": [{"graph": nx.path_graph(3)} for _ in range(82)],
            "infeasible_candidates": (
                [partial_failure for _ in range(414)]
                + [lookahead_failure for _ in range(5)]
                + [completion_failure for _ in range(98)]
                + [final_failure for _ in range(7)]
            ),
        },
        start_graph=nx.path_graph(2),
        n_edges=5,
        next_depth=3,
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
    lines = out.splitlines()
    assert "[graph 0] remaining_edges=2" in lines
    assert any(
        line.startswith("search_phase=2/5 depth=3 step_time=") and " eta=" in line
        for line in lines
    )
    assert (
        "tried=594 generated=594 partial_infeasible=414 "
        "partial_feasible=180 lookahead_infeasible=5 final_infeasible=7 "
        "viable=82 retained=1"
    ) in out


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


def test_draw_graphs_retries_with_titles_when_layout_kwargs_are_rejected() -> None:
    generator = EdgeGenerator(feasibility_estimator=object(), graph_estimator=object())
    draw_calls = []

    def fake_draw(graphs, **kwargs):
        if "n_graphs_per_line" in kwargs:
            raise TypeError("multiple values for n_graphs_per_line")
        draw_calls.append((graphs, kwargs))

    generator._draw_graphs(
        fake_draw,
        [nx.path_graph(2), nx.path_graph(3)],
        n_graphs_per_line=2,
        titles=["original input", "feasible input"],
    )

    assert len(draw_calls) == 1
    assert draw_calls[0][1] == {"titles": ["original input", "feasible input"]}


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
    assert "[graph 0] repair_fallback=2/4" in out
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


def test_trace_failure_ratio_counts_completion_and_blocked_failures() -> None:
    risk_estimator = _RecordingRiskEstimator()
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
        edge_risk_estimator=risk_estimator,
    )
    generator._reset_edge_risk_attempt_trace()
    root = generator._make_state(nx.path_graph(2), parent=None, score=1.0, depth=0)
    decision = generator._make_state(nx.path_graph(3), parent=root, score=0.9, depth=1)
    completion_failure = generator._make_state(nx.path_graph(4), parent=decision, score=0.0, depth=2)
    blocked_failure = generator._make_state(nx.path_graph(5), parent=decision, score=0.0, depth=2)
    pruned = generator._make_state(nx.path_graph(3), parent=decision, score=0.8, depth=2)

    generator._mark_trace_state_status(decision, "retained")
    generator._mark_trace_state_status(completion_failure, "completion_infeasible")
    generator._mark_trace_state_status(blocked_failure, "blocked")
    generator._mark_trace_state_status(pruned, "pruned")

    assert generator._trace_failure_ratio_for_state(decision["state_id"]) == pytest.approx(0.5)


def test_generate_accepts_final_feasible_unexpandable_beam(monkeypatch) -> None:
    class _AlwaysFeasibleEstimator:
        def fit(self, graphs):
            return self

        def predict(self, graphs):
            return np.ones(len(graphs), dtype=bool)

    graph = nx.path_graph(3)
    generator = EdgeGenerator(
        partial_feasibility_estimator=_AlwaysFeasibleEstimator(),
        final_feasibility_estimator=_AlwaysFeasibleEstimator(),
        graph_estimator=_RecordingGraphEstimator(),
        require_single_connected_component=True,
    )
    generator.edge_attribute_templates_ = [{}]
    generator._positive_scores = lambda graphs: np.asarray([0.9] * len(graphs), dtype=float)
    generator._target_scores = lambda graphs, *, target: np.zeros(len(graphs))
    generator._edge_risk_scores = lambda candidates: np.zeros(len(candidates))
    monkeypatch.setattr(
        generator,
        "_max_total_edges_for_generation",
        lambda start_graph, n_edges: start_graph.number_of_edges(),
    )

    def fail_mark_completion_infeasible(search):
        raise AssertionError("final-feasible unexpandable beam should be accepted")

    monkeypatch.setattr(
        generator,
        "_mark_unexpandable_beam_as_completion_infeasible",
        fail_mark_completion_infeasible,
    )

    path = generator._generate_one(
        graph,
        graph.number_of_edges() + 1,
        target=None,
        target_lambda=1.0,
        verbose=False,
    )

    assert len(path) == 1
    assert sorted(path[0].edges()) == sorted(graph.edges())


def test_partition_candidates_keeps_partial_feasible_non_terminal_candidates() -> None:
    class _AlwaysFeasibleEstimator:
        def predict(self, graphs):
            return np.ones(len(graphs), dtype=bool)

    generator = EdgeGenerator(
        feasibility_estimator=_AlwaysFeasibleEstimator(),
        partial_feasibility_estimator=_AlwaysFeasibleEstimator(),
        final_feasibility_estimator=_AlwaysFeasibleEstimator(),
        graph_estimator=object(),
        require_single_connected_component=True,
    )
    generator._positive_scores = lambda graphs: np.asarray([0.9], dtype=float)
    generator._target_scores = lambda graphs, *, target: np.asarray([0.0], dtype=float)

    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_edge(0, 1)
    root = generator._make_state(nx.empty_graph(4), parent=None, score=1.0, depth=0)
    cand = generator._make_state(graph, parent=root, score=None, depth=1)
    feasible_candidates = []
    infeasible_candidates = []

    generator._partition_candidates_by_feasibility(
        [cand],
        n_edges=3,
        max_total_edges=2,
        target=None,
        target_lambda=1.0,
        feasible_candidates=feasible_candidates,
        infeasible_candidates=infeasible_candidates,
    )

    assert feasible_candidates == [cand]
    assert infeasible_candidates == []
    assert "feasibility_stage" not in cand


def test_partition_candidates_does_not_call_lookahead_when_omitted() -> None:
    class _AlwaysFeasibleEstimator:
        def predict(self, graphs):
            return np.ones(len(graphs), dtype=bool)

    generator = EdgeGenerator(
        partial_feasibility_estimator=_AlwaysFeasibleEstimator(),
        final_feasibility_estimator=_AlwaysFeasibleEstimator(),
        graph_estimator=object(),
    )
    generator._positive_scores = lambda graphs: np.asarray([0.9], dtype=float)
    generator._target_scores = lambda graphs, *, target: np.asarray([0.0], dtype=float)

    root = generator._make_state(nx.path_graph(2), parent=None, score=1.0, depth=0)
    cand = generator._make_state(nx.path_graph(3), parent=root, score=None, depth=1)
    feasible_candidates = []
    infeasible_candidates = []

    generator._partition_candidates_by_feasibility(
        [cand],
        n_edges=4,
        max_total_edges=4,
        target=None,
        target_lambda=1.0,
        feasible_candidates=feasible_candidates,
        infeasible_candidates=infeasible_candidates,
    )

    assert feasible_candidates == [cand]
    assert infeasible_candidates == []
    assert generator.lookahead_pruning_active_ is False


def test_partition_candidates_prunes_lookahead_violations_over_stage_envelope() -> None:
    class _AlwaysFeasibleEstimator:
        def predict(self, graphs):
            return np.ones(len(graphs), dtype=bool)

    generator = EdgeGenerator(
        partial_feasibility_estimator=_AlwaysFeasibleEstimator(),
        final_feasibility_estimator=_CountingViolationsFeasibilityEstimator(3),
        graph_estimator=object(),
        lookahead_rejection_model="max_envelope",
    )
    generator.lookahead_pruning_active_ = True
    generator.lookahead_violation_thresholds_ = {1: 2.0}
    generator._positive_scores = lambda graphs: np.asarray([0.9], dtype=float)
    generator._target_scores = lambda graphs, *, target: np.asarray([0.0], dtype=float)

    root = generator._make_state(nx.path_graph(2), parent=None, score=1.0, depth=0)
    cand = generator._make_state(nx.path_graph(3), parent=root, score=None, depth=1)
    feasible_candidates = []
    infeasible_candidates = []

    generator._partition_candidates_by_feasibility(
        [cand],
        n_edges=3,
        max_total_edges=5,
        target=None,
        target_lambda=1.0,
        feasible_candidates=feasible_candidates,
        infeasible_candidates=infeasible_candidates,
    )

    assert feasible_candidates == []
    assert infeasible_candidates == [cand]
    assert cand["feasibility_stage"] == "lookahead"
    assert cand["lookahead_violation_count"] == pytest.approx(3.0)
    assert cand["lookahead_violation_threshold"] == pytest.approx(2.0)
    assert cand["remaining_moves"] == 1


def test_partition_candidates_keeps_lookahead_violations_equal_to_stage_envelope() -> None:
    class _AlwaysFeasibleEstimator:
        def predict(self, graphs):
            return np.ones(len(graphs), dtype=bool)

    generator = EdgeGenerator(
        partial_feasibility_estimator=_AlwaysFeasibleEstimator(),
        final_feasibility_estimator=_CountingViolationsFeasibilityEstimator(2),
        graph_estimator=object(),
        lookahead_rejection_model="max_envelope",
    )
    generator.lookahead_pruning_active_ = True
    generator.lookahead_violation_thresholds_ = {1: 2.0}
    generator._positive_scores = lambda graphs: np.asarray([0.9], dtype=float)
    generator._target_scores = lambda graphs, *, target: np.asarray([0.0], dtype=float)

    root = generator._make_state(nx.path_graph(2), parent=None, score=1.0, depth=0)
    cand = generator._make_state(nx.path_graph(3), parent=root, score=None, depth=1)
    feasible_candidates = []
    infeasible_candidates = []

    generator._partition_candidates_by_feasibility(
        [cand],
        n_edges=3,
        max_total_edges=5,
        target=None,
        target_lambda=1.0,
        feasible_candidates=feasible_candidates,
        infeasible_candidates=infeasible_candidates,
    )

    assert feasible_candidates == [cand]
    assert infeasible_candidates == []
    assert cand["lookahead_violation_count"] == pytest.approx(2.0)
    assert cand["lookahead_violation_threshold"] == pytest.approx(2.0)
    assert cand["remaining_moves"] == 1


def test_partition_candidates_skips_lookahead_when_stage_has_no_envelope() -> None:
    class _AlwaysFeasibleEstimator:
        def predict(self, graphs):
            return np.ones(len(graphs), dtype=bool)

    generator = EdgeGenerator(
        partial_feasibility_estimator=_AlwaysFeasibleEstimator(),
        final_feasibility_estimator=_CountingViolationsFeasibilityEstimator(99),
        graph_estimator=object(),
        lookahead_rejection_model="max_envelope",
    )
    generator.lookahead_pruning_active_ = True
    generator.lookahead_violation_thresholds_ = {2: 0.0}
    generator._positive_scores = lambda graphs: np.asarray([0.9], dtype=float)
    generator._target_scores = lambda graphs, *, target: np.asarray([0.0], dtype=float)

    root = generator._make_state(nx.path_graph(2), parent=None, score=1.0, depth=0)
    cand = generator._make_state(nx.path_graph(3), parent=root, score=None, depth=1)
    feasible_candidates = []
    infeasible_candidates = []

    generator._partition_candidates_by_feasibility(
        [cand],
        n_edges=3,
        max_total_edges=5,
        target=None,
        target_lambda=1.0,
        feasible_candidates=feasible_candidates,
        infeasible_candidates=infeasible_candidates,
    )

    assert feasible_candidates == [cand]
    assert infeasible_candidates == []
    assert "lookahead_violation_count" not in cand


def test_partition_candidates_samples_lognormal_tail_rejection() -> None:
    class _AlwaysFeasibleEstimator:
        def predict(self, graphs):
            return np.ones(len(graphs), dtype=bool)

    generator = EdgeGenerator(
        partial_feasibility_estimator=_AlwaysFeasibleEstimator(),
        final_feasibility_estimator=_CountingViolationsFeasibilityEstimator(100),
        graph_estimator=object(),
        lookahead_rejection_model="lognormal_tail",
        lookahead_rejection_quantile=0.0,
        seed=0,
    )
    generator.lookahead_pruning_active_ = True
    generator.lookahead_lognormal_params_ = {
        1: {"n": 3, "mu": 0.0, "sigma": 1.0, "max": 2.0}
    }
    generator._positive_scores = lambda graphs: np.asarray([0.9], dtype=float)
    generator._target_scores = lambda graphs, *, target: np.asarray([0.0], dtype=float)

    root = generator._make_state(nx.path_graph(2), parent=None, score=1.0, depth=0)
    cand = generator._make_state(nx.path_graph(3), parent=root, score=None, depth=1)
    feasible_candidates = []
    infeasible_candidates = []

    generator._partition_candidates_by_feasibility(
        [cand],
        n_edges=3,
        max_total_edges=5,
        target=None,
        target_lambda=1.0,
        feasible_candidates=feasible_candidates,
        infeasible_candidates=infeasible_candidates,
    )

    assert feasible_candidates == []
    assert infeasible_candidates == [cand]
    assert cand["feasibility_stage"] == "lookahead"
    assert cand["lookahead_reject_prob"] > 0.99


def test_partition_candidates_keeps_lognormal_tail_below_quantile() -> None:
    class _AlwaysFeasibleEstimator:
        def predict(self, graphs):
            return np.ones(len(graphs), dtype=bool)

    generator = EdgeGenerator(
        partial_feasibility_estimator=_AlwaysFeasibleEstimator(),
        final_feasibility_estimator=_CountingViolationsFeasibilityEstimator(1),
        graph_estimator=object(),
        lookahead_rejection_model="lognormal_tail",
        lookahead_rejection_quantile=0.95,
        seed=0,
    )
    generator.lookahead_pruning_active_ = True
    generator.lookahead_lognormal_params_ = {
        1: {"n": 3, "mu": math.log1p(1.0), "sigma": 1.0, "max": 2.0}
    }
    generator._positive_scores = lambda graphs: np.asarray([0.9], dtype=float)
    generator._target_scores = lambda graphs, *, target: np.asarray([0.0], dtype=float)

    root = generator._make_state(nx.path_graph(2), parent=None, score=1.0, depth=0)
    cand = generator._make_state(nx.path_graph(3), parent=root, score=None, depth=1)
    feasible_candidates = []
    infeasible_candidates = []

    generator._partition_candidates_by_feasibility(
        [cand],
        n_edges=3,
        max_total_edges=5,
        target=None,
        target_lambda=1.0,
        feasible_candidates=feasible_candidates,
        infeasible_candidates=infeasible_candidates,
    )

    assert feasible_candidates == [cand]
    assert infeasible_candidates == []
    assert cand["lookahead_reject_prob"] == pytest.approx(0.0)


def test_partition_candidates_skips_lookahead_for_terminal_candidates() -> None:
    class _AlwaysFeasibleEstimator:
        def predict(self, graphs):
            return np.ones(len(graphs), dtype=bool)

        def number_of_violations(self, graphs):
            return np.zeros(len(graphs), dtype=int)

    class _RejectFinalEstimator(_AlwaysFeasibleEstimator):
        def predict(self, graphs):
            return np.zeros(len(graphs), dtype=bool)

    generator = EdgeGenerator(
        partial_feasibility_estimator=_AlwaysFeasibleEstimator(),
        final_feasibility_estimator=_RejectFinalEstimator(),
        graph_estimator=object(),
    )
    generator.lookahead_pruning_active_ = True
    generator.lookahead_violation_thresholds_ = {1: 0.0}
    generator._positive_scores = lambda graphs: np.asarray([0.9], dtype=float)
    generator._target_scores = lambda graphs, *, target: np.asarray([0.0], dtype=float)

    root = generator._make_state(nx.path_graph(2), parent=None, score=1.0, depth=0)
    cand = generator._make_state(nx.path_graph(3), parent=root, score=None, depth=1)
    feasible_candidates = []
    infeasible_candidates = []

    generator._partition_candidates_by_feasibility(
        [cand],
        n_edges=2,
        max_total_edges=5,
        target=None,
        target_lambda=1.0,
        feasible_candidates=feasible_candidates,
        infeasible_candidates=infeasible_candidates,
    )

    assert feasible_candidates == []
    assert infeasible_candidates == [cand]
    assert cand["feasibility_stage"] == "final"


def test_partition_candidates_skips_lookahead_when_final_estimator_lacks_violations() -> None:
    class _AlwaysFeasibleEstimator:
        def predict(self, graphs):
            return np.ones(len(graphs), dtype=bool)

    generator = EdgeGenerator(
        partial_feasibility_estimator=_AlwaysFeasibleEstimator(),
        final_feasibility_estimator=_AlwaysFeasibleEstimator(),
        graph_estimator=object(),
    )
    generator._positive_scores = lambda graphs: np.asarray([0.9], dtype=float)
    generator._target_scores = lambda graphs, *, target: np.asarray([0.0], dtype=float)

    root = generator._make_state(nx.path_graph(2), parent=None, score=1.0, depth=0)
    cand = generator._make_state(nx.path_graph(3), parent=root, score=None, depth=1)
    feasible_candidates = []
    infeasible_candidates = []

    generator._partition_candidates_by_feasibility(
        [cand],
        n_edges=3,
        max_total_edges=5,
        target=None,
        target_lambda=1.0,
        feasible_candidates=feasible_candidates,
        infeasible_candidates=infeasible_candidates,
    )

    assert feasible_candidates == [cand]
    assert infeasible_candidates == []


def test_rank_feasible_candidates_prioritizes_one_step_final_completion() -> None:
    class _AlwaysFeasibleEstimator:
        def predict(self, graphs):
            return np.ones(len(graphs), dtype=bool)

    class _PathFinalEstimator:
        def predict(self, graphs):
            return np.asarray(
                [
                    set(graph.edges()) == {(0, 1), (1, 2)}
                    for graph in graphs
                ],
                dtype=bool,
            )

    generator = EdgeGenerator(
        partial_feasibility_estimator=_AlwaysFeasibleEstimator(),
        final_feasibility_estimator=_PathFinalEstimator(),
        graph_estimator=object(),
    )
    generator.edge_attribute_templates_ = [{}]
    generator._positive_scores = lambda graphs: np.asarray([0.2, 0.9], dtype=float)
    generator._target_scores = lambda graphs, *, target: np.zeros(len(graphs), dtype=float)
    generator._repulsion_values = lambda graphs, *, fallback_index: (
        np.zeros(len(graphs), dtype=float),
        0.0,
    )

    graph_with_completion = nx.Graph()
    graph_with_completion.add_nodes_from([0, 1, 2])
    graph_with_completion.add_edge(0, 1)
    graph_without_completion = nx.Graph()
    graph_without_completion.add_nodes_from([0, 1, 2])
    graph_without_completion.add_edge(0, 2)
    root = generator._make_state(nx.empty_graph(3), parent=None, score=1.0, depth=0)
    cand_with_completion = generator._make_state(
        graph_with_completion,
        parent=root,
        score=None,
        depth=1,
    )
    cand_without_completion = generator._make_state(
        graph_without_completion,
        parent=root,
        score=None,
        depth=1,
    )
    feasible_candidates = []
    infeasible_candidates = []

    generator._partition_candidates_by_feasibility(
        [cand_with_completion, cand_without_completion],
        n_edges=2,
        max_total_edges=2,
        target=None,
        target_lambda=1.0,
        feasible_candidates=feasible_candidates,
        infeasible_candidates=infeasible_candidates,
    )
    generator._rank_feasible_candidates(feasible_candidates, fallback_index=-1)

    assert feasible_candidates[0] is cand_with_completion
    assert cand_with_completion["final_lookahead_feasible"] is True
    assert cand_without_completion["final_lookahead_feasible"] is False


def test_retain_unseen_candidates_allows_final_feasible_diversity_hit() -> None:
    generator = EdgeGenerator(
        feasibility_estimator=object(),
        graph_estimator=object(),
        enforce_diversity=True,
    )
    graph = nx.path_graph(3)
    root = generator._make_state(nx.empty_graph(3), parent=None, score=1.0, depth=0)
    cand = generator._make_state(graph, parent=root, score=1.0, depth=1)
    cand["connected_components"] = 1
    generator.diversity_memory_hash_set_ = {cand["graph_hash"]}
    search = generator._initialize_search_state(nx.empty_graph(3))

    retained = generator._retain_unseen_candidates(
        [cand],
        search=search,
        next_depth=1,
        beam_limit=1,
    )

    assert retained == [cand]


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
        max_total_edges=5,
        target=7,
        target_lambda=0.5,
        feasible_candidates=feasible_candidates,
        infeasible_candidates=infeasible_candidates,
    )

    assert len(feasible_candidates) == 1
    assert infeasible_candidates == []
    assert feasible_candidates[0]["risk_score"] == pytest.approx(0.4)
    assert feasible_candidates[0]["selection_score"] == pytest.approx(0.2)
