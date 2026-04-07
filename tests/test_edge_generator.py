from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from abstractgraph_generative.edge_generator import EdgeGenerator


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
