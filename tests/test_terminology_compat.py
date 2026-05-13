from __future__ import annotations

import random

import networkx as nx
import pytest

from abstractgraph import node as node_operator
from abstractgraph.graphs import AbstractGraph, graph_to_abstract_graph
from abstractgraph_generative.autoregressive import generate_pruning_sequences
from abstractgraph_generative.conditional import ConditionalAutoregressiveGenerator


def test_conditional_component_builder_reads_mapped_subgraph() -> None:
    graph = nx.path_graph(3)
    for node_id in graph.nodes:
        graph.nodes[node_id]["label"] = str(node_id)

    ag = AbstractGraph(graph=graph)
    ag.create_default_interpretation_node()
    ag.create_interpretation_node_with_subgraph_from_nodes([0, 1])
    ag.create_interpretation_node_with_subgraph_from_nodes([1, 2])

    generator = ConditionalAutoregressiveGenerator(decomposition_function=lambda x: x, nbits=6)
    component = generator._build_component_instance(ag, interpretation_node=1, comp_id=7)
    assert component.comp_id == 7
    assert isinstance(component.interpretation_type, int)
    assert component.subgraph.number_of_nodes() == 2


def test_conditional_component_builder_accepts_directed_mapped_subgraph() -> None:
    graph = nx.DiGraph()
    graph.add_node(0, label="0")
    graph.add_node(1, label="1")
    graph.add_node(2, label="2")
    graph.add_edge(0, 1, label="x")
    graph.add_edge(1, 2, label="y")

    ag = AbstractGraph(graph=graph)
    ag.create_default_interpretation_node()
    ag.create_interpretation_node_with_subgraph_from_edges([(0, 1)])
    ag.create_interpretation_node_with_subgraph_from_edges([(1, 2)])

    generator = ConditionalAutoregressiveGenerator(decomposition_function=lambda x: x, nbits=6)
    component = generator._build_component_instance(ag, interpretation_node=1, comp_id=3)

    assert component.comp_id == 3
    assert component.subgraph.is_directed()
    assert component.subgraph.number_of_edges() == 1


def test_conditional_generator_supports_canonical_radius_names() -> None:
    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=lambda ag: ag,
        nbits=6,
        base_cut_radius=2,
        interpretation_cut_radius=3,
    )
    assert generator.base_cut_radius == 2
    assert generator.interpretation_cut_radius == 3


def test_conditional_generator_debug_false_disables_positive_debug_level() -> None:
    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=lambda ag: ag,
        nbits=6,
        debug=False,
        debug_level=1,
    )

    assert generator.debug is False
    assert generator.debug_level == 0


def test_generate_accepts_interpretation_graphs_alias() -> None:
    graph = nx.path_graph(3)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)

    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=node_operator(),
        nbits=6,
        base_cut_radius=0,
        interpretation_cut_radius=0,
    )
    generator.fit([graph])

    outputs = generator.generate(
        n_samples=1,
        interpretation_graphs=[generator.interpretation_graph_pool[0].copy()],
        random_state=0,
        max_backtracks=10,
        max_attempts_per_sample=1,
        max_total_attempts=1,
    )
    assert isinstance(outputs, list)


@pytest.mark.parametrize("label_mode", ["operator_hash", "graph_hash", "histogram"])
def test_conditional_generator_checks_generated_interpretation_postcondition(label_mode: str) -> None:
    def decomposition(ag: AbstractGraph) -> AbstractGraph:
        ag.interpretation_graph = nx.Graph()
        if ag.base_graph.graph.get("kind") == "cycle_tree":
            ag.create_interpretation_node_with_subgraph_from_nodes(
                ag.base_graph.nodes,
                meta={"user_name": "cyc"},
            )
            ag.create_interpretation_node_with_subgraph_from_nodes(
                ag.base_graph.nodes,
                meta={"user_name": "tree"},
            )
            ag.interpretation_graph.add_edge(0, 1)
        else:
            ag.create_interpretation_node_with_subgraph_from_nodes(
                ag.base_graph.nodes,
                meta={"user_name": "tree"},
            )
        return ag

    target_graph = nx.path_graph(3)
    target_graph.graph["kind"] = "cycle_tree"
    generated_graph = nx.path_graph(3)
    generated_graph.graph["kind"] = "tree_only"

    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=decomposition,
        nbits=6,
        label_mode=label_mode,
    )

    target_ag = graph_to_abstract_graph(
        target_graph,
        decomposition_function=decomposition,
        nbits=6,
        label_mode=label_mode,
    )

    assert not generator._matches_target_interpretation(
        generated_graph,
        target_ag.interpretation_graph,
    )
    assert generator._matches_target_interpretation(
        target_graph,
        target_ag.interpretation_graph,
    )


def test_conditional_generator_store_prepares_local_neighbor_context() -> None:
    graphs = []
    for n_nodes in (3, 4, 5):
        graph = nx.path_graph(n_nodes)
        for node in graph.nodes:
            graph.nodes[node]["label"] = str(node % 2)
        graphs.append(graph)

    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=node_operator(),
        nbits=6,
        base_cut_radius=0,
        interpretation_cut_radius=0,
        n_jobs=1,
    )
    generator.store(graphs)

    pool = generator._prepare_stored_generation_context(
        rng=random.Random(0),
        n_neighbors=1,
    )

    assert len(pool) == 1
    assert generator.last_sampled_index_ is not None
    assert len(generator.last_neighbor_indices_) == 1
    assert len(generator.last_generation_training_graphs_) == 1
    assert generator._is_fitted


def test_conditional_generator_expands_neighbors_for_signature_coverage(monkeypatch) -> None:
    graphs = []
    for n_nodes in (3, 4, 5):
        graph = nx.path_graph(n_nodes)
        for node in graph.nodes:
            graph.nodes[node]["label"] = str(node % 2)
        graphs.append(graph)

    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=node_operator(),
        nbits=6,
        base_cut_radius=0,
        interpretation_cut_radius=0,
        n_jobs=1,
        debug=False,
        debug_level=0,
    )
    generator.store(graphs)

    target = nx.Graph()
    target.add_node(0, signature=("a", 1))
    target.add_node(1, signature=("b", 1))
    target.add_edge(0, 1)

    monkeypatch.setattr(
        generator,
        "_stored_neighbor_indices",
        lambda _index, *, n_neighbors: [1, 2][:n_neighbors],
    )
    monkeypatch.setattr(generator, "_stored_interpretation_graph", lambda _index: target.copy())
    monkeypatch.setattr(
        generator,
        "_compute_target_signatures",
        lambda graph: {node: graph.nodes[node]["signature"] for node in graph.nodes},
    )
    monkeypatch.setattr(
        generator,
        "_stored_signature_set",
        lambda index: {("a", 1)} if index == 1 else {("b", 1)},
    )

    pool = generator._prepare_stored_generation_context(
        rng=random.Random(0),
        n_neighbors=1,
        neighbor_coverage_factor=3,
        max_seed_retries=1,
    )

    assert len(pool) == 1
    assert generator.last_neighbor_indices_ == [1, 2]
    assert len(generator.last_generation_training_graphs_) == 2
    assert generator._is_fitted


def test_conditional_generator_skips_uncovered_stored_seeds(monkeypatch) -> None:
    graphs = []
    for n_nodes in (3, 4, 5):
        graph = nx.path_graph(n_nodes)
        for node in graph.nodes:
            graph.nodes[node]["label"] = str(node % 2)
        graphs.append(graph)

    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=node_operator(),
        nbits=6,
        base_cut_radius=0,
        interpretation_cut_radius=0,
        n_jobs=1,
        debug=False,
        debug_level=0,
    )
    generator.store(graphs)

    target = nx.Graph()
    target.add_node(0, signature=("missing", 1))

    monkeypatch.setattr(
        generator,
        "_stored_neighbor_indices",
        lambda _index, *, n_neighbors: [1, 2][:n_neighbors],
    )
    monkeypatch.setattr(generator, "_stored_interpretation_graph", lambda _index: target.copy())
    monkeypatch.setattr(
        generator,
        "_compute_target_signatures",
        lambda graph: {node: graph.nodes[node]["signature"] for node in graph.nodes},
    )
    monkeypatch.setattr(generator, "_stored_signature_set", lambda _index: {("covered", 1)})

    with pytest.warns(RuntimeWarning, match="Could not find a stored seed graph"):
        pool = generator._prepare_stored_generation_context(
            rng=random.Random(0),
            n_neighbors=1,
            neighbor_coverage_factor=2,
            max_seed_retries=2,
        )

    assert pool == []
    assert generator.last_sampled_index_ is None
    assert generator.last_neighbor_indices_ == []
    assert generator.last_generation_training_graphs_ == []


def test_conditional_generator_sample_forwards_without_stored_graphs(monkeypatch) -> None:
    graph = nx.path_graph(3)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)

    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=node_operator(),
        nbits=6,
        base_cut_radius=0,
        interpretation_cut_radius=0,
    )

    seen = {}

    def fake_generate(*, n_samples=1, **kwargs):
        seen["n_samples"] = n_samples
        seen["kwargs"] = kwargs
        return [graph]

    monkeypatch.setattr(generator, "generate", fake_generate)

    assert generator.sample(n_samples=3, n_neighbors=2) == [graph]
    assert seen == {"n_samples": 3, "kwargs": {"n_neighbors": 2}}


def test_conditional_generator_sample_uses_n_samples_as_stored_seed_count(monkeypatch) -> None:
    graph = nx.path_graph(3)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)

    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=node_operator(),
        nbits=6,
        base_cut_radius=0,
        interpretation_cut_radius=0,
    )
    generator.stored_graphs_ = [graph]

    calls = []

    def fake_generate(*, n_samples=1, **kwargs):
        calls.append({"n_samples": n_samples, "kwargs": kwargs})
        return [graph.copy() for _ in range(n_samples)]

    monkeypatch.setattr(generator, "generate", fake_generate)

    outputs = generator.sample(
        n_samples=3,
        n_instances_per_sample=2,
        n_neighbors=2,
        random_state=0,
    )

    assert len(outputs) == 6
    assert [call["n_samples"] for call in calls] == [2, 2, 2]
    assert [call["kwargs"]["n_neighbors"] for call in calls] == [2, 2, 2]
    assert all(isinstance(call["kwargs"]["random_state"], int) for call in calls)
    assert len({call["kwargs"]["random_state"] for call in calls}) == 3
    assert generator.last_sampled_indices_ == []


def test_conditional_generator_sample_records_stored_seed_history(monkeypatch) -> None:
    graph = nx.path_graph(3)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)

    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=node_operator(),
        nbits=6,
        base_cut_radius=0,
        interpretation_cut_radius=0,
    )
    generator.stored_graphs_ = [graph, graph.copy(), graph.copy()]

    sampled_indices = iter([2, 0])

    def fake_generate(*, n_samples=1, **_kwargs):
        sampled_index = next(sampled_indices)
        generator.last_sampled_index_ = sampled_index
        generator.last_neighbor_indices_ = [1]
        generator.last_generation_training_graphs_ = [graph]
        return [graph.copy() for _ in range(n_samples)]

    monkeypatch.setattr(generator, "generate", fake_generate)

    outputs = generator.sample(n_samples=2, n_instances_per_sample=1, random_state=0)

    assert len(outputs) == 2
    assert generator.last_sampled_indices_ == [2, 0]
    assert generator.last_neighbor_indices_history_ == [[1], [1]]
    assert [len(graphs) for graphs in generator.last_generation_training_graphs_history_] == [1, 1]


def test_generate_pruning_sequences_supports_canonical_interpretation_aliases() -> None:
    graph = nx.path_graph(3)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)

    interpretation_graph = AbstractGraph(graph=graph)
    interpretation_graph.create_interpretation_node_with_subgraph_from_nodes([0, 1])
    fixed_interpretation_graph = interpretation_graph.interpretation_graph.copy()

    outputs, interpretation_steps = generate_pruning_sequences(
        graph,
        min_nodes_for_pruning=1,
        decomposition_function=node_operator(),
        nbits=6,
        association_aware=True,
        fixed_interpretation_graph=fixed_interpretation_graph,
        return_interpretation_steps=True,
        include_start=True,
        seed=0,
    )
    assert isinstance(outputs, list)
    assert isinstance(interpretation_steps, list)
    assert interpretation_steps
