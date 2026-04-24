from __future__ import annotations

import warnings

import networkx as nx

from abstractgraph import node as node_operator
from abstractgraph.graphs import AbstractGraph
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
    component = generator._build_component_instance(ag, image_node=1, comp_id=7)
    assert component.comp_id == 7
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
    component = generator._build_component_instance(ag, image_node=1, comp_id=3)

    assert component.comp_id == 3
    assert component.subgraph.is_directed()
    assert set(component.subgraph.edges()) == {(0, 1)}


def test_conditional_generator_supports_canonical_radius_names() -> None:
    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=lambda ag: ag,
        nbits=6,
        base_cut_radius=2,
        interpretation_cut_radius=3,
    )
    assert generator.base_cut_radius == 2
    assert generator.interpretation_cut_radius == 3

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert generator.preimage_cut_radius == 2
        assert generator.image_cut_radius == 3

    assert len(caught) == 2


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
