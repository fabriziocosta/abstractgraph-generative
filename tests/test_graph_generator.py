from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from abstractgraph import node as node_operator
from abstractgraph_generative.graph_generator import GraphGenerator


class SizeVectorizer:
    def fit_transform(self, graphs):
        return self.transform(graphs)

    def transform(self, graphs):
        return np.asarray(
            [
                [
                    float(graph.number_of_nodes()),
                    float(graph.number_of_edges()),
                ]
                for graph in graphs
            ],
            dtype=float,
        )


class FakeEdgeGenerator:
    def __init__(self, generated_graph=None, *, fit_exception=None):
        self.generated_graph = generated_graph
        self.fit_exception = fit_exception
        self.store_calls = []
        self.fit_calls = []
        self.generate_calls = []

    def store(self, graphs, targets=None):
        self.store_calls.append((list(graphs), targets))
        return self

    def fit(self, graphs, targets=None):
        self.fit_calls.append((list(graphs), targets))
        if self.fit_exception is not None:
            raise self.fit_exception
        return self

    def generate(self, graph, n_edges, **kwargs):
        self.generate_calls.append((graph.copy(), n_edges, dict(kwargs)))
        if self.generated_graph is None:
            return None
        return self.generated_graph.copy()


class FakeConditionalGenerator:
    def __init__(self):
        self.fit_calls = []
        self.generate_calls = []

    def fit(self, graphs):
        self.fit_calls.append(list(graphs))
        return self

    def generate(self, n_samples=1, *, interpretation_graphs=None, **kwargs):
        self.generate_calls.append(
            {
                "n_samples": n_samples,
                "interpretation_graphs": list(interpretation_graphs or []),
                "kwargs": dict(kwargs),
            }
        )
        return [nx.path_graph(2) for _ in range(n_samples)]


def _labeled_path(n_nodes: int) -> nx.Graph:
    graph = nx.path_graph(n_nodes)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)
    return graph


def test_store_computes_and_aligns_interpretation_graphs() -> None:
    graphs = [_labeled_path(3), _labeled_path(4)]
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    )

    result = generator.store(graphs, targets=[0, 1])

    assert result is generator
    assert [graph.number_of_nodes() for graph in generator.stored_graphs_] == [3, 4]
    assert len(generator.stored_interpretation_graphs_) == len(graphs)
    assert generator.stored_targets_ == [0, 1]
    assert len(generator.stored_interpretation_hash_to_index_) == 2


def test_sample_uses_interpretation_neighbors_then_conditional_base_neighbors() -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3), nx.path_graph(4)]
    base_graphs = [_labeled_path(2), _labeled_path(3), _labeled_path(4)]
    generated_interpretation_graph = nx.path_graph(4)
    edge_generator = FakeEdgeGenerator(generated_interpretation_graph)
    conditional_generator = FakeConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=conditional_generator,
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
        seed=0,
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=1,
        n_conditional_neighbors=2,
        n_instances_per_sample=3,
        interpretation_edge_removal_size=0,
        random_state=0,
    )

    assert len(outputs) == 3
    assert len(edge_generator.fit_calls) == 1
    assert len(edge_generator.fit_calls[0][0]) == 1
    sampled_idx = generator.last_sampled_indices_[0]
    assert (
        edge_generator.generate_calls[0][1]
        == interpretation_graphs[sampled_idx].number_of_edges()
    )
    assert len(conditional_generator.fit_calls) == 1
    assert [graph.number_of_nodes() for graph in conditional_generator.fit_calls[0]] == [
        4,
        3,
    ]
    assert conditional_generator.generate_calls[0]["n_samples"] == 3
    assert len(conditional_generator.generate_calls[0]["interpretation_graphs"]) == 1


def test_sample_records_histories_for_successful_generation() -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3), nx.path_graph(4)]
    base_graphs = [_labeled_path(2), _labeled_path(3), _labeled_path(4)]
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(3)),
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=2,
        n_conditional_neighbors=2,
        n_instances_per_sample=2,
        random_state=1,
    )

    assert len(outputs) == 2
    assert len(generator.last_sampled_indices_) == 1
    assert len(generator.last_interpretation_neighbor_indices_history_) == 1
    assert len(generator.last_conditional_neighbor_indices_history_) == 1
    assert len(generator.last_generated_interpretation_graphs_) == 1
    assert len(generator.last_edge_generation_paths_) == 1
    assert len(generator.last_conditional_training_graphs_history_) == 1


def test_edge_stage_failure_skips_seed_without_success_histories() -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3)]
    base_graphs = [_labeled_path(2), _labeled_path(3)]
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(generated_graph=None),
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    with pytest.warns(RuntimeWarning, match="Edge stage failed"):
        outputs = generator.sample(n_samples=1, random_state=0)

    assert outputs == []
    assert len(generator.last_sampled_indices_) == 1
    assert len(generator.last_interpretation_neighbor_indices_history_) == 1
    assert generator.last_conditional_neighbor_indices_history_ == []
    assert generator.last_generated_interpretation_graphs_ == []
    assert generator.last_edge_generation_paths_ == []
    assert generator.last_conditional_training_graphs_history_ == []


def test_edge_fit_failure_skips_seed_without_success_histories() -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3)]
    base_graphs = [_labeled_path(2), _labeled_path(3)]
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(
            generated_graph=nx.path_graph(3),
            fit_exception=ValueError("bad fit"),
        ),
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    with pytest.warns(RuntimeWarning, match="failed while fitting"):
        outputs = generator.sample(n_samples=1, random_state=0)

    assert outputs == []
    assert len(generator.last_sampled_indices_) == 1
    assert len(generator.last_interpretation_neighbor_indices_history_) == 1
    assert generator.last_conditional_neighbor_indices_history_ == []
    assert generator.last_generated_interpretation_graphs_ == []
    assert generator.last_edge_generation_paths_ == []
    assert generator.last_conditional_training_graphs_history_ == []


def test_sample_adds_seed_to_edge_training_when_neighbors_have_no_edges() -> None:
    seed_interpretation = nx.Graph()
    seed_interpretation.add_edge(0, 1)
    empty_neighbor = nx.Graph()
    empty_neighbor.add_nodes_from([0, 1])
    interpretation_graphs = [empty_neighbor, seed_interpretation]
    base_graphs = [_labeled_path(2), _labeled_path(3)]
    edge_generator = FakeEdgeGenerator(nx.path_graph(2))
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    generator.sample(
        n_samples=1,
        n_interpretation_neighbors=1,
        n_conditional_neighbors=1,
        random_state=0,
    )

    assert generator.last_sampled_indices_ == [1]
    assert generator.last_interpretation_neighbor_indices_history_ == [[0, 1]]
    assert [graph.number_of_edges() for graph in edge_generator.fit_calls[0][0]] == [
        0,
        1,
    ]


def test_store_requires_at_least_two_graphs() -> None:
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    )

    with pytest.raises(ValueError, match="at least two"):
        generator.store([_labeled_path(2)])
