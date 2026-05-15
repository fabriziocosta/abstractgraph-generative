from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from abstractgraph import node as node_operator
from abstractgraph.hashing import hash_graph
from abstractgraph_ml.estimators import GraphEstimator
from abstractgraph_generative.edge_generator import EdgeGenerator
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


class EdgeOnlyVectorizer:
    def fit_transform(self, graphs, targets=None):
        return self.transform(graphs)

    def transform(self, graphs, targets=None):
        rows = []
        for graph in graphs:
            if graph.number_of_edges() == 0:
                raise ValueError("edgeless graphs are unsupported")
            rows.append([float(graph.number_of_nodes()), float(graph.number_of_edges())])
        return np.asarray(rows, dtype=float)


class FakeEdgeGenerator:
    def __init__(self, generated_graph=None, *, generated_graphs=None, fit_exception=None):
        self.generated_graph = generated_graph
        self.generated_graphs = None if generated_graphs is None else list(generated_graphs)
        self.fit_exception = fit_exception
        self.verbose = True
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
        if self.generated_graphs is not None:
            graph_index = min(len(self.generate_calls) - 1, len(self.generated_graphs) - 1)
            generated_graph = self.generated_graphs[graph_index]
        else:
            generated_graph = self.generated_graph
        if generated_graph is None:
            return None
        return generated_graph.copy()


class FakeConditionalGenerator:
    def __init__(self, *, match_results=None):
        self.debug = True
        self.debug_level = 2
        self.match_results = None if match_results is None else list(match_results)
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

    def _matches_target_interpretation(self, graph, target_interpretation_graph):
        if self.match_results is None:
            return True
        match_index = min(
            len(self.generate_calls[-1].setdefault("match_checks", [])),
            len(self.match_results) - 1,
        )
        self.generate_calls[-1]["match_checks"].append(
            (graph.copy(), target_interpretation_graph.copy())
        )
        return bool(self.match_results[match_index])

    def component_subgraph_hashes_for_graph(self, graph):
        return {hash_graph(graph)}


class EmptyWhenAvoidingSeedConditionalGenerator(FakeConditionalGenerator):
    def generate(self, n_samples=1, *, interpretation_graphs=None, **kwargs):
        self.generate_calls.append(
            {
                "n_samples": n_samples,
                "interpretation_graphs": list(interpretation_graphs or []),
                "kwargs": dict(kwargs),
            }
        )
        if "avoid_component_subgraph_hashes" in kwargs:
            return []
        return [nx.path_graph(2) for _ in range(n_samples)]


class EmptyUntilSeedFallbackConditionalGenerator(FakeConditionalGenerator):
    def generate(self, n_samples=1, *, interpretation_graphs=None, **kwargs):
        self.generate_calls.append(
            {
                "n_samples": n_samples,
                "interpretation_graphs": list(interpretation_graphs or []),
                "kwargs": dict(kwargs),
                "fit_size": len(self.fit_calls[-1]),
            }
        )
        if len(self.fit_calls[-1]) < 2:
            return []
        return [nx.path_graph(2) for _ in range(n_samples)]


class ShortFirstBatchConditionalGenerator(FakeConditionalGenerator):
    def generate(self, n_samples=1, *, interpretation_graphs=None, **kwargs):
        self.generate_calls.append(
            {
                "n_samples": n_samples,
                "interpretation_graphs": list(interpretation_graphs or []),
                "kwargs": dict(kwargs),
            }
        )
        batch_size = 1 if len(self.generate_calls) == 1 else int(n_samples)
        return [nx.path_graph(2) for _ in range(batch_size)]


class AlwaysFeasible:
    def fit(self, graphs):
        return self

    def predict(self, graphs):
        return [1] * len(graphs)


def _labeled_path(n_nodes: int) -> nx.Graph:
    graph = nx.path_graph(n_nodes)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)
    return graph


def _single_label_graph(label: str) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, label=label)
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


def test_constructor_propagates_default_debug_false_to_generators() -> None:
    edge_generator = FakeEdgeGenerator(nx.path_graph(2))
    conditional_generator = FakeConditionalGenerator()

    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=conditional_generator,
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    )

    assert generator.debug is False
    assert edge_generator.debug is False
    assert edge_generator.verbose is False
    assert conditional_generator.debug is False
    assert conditional_generator.debug_level == 0


def test_constructor_propagates_debug_true_to_generators() -> None:
    edge_generator = FakeEdgeGenerator(nx.path_graph(2))
    conditional_generator = FakeConditionalGenerator()
    conditional_generator.debug_level = 0

    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=conditional_generator,
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
        debug=True,
    )

    assert generator.debug is True
    assert edge_generator.debug is True
    assert edge_generator.verbose is True
    assert conditional_generator.debug is True
    assert conditional_generator.debug_level == 1


def test_sample_bypasses_edge_generator_when_no_interpretation_edges_removed() -> None:
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
    assert edge_generator.fit_calls == []
    assert edge_generator.generate_calls == []
    sampled_idx = generator.last_sampled_indices_[0]
    assert nx.is_isomorphic(
        generator.last_generated_interpretation_graphs_[0],
        interpretation_graphs[sampled_idx],
    )
    assert len(conditional_generator.fit_calls) == 1
    assert [graph.number_of_nodes() for graph in conditional_generator.fit_calls[0]] == [
        2,
        4,
    ]
    assert conditional_generator.generate_calls[0]["n_samples"] == 3
    assert len(conditional_generator.generate_calls[0]["interpretation_graphs"]) == 1
    assert generator.last_interpretation_neighbor_indices_history_ == [[]]
    assert len(generator.last_edge_generation_paths_) == 1
    assert len(generator.last_edge_generation_paths_[0]) == 2


def test_sample_deduplicates_conditional_neighbors_by_interpretation_hash() -> None:
    duplicate_a = nx.path_graph(2)
    duplicate_b = nx.path_graph(2)
    seed_interpretation = nx.path_graph(3)
    distinct_neighbor = nx.path_graph(4)
    interpretation_graphs = [
        duplicate_a,
        duplicate_b,
        seed_interpretation,
        distinct_neighbor,
    ]
    base_graphs = [
        _labeled_path(2),
        _labeled_path(20),
        _labeled_path(3),
        _labeled_path(4),
    ]
    conditional_generator = FakeConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(3)),
        conditional_generator=conditional_generator,
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
        require_new_interpretation_graph=False,
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=1,
        n_conditional_neighbors=3,
        interpretation_edge_removal_size=0,
        random_state=0,
        max_seed_attempts=1,
    )

    assert len(outputs) == 1
    assert generator.last_sampled_indices_ == [3]
    assert generator.last_conditional_neighbor_indices_history_ == [[2, 0]]
    assert [graph.number_of_nodes() for graph in conditional_generator.fit_calls[0]] == [
        3,
        2,
    ]


def test_sample_can_include_seed_in_conditional_neighbors_when_requested() -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3), nx.path_graph(4)]
    base_graphs = [_labeled_path(2), _labeled_path(3), _labeled_path(4)]
    conditional_generator = FakeConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(3)),
        conditional_generator=conditional_generator,
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
        require_new_interpretation_graph=False,
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=1,
        n_conditional_neighbors=2,
        interpretation_edge_removal_size=0,
        exclude_seed_from_conditional_neighbors=False,
        random_state=0,
        max_seed_attempts=1,
    )

    assert len(outputs) == 1
    assert generator.last_sampled_indices_ == [1]
    assert generator.last_conditional_neighbor_indices_history_ == [[1, 0]]


def test_sample_full_interpretation_edge_removal_removes_all_seed_edges() -> None:
    interpretation_graphs = [nx.path_graph(3), nx.cycle_graph(4), nx.path_graph(5)]
    base_graphs = [_labeled_path(3), _labeled_path(4), _labeled_path(5)]
    edge_generator = FakeEdgeGenerator(nx.path_graph(4))
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=2,
        n_conditional_neighbors=1,
        interpretation_edge_removal_size=1.0,
        random_state=0,
    )

    assert len(outputs) == 1
    assert len(edge_generator.generate_calls) == 1
    sampled_idx = generator.last_sampled_indices_[0]
    start_graph, target_edges, _kwargs = edge_generator.generate_calls[0]
    assert start_graph.number_of_edges() == 0
    assert target_edges == interpretation_graphs[sampled_idx].number_of_edges()


def test_sample_deduplicates_interpretation_neighbors_by_graph_hash() -> None:
    duplicate_a = nx.path_graph(2)
    duplicate_b = nx.path_graph(2)
    distinct_neighbor = nx.path_graph(4)
    seed_interpretation = nx.path_graph(3)
    interpretation_graphs = [
        duplicate_a,
        duplicate_b,
        distinct_neighbor,
        seed_interpretation,
    ]
    base_graphs = [
        _labeled_path(2),
        _labeled_path(2),
        _labeled_path(4),
        _labeled_path(3),
    ]
    edge_generator = FakeEdgeGenerator(nx.path_graph(4))
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=3,
        n_conditional_neighbors=1,
        random_state=0,
        max_seed_attempts=1,
    )

    assert len(outputs) == 1
    edge_training_graphs = edge_generator.fit_calls[0][0]
    edge_training_hashes = [hash_graph(graph) for graph in edge_training_graphs]
    assert len(edge_training_hashes) == 2
    assert len(set(edge_training_hashes)) == len(edge_training_hashes)
    assert generator.last_interpretation_neighbor_indices_history_ == [[0, 2]]


def test_sample_logs_edge_neighbor_distances_when_debug_enabled(capsys) -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3), nx.path_graph(4)]
    base_graphs = [_labeled_path(2), _labeled_path(3), _labeled_path(4)]
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
        debug=True,
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=2,
        n_conditional_neighbors=1,
        random_state=0,
        max_seed_attempts=1,
    )

    assert len(outputs) == 1
    out = capsys.readouterr().out
    assert "[graph-generator edge]" in out
    assert "[graph-generator sample]" in out
    assert "event=seed_start" in out
    assert "event=seed_success" in out
    assert "currently_generated=1/1" in out
    assert "attempted_seeds=1/1" in out
    assert "generated_graphs=1" in out
    assert "seed_idx=1" in out
    assert "neighbor_indices=[0, 2]" in out
    assert "neighbor_distances=[1.4142, 1.4142]" in out
    assert generator.last_interpretation_neighbor_distances_history_ == [
        pytest.approx([np.sqrt(2.0), np.sqrt(2.0)])
    ]


def test_sample_logs_skip_progress_when_debug_enabled(capsys) -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3)]
    base_graphs = [_labeled_path(2), _labeled_path(3)]
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(3)),
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
        debug=True,
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    with pytest.warns(RuntimeWarning, match="same interpretation graph"):
        outputs = generator.sample(
            n_samples=1,
            n_interpretation_neighbors=1,
            n_conditional_neighbors=1,
            random_state=0,
            max_seed_attempts=1,
        )

    assert outputs == []
    out = capsys.readouterr().out
    assert "event=seed_start" in out
    assert "event=seed_skip" in out
    assert "currently_generated=0/1" in out
    assert "attempted_seeds=1/1" in out
    assert "reason=edge_generation_failed" in out


def test_sample_rejects_generated_interpretation_graph_matching_seed_by_default() -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3)]
    base_graphs = [_labeled_path(2), _labeled_path(3)]
    edge_generator = FakeEdgeGenerator(nx.path_graph(3))
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    with pytest.warns(RuntimeWarning, match="same interpretation graph"):
        outputs = generator.sample(
            n_samples=1,
            n_interpretation_neighbors=1,
            n_conditional_neighbors=1,
            random_state=0,
            max_seed_attempts=1,
        )

    assert outputs == []
    assert generator.last_sampled_indices_ == [1]
    assert generator.last_successful_sampled_indices_ == []
    assert generator.last_seed_graphs_ == []
    assert generator.last_seed_interpretation_graphs_ == []
    assert generator.last_generated_interpretation_graphs_ == []
    assert len(edge_generator.generate_calls) == 4


def test_sample_retries_same_interpretation_graph_until_new_graph() -> None:
    seed_interpretation = nx.path_graph(3)
    new_interpretation = nx.path_graph(4)
    interpretation_graphs = [nx.path_graph(2), seed_interpretation]
    base_graphs = [_labeled_path(2), _labeled_path(3)]
    edge_generator = FakeEdgeGenerator(
        generated_graphs=[
            seed_interpretation,
            seed_interpretation,
            new_interpretation,
        ]
    )
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=1,
        n_conditional_neighbors=1,
        random_state=0,
        max_seed_attempts=1,
    )

    assert len(outputs) == 1
    assert len(edge_generator.generate_calls) == 3
    assert nx.is_isomorphic(
        generator.last_generated_interpretation_graphs_[0],
        new_interpretation,
    )


def test_sample_filters_conditional_outputs_that_do_not_match_target() -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3), nx.path_graph(4)]
    base_graphs = [_labeled_path(2), _labeled_path(3), _labeled_path(4)]
    conditional_generator = FakeConditionalGenerator(match_results=[False, True])
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generator=conditional_generator,
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=2,
        n_conditional_neighbors=1,
        n_instances_per_sample=2,
        random_state=0,
        max_seed_attempts=1,
    )

    assert len(outputs) == 1
    assert len(conditional_generator.generate_calls[0]["match_checks"]) == 2
    assert len(generator.last_generated_interpretation_graphs_) == 1


def test_sample_passes_seed_subgraph_hashes_to_conditional_generation() -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3), nx.path_graph(4)]
    base_graphs = [_labeled_path(2), _labeled_path(3), _labeled_path(4)]
    conditional_generator = FakeConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generator=conditional_generator,
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=2,
        n_conditional_neighbors=1,
        n_instances_per_sample=1,
        random_state=0,
        max_seed_attempts=1,
    )

    assert len(outputs) == 1
    sampled_idx = generator.last_sampled_indices_[0]
    assert conditional_generator.generate_calls[0]["kwargs"][
        "avoid_component_subgraph_hashes"
    ] == {hash_graph(base_graphs[sampled_idx])}


def test_sample_retries_conditional_generation_without_seed_avoidance() -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3), nx.path_graph(4)]
    base_graphs = [_labeled_path(2), _labeled_path(3), _labeled_path(4)]
    conditional_generator = EmptyWhenAvoidingSeedConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generator=conditional_generator,
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=2,
        n_conditional_neighbors=1,
        n_instances_per_sample=1,
        random_state=0,
        max_seed_attempts=1,
    )

    assert len(outputs) == 1
    assert len(conditional_generator.generate_calls) == 2
    assert "avoid_component_subgraph_hashes" in conditional_generator.generate_calls[0][
        "kwargs"
    ]
    assert "avoid_component_subgraph_hashes" not in conditional_generator.generate_calls[
        1
    ]["kwargs"]


def test_sample_refits_with_seed_when_seedless_neighbors_cannot_generate() -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3), nx.path_graph(4)]
    base_graphs = [_labeled_path(2), _labeled_path(3), _labeled_path(4)]
    conditional_generator = EmptyUntilSeedFallbackConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generator=conditional_generator,
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=2,
        n_conditional_neighbors=1,
        n_instances_per_sample=1,
        interpretation_edge_removal_size=0,
        random_state=0,
        max_seed_attempts=1,
    )

    assert len(outputs) == 1
    assert len(conditional_generator.fit_calls) == 2
    assert len(conditional_generator.fit_calls[0]) == 1
    assert len(conditional_generator.fit_calls[1]) == 2
    assert (
        generator.last_sampled_indices_[0]
        in generator.last_conditional_neighbor_indices_history_[0]
    )


def test_sample_records_generated_graph_batches_per_successful_seed() -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3), nx.path_graph(4)]
    base_graphs = [_labeled_path(2), _labeled_path(3), _labeled_path(4)]
    conditional_generator = ShortFirstBatchConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generator=conditional_generator,
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=2,
        n_interpretation_neighbors=2,
        n_conditional_neighbors=1,
        n_instances_per_sample=3,
        interpretation_edge_removal_size=0,
        random_state=0,
        max_seed_attempts=2,
        avoid_seed_components=False,
    )

    assert len(outputs) == 4
    assert [len(batch) for batch in generator.last_generated_graphs_history_] == [
        1,
        3,
    ]


def test_sample_uses_configured_same_interpretation_retry_limit() -> None:
    seed_interpretation = nx.path_graph(3)
    interpretation_graphs = [nx.path_graph(2), seed_interpretation]
    base_graphs = [_labeled_path(2), _labeled_path(3)]
    edge_generator = FakeEdgeGenerator(
        generated_graphs=[
            seed_interpretation,
            seed_interpretation,
            nx.path_graph(4),
        ]
    )
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
        max_same_interpretation_retries=1,
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    with pytest.warns(RuntimeWarning, match="after 1 retries"):
        outputs = generator.sample(
            n_samples=1,
            n_interpretation_neighbors=1,
            n_conditional_neighbors=1,
            random_state=0,
            max_seed_attempts=1,
        )

    assert outputs == []
    assert len(edge_generator.generate_calls) == 2


def test_constructor_rejects_negative_same_interpretation_retry_limit() -> None:
    with pytest.raises(ValueError, match="max_same_interpretation_retries"):
        GraphGenerator(
            edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
            conditional_generator=FakeConditionalGenerator(),
            decomposition_function=node_operator(),
            nbits=6,
            interpretation_neighbor_vectorizer=SizeVectorizer(),
            max_same_interpretation_retries=-1,
        )


def test_sample_can_allow_generated_interpretation_graph_matching_seed() -> None:
    interpretation_graphs = [nx.path_graph(2), nx.path_graph(3)]
    base_graphs = [_labeled_path(2), _labeled_path(3)]
    edge_generator = FakeEdgeGenerator(nx.path_graph(3))
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
        require_new_interpretation_graph=False,
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=1,
        n_conditional_neighbors=1,
        random_state=0,
        max_seed_attempts=1,
    )

    assert len(outputs) == 1
    assert generator.last_successful_sampled_indices_ == [1]
    assert len(generator.last_seed_graphs_) == 1
    assert len(generator.last_seed_interpretation_graphs_) == 1


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
    assert len(generator.last_successful_sampled_indices_) == 1
    assert generator.last_successful_sampled_indices_ == generator.last_sampled_indices_
    assert len(generator.last_interpretation_neighbor_indices_history_) == 1
    assert len(generator.last_conditional_neighbor_indices_history_) == 1
    assert len(generator.last_seed_graphs_) == 1
    assert len(generator.last_seed_interpretation_graphs_) == 1
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
        outputs = generator.sample(n_samples=1, random_state=0, max_seed_attempts=1)

    assert outputs == []
    assert len(generator.last_sampled_indices_) == 1
    assert generator.last_successful_sampled_indices_ == []
    assert len(generator.last_interpretation_neighbor_indices_history_) == 1
    assert generator.last_conditional_neighbor_indices_history_ == []
    assert generator.last_seed_graphs_ == []
    assert generator.last_seed_interpretation_graphs_ == []
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
        outputs = generator.sample(n_samples=1, random_state=0, max_seed_attempts=1)

    assert outputs == []
    assert len(generator.last_sampled_indices_) == 1
    assert generator.last_successful_sampled_indices_ == []
    assert len(generator.last_interpretation_neighbor_indices_history_) == 1
    assert generator.last_conditional_neighbor_indices_history_ == []
    assert generator.last_seed_graphs_ == []
    assert generator.last_seed_interpretation_graphs_ == []
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
        require_new_interpretation_graph=False,
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


def test_sample_extends_interpretation_neighbors_for_label_coverage() -> None:
    seed_interpretation = _single_label_graph("rare")
    near_neighbor = _single_label_graph("common")
    covering_neighbor = _single_label_graph("rare")
    interpretation_graphs = [near_neighbor, seed_interpretation, covering_neighbor]
    base_graphs = [_labeled_path(2), _labeled_path(3), _labeled_path(4)]
    edge_generator = FakeEdgeGenerator(nx.path_graph(1))
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
    assert generator.last_interpretation_neighbor_indices_history_ == [[2]]


def test_sample_skips_seed_when_interpretation_labels_are_not_covered() -> None:
    seed_interpretation = _single_label_graph("rare")
    neighbor = _single_label_graph("common")
    interpretation_graphs = [neighbor, seed_interpretation]
    base_graphs = [_labeled_path(2), _labeled_path(3)]
    edge_generator = FakeEdgeGenerator(nx.path_graph(1))
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    with pytest.warns(RuntimeWarning, match="covers the sampled seed"):
        outputs = generator.sample(
            n_samples=1,
            n_interpretation_neighbors=1,
            n_conditional_neighbors=1,
            random_state=0,
            max_seed_attempts=1,
        )

    assert outputs == []
    assert generator.last_sampled_indices_ == [1]
    assert generator.last_successful_sampled_indices_ == []
    assert generator.last_interpretation_neighbor_indices_history_ == []
    assert edge_generator.fit_calls == []


def test_sample_zero_edge_removal_does_not_require_label_coverage() -> None:
    seed_interpretation = _single_label_graph("rare")
    neighbor = _single_label_graph("common")
    interpretation_graphs = [neighbor, seed_interpretation]
    base_graphs = [_labeled_path(2), _labeled_path(3)]
    edge_generator = FakeEdgeGenerator(nx.path_graph(1))
    conditional_generator = FakeConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=conditional_generator,
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=1,
        n_conditional_neighbors=1,
        interpretation_edge_removal_size=0,
        random_state=0,
        max_seed_attempts=1,
    )

    assert len(outputs) == 1
    assert generator.last_sampled_indices_ == [1]
    assert generator.last_successful_sampled_indices_ == [1]
    assert generator.last_interpretation_neighbor_indices_history_ == [[]]
    assert edge_generator.fit_calls == []
    assert edge_generator.generate_calls == []


def test_sample_retries_distinct_seeds_until_success() -> None:
    unsupported = _single_label_graph("rare")
    neighbor = _single_label_graph("common")
    supported = _single_label_graph("common")
    interpretation_graphs = [neighbor, unsupported, supported]
    base_graphs = [_labeled_path(2), _labeled_path(3), _labeled_path(4)]
    edge_generator = FakeEdgeGenerator(nx.path_graph(1))
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=FakeConditionalGenerator(),
        decomposition_function=node_operator(),
        nbits=6,
        interpretation_neighbor_vectorizer=SizeVectorizer(),
    ).store(base_graphs, interpretation_graphs=interpretation_graphs)

    with pytest.warns(RuntimeWarning, match="covers the sampled seed"):
        outputs = generator.sample(
            n_samples=1,
            n_interpretation_neighbors=1,
            n_conditional_neighbors=1,
            random_state=0,
            max_seed_attempts=3,
        )

    assert len(outputs) == 1
    assert len(generator.last_sampled_indices_) > 1
    assert generator.last_sampled_indices_[0] == 1
    assert generator.last_successful_sampled_indices_ == [2]
    assert len(edge_generator.fit_calls) == 1


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


def test_edge_generator_does_not_train_graph_estimator_on_edgeless_fragments() -> None:
    graph_estimator = GraphEstimator(
        transformer=EdgeOnlyVectorizer(),
        estimator=RandomForestClassifier(n_estimators=5, random_state=0),
        postprocessor=None,
    )
    generator = EdgeGenerator(
        partial_feasibility_estimator=AlwaysFeasible(),
        final_feasibility_estimator=AlwaysFeasible(),
        graph_estimator=graph_estimator,
        n_negative_per_positive=1,
        n_replicates=1,
        fit_n_jobs=1,
        seed=0,
    )

    generator.fit([nx.path_graph(3)])

    assert any(graph.number_of_edges() == 0 for graph, _ in generator.dataset_)
    assert generator.targets_.tolist() == [1, 0]
