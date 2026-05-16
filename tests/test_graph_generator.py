from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances

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
        self.retrieval_transformer_ = None
        self.stored_retrieval_vectors_ = None
        self.stored_distance_matrix_ = None
        self.store_calls = []
        self.fit_calls = []
        self.generate_calls = []

    def store(self, graphs, targets=None):
        graph_list = list(graphs)
        self.store_calls.append((graph_list, targets))
        self.retrieval_transformer_ = SizeVectorizer()
        self.stored_retrieval_vectors_ = self.retrieval_transformer_.fit_transform(
            graph_list
        )
        self.stored_distance_matrix_ = pairwise_distances(
            self.stored_retrieval_vectors_
        )
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
    def __init__(
        self,
        *,
        generated_graph=None,
        generated_graphs=None,
        match_results=None,
        decomposition_function=None,
        nbits: int = 6,
        label_mode: str = "operator",
        context_vectorizer=None,
    ):
        self.debug = True
        self.debug_level = 2
        self.generated_graph = generated_graph
        self.generated_graphs = None if generated_graphs is None else list(generated_graphs)
        self.match_results = None if match_results is None else list(match_results)
        self.decomposition_function = (
            node_operator() if decomposition_function is None else decomposition_function
        )
        self.nbits = int(nbits)
        self.label_mode = label_mode
        self.context_vectorizer = context_vectorizer
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
        if self.generated_graphs is not None:
            outputs = self.generated_graphs[: int(n_samples)]
            if len(outputs) < int(n_samples):
                outputs.extend([self.generated_graphs[-1]] * (int(n_samples) - len(outputs)))
        elif self.generated_graph is not None:
            outputs = [self.generated_graph for _ in range(n_samples)]
        else:
            outputs = [nx.path_graph(2) for _ in range(n_samples)]
        return [graph.copy() for graph in outputs]

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


def test_constructor_accepts_single_generator_shorthand() -> None:
    conditional_generator = FakeConditionalGenerator(nbits=9, label_mode="histogram")

    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
        conditional_generator=conditional_generator,
    )

    assert generator.conditional_generator is conditional_generator
    assert generator.conditional_generators == [conditional_generator]


def test_constructor_accepts_conditional_generator_sequence() -> None:
    generators = [FakeConditionalGenerator(), FakeConditionalGenerator(nbits=8)]

    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
        conditional_generators=generators,
    )

    assert generator.conditional_generators == generators


def test_constructor_rejects_ambiguous_generator_arguments() -> None:
    with pytest.raises(ValueError, match="either conditional_generator"):
        GraphGenerator(
            edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
            conditional_generator=FakeConditionalGenerator(),
            conditional_generators=[FakeConditionalGenerator()],
        )


def test_constructor_requires_stage_interpretation_config() -> None:
    conditional_generator = FakeConditionalGenerator()
    del conditional_generator.decomposition_function

    with pytest.raises(ValueError, match=r"conditional_generators\[0\].decomposition_function"):
        GraphGenerator(
            edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
            conditional_generators=[conditional_generator],
        )

    conditional_generator = FakeConditionalGenerator()
    del conditional_generator.nbits

    with pytest.raises(ValueError, match=r"conditional_generators\[0\].nbits"):
        GraphGenerator(
            edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
            conditional_generators=[conditional_generator],
        )

    conditional_generator = FakeConditionalGenerator()
    del conditional_generator.label_mode

    with pytest.raises(ValueError, match=r"conditional_generators\[0\].label_mode"):
        GraphGenerator(
            edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
            conditional_generators=[conditional_generator],
        )


def test_constructor_propagates_debug_to_all_generators() -> None:
    edge_generator = FakeEdgeGenerator(nx.path_graph(2))
    first = FakeConditionalGenerator()
    second = FakeConditionalGenerator()
    first.debug_level = 0
    second.debug_level = 0

    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generators=[first, second],
        debug=True,
    )

    assert generator.debug is True
    assert edge_generator.debug is True
    assert edge_generator.verbose is True
    assert first.debug is True
    assert first.debug_level == 1
    assert second.debug is True
    assert second.debug_level == 1


def test_store_computes_hierarchy_levels_and_indexes_top_level() -> None:
    graphs = [_labeled_path(2), _labeled_path(3), _labeled_path(4)]
    edge_generator = FakeEdgeGenerator(nx.path_graph(2))
    lower_generator = FakeConditionalGenerator(context_vectorizer=SizeVectorizer())
    upper_generator = FakeConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generators=[lower_generator, upper_generator],
    )

    result = generator.store(graphs, targets=[0, 1, 1])

    assert result is generator
    assert len(generator.stored_level_graphs_) == 3
    assert [len(level) for level in generator.stored_level_graphs_] == [3, 3, 3]
    assert [graph.number_of_nodes() for graph in generator.stored_graphs_] == [2, 3, 4]
    assert [graph.number_of_nodes() for graph in generator.stored_interpretation_graphs_] == [
        graph.number_of_nodes() for graph in generator.stored_level_graphs_[-1]
    ]
    assert edge_generator.store_calls[0][1] == [0, 1, 1]
    assert len(edge_generator.store_calls[0][0]) == 3
    assert generator.conditional_retrieval_states_[0] is not None
    assert generator.conditional_retrieval_states_[1] is None


def test_store_requires_at_least_two_graphs() -> None:
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
        conditional_generator=FakeConditionalGenerator(),
    )

    with pytest.raises(ValueError, match="at least two"):
        generator.store([_labeled_path(2)])


def test_sample_bypasses_edge_stage_with_zero_edge_removal() -> None:
    edge_generator = FakeEdgeGenerator(nx.path_graph(4))
    conditional_generator = FakeConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=conditional_generator,
    ).store([_labeled_path(2), _labeled_path(3), _labeled_path(4)])

    outputs = generator.sample(
        n_samples=1,
        n_conditional_neighbors=2,
        n_instances_per_sample=3,
        interpretation_edge_removal_size=0,
        random_state=0,
        avoid_seed_components=False,
    )

    assert len(outputs) == 3
    assert edge_generator.fit_calls == []
    assert edge_generator.generate_calls == []
    assert conditional_generator.generate_calls[0]["n_samples"] == 3
    assert len(generator.last_level_generated_graphs_history_[0]) == 2
    assert len(generator.last_edge_generation_paths_[0]) == 2


def test_sample_uses_top_edge_stage_then_single_conditional_stage() -> None:
    edge_generator = FakeEdgeGenerator(nx.path_graph(5))
    conditional_generator = FakeConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=conditional_generator,
        require_new_interpretation_graph=False,
    ).store([_labeled_path(2), _labeled_path(3), _labeled_path(4)])

    outputs = generator.sample(
        n_samples=1,
        n_interpretation_neighbors=2,
        n_conditional_neighbors=1,
        n_instances_per_sample=2,
        random_state=0,
        max_seed_attempts=1,
        avoid_seed_components=False,
    )

    assert len(outputs) == 2
    assert len(edge_generator.fit_calls) == 1
    assert len(edge_generator.generate_calls) == 1
    assert len(conditional_generator.fit_calls) == 1
    assert generator.last_successful_sampled_indices_ == generator.last_sampled_indices_
    assert len(generator.last_level_neighbor_indices_history_[0][0]) == 1


def test_sample_walks_two_level_chain_without_multiplicative_fanout() -> None:
    top_generator = FakeConditionalGenerator(
        generated_graphs=[nx.path_graph(10), nx.path_graph(11), nx.path_graph(12)]
    )
    base_generator = FakeConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generators=[base_generator, top_generator],
        require_new_interpretation_graph=False,
    ).store([_labeled_path(2), _labeled_path(3), _labeled_path(4)])

    outputs = generator.sample(
        n_samples=1,
        n_conditional_neighbors=[2, 1],
        n_instances_per_sample=3,
        interpretation_edge_removal_size=0,
        random_state=0,
        max_seed_attempts=1,
        avoid_seed_components=False,
    )

    assert len(outputs) == 3
    assert top_generator.generate_calls[0]["n_samples"] == 3
    assert sum(call["n_samples"] for call in base_generator.generate_calls) == 3
    assert len(generator.last_level_generated_graphs_history_[0]) == 3
    assert len(generator.last_level_generated_graphs_history_[0][0]) == 3
    assert len(generator.last_level_generated_graphs_history_[0][1]) == 3
    assert len(generator.last_level_generated_graphs_history_[0][2]) == 1


def test_sample_accepts_per_stage_conditional_kwargs() -> None:
    base_generator = FakeConditionalGenerator()
    top_generator = FakeConditionalGenerator(generated_graph=nx.path_graph(6))
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generators=[base_generator, top_generator],
        require_new_interpretation_graph=False,
    ).store([_labeled_path(2), _labeled_path(3), _labeled_path(4)])

    outputs = generator.sample(
        n_samples=1,
        n_instances_per_sample=1,
        interpretation_edge_removal_size=0,
        random_state=0,
        max_seed_attempts=1,
        conditional_generate_kwargs=[
            {"random_state": 1, "n_samples": 999},
            {"random_state": 2, "interpretation_graphs": ["ignored"]},
        ],
        avoid_seed_components=False,
    )

    assert len(outputs) == 1
    assert base_generator.generate_calls[0]["kwargs"] == {"random_state": 1}
    assert top_generator.generate_calls[0]["kwargs"] == {"random_state": 2}


def test_sample_accepts_scalar_conditional_kwargs_for_every_stage() -> None:
    base_generator = FakeConditionalGenerator()
    top_generator = FakeConditionalGenerator(generated_graph=nx.path_graph(6))
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generators=[base_generator, top_generator],
        require_new_interpretation_graph=False,
    ).store([_labeled_path(2), _labeled_path(3), _labeled_path(4)])

    outputs = generator.sample(
        n_samples=1,
        n_instances_per_sample=1,
        interpretation_edge_removal_size=0,
        random_state=0,
        max_seed_attempts=1,
        conditional_generate_kwargs={"random_state": 7},
        avoid_seed_components=False,
    )

    assert len(outputs) == 1
    assert base_generator.generate_calls[0]["kwargs"] == {"random_state": 7}
    assert top_generator.generate_calls[0]["kwargs"] == {"random_state": 7}


def test_sample_retries_without_seed_avoidance() -> None:
    conditional_generator = EmptyWhenAvoidingSeedConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generator=conditional_generator,
    ).store([_labeled_path(2), _labeled_path(3), _labeled_path(4)])

    outputs = generator.sample(
        n_samples=1,
        n_conditional_neighbors=1,
        n_instances_per_sample=1,
        interpretation_edge_removal_size=0,
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
    conditional_generator = EmptyUntilSeedFallbackConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generator=conditional_generator,
    ).store([_labeled_path(2), _labeled_path(3), _labeled_path(4)])

    outputs = generator.sample(
        n_samples=1,
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
        in generator.last_level_neighbor_indices_history_[0][0][0]
    )


def test_sample_filters_outputs_that_do_not_match_stage_target() -> None:
    conditional_generator = FakeConditionalGenerator(match_results=[False, True])
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generator=conditional_generator,
    ).store([_labeled_path(2), _labeled_path(3), _labeled_path(4)])

    outputs = generator.sample(
        n_samples=1,
        n_conditional_neighbors=1,
        n_instances_per_sample=2,
        interpretation_edge_removal_size=0,
        random_state=0,
        max_seed_attempts=1,
        avoid_seed_components=False,
    )

    assert len(outputs) == 1
    assert len(conditional_generator.generate_calls[0]["match_checks"]) == 2


def test_sample_rejects_same_top_interpretation_graph_by_default() -> None:
    base_graphs = [_labeled_path(2), _labeled_path(3)]
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(),
        conditional_generator=FakeConditionalGenerator(),
    ).store(base_graphs)
    seed_idx = 1
    generator.edge_generator.generated_graph = generator.stored_interpretation_graphs_[
        seed_idx
    ]

    with pytest.warns(RuntimeWarning, match="same interpretation graph"):
        outputs = generator.sample(
            n_samples=1,
            n_interpretation_neighbors=1,
            n_conditional_neighbors=1,
            random_state=0,
            max_seed_attempts=1,
        )

    assert outputs == []
    assert generator.last_successful_sampled_indices_ == []
    assert generator.last_level_generated_graphs_history_ == []


def test_sample_uses_configured_same_interpretation_retry_limit() -> None:
    base_graphs = [_labeled_path(2), _labeled_path(3)]
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(),
        conditional_generator=FakeConditionalGenerator(),
        max_same_interpretation_retries=1,
    ).store(base_graphs)
    seed_graph = generator.stored_interpretation_graphs_[1]
    generator.edge_generator.generated_graphs = [seed_graph, seed_graph, nx.path_graph(4)]

    with pytest.warns(RuntimeWarning, match="after 1 retries"):
        outputs = generator.sample(
            n_samples=1,
            n_interpretation_neighbors=1,
            n_conditional_neighbors=1,
            random_state=0,
            max_seed_attempts=1,
        )

    assert outputs == []
    assert len(generator.edge_generator.generate_calls) == 2


def test_constructor_rejects_negative_same_interpretation_retry_limit() -> None:
    with pytest.raises(ValueError, match="max_same_interpretation_retries"):
        GraphGenerator(
            edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
            conditional_generator=FakeConditionalGenerator(),
            max_same_interpretation_retries=-1,
        )


def test_sample_records_chain_histories_for_successful_generation() -> None:
    base_generator = FakeConditionalGenerator()
    top_generator = FakeConditionalGenerator(generated_graph=nx.path_graph(5))
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(4)),
        conditional_generators=[base_generator, top_generator],
        require_new_interpretation_graph=False,
    ).store([_labeled_path(2), _labeled_path(3), _labeled_path(4)])

    outputs = generator.sample(
        n_samples=1,
        n_conditional_neighbors=[1, 1],
        n_instances_per_sample=2,
        random_state=0,
        max_seed_attempts=1,
        avoid_seed_components=False,
    )

    assert len(outputs) == 2
    assert len(generator.last_level_seed_graphs_history_) == 1
    assert len(generator.last_level_seed_graphs_history_[0]) == 3
    assert len(generator.last_level_generated_graphs_history_[0]) == 3
    assert len(generator.last_level_neighbor_indices_history_[0]) == 2
    assert len(generator.last_level_training_graphs_history_[0]) == 2
    assert len(generator.last_edge_generation_paths_) == 1


def test_edge_stage_failure_skips_seed_without_success_histories() -> None:
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(generated_graph=None),
        conditional_generator=FakeConditionalGenerator(),
    ).store([_labeled_path(2), _labeled_path(3)])

    with pytest.warns(RuntimeWarning, match="Edge stage failed"):
        outputs = generator.sample(n_samples=1, random_state=0, max_seed_attempts=1)

    assert outputs == []
    assert len(generator.last_sampled_indices_) == 1
    assert generator.last_successful_sampled_indices_ == []
    assert generator.last_level_generated_graphs_history_ == []
    assert generator.last_edge_generation_paths_ == []


def test_edge_fit_failure_skips_seed_without_success_histories() -> None:
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(
            generated_graph=nx.path_graph(3),
            fit_exception=ValueError("bad fit"),
        ),
        conditional_generator=FakeConditionalGenerator(),
    ).store([_labeled_path(2), _labeled_path(3)])

    with pytest.warns(RuntimeWarning, match="failed while fitting"):
        outputs = generator.sample(n_samples=1, random_state=0, max_seed_attempts=1)

    assert outputs == []
    assert len(generator.last_sampled_indices_) == 1
    assert generator.last_successful_sampled_indices_ == []
    assert generator.last_level_generated_graphs_history_ == []
    assert generator.last_edge_generation_paths_ == []


def test_sample_zero_edge_removal_does_not_require_label_coverage() -> None:
    edge_generator = FakeEdgeGenerator(nx.path_graph(1))
    conditional_generator = FakeConditionalGenerator()
    generator = GraphGenerator(
        edge_generator=edge_generator,
        conditional_generator=conditional_generator,
    ).store([_labeled_path(2), _labeled_path(3)])

    outputs = generator.sample(
        n_samples=1,
        n_conditional_neighbors=1,
        interpretation_edge_removal_size=0,
        random_state=0,
        max_seed_attempts=1,
    )

    assert len(outputs) == 1
    assert generator.last_interpretation_neighbor_indices_history_ == [[]]
    assert edge_generator.fit_calls == []
    assert edge_generator.generate_calls == []


def test_sample_rejects_invalid_per_stage_parameter_length() -> None:
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
        conditional_generators=[FakeConditionalGenerator(), FakeConditionalGenerator()],
    ).store([_labeled_path(2), _labeled_path(3)])

    with pytest.raises(ValueError, match="n_conditional_neighbors"):
        generator.sample(
            n_samples=1,
            n_conditional_neighbors=[1],
            interpretation_edge_removal_size=0,
        )


def test_sample_rejects_non_positive_neighbor_count() -> None:
    generator = GraphGenerator(
        edge_generator=FakeEdgeGenerator(nx.path_graph(2)),
        conditional_generator=FakeConditionalGenerator(),
    ).store([_labeled_path(2), _labeled_path(3)])

    with pytest.raises(ValueError, match="n_conditional_neighbors"):
        generator.sample(n_samples=1, n_conditional_neighbors=0)


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
