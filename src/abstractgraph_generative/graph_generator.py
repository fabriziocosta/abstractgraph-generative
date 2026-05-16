"""Hierarchical graph generation over interpretation graph levels."""

from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
from abstractgraph.graphs import graph_to_abstract_graph, is_simple_graph
from abstractgraph.hashing import hash_graph
from sklearn.metrics import pairwise_distances

from abstractgraph_generative.conditional import ConditionalAutoregressiveGenerator
from abstractgraph_generative.edge_generator import EdgeGenerator, remove_edges


class _FallbackGraphVectorizer:
    """Simple graph descriptor vectorizer for conditional-level retrieval."""

    def fit_transform(self, graphs):
        return self.transform(graphs)

    def transform(self, graphs):
        rows = []
        for graph in graphs:
            degrees = [int(deg) for _node, deg in graph.degree()]
            rows.append(
                [
                    float(graph.number_of_nodes()),
                    float(graph.number_of_edges()),
                    float(max(degrees, default=0)),
                    float(sum(degrees) / max(1, len(degrees))),
                ]
            )
        return np.asarray(rows, dtype=float)


class GraphGenerator:
    """Generate base graphs through a hierarchy of conditional generators.

    ``conditional_generators`` are ordered bottom-up. Generator ``i`` maps
    stored level ``i`` graphs to level ``i + 1`` interpretation graphs. The
    edge generator operates on the final top interpretation level. Sampling
    creates a top-level target, then applies the conditional generators in
    reverse order until base graphs are produced.
    """

    def __init__(
        self,
        edge_generator: EdgeGenerator,
        conditional_generator: ConditionalAutoregressiveGenerator | None = None,
        *,
        conditional_generators: Sequence[ConditionalAutoregressiveGenerator]
        | None = None,
        seed: int | None = None,
        debug: bool = False,
        require_new_interpretation_graph: bool = True,
        max_same_interpretation_retries: int = 3,
    ):
        max_same_interpretation_retries = int(max_same_interpretation_retries)
        if max_same_interpretation_retries < 0:
            raise ValueError("max_same_interpretation_retries must be >= 0")

        self.conditional_generators = self._normalize_conditional_generators(
            conditional_generator,
            conditional_generators,
        )
        self._validate_conditional_generators(self.conditional_generators)

        self.edge_generator = edge_generator
        self.conditional_generator = self.conditional_generators[0]
        self.seed = seed
        self.debug = bool(debug)
        self.require_new_interpretation_graph = bool(require_new_interpretation_graph)
        self.max_same_interpretation_retries = max_same_interpretation_retries
        self._propagate_debug()

        self.stored_level_graphs_: list[list[nx.Graph]] | None = None
        self.stored_graphs_: list[nx.Graph] | None = None
        self.stored_interpretation_graphs_: list[nx.Graph] | None = None
        self.stored_targets_: list[Any] | None = None
        self.stored_interpretation_hash_to_index_: dict[Any, int] = {}
        self.stored_interpretation_distance_matrix_: np.ndarray | None = None
        self.conditional_retrieval_states_: list[tuple[object, np.ndarray, np.ndarray] | None] = []

        self.last_sampled_indices_: list[int] = []
        self.last_successful_sampled_indices_: list[int] = []
        self.last_interpretation_neighbor_indices_history_: list[list[int]] = []
        self.last_interpretation_neighbor_distances_history_: list[list[float]] = []
        self.last_edge_generation_paths_: list[list[nx.Graph]] = []
        self.last_level_seed_graphs_history_: list[list[nx.Graph]] = []
        self.last_level_generated_graphs_history_: list[list[list[nx.Graph]]] = []
        self.last_level_neighbor_indices_history_: list[list[list[list[int]]]] = []
        self.last_level_training_graphs_history_: list[list[list[nx.Graph]]] = []

    @staticmethod
    def _normalize_conditional_generators(
        conditional_generator,
        conditional_generators,
    ) -> list[ConditionalAutoregressiveGenerator]:
        if conditional_generators is not None and conditional_generator is not None:
            raise ValueError(
                "Pass either conditional_generator or conditional_generators, not both."
            )
        if conditional_generators is None:
            if conditional_generator is None:
                raise ValueError(
                    "conditional_generator or conditional_generators is required."
                )
            return [conditional_generator]
        generators = list(conditional_generators)
        if not generators:
            raise ValueError("conditional_generators must be non-empty.")
        return generators

    @staticmethod
    def _validate_conditional_generators(generators: Sequence[object]) -> None:
        for stage, generator in enumerate(generators):
            if getattr(generator, "decomposition_function", None) is None:
                raise ValueError(
                    f"conditional_generators[{stage}].decomposition_function is required."
                )
            if getattr(generator, "nbits", None) is None:
                raise ValueError(f"conditional_generators[{stage}].nbits is required.")
            if getattr(generator, "label_mode", None) is None:
                raise ValueError(
                    f"conditional_generators[{stage}].label_mode is required."
                )

    def store(
        self,
        graphs,
        *,
        targets=None,
    ) -> "GraphGenerator":
        """Store base graphs and all computed interpretation hierarchy levels."""
        graph_list = self._as_graph_list(graphs)
        if len(graph_list) < 2:
            raise ValueError("store(graphs, ...) requires at least two graphs")

        target_list = None
        if targets is not None:
            if isinstance(targets, (str, bytes)):
                target_list = [targets] * len(graph_list)
            else:
                try:
                    target_list = list(targets)
                except TypeError:
                    target_list = [targets] * len(graph_list)
            if len(target_list) != len(graph_list):
                raise ValueError("targets and graphs must have the same length")

        stored_levels = [[graph.copy() for graph in graph_list]]
        for stage, generator in enumerate(self.conditional_generators):
            lower_graphs = stored_levels[stage]
            stored_levels.append(
                [
                    self._interpretation_graph_for_stage(stage, graph)
                    for graph in lower_graphs
                ]
            )

        self.stored_level_graphs_ = stored_levels
        self.stored_graphs_ = [graph.copy() for graph in stored_levels[0]]
        self.stored_interpretation_graphs_ = [
            graph.copy() for graph in stored_levels[-1]
        ]
        self.stored_targets_ = None if target_list is None else list(target_list)

        self.stored_interpretation_hash_to_index_ = {}
        for idx, graph in enumerate(self.stored_interpretation_graphs_):
            graph_hash = hash_graph(graph)
            if graph_hash not in self.stored_interpretation_hash_to_index_:
                self.stored_interpretation_hash_to_index_[graph_hash] = int(idx)

        self._fit_interpretation_retrieval_index(targets=self.stored_targets_)
        self._fit_conditional_retrieval_indexes()
        self._reset_histories()
        return self

    def sample(
        self,
        n_samples: int = 1,
        *,
        n_interpretation_neighbors: int = 30,
        n_conditional_neighbors: int | Sequence[int] = 30,
        n_instances_per_sample: int = 1,
        interpretation_edge_removal_size: float = 0.5,
        random_state: int | None = None,
        max_seed_attempts: int | None = None,
        edge_generate_kwargs: dict | None = None,
        conditional_generate_kwargs: dict | Sequence[dict | None] | None = None,
        deduplicate_conditional_neighbors: bool | Sequence[bool] = True,
        exclude_seed_from_conditional_neighbors: bool | Sequence[bool] = True,
        avoid_seed_components: bool | Sequence[bool] = True,
    ) -> list[nx.Graph]:
        """Generate base graphs through top-level exploration and staged descent."""
        stored_levels = self._require_stored_levels()
        top_level_graphs = stored_levels[-1]
        n_stages = len(self.conditional_generators)
        n_samples = int(n_samples)
        n_instances_per_sample = int(n_instances_per_sample)
        if n_samples <= 0 or n_instances_per_sample <= 0:
            self._reset_histories()
            return []

        n_conditional_neighbors_by_stage = self._normalize_stage_parameter(
            n_conditional_neighbors,
            n_stages,
            name="n_conditional_neighbors",
            scalar_types=(int, np.integer),
        )
        for value in n_conditional_neighbors_by_stage:
            if int(value) < 1:
                raise ValueError("n_conditional_neighbors values must be >= 1")

        conditional_kwargs_by_stage = self._normalize_stage_parameter(
            conditional_generate_kwargs,
            n_stages,
            name="conditional_generate_kwargs",
            default=None,
            scalar_types=(dict, type(None)),
        )
        conditional_kwargs_by_stage = [
            self._clean_conditional_generate_kwargs(kwargs)
            for kwargs in conditional_kwargs_by_stage
        ]
        deduplicate_by_stage = [
            bool(value)
            for value in self._normalize_stage_parameter(
                deduplicate_conditional_neighbors,
                n_stages,
                name="deduplicate_conditional_neighbors",
                scalar_types=(bool, np.bool_),
            )
        ]
        exclude_seed_by_stage = [
            bool(value)
            for value in self._normalize_stage_parameter(
                exclude_seed_from_conditional_neighbors,
                n_stages,
                name="exclude_seed_from_conditional_neighbors",
                scalar_types=(bool, np.bool_),
            )
        ]
        avoid_seed_by_stage = [
            bool(value)
            for value in self._normalize_stage_parameter(
                avoid_seed_components,
                n_stages,
                name="avoid_seed_components",
                scalar_types=(bool, np.bool_),
            )
        ]

        if max_seed_attempts is None:
            max_seed_attempts = min(
                len(top_level_graphs),
                max(n_samples, n_samples * 10),
            )
        max_seed_attempts = int(max_seed_attempts)
        if max_seed_attempts < 1:
            raise ValueError("max_seed_attempts must be >= 1")

        rng = random.Random(self.seed if random_state is None else random_state)
        edge_generate_kwargs = dict(edge_generate_kwargs or {})
        edge_generate_kwargs.pop("return_path", None)

        self._reset_histories()
        generated_base_graphs: list[nx.Graph] = []
        successful_samples = 0
        seed_order: list[int] = []
        skip_edge_stage = float(interpretation_edge_removal_size) == 0.0

        for _ in range(max_seed_attempts):
            if successful_samples >= n_samples:
                break
            if not seed_order:
                seed_order = list(range(len(top_level_graphs)))
                rng.shuffle(seed_order)
            seed_idx = seed_order.pop()
            seed_top_graph = top_level_graphs[seed_idx].copy()
            self.last_sampled_indices_.append(seed_idx)
            self._log_sample_progress(
                event="seed_start",
                successful_samples=successful_samples,
                requested_samples=n_samples,
                attempted_seeds=len(self.last_sampled_indices_),
                max_seed_attempts=max_seed_attempts,
                generated_graphs=len(generated_base_graphs),
                seed_idx=seed_idx,
            )

            if skip_edge_stage:
                start_graph = seed_top_graph.copy()
                generated_top_graph = seed_top_graph.copy()
                self.last_interpretation_neighbor_indices_history_.append([])
                self.last_interpretation_neighbor_distances_history_.append([])
            else:
                edge_result = self._generate_top_interpretation_for_seed(
                    seed_idx=seed_idx,
                    seed_top_graph=seed_top_graph,
                    n_interpretation_neighbors=n_interpretation_neighbors,
                    interpretation_edge_removal_size=interpretation_edge_removal_size,
                    rng=rng,
                    edge_generate_kwargs=edge_generate_kwargs,
                    successful_samples=successful_samples,
                    requested_samples=n_samples,
                    max_seed_attempts=max_seed_attempts,
                    generated_graphs=len(generated_base_graphs),
                )
                if edge_result is None:
                    continue
                start_graph, generated_top_graph = edge_result

            generated_levels = [[] for _ in range(n_stages + 1)]
            generated_levels[-1] = [generated_top_graph.copy()]
            level_neighbor_history: list[list[list[int]]] = [
                [] for _ in range(n_stages)
            ]
            level_training_history: list[list[nx.Graph]] = [
                [] for _ in range(n_stages)
            ]

            current_targets = [generated_top_graph.copy()]
            failed_stage = None
            for stage in reversed(range(n_stages)):
                generated_lower = []
                for target_graph in current_targets:
                    if len(generated_lower) >= n_instances_per_sample:
                        break
                    remaining = n_instances_per_sample - len(generated_lower)
                    stage_outputs = self._generate_stage_batch(
                        stage=stage,
                        target_graph=target_graph,
                        seed_idx=seed_idx,
                        n_neighbors=int(n_conditional_neighbors_by_stage[stage]),
                        n_samples=remaining,
                        generate_kwargs=conditional_kwargs_by_stage[stage],
                        deduplicate_neighbors=deduplicate_by_stage[stage],
                        exclude_seed=exclude_seed_by_stage[stage],
                        avoid_seed_components=avoid_seed_by_stage[stage],
                        neighbor_history=level_neighbor_history[stage],
                        training_history=level_training_history[stage],
                    )
                    generated_lower.extend(stage_outputs)
                if not generated_lower:
                    failed_stage = stage
                    break
                generated_lower = generated_lower[:n_instances_per_sample]
                generated_levels[stage] = [graph.copy() for graph in generated_lower]
                current_targets = generated_lower

            if failed_stage is not None:
                warnings.warn(
                    "Conditional stage generated no graphs matching the generated "
                    f"target at stage {failed_stage}; skipping seed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._log_sample_progress(
                    event="seed_skip",
                    successful_samples=successful_samples,
                    requested_samples=n_samples,
                    attempted_seeds=len(self.last_sampled_indices_),
                    max_seed_attempts=max_seed_attempts,
                    generated_graphs=len(generated_base_graphs),
                    seed_idx=seed_idx,
                    reason="conditional_generation_failed",
                )
                continue

            final_batch = generated_levels[0]
            self.last_successful_sampled_indices_.append(seed_idx)
            self.last_edge_generation_paths_.append(
                [start_graph.copy(), generated_top_graph.copy()]
            )
            self.last_level_seed_graphs_history_.append(
                [level[seed_idx].copy() for level in stored_levels]
            )
            self.last_level_generated_graphs_history_.append(generated_levels)
            self.last_level_neighbor_indices_history_.append(level_neighbor_history)
            self.last_level_training_graphs_history_.append(level_training_history)
            generated_base_graphs.extend(graph.copy() for graph in final_batch)
            successful_samples += 1
            self._log_sample_progress(
                event="seed_success",
                successful_samples=successful_samples,
                requested_samples=n_samples,
                attempted_seeds=len(self.last_sampled_indices_),
                max_seed_attempts=max_seed_attempts,
                generated_graphs=len(generated_base_graphs),
                seed_idx=seed_idx,
            )

        self._log_sample_summary(
            successful_samples=successful_samples,
            requested_samples=n_samples,
            attempted_seeds=len(self.last_sampled_indices_),
            max_seed_attempts=max_seed_attempts,
            generated_graphs=len(generated_base_graphs),
        )
        return generated_base_graphs

    def _generate_top_interpretation_for_seed(
        self,
        *,
        seed_idx: int,
        seed_top_graph: nx.Graph,
        n_interpretation_neighbors: int,
        interpretation_edge_removal_size: float,
        rng: random.Random,
        edge_generate_kwargs: dict,
        successful_samples: int,
        requested_samples: int,
        max_seed_attempts: int,
        generated_graphs: int,
    ) -> tuple[nx.Graph, nx.Graph] | None:
        top_level_graphs = self._require_stored_levels()[-1]
        interpretation_candidate_indices = self._nearest_top_level_indices(
            seed_top_graph,
            n_neighbors=len(top_level_graphs),
            query_index=seed_idx,
            exclude_query=True,
        )
        interpretation_candidate_indices = self._deduplicate_level_indices_by_hash(
            level=len(self.conditional_generators),
            indices=interpretation_candidate_indices,
        )
        (
            interpretation_neighbor_indices,
            interpretation_labels_covered,
        ) = self._augment_interpretation_indices_for_label_coverage(
            seed_top_graph,
            interpretation_candidate_indices,
            n_neighbors=n_interpretation_neighbors,
        )
        if not interpretation_labels_covered:
            warnings.warn(
                "No interpretation-neighbor context covers the sampled seed's "
                "interpretation node labels; skipping seed.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._log_sample_progress(
                event="seed_skip",
                successful_samples=successful_samples,
                requested_samples=requested_samples,
                attempted_seeds=len(self.last_sampled_indices_),
                max_seed_attempts=max_seed_attempts,
                generated_graphs=generated_graphs,
                seed_idx=seed_idx,
                reason="interpretation_label_coverage",
            )
            return None

        self.last_interpretation_neighbor_indices_history_.append(
            interpretation_neighbor_indices
        )
        self.last_interpretation_neighbor_distances_history_.append(
            self._top_level_distances_for_indices(
                seed_top_graph,
                interpretation_neighbor_indices,
                query_index=seed_idx,
            )
        )

        edge_training_graphs = [
            top_level_graphs[idx].copy() for idx in interpretation_neighbor_indices
        ]
        if (
            seed_top_graph.number_of_edges() > 0
            and all(graph.number_of_edges() == 0 for graph in edge_training_graphs)
            and seed_idx not in interpretation_neighbor_indices
        ):
            edge_training_graphs.append(seed_top_graph.copy())
            interpretation_neighbor_indices = list(interpretation_neighbor_indices) + [
                seed_idx
            ]
            self.last_interpretation_neighbor_indices_history_[-1] = list(
                interpretation_neighbor_indices
            )
            self.last_interpretation_neighbor_distances_history_[-1] = (
                self._top_level_distances_for_indices(
                    seed_top_graph,
                    interpretation_neighbor_indices,
                    query_index=seed_idx,
                )
            )
        self._log_edge_neighbor_context(
            seed_idx=seed_idx,
            neighbor_indices=interpretation_neighbor_indices,
            neighbor_distances=self.last_interpretation_neighbor_distances_history_[-1],
        )
        edge_targets = self._select_targets(interpretation_neighbor_indices)
        try:
            self._fit_edge_generator(edge_training_graphs, edge_targets)
        except Exception as exc:
            warnings.warn(
                "Edge stage failed while fitting on interpretation neighbors; "
                f"skipping seed. Error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._log_sample_progress(
                event="seed_skip",
                successful_samples=successful_samples,
                requested_samples=requested_samples,
                attempted_seeds=len(self.last_sampled_indices_),
                max_seed_attempts=max_seed_attempts,
                generated_graphs=generated_graphs,
                seed_idx=seed_idx,
                reason="edge_fit_failed",
            )
            return None

        start_graph, original_edge_count = remove_edges(
            seed_top_graph,
            size=self._interpretation_edge_removal_size_for_remove_edges(
                seed_top_graph,
                interpretation_edge_removal_size,
            ),
            rng=rng,
        )
        generated_top_graph = self._generate_interpretation_graph_with_retries(
            start_graph,
            original_edge_count,
            seed_interpretation_graph=seed_top_graph,
            edge_generate_kwargs=edge_generate_kwargs,
        )
        if generated_top_graph is None:
            self._log_sample_progress(
                event="seed_skip",
                successful_samples=successful_samples,
                requested_samples=requested_samples,
                attempted_seeds=len(self.last_sampled_indices_),
                max_seed_attempts=max_seed_attempts,
                generated_graphs=generated_graphs,
                seed_idx=seed_idx,
                reason="edge_generation_failed",
            )
            return None
        return start_graph, generated_top_graph.copy()

    def _generate_stage_batch(
        self,
        *,
        stage: int,
        target_graph: nx.Graph,
        seed_idx: int,
        n_neighbors: int,
        n_samples: int,
        generate_kwargs: dict,
        deduplicate_neighbors: bool,
        exclude_seed: bool,
        avoid_seed_components: bool,
        neighbor_history: list[list[int]],
        training_history: list[nx.Graph],
    ) -> list[nx.Graph]:
        stored_levels = self._require_stored_levels()
        upper_level = stage + 1
        lower_level = stage
        candidate_indices = self._nearest_level_indices(
            level=upper_level,
            graph=target_graph,
            n_neighbors=len(stored_levels[upper_level]),
            exclude_query=False,
        )
        if deduplicate_neighbors:
            candidate_indices = self._deduplicate_level_indices_by_hash(
                level=upper_level,
                indices=candidate_indices,
            )
        if exclude_seed:
            seedless_candidate_indices = [
                idx for idx in candidate_indices if idx != seed_idx
            ]
            if seedless_candidate_indices:
                candidate_indices = seedless_candidate_indices
        neighbor_indices = candidate_indices[
            : min(int(n_neighbors), len(candidate_indices))
        ]
        if not neighbor_indices:
            return []

        lower_training_graphs = [
            stored_levels[lower_level][idx].copy() for idx in neighbor_indices
        ]
        generator = self.conditional_generators[stage]
        generator.fit(lower_training_graphs)
        training_history.extend(graph.copy() for graph in lower_training_graphs)

        attempt_kwargs = dict(generate_kwargs)
        avoidance_applied = False
        if (
            avoid_seed_components
            and "avoid_component_subgraph_hashes" not in attempt_kwargs
            and hasattr(generator, "component_subgraph_hashes_for_graph")
        ):
            attempt_kwargs["avoid_component_subgraph_hashes"] = (
                generator.component_subgraph_hashes_for_graph(
                    stored_levels[lower_level][seed_idx]
                )
            )
            avoidance_applied = True

        def generate_with_kwargs(kwargs: dict) -> list[nx.Graph]:
            generated = generator.generate(
                n_samples=n_samples,
                interpretation_graphs=[target_graph],
                **kwargs,
            )
            return [
                graph.copy()
                for graph in self._as_output_graph_list(generated)
                if self._conditional_output_matches_interpretation(
                    stage,
                    graph,
                    target_graph,
                )
            ]

        generated_batch = generate_with_kwargs(attempt_kwargs)
        if not generated_batch and avoidance_applied:
            generated_batch = generate_with_kwargs(dict(generate_kwargs))
        if not generated_batch and seed_idx not in neighbor_indices:
            neighbor_indices = list(neighbor_indices) + [seed_idx]
            seed_graph = stored_levels[lower_level][seed_idx].copy()
            lower_training_graphs = lower_training_graphs + [seed_graph]
            generator.fit(lower_training_graphs)
            training_history.append(seed_graph.copy())
            generated_batch = generate_with_kwargs(dict(generate_kwargs))

        if generated_batch:
            neighbor_history.append(list(neighbor_indices))
        return generated_batch[:n_samples]

    def _reset_histories(self) -> None:
        self.last_sampled_indices_ = []
        self.last_successful_sampled_indices_ = []
        self.last_interpretation_neighbor_indices_history_ = []
        self.last_interpretation_neighbor_distances_history_ = []
        self.last_edge_generation_paths_ = []
        self.last_level_seed_graphs_history_ = []
        self.last_level_generated_graphs_history_ = []
        self.last_level_neighbor_indices_history_ = []
        self.last_level_training_graphs_history_ = []

    def _propagate_debug(self) -> None:
        for generator in [self.edge_generator, *self.conditional_generators]:
            setattr(generator, "debug", self.debug)
            if hasattr(generator, "debug_level"):
                if self.debug:
                    generator.debug_level = max(1, int(generator.debug_level))
                else:
                    generator.debug_level = 0
            if hasattr(generator, "verbose"):
                generator.verbose = self.debug

    def _require_stored_levels(self) -> list[list[nx.Graph]]:
        if self.stored_level_graphs_ is None:
            raise ValueError("Call store(graphs, ...) before sample(...)")
        return self.stored_level_graphs_

    def _interpretation_graph_for_stage(self, stage: int, graph: nx.Graph) -> nx.Graph:
        generator = self.conditional_generators[int(stage)]
        abstract_graph = graph_to_abstract_graph(
            graph,
            decomposition_function=generator.decomposition_function,
            nbits=int(generator.nbits),
            label_mode=generator.label_mode,
        )
        return abstract_graph.interpretation_graph.copy()

    def _fit_interpretation_retrieval_index(self, *, targets) -> None:
        top_level_graphs = self._require_stored_levels()[-1]
        self.edge_generator.store(top_level_graphs, targets=targets)
        _transformer, vectors, distances = self._edge_retrieval_state()
        if vectors.shape[0] != len(top_level_graphs):
            raise ValueError(
                "edge_generator.store(...) must initialize one retrieval row per graph"
            )
        self.stored_interpretation_distance_matrix_ = distances.copy()
        np.fill_diagonal(self.stored_interpretation_distance_matrix_, 0.0)

    def _fit_conditional_retrieval_indexes(self) -> None:
        stored_levels = self._require_stored_levels()
        n_stages = len(self.conditional_generators)
        self.conditional_retrieval_states_ = [None for _ in range(n_stages)]
        for stage in range(n_stages - 1):
            upper_graphs = stored_levels[stage + 1]
            vectorizer = getattr(
                self.conditional_generators[stage],
                "context_vectorizer",
                None,
            )
            if vectorizer is None:
                vectorizer = _FallbackGraphVectorizer()
            vectors = self._vectorize_graphs(vectorizer, upper_graphs, fit=True)
            distances = pairwise_distances(vectors)
            np.fill_diagonal(distances, 0.0)
            self.conditional_retrieval_states_[stage] = (
                vectorizer,
                vectors,
                distances,
            )

    def _nearest_level_indices(
        self,
        *,
        level: int,
        graph: nx.Graph,
        n_neighbors: int,
        query_index: int | None = None,
        exclude_query: bool = False,
    ) -> list[int]:
        if level == len(self.conditional_generators):
            return self._nearest_top_level_indices(
                graph,
                n_neighbors=n_neighbors,
                query_index=query_index,
                exclude_query=exclude_query,
            )
        stage = level - 1
        state = self.conditional_retrieval_states_[stage]
        if state is None:
            raise ValueError(f"No retrieval index is available for level {level}.")
        return self._nearest_indices_from_state(
            state,
            graph,
            n_neighbors=n_neighbors,
            query_index=query_index,
            exclude_query=exclude_query,
        )

    def _nearest_top_level_indices(
        self,
        graph: nx.Graph,
        *,
        n_neighbors: int,
        query_index: int | None = None,
        exclude_query: bool = False,
    ) -> list[int]:
        return self._nearest_indices_from_state(
            self._edge_retrieval_state(),
            graph,
            n_neighbors=n_neighbors,
            query_index=query_index,
            exclude_query=exclude_query,
            hash_to_index=self.stored_interpretation_hash_to_index_,
        )

    def _nearest_indices_from_state(
        self,
        state,
        graph: nx.Graph,
        *,
        n_neighbors: int,
        query_index: int | None = None,
        exclude_query: bool = False,
        hash_to_index: dict[Any, int] | None = None,
    ) -> list[int]:
        _transformer, vectors, distances = state
        if n_neighbors <= 0 or vectors.shape[0] == 0:
            return []
        n_neighbors = min(int(n_neighbors), vectors.shape[0])

        if query_index is not None:
            graph_distances = np.asarray(distances[int(query_index)], dtype=float)
        else:
            transformer, vectors, _distances = state
            query_vector = self._vectorize_graphs(transformer, [graph], fit=False)
            graph_distances = pairwise_distances(query_vector, vectors)[0]
            if hash_to_index is not None:
                query_index = hash_to_index.get(hash_graph(graph))

        order = np.argsort(graph_distances, kind="mergesort")
        indices = []
        for idx in order:
            int_idx = int(idx)
            if exclude_query and query_index is not None and int_idx == int(query_index):
                continue
            indices.append(int_idx)
            if len(indices) >= n_neighbors:
                break
        return indices

    def _top_level_distances_for_indices(
        self,
        graph: nx.Graph,
        indices: Sequence[int],
        *,
        query_index: int | None = None,
    ) -> list[float]:
        if not indices:
            return []
        if (
            query_index is not None
            and self.stored_interpretation_distance_matrix_ is not None
        ):
            distances = np.asarray(
                self.stored_interpretation_distance_matrix_[int(query_index)],
                dtype=float,
            )
        else:
            transformer, vectors, _distances = self._edge_retrieval_state()
            query_vector = self._vectorize_graphs(transformer, [graph], fit=False)
            distances = pairwise_distances(query_vector, vectors)[0]
        return [float(distances[int(idx)]) for idx in indices]

    def _edge_retrieval_state(self):
        transformer = getattr(self.edge_generator, "retrieval_transformer_", None)
        vectors = getattr(self.edge_generator, "stored_retrieval_vectors_", None)
        distances = getattr(self.edge_generator, "stored_distance_matrix_", None)
        if transformer is None or vectors is None or distances is None:
            raise ValueError(
                "edge_generator.store(...) did not initialize retrieval state"
            )
        return (
            transformer,
            self._as_dense_matrix(vectors),
            self._as_dense_matrix(distances),
        )

    def _log_edge_neighbor_context(
        self,
        *,
        seed_idx: int,
        neighbor_indices: Sequence[int],
        neighbor_distances: Sequence[float],
    ) -> None:
        if not self.debug:
            return
        rounded_distances = [round(float(distance), 4) for distance in neighbor_distances]
        print(
            "[graph-generator edge] "
            f"seed_idx={int(seed_idx)} "
            f"n_neighbors={len(neighbor_indices)} "
            f"neighbor_indices={list(neighbor_indices)} "
            f"neighbor_distances={rounded_distances}"
        )

    def _log_sample_progress(
        self,
        *,
        event: str,
        successful_samples: int,
        requested_samples: int,
        attempted_seeds: int,
        max_seed_attempts: int,
        generated_graphs: int,
        seed_idx: int | None = None,
        reason: str | None = None,
    ) -> None:
        if not self.debug:
            return
        parts = [
            "[graph-generator sample]",
            f"event={event}",
            f"currently_generated={int(successful_samples)}/{int(requested_samples)}",
            f"attempted_seeds={int(attempted_seeds)}/{int(max_seed_attempts)}",
            f"generated_graphs={int(generated_graphs)}",
        ]
        if seed_idx is not None:
            parts.append(f"seed_idx={int(seed_idx)}")
        if reason is not None:
            parts.append(f"reason={reason}")
        print(" ".join(parts))

    def _log_sample_summary(
        self,
        *,
        successful_samples: int,
        requested_samples: int,
        attempted_seeds: int,
        max_seed_attempts: int,
        generated_graphs: int,
    ) -> None:
        if not self.debug:
            return
        print(
            "[graph-generator sample] "
            "summary "
            f"currently_generated={int(successful_samples)}/{int(requested_samples)} "
            f"attempted_seeds={int(attempted_seeds)}/{int(max_seed_attempts)} "
            f"generated_graphs={int(generated_graphs)}"
        )

    def _deduplicate_level_indices_by_hash(
        self,
        *,
        level: int,
        indices: Sequence[int],
    ) -> list[int]:
        stored_levels = self._require_stored_levels()
        deduplicated = []
        seen_hashes = set()
        for idx in indices:
            int_idx = int(idx)
            graph_hash = hash_graph(stored_levels[level][int_idx])
            if graph_hash in seen_hashes:
                continue
            seen_hashes.add(graph_hash)
            deduplicated.append(int_idx)
        return deduplicated

    def _augment_interpretation_indices_for_label_coverage(
        self,
        graph: nx.Graph,
        candidate_indices: Sequence[int],
        *,
        n_neighbors: int,
    ) -> tuple[list[int], bool]:
        """Select nearest top-level neighbors using repair-style label coverage."""
        top_level_graphs = self._require_stored_levels()[-1]
        candidate_indices = list(dict.fromkeys(int(idx) for idx in candidate_indices))
        requested_neighbors = max(0, int(n_neighbors))
        required_labels = self._graph_unique_node_labels(graph)
        selected_indices = []
        selected_set = set()

        if required_labels:
            missing_labels = set(required_labels)
            for idx in candidate_indices:
                candidate_labels = self._graph_unique_node_labels(
                    top_level_graphs[idx]
                )
                if not missing_labels.intersection(candidate_labels):
                    continue
                selected_indices.append(idx)
                selected_set.add(idx)
                missing_labels.difference_update(candidate_labels)
                if not missing_labels:
                    break
            if missing_labels:
                return selected_indices, False

        for idx in candidate_indices:
            if len(selected_indices) >= requested_neighbors:
                break
            if idx in selected_set:
                continue
            selected_indices.append(idx)
            selected_set.add(idx)

        return selected_indices, True

    @staticmethod
    def _graph_unique_node_labels(graph: nx.Graph) -> set:
        return {
            data.get("label")
            for _node, data in graph.nodes(data=True)
            if data.get("label") is not None
        }

    def _fit_edge_generator(
        self,
        graphs: Sequence[nx.Graph],
        targets: Sequence[Any] | None,
    ) -> None:
        if targets is None:
            self.edge_generator.fit(graphs)
        else:
            self.edge_generator.fit(graphs, targets=targets)

    def _generate_interpretation_graph_with_retries(
        self,
        start_graph: nx.Graph,
        original_edge_count: int,
        *,
        seed_interpretation_graph: nx.Graph,
        edge_generate_kwargs: dict,
    ) -> nx.Graph | None:
        seed_hash = hash_graph(seed_interpretation_graph)
        same_graph_retries = 0

        while True:
            try:
                generated_interpretation_graph = self.edge_generator.generate(
                    start_graph,
                    original_edge_count,
                    return_path=False,
                    **edge_generate_kwargs,
                )
            except Exception as exc:
                warnings.warn(
                    "Edge stage failed while generating an interpretation graph; "
                    f"skipping seed. Error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            if isinstance(generated_interpretation_graph, list):
                generated_interpretation_graph = (
                    generated_interpretation_graph[-1]
                    if generated_interpretation_graph
                    else None
                )
            if generated_interpretation_graph is None:
                warnings.warn(
                    "Edge stage failed to generate an interpretation graph; skipping seed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None

            generated_interpretation_graph = generated_interpretation_graph.copy()
            if (
                not self.require_new_interpretation_graph
                or hash_graph(generated_interpretation_graph) != seed_hash
            ):
                return generated_interpretation_graph

            if same_graph_retries >= self.max_same_interpretation_retries:
                warnings.warn(
                    "Edge stage generated the same interpretation graph as the "
                    "sampled seed after "
                    f"{same_graph_retries} retries; skipping seed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None

            same_graph_retries += 1
            if self.debug:
                print(
                    "[graph-generator edge] "
                    "generated same interpretation graph as seed; "
                    f"retry={same_graph_retries}/"
                    f"{self.max_same_interpretation_retries}"
                )

    @staticmethod
    def _interpretation_edge_removal_size_for_remove_edges(
        graph: nx.Graph,
        size: float,
    ) -> float | int:
        if float(size) == 1.0:
            return graph.number_of_edges()
        return size

    def _select_targets(self, indices: Sequence[int]) -> list[Any] | None:
        if self.stored_targets_ is None:
            return None
        return [self.stored_targets_[idx] for idx in indices]

    def _conditional_output_matches_interpretation(
        self,
        stage: int,
        graph: nx.Graph,
        interpretation_graph: nx.Graph,
    ) -> bool:
        generator = self.conditional_generators[int(stage)]
        if hasattr(generator, "_matches_target_interpretation"):
            return bool(
                generator._matches_target_interpretation(
                    graph,
                    interpretation_graph,
                )
            )
        generated_interpretation_graph = self._interpretation_graph_for_stage(
            stage,
            graph,
        )
        return hash_graph(generated_interpretation_graph) == hash_graph(
            interpretation_graph
        )

    @staticmethod
    def _normalize_stage_parameter(
        value,
        n_stages: int,
        *,
        name: str,
        default=None,
        scalar_types: tuple[type, ...],
    ) -> list:
        if value is None:
            value = default
        if isinstance(value, scalar_types):
            return [value for _ in range(n_stages)]
        if isinstance(value, (str, bytes)):
            return [value for _ in range(n_stages)]
        try:
            values = list(value)
        except TypeError:
            return [value for _ in range(n_stages)]
        if len(values) != n_stages:
            raise ValueError(f"{name} must have one value per conditional generator")
        return values

    @staticmethod
    def _clean_conditional_generate_kwargs(kwargs) -> dict:
        cleaned = dict(kwargs or {})
        cleaned.pop("interpretation_graphs", None)
        cleaned.pop("n_samples", None)
        return cleaned

    def _vectorize_graphs(self, transformer, graphs, *, fit: bool) -> np.ndarray:
        if fit and hasattr(transformer, "fit_transform"):
            features = transformer.fit_transform(graphs)
        elif fit and hasattr(transformer, "fit") and hasattr(transformer, "transform"):
            transformer.fit(graphs)
            features = transformer.transform(graphs)
        elif hasattr(transformer, "transform"):
            features = transformer.transform(graphs)
        else:
            raise ValueError(
                "retrieval transformer must provide fit_transform(...) or transform(...)"
            )
        return self._as_dense_matrix(features)

    @staticmethod
    def _as_dense_matrix(features) -> np.ndarray:
        if hasattr(features, "toarray"):
            features = features.toarray()
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        return matrix

    @staticmethod
    def _as_graph_list(graphs) -> list[nx.Graph]:
        if graphs is None:
            raise ValueError("graphs is required")
        if is_simple_graph(graphs):
            return [graphs.copy()]
        return [graph.copy() for graph in graphs]

    @staticmethod
    def _as_output_graph_list(graphs) -> list[nx.Graph]:
        if graphs is None:
            return []
        if is_simple_graph(graphs):
            return [graphs]
        return list(graphs)
