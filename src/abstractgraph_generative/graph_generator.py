"""Two-stage graph generation over interpretation graphs and base graphs."""

from __future__ import annotations

import copy
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


class GraphGenerator:
    """Combine interpretation-graph generation with conditional base instantiation.

    ``GraphGenerator`` stores a base graph corpus together with each graph's
    interpretation graph. Sampling first trains an ``EdgeGenerator`` on nearby
    interpretation graphs and regrows a new target interpretation graph. It then
    trains a ``ConditionalAutoregressiveGenerator`` on base graphs whose
    interpretation graphs are near that generated target.
    """

    def __init__(
        self,
        edge_generator: EdgeGenerator,
        conditional_generator: ConditionalAutoregressiveGenerator,
        decomposition_function,
        nbits: int,
        label_mode: str = "operator_hash",
        interpretation_neighbor_vectorizer=None,
        seed: int | None = None,
    ):
        self.edge_generator = edge_generator
        self.conditional_generator = conditional_generator
        self.decomposition_function = decomposition_function
        self.nbits = int(nbits)
        self.label_mode = label_mode
        self.interpretation_neighbor_vectorizer = interpretation_neighbor_vectorizer
        self.seed = seed

        self.stored_graphs_: list[nx.Graph] | None = None
        self.stored_interpretation_graphs_: list[nx.Graph] | None = None
        self.stored_targets_: list[Any] | None = None
        self.stored_interpretation_hash_to_index_: dict[Any, int] = {}
        self.interpretation_retrieval_transformer_ = None
        self.stored_interpretation_retrieval_vectors_: np.ndarray | None = None
        self.stored_interpretation_distance_matrix_: np.ndarray | None = None

        self.last_sampled_indices_: list[int] = []
        self.last_interpretation_neighbor_indices_history_: list[list[int]] = []
        self.last_conditional_neighbor_indices_history_: list[list[int]] = []
        self.last_generated_interpretation_graphs_: list[nx.Graph] = []
        self.last_edge_generation_paths_: list[list[nx.Graph]] = []
        self.last_conditional_training_graphs_history_: list[list[nx.Graph]] = []

    def store(
        self,
        graphs,
        *,
        interpretation_graphs=None,
        targets=None,
    ) -> "GraphGenerator":
        """Store aligned base graphs and interpretation graphs for retrieval."""
        graph_list = self._as_graph_list(graphs)
        if len(graph_list) < 2:
            raise ValueError("store(graphs, ...) requires at least two graphs")

        if interpretation_graphs is None:
            interpretation_graph_list = [
                self._interpretation_graph_for(graph) for graph in graph_list
            ]
        else:
            interpretation_graph_list = self._as_graph_list(interpretation_graphs)
            if len(interpretation_graph_list) != len(graph_list):
                raise ValueError(
                    "interpretation_graphs and graphs must have the same length"
                )

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

        self.stored_graphs_ = [graph.copy() for graph in graph_list]
        self.stored_interpretation_graphs_ = [
            graph.copy() for graph in interpretation_graph_list
        ]
        self.stored_targets_ = None if target_list is None else list(target_list)

        self.stored_interpretation_hash_to_index_ = {}
        for idx, graph in enumerate(self.stored_interpretation_graphs_):
            graph_hash = hash_graph(graph)
            if graph_hash not in self.stored_interpretation_hash_to_index_:
                self.stored_interpretation_hash_to_index_[graph_hash] = int(idx)

        self._fit_interpretation_retrieval_index(targets=self.stored_targets_)
        self._reset_histories()
        return self

    def sample(
        self,
        n_samples: int = 1,
        *,
        n_interpretation_neighbors: int = 30,
        n_conditional_neighbors: int = 30,
        n_instances_per_sample: int = 1,
        interpretation_edge_removal_size: float = 0.5,
        random_state: int | None = None,
        edge_generate_kwargs: dict | None = None,
        conditional_generate_kwargs: dict | None = None,
    ) -> list[nx.Graph]:
        """Generate base graphs through interpretation-graph exploration."""
        stored_graphs, stored_interpretation_graphs = self._require_stored()
        n_samples = int(n_samples)
        n_instances_per_sample = int(n_instances_per_sample)
        if n_samples <= 0 or n_instances_per_sample <= 0:
            self._reset_histories()
            return []
        if n_conditional_neighbors < 1:
            raise ValueError("n_conditional_neighbors must be >= 1")

        rng = random.Random(self.seed if random_state is None else random_state)
        edge_generate_kwargs = dict(edge_generate_kwargs or {})
        edge_generate_kwargs.pop("return_path", None)
        conditional_generate_kwargs = dict(conditional_generate_kwargs or {})
        conditional_generate_kwargs.pop("interpretation_graphs", None)
        conditional_generate_kwargs.pop("n_samples", None)

        self._reset_histories()
        generated_base_graphs: list[nx.Graph] = []

        for _ in range(n_samples):
            seed_idx = rng.randrange(len(stored_interpretation_graphs))
            seed_interpretation_graph = stored_interpretation_graphs[seed_idx].copy()
            self.last_sampled_indices_.append(seed_idx)

            interpretation_neighbor_indices = self._nearest_interpretation_indices(
                seed_interpretation_graph,
                n_neighbors=n_interpretation_neighbors,
                query_index=seed_idx,
                exclude_query=True,
            )
            if not interpretation_neighbor_indices:
                interpretation_neighbor_indices = [seed_idx]
            self.last_interpretation_neighbor_indices_history_.append(
                list(interpretation_neighbor_indices)
            )

            edge_training_graphs = [
                stored_interpretation_graphs[idx].copy()
                for idx in interpretation_neighbor_indices
            ]
            edge_targets = self._select_targets(interpretation_neighbor_indices)
            self._fit_edge_generator(edge_training_graphs, edge_targets)

            start_graph, original_edge_count = remove_edges(
                seed_interpretation_graph,
                size=interpretation_edge_removal_size,
                rng=rng,
            )
            generated_interpretation_graph = self.edge_generator.generate(
                start_graph,
                original_edge_count,
                return_path=False,
                **edge_generate_kwargs,
            )
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
                continue

            generated_interpretation_graph = generated_interpretation_graph.copy()
            conditional_neighbor_indices = self._nearest_interpretation_indices(
                generated_interpretation_graph,
                n_neighbors=n_conditional_neighbors,
                exclude_query=False,
            )
            if not conditional_neighbor_indices:
                warnings.warn(
                    "No conditional neighbors found for generated interpretation graph; skipping seed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            conditional_training_graphs = [
                stored_graphs[idx].copy() for idx in conditional_neighbor_indices
            ]
            self.conditional_generator.fit(conditional_training_graphs)
            generated_batch = self.conditional_generator.generate(
                n_samples=n_instances_per_sample,
                interpretation_graphs=[generated_interpretation_graph],
                **conditional_generate_kwargs,
            )
            generated_batch = self._as_output_graph_list(generated_batch)

            self.last_conditional_neighbor_indices_history_.append(
                list(conditional_neighbor_indices)
            )
            self.last_generated_interpretation_graphs_.append(
                generated_interpretation_graph.copy()
            )
            self.last_edge_generation_paths_.append(
                [start_graph.copy(), generated_interpretation_graph.copy()]
            )
            self.last_conditional_training_graphs_history_.append(
                [graph.copy() for graph in conditional_training_graphs]
            )
            generated_base_graphs.extend(graph.copy() for graph in generated_batch)

        return generated_base_graphs

    def _reset_histories(self) -> None:
        self.last_sampled_indices_ = []
        self.last_interpretation_neighbor_indices_history_ = []
        self.last_conditional_neighbor_indices_history_ = []
        self.last_generated_interpretation_graphs_ = []
        self.last_edge_generation_paths_ = []
        self.last_conditional_training_graphs_history_ = []

    def _require_stored(self) -> tuple[list[nx.Graph], list[nx.Graph]]:
        if self.stored_graphs_ is None or self.stored_interpretation_graphs_ is None:
            raise ValueError("Call store(graphs, ...) before sample(...)")
        return self.stored_graphs_, self.stored_interpretation_graphs_

    def _interpretation_graph_for(self, graph: nx.Graph) -> nx.Graph:
        abstract_graph = graph_to_abstract_graph(
            graph,
            decomposition_function=self.decomposition_function,
            nbits=self.nbits,
            label_mode=self.label_mode,
        )
        return abstract_graph.interpretation_graph.copy()

    def _fit_interpretation_retrieval_index(self, *, targets) -> None:
        _stored_graphs, interpretation_graphs = self._require_stored()
        if self.interpretation_neighbor_vectorizer is None:
            self.edge_generator.store(interpretation_graphs, targets=targets)
            self.interpretation_retrieval_transformer_ = copy.deepcopy(
                getattr(self.edge_generator, "retrieval_transformer_", None)
            )
            vectors = getattr(self.edge_generator, "stored_retrieval_vectors_", None)
            distances = getattr(self.edge_generator, "stored_distance_matrix_", None)
            if (
                self.interpretation_retrieval_transformer_ is None
                or vectors is None
                or distances is None
            ):
                raise ValueError(
                    "edge_generator.store(...) did not initialize retrieval vectors"
                )
            self.stored_interpretation_retrieval_vectors_ = self._as_dense_matrix(vectors)
            self.stored_interpretation_distance_matrix_ = self._as_dense_matrix(
                distances
            )
            return

        transformer = self.interpretation_neighbor_vectorizer
        self.interpretation_retrieval_transformer_ = transformer
        vectors = self._vectorize_graphs(transformer, interpretation_graphs, fit=True)
        if vectors.shape[0] != len(interpretation_graphs):
            raise ValueError(
                "interpretation_neighbor_vectorizer must return one row per graph"
            )
        self.stored_interpretation_retrieval_vectors_ = vectors
        self.stored_interpretation_distance_matrix_ = pairwise_distances(vectors)
        np.fill_diagonal(self.stored_interpretation_distance_matrix_, 0.0)

    def _nearest_interpretation_indices(
        self,
        graph: nx.Graph,
        *,
        n_neighbors: int,
        query_index: int | None = None,
        exclude_query: bool = False,
    ) -> list[int]:
        _stored_graphs, stored_interpretation_graphs = self._require_stored()
        if n_neighbors <= 0 or not stored_interpretation_graphs:
            return []
        n_neighbors = min(int(n_neighbors), len(stored_interpretation_graphs))

        if (
            query_index is not None
            and self.stored_interpretation_distance_matrix_ is not None
        ):
            distances = np.asarray(
                self.stored_interpretation_distance_matrix_[int(query_index)],
                dtype=float,
            )
        else:
            vectors = self.stored_interpretation_retrieval_vectors_
            transformer = self.interpretation_retrieval_transformer_
            if vectors is None or transformer is None:
                raise ValueError("interpretation retrieval index is not initialized")
            query_vector = self._vectorize_graphs(transformer, [graph], fit=False)
            distances = pairwise_distances(query_vector, vectors)[0]
            if query_index is None:
                query_index = self.stored_interpretation_hash_to_index_.get(
                    hash_graph(graph)
                )

        order = np.argsort(distances, kind="mergesort")
        indices = []
        for idx in order:
            int_idx = int(idx)
            if exclude_query and query_index is not None and int_idx == int(query_index):
                continue
            indices.append(int_idx)
            if len(indices) >= n_neighbors:
                break
        return indices

    def _fit_edge_generator(
        self,
        graphs: Sequence[nx.Graph],
        targets: Sequence[Any] | None,
    ) -> None:
        if targets is None:
            self.edge_generator.fit(graphs)
        else:
            self.edge_generator.fit(graphs, targets=targets)

    def _select_targets(self, indices: Sequence[int]) -> list[Any] | None:
        if self.stored_targets_ is None:
            return None
        return [self.stored_targets_[idx] for idx in indices]

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
                "interpretation_neighbor_vectorizer must provide fit_transform(...) or transform(...)"
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
