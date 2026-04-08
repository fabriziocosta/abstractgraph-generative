from __future__ import annotations

import copy
import math
import random
import time
from itertools import combinations, permutations
from typing import Callable

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from abstractgraph.graphs import graph_to_abstract_graph
from abstractgraph.hashing import hash_graph
from sklearn.metrics import pairwise_distances
from abstractgraph_generative.interpolate import _build_adjacency


DrawGraphsFn = Callable[..., object]


def mix_connected_components(
    graph1: nx.Graph,
    graph2: nx.Graph,
    *,
    target_n_nodes: int | None = None,
    n_trials: int = 128,
    seed: int | None = None,
):
    if graph1.number_of_nodes() < 1 or graph2.number_of_nodes() < 1:
        raise ValueError("Both input graphs must contain at least one node")
    if nx.is_directed(graph1) != nx.is_directed(graph2):
        raise ValueError("Both input graphs must have the same directedness")
    if graph1.is_multigraph() != graph2.is_multigraph():
        raise ValueError("Both input graphs must both be simple or both multigraphs")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    rng = random.Random(seed)
    component_graphs1 = _connected_component_subgraphs(graph1)
    component_graphs2 = _connected_component_subgraphs(graph2)

    max_pairs = min(len(component_graphs1), len(component_graphs2))
    if max_pairs < 1:
        raise ValueError("Both graphs must expose at least one connected component")

    if target_n_nodes is None:
        target_n_nodes = int(round((graph1.number_of_nodes() + graph2.number_of_nodes()) / 2))
    target_n_nodes = max(1, int(target_n_nodes))

    best_choice = None
    best_score = None

    for n_components_per_graph in range(1, max_pairs + 1):
        trial_count = 1 if (
            n_components_per_graph == len(component_graphs1)
            and n_components_per_graph == len(component_graphs2)
        ) else n_trials
        for _ in range(trial_count):
            selected1 = _sample_components(component_graphs1, n_components_per_graph, rng)
            selected2 = _sample_components(component_graphs2, n_components_per_graph, rng)
            total_nodes = sum(g.number_of_nodes() for g in selected1) + sum(
                g.number_of_nodes() for g in selected2
            )
            score = abs(total_nodes - target_n_nodes)
            tie_break = total_nodes
            if best_score is None or (score, tie_break) < best_score:
                best_score = (score, tie_break)
                best_choice = (selected1, selected2)
                if score == 0:
                    break
        if best_score is not None and best_score[0] == 0:
            break

    if best_choice is None:
        raise ValueError("Could not select connected components from the input graphs")

    selected1, selected2 = best_choice
    return _merge_component_graphs(selected1 + selected2, graph1)


def edge_neighbors(
    G: nx.Graph,
    *,
    n_samples: int = 1,
    seed: int | None = None,
    allow_self_loops: bool = False,
):
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("random_edge_move_copies supports only simple NetworkX graphs")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if G.number_of_nodes() < 2:
        raise ValueError("Input graph must have at least 2 nodes")
    if G.number_of_edges() < 1:
        raise ValueError("Input graph must have at least 1 edge")

    rng = random.Random(seed)
    nodes = list(G.nodes())
    edges = [tuple(edge) for edge in G.edges()]
    occupied = set(edges)

    if nx.is_directed(G):
        candidate_edges = list(permutations(nodes, 2))
    else:
        candidate_edges = list(combinations(nodes, 2))
    if allow_self_loops:
        candidate_edges += [(node, node) for node in nodes]

    sampled_graphs = []
    for _ in range(n_samples):
        old_edge = rng.choice(edges)
        new_edge_options = [
            edge for edge in candidate_edges if edge != old_edge and edge not in occupied
        ]
        if not new_edge_options:
            raise ValueError("No valid destination edge is available for moving an edge")

        new_edge = rng.choice(new_edge_options)
        H = G.copy()
        H.remove_edge(*old_edge)
        H.add_edge(*new_edge, **dict(G.edges[old_edge]))
        sampled_graphs.append(H)

    return sampled_graphs


def remove_edges(G: nx.Graph, size=0.1):
    H = G.copy()
    n_edges = H.number_of_edges()
    edges = list(H.edges())

    if size < 0:
        raise ValueError("size must be >= 0")
    if size < 1:
        n_remove = int(round(size * n_edges))
    else:
        n_remove = int(size)
    n_remove = max(0, min(n_edges, n_remove))

    if n_remove > 0:
        removed_edges = random.sample(edges, k=n_remove)
        H.remove_edges_from(removed_edges)

    return H, n_edges


def make_edge_regression_dataset(
    seed_graph: nx.Graph,
    *,
    n_negative_per_positive: int,
    n_replicates: int = 1,
    seed: int | None = None,
    allow_self_loops: bool = False,
):
    rng = random.Random(seed)
    positives = []
    negatives = []
    dataset = []

    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")

    for _ in range(n_replicates):
        current_graph = seed_graph.copy()
        while current_graph.number_of_edges() > 0:
            edge = rng.choice(list(current_graph.edges()))
            positive_graph = current_graph.copy()
            positive_graph.remove_edge(*edge)

            positives.append(positive_graph)
            dataset.append((positive_graph, 1))

            if positive_graph.number_of_edges() > 0:
                negative_graphs = edge_neighbors(
                    positive_graph,
                    n_samples=n_negative_per_positive,
                    seed=rng.randrange(10**9),
                    allow_self_loops=allow_self_loops,
                )
                negatives.extend(negative_graphs)
                dataset.extend((negative_graph, 0) for negative_graph in negative_graphs)

            current_graph = positive_graph

    return positives, negatives, dataset


def make_edge_regression_dataset_subgraph_ordered(
    seed_graph: nx.Graph,
    *,
    decomposition_function,
    nbits: int,
    n_negative_per_positive: int,
    n_replicates: int = 1,
    seed: int | None = None,
    allow_self_loops: bool = False,
):
    rng = random.Random(seed)
    positives = []
    negatives = []
    dataset = []

    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")

    edge_groups = _decomposition_edge_groups(
        seed_graph,
        decomposition_function=decomposition_function,
        nbits=nbits,
    )

    for _ in range(n_replicates):
        current_graph = seed_graph.copy()
        replicate_groups = [list(group) for group in edge_groups]
        rng.shuffle(replicate_groups)

        for group in replicate_groups:
            while True:
                remaining_group_edges = [
                    edge for edge in group if _graph_has_canonical_edge(current_graph, edge)
                ]
                if not remaining_group_edges:
                    break
                edge = rng.choice(remaining_group_edges)
                positive_graph = current_graph.copy()
                positive_graph.remove_edge(*edge)

                positives.append(positive_graph)
                dataset.append((positive_graph, 1))

                if positive_graph.number_of_edges() > 0:
                    negative_graphs = edge_neighbors(
                        positive_graph,
                        n_samples=n_negative_per_positive,
                        seed=rng.randrange(10**9),
                        allow_self_loops=allow_self_loops,
                    )
                    negatives.extend(negative_graphs)
                    dataset.extend((negative_graph, 0) for negative_graph in negative_graphs)

                current_graph = positive_graph
                if current_graph.number_of_edges() == 0:
                    break

            if current_graph.number_of_edges() == 0:
                break

    return positives, negatives, dataset


def _decomposition_edge_groups(
    graph: nx.Graph,
    *,
    decomposition_function,
    nbits: int,
):
    abstract_graph = graph_to_abstract_graph(
        graph,
        decomposition_function=decomposition_function,
        nbits=nbits,
    )
    groups = []
    seen_groups = set()
    full_graph_edges = frozenset(_canonicalize_edge(edge, graph) for edge in graph.edges())

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        if subgraph is None or subgraph.number_of_edges() == 0:
            continue
        group_edges = frozenset(_canonicalize_edge(edge, graph) for edge in subgraph.edges())
        if not group_edges:
            continue
        # Skip the default full-graph interpretation node; keep only actual decomposition groups.
        if group_edges == full_graph_edges:
            continue
        if group_edges in seen_groups:
            continue
        seen_groups.add(group_edges)
        groups.append(list(group_edges))

    covered_edges = set()
    for group in groups:
        covered_edges.update(group)
    leftover_edges = [edge for edge in full_graph_edges if edge not in covered_edges]
    if leftover_edges:
        groups.append(leftover_edges)

    return groups if groups else [list(full_graph_edges)]


def _canonicalize_edge(edge, graph: nx.Graph):
    u, v = edge[:2]
    if nx.is_directed(graph):
        return (u, v)
    return tuple(sorted((u, v)))


def _graph_has_canonical_edge(graph: nx.Graph, edge) -> bool:
    u, v = edge
    return graph.has_edge(u, v)


def _connected_component_subgraphs(graph: nx.Graph):
    if nx.is_directed(graph):
        components = nx.weakly_connected_components(graph)
    else:
        components = nx.connected_components(graph)
    return [graph.subgraph(nodes).copy() for nodes in components]


def _sample_components(component_graphs, n_components, rng: random.Random):
    if n_components >= len(component_graphs):
        return list(component_graphs)
    indices = rng.sample(range(len(component_graphs)), k=n_components)
    return [component_graphs[idx] for idx in indices]


def _merge_component_graphs(component_graphs, template_graph: nx.Graph):
    merged = template_graph.__class__()
    next_node_id = 0
    for component in component_graphs:
        relabel_map = {}
        for node, node_attrs in component.nodes(data=True):
            relabel_map[node] = next_node_id
            merged.add_node(next_node_id, **dict(node_attrs))
            next_node_id += 1
        relabeled_component = nx.relabel_nodes(component, relabel_map, copy=True)
        if merged.is_multigraph():
            for u, v, key, edge_attrs in relabeled_component.edges(keys=True, data=True):
                merged.add_edge(u, v, key=key, **dict(edge_attrs))
        else:
            for u, v, edge_attrs in relabeled_component.edges(data=True):
                merged.add_edge(u, v, **dict(edge_attrs))
    merged.graph.update(dict(template_graph.graph))
    return merged


class EdgeGenerator:
    def __init__(
        self,
        feasibility_estimator=None,
        graph_estimator=None,
        target_estimator=None,
        target_estimator_mode: str = "classification",
        decomposition_function=None,
        *,
        partial_feasibility_estimator=None,
        final_feasibility_estimator=None,
        n_negative_per_positive: int = 3,
        n_replicates: int = 1,
        beam_size: int = 10,
        max_restarts: int = 3,
        fallback_base_steps: int = 2,
        fallback_growth_factor: float = 2.0,
        beam_growth_factor: float = 1.5,
        max_beam_size: int | None = None,
        enforce_diversity: bool = True,
        use_similarity_repulsion: bool = True,
        repulsion_weight: float = 0.2,
        repulsion_growth_factor: float = 1.5,
        max_repulsion_memory: int = 256,
        allow_self_loops: bool = False,
        fit_n_jobs: int = -1,
        fit_backend: str = "loky",
        verbose: bool = False,
        seed: int | None = None,
    ):
        (
            self.partial_feasibility_estimator,
            self.final_feasibility_estimator,
        ) = self._resolve_feasibility_estimators(
            feasibility_estimator=feasibility_estimator,
            partial_feasibility_estimator=partial_feasibility_estimator,
            final_feasibility_estimator=final_feasibility_estimator,
        )
        if graph_estimator is None:
            raise ValueError("graph_estimator is required")
        # Backward-compatible alias for code that still expects one search-time estimator.
        self.feasibility_estimator = self.partial_feasibility_estimator
        self.graph_estimator = graph_estimator
        self.target_estimator = target_estimator
        if target_estimator_mode not in {"classification", "regression"}:
            raise ValueError(
                "target_estimator_mode must be 'classification' or 'regression'"
            )
        self.target_estimator_mode = target_estimator_mode
        self.decomposition_function = decomposition_function
        self.n_negative_per_positive = n_negative_per_positive
        self.n_replicates = n_replicates
        self.beam_size = beam_size
        self.max_restarts = max_restarts
        self.fallback_base_steps = fallback_base_steps
        self.fallback_growth_factor = fallback_growth_factor
        self.beam_growth_factor = beam_growth_factor
        self.max_beam_size = max_beam_size
        self.enforce_diversity = enforce_diversity
        self.use_similarity_repulsion = use_similarity_repulsion
        self.repulsion_weight = repulsion_weight
        self.repulsion_growth_factor = repulsion_growth_factor
        self.max_repulsion_memory = max_repulsion_memory
        self.allow_self_loops = allow_self_loops
        self.fit_n_jobs = fit_n_jobs
        self.fit_backend = fit_backend
        self.verbose = verbose
        self.seed = seed
        self.rng = random.Random(seed)
        self.seed_graphs_ = None
        self.positives_ = None
        self.negatives_ = None
        self.dataset_ = None
        self.targets_ = None
        self.target_graphs_ = None
        self.target_values_ = None
        self.edge_attribute_templates_ = None
        self.top_k_ = beam_size
        self.n_tried_ = 0
        self.max_depth_ = 0
        self.embedding_cache_ = {}
        self.diversity_memory_hashes_ = []
        self.diversity_memory_hash_set_ = set()
        self.failed_memory_hashes_ = []
        self.failed_memory_hash_set_ = set()
        self.stored_graphs_ = None
        self.stored_targets_ = None
        self.stored_graph_hash_to_index_ = {}
        self.retrieval_transformer_ = None
        self.stored_retrieval_vectors_ = None
        self.stored_distance_matrix_ = None
        self.surgical_backtracking_ = True
        self.last_pair_session_ = None

    def _resolve_feasibility_estimators(
        self,
        *,
        feasibility_estimator,
        partial_feasibility_estimator,
        final_feasibility_estimator,
    ):
        resolved_partial = (
            partial_feasibility_estimator
            if partial_feasibility_estimator is not None
            else feasibility_estimator
        )
        resolved_final = (
            final_feasibility_estimator
            if final_feasibility_estimator is not None
            else feasibility_estimator
        )
        if resolved_partial is None or resolved_final is None:
            raise ValueError(
                "Provide feasibility_estimator or both partial_feasibility_estimator "
                "and final_feasibility_estimator"
            )
        if resolved_partial is resolved_final:
            resolved_final = copy.deepcopy(resolved_final)
        return resolved_partial, resolved_final

    def fit(self, graphs, targets=None):
        graph_list = self._as_graph_list(graphs)
        self.seed_graphs_ = [graph.copy() for graph in graph_list]
        dataset_parts = self._build_fragment_datasets(self.seed_graphs_)
        self._store_graph_estimator_training_data(dataset_parts)

        partial_fit_graphs = self._unique_graphs(
            list(self.seed_graphs_)
            + [graph for graph in self.positives_ if graph.number_of_edges() > 0]
        )
        final_fit_graphs = self._unique_graphs(list(self.seed_graphs_))
        partial_feasibility_fit_start = time.perf_counter()
        self.partial_feasibility_estimator.fit(partial_fit_graphs)
        partial_feasibility_fit_time = time.perf_counter() - partial_feasibility_fit_start
        final_feasibility_fit_start = time.perf_counter()
        self.final_feasibility_estimator.fit(final_fit_graphs)
        final_feasibility_fit_time = time.perf_counter() - final_feasibility_fit_start
        if self.verbose:
            partial_fit_min = int(partial_feasibility_fit_time // 60)
            partial_fit_sec = partial_feasibility_fit_time - 60 * partial_fit_min
            final_fit_min = int(final_feasibility_fit_time // 60)
            final_fit_sec = final_feasibility_fit_time - 60 * final_fit_min
            print(
                f"[fit] partial_feasibility_graphs={len(partial_fit_graphs)} "
                f"final_feasibility_graphs={len(final_fit_graphs)} "
                f"positives={len(self.positives_)} negatives={len(self.negatives_)} "
                f"dataset={len(self.dataset_)} "
                f"partial_time={partial_fit_min}m {partial_fit_sec:.1f}s "
                f"final_time={final_fit_min}m {final_fit_sec:.1f}s"
            )

        self.targets_ = np.array([label for graph, label in self.dataset_], dtype=int)
        train_graphs = [graph for graph, label in self.dataset_]
        graph_estimator_fit_start = time.perf_counter()
        self.graph_estimator.fit(train_graphs, self.targets_)
        graph_estimator_fit_time = time.perf_counter() - graph_estimator_fit_start
        if self.verbose:
            n_positive = int(np.sum(self.targets_ == 1))
            n_negative = int(np.sum(self.targets_ == 0))
            graph_estimator_fit_min = int(graph_estimator_fit_time // 60)
            graph_estimator_fit_sec = graph_estimator_fit_time - 60 * graph_estimator_fit_min
            print(
                f"[fit] graph_estimator_graphs={len(train_graphs)} "
                f"positive_labels={n_positive} negative_labels={n_negative} "
                f"time={graph_estimator_fit_min}m {graph_estimator_fit_sec:.1f}s"
            )

        if targets is not None:
            target_list = self._coerce_optional_per_graph_argument(
                targets,
                self.seed_graphs_,
                name="targets",
            )
            target_graphs, target_values = self._build_target_fragment_dataset(
                dataset_parts,
                target_list,
            )
            self._fit_target_estimator_from_fragments(
                target_graphs,
                target_values,
                verbose_prefix="[fit]",
            )
        else:
            self.target_graphs_ = None
            self.target_values_ = None

        self.edge_attribute_templates_ = self._collect_edge_attribute_templates(
            self.seed_graphs_
        )
        self.embedding_cache_ = {}
        self._initialize_diversity_memory(self.seed_graphs_)
        self.failed_memory_hashes_ = []
        self.failed_memory_hash_set_ = set()
        return self

    def fit_target_estimator(self, graphs, targets):
        if self.target_estimator is None:
            raise ValueError("target_estimator is None; provide one before fitting")
        graph_list = [graph.copy() for graph in self._as_graph_list(graphs)]
        target_list = self._coerce_optional_per_graph_argument(
            targets,
            graph_list,
            name="targets",
        )
        dataset_parts = self._build_fragment_datasets(graph_list)
        target_graphs, target_values = self._build_target_fragment_dataset(
            dataset_parts,
            target_list,
        )
        self._fit_target_estimator_from_fragments(
            target_graphs,
            target_values,
            verbose_prefix="[fit_target_estimator]",
        )
        return self

    def store(self, graphs, targets=None):
        graph_list = [graph.copy() for graph in self._as_graph_list(graphs)]
        if len(graph_list) < 2:
            raise ValueError("store(graphs, ...) requires at least two graphs")

        self.stored_graphs_ = graph_list
        if targets is None:
            self.stored_targets_ = None
        else:
            self.stored_targets_ = list(
                self._coerce_optional_per_graph_argument(
                    targets,
                    self.stored_graphs_,
                    name="targets",
                )
            )

        self.stored_graph_hash_to_index_ = {}
        for idx, graph in enumerate(self.stored_graphs_):
            graph_hash = hash_graph(graph)
            if graph_hash not in self.stored_graph_hash_to_index_:
                self.stored_graph_hash_to_index_[graph_hash] = idx

        transformer = self._make_retrieval_transformer()
        self.retrieval_transformer_ = transformer
        self.stored_retrieval_vectors_ = self._vectorize_retrieval_graphs(
            transformer,
            self.stored_graphs_,
            fit=True,
        )
        self.stored_distance_matrix_ = pairwise_distances(self.stored_retrieval_vectors_)
        np.fill_diagonal(self.stored_distance_matrix_, 0.0)
        return self

    def generate(
        self,
        graphs,
        n_edges,
        *,
        target=None,
        target_lambda: float = 1.0,
        return_path: bool = True,
        draw_graphs_fn: DrawGraphsFn | None = None,
        verbose: bool | None = None,
    ):
        if self.edge_attribute_templates_ is None:
            raise ValueError("EdgeGenerator must be fit before calling generate")
        verbose = self.verbose if verbose is None else verbose

        graph_list = self._as_graph_list(graphs)
        if isinstance(n_edges, int):
            edge_counts = [n_edges]
        else:
            edge_counts = list(n_edges)
        target_values = self._coerce_optional_per_graph_argument(
            target,
            graph_list,
            name="target",
        )

        if len(graph_list) != len(edge_counts):
            raise ValueError("graphs and n_edges must have the same length")

        paths = []
        for i, (graph, target_n_edges, target_value) in enumerate(
            zip(graph_list, edge_counts, target_values if target_values is not None else [None] * len(graph_list))
        ):
            try:
                path = self._generate_one(
                    graph,
                    target_n_edges,
                    target=target_value,
                    target_lambda=target_lambda,
                    draw_graphs_fn=draw_graphs_fn,
                    verbose=verbose,
                    graph_index=i,
                )
            except ValueError as exc:
                if verbose:
                    print(
                        f"[graph {i}] failed target_edges={target_n_edges} "
                        f"reason={exc}"
                    )
                continue
            paths.append(path)

        if self._is_single_graph_input(graphs):
            if not paths:
                return [] if return_path else None
            return paths[0] if return_path else paths[0][-1]
        if return_path:
            return paths
        return [path[-1] for path in paths]

    def generate_from_pair(
        self,
        graph_a,
        graph_b,
        *,
        size_of_edge_removal=0.5,
        n_paths: int = 3,
        path_k: int = 3,
        n_neighbors_per_path_graph: int = 3,
        target=None,
        target_lambda: float = 1.0,
        return_path: bool = True,
        draw_graphs_fn: DrawGraphsFn | None = None,
        verbose: bool | None = None,
    ):
        verbose = self.verbose if verbose is None else verbose
        if (graph_a is None) != (graph_b is None):
            raise ValueError("graph_a and graph_b must either both be provided or both be None")

        if graph_a is None and graph_b is None:
            session = self._require_cached_pair_session()
            if verbose:
                print("[pair] reusing cached pair session and fitted estimators")
            return self._generate_from_cached_pair_session(
                session,
                target_lambda=target_lambda,
                return_path=return_path,
                draw_graphs_fn=draw_graphs_fn,
                verbose=verbose,
            )

        self._require_stored_dataset()

        training_set_start = time.perf_counter()
        query = self._build_pair_query_corpus(graph_a, graph_b)
        path_matrix = self._path_matrix_from_distance_matrix(
            query["distance_matrix"],
            k=path_k,
        )
        paths = self._shortest_paths_from_matrix(
            path_matrix,
            query["source_idx"],
            query["dest_idx"],
            n_paths=n_paths,
        )
        if not paths:
            raise ValueError("Could not find shortest paths between the requested graphs")

        selected_indices = sorted({idx for path in paths for idx in path})
        selected_indices = self._augment_indices_with_nearest_neighbors(
            query["distance_matrix"],
            selected_indices,
            k=n_neighbors_per_path_graph,
        )
        fit_graphs = [query["graphs"][idx].copy() for idx in selected_indices]
        fit_targets = None
        if query["targets"] is not None:
            fit_targets = [query["targets"][idx] for idx in selected_indices]
        training_set_elapsed = time.perf_counter() - training_set_start

        if verbose:
            path_lengths = [len(path) for path in paths]
            print(
                f"[pair] source_idx={query['source_idx']} dest_idx={query['dest_idx']} "
                f"n_paths={len(paths)} selected_graphs={len(fit_graphs)} "
                f"path_k={path_k} "
                f"path_lengths={path_lengths} "
                f"training_set_time={self._format_minutes_seconds(training_set_elapsed)}"
            )
            print(f"[pair] selected_indices={selected_indices}")
            for path_idx, path in enumerate(paths, start=1):
                row_indices = list(path)
                row_graphs = [query["graphs"][idx] for idx in row_indices]
                row_titles = []
                for position, idx in enumerate(row_indices):
                    target_value = (
                        query["targets"][idx]
                        if query["targets"] is not None and idx < len(query["targets"])
                        else None
                    )
                    label = f"idx={idx}"
                    if position == 0:
                        label = f"src\n{label}"
                    elif position == len(row_indices) - 1:
                        label = f"dest\n{label}"
                    if target_value is not None:
                        label = f"{label}\ntgt={target_value}"
                    row_titles.append(label)
                print(f"[pair] path {path_idx}/{len(paths)} indices={path}")
                self._draw_graphs(
                    draw_graphs_fn,
                    row_graphs,
                    n_graphs_per_line=min(len(row_graphs), 7),
                    titles=row_titles,
                )
            training_titles = []
            for idx in selected_indices:
                target_value = (
                    query["targets"][idx]
                    if query["targets"] is not None and idx < len(query["targets"])
                    else None
                )
                label = f"idx={idx}"
                if target_value is not None:
                    label = f"{label}\ntgt={target_value}"
                training_titles.append(label)
            print(f"[pair] training_set_indices={selected_indices}")
            self._draw_graphs(
                draw_graphs_fn,
                fit_graphs,
                n_graphs_per_line=min(len(fit_graphs), 7),
                titles=training_titles,
            )

        if fit_targets is not None and all(target_value is not None for target_value in fit_targets):
            self.fit(fit_graphs, targets=fit_targets)
        else:
            self.fit(fit_graphs)
            if self.target_estimator is not None and fit_targets is not None:
                labeled_pairs = [
                    (graph, target_value)
                    for graph, target_value in zip(fit_graphs, fit_targets)
                    if target_value is not None
                ]
                if labeled_pairs:
                    labeled_graphs, labeled_targets = zip(*labeled_pairs)
                    self.fit_target_estimator(list(labeled_graphs), list(labeled_targets))

        resolved_target = target
        if resolved_target is None:
            resolved_target = self._infer_pair_target(
                query["targets"][query["source_idx"]] if query["targets"] is not None else None,
                query["targets"][query["dest_idx"]] if query["targets"] is not None else None,
            )
        self._cache_pair_session(
            graph_a=graph_a,
            graph_b=graph_b,
            size_of_edge_removal=size_of_edge_removal,
            target=resolved_target,
        )

        return self._generate_from_cached_pair_session(
            self.last_pair_session_,
            target_lambda=target_lambda,
            return_path=return_path,
            draw_graphs_fn=draw_graphs_fn,
            verbose=verbose,
        )

    def _cache_pair_session(
        self,
        *,
        graph_a,
        graph_b,
        size_of_edge_removal,
        target,
    ) -> None:
        self.last_pair_session_ = {
            "graph_a": None if graph_a is None else graph_a.copy(),
            "graph_b": None if graph_b is None else graph_b.copy(),
            "size_of_edge_removal": float(size_of_edge_removal),
            "target": target,
        }

    def _require_cached_pair_session(self):
        if self.last_pair_session_ is None:
            raise ValueError(
                "No cached pair session is available; call generate_from_pair(graph_a, graph_b, ...)"
            )
        return self.last_pair_session_

    def _generate_from_cached_pair_session(
        self,
        session,
        *,
        target_lambda: float,
        return_path: bool,
        draw_graphs_fn: DrawGraphsFn | None,
        verbose: bool,
    ):
        graph_a = session["graph_a"]
        graph_b = session["graph_b"]
        size_of_edge_removal = session["size_of_edge_removal"]
        resolved_target = session["target"]

        start_graph_a, target_n_edges_a = remove_edges(
            graph_a,
            size=size_of_edge_removal,
        )
        start_graph_b, target_n_edges_b = remove_edges(
            graph_b,
            size=size_of_edge_removal,
        )
        mixed_graph = mix_connected_components(
            start_graph_a,
            start_graph_b,
            seed=self.rng.randrange(10**9),
        )
        mixed_target_n_edges = int(round(np.mean([target_n_edges_a, target_n_edges_b])))
        return self.generate(
            mixed_graph,
            mixed_target_n_edges,
            target=resolved_target,
            target_lambda=target_lambda,
            return_path=return_path,
            draw_graphs_fn=draw_graphs_fn,
            verbose=verbose,
        )

    def _augment_indices_with_nearest_neighbors(
        self,
        distance_matrix,
        selected_indices,
        *,
        k: int,
    ):
        if k < 0:
            raise ValueError("n_neighbors_per_path_graph must be >= 0")

        selected = []
        seen = set()
        for idx in selected_indices:
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(int(idx))

        if k == 0 or not selected:
            return selected

        distances = np.asarray(distance_matrix, dtype=float)
        n_nodes = distances.shape[0]
        for idx in list(selected):
            if idx < 0 or idx >= n_nodes:
                continue
            row = distances[idx]
            neighbor_order = np.argsort(row, kind="stable")
            n_added = 0
            for neighbor_idx in neighbor_order:
                neighbor_idx = int(neighbor_idx)
                if neighbor_idx == idx or neighbor_idx in seen:
                    continue
                if not np.isfinite(row[neighbor_idx]):
                    continue
                seen.add(neighbor_idx)
                selected.append(neighbor_idx)
                n_added += 1
                if n_added >= k:
                    break
        return sorted(selected)

    def _generate_one(
        self,
        graph: nx.Graph,
        n_edges: int,
        *,
        target=None,
        target_lambda: float = 1.0,
        draw_graphs_fn: DrawGraphsFn | None = None,
        verbose: bool = False,
        graph_index: int = 0,
    ):
        start_graph = graph.copy()
        if start_graph.number_of_edges() > n_edges:
            raise ValueError("Input graph already has more edges than n_edges")

        self.n_tried_ = 0
        self.max_depth_ = 0
        n_fallbacks = max(0, self.max_restarts)
        total_phases = n_fallbacks + 1
        start_time = time.perf_counter()
        if verbose:
            remaining_edges = n_edges - start_graph.number_of_edges()
            start_parts = [
                f"[graph {graph_index}] start",
                f"start_edges={start_graph.number_of_edges()}",
                f"target_edges={n_edges}",
                f"remaining_edges={remaining_edges}",
            ]
            if target is not None:
                start_parts.append(f"target={target}")
                start_parts.append(f"target_lambda={target_lambda:.3f}")
            print(" ".join(start_parts))
            self._draw_graphs(draw_graphs_fn, [start_graph])

        if start_graph.number_of_edges() == n_edges:
            if not bool(self.final_feasibility_estimator.predict([start_graph])[0]):
                raise ValueError("Start graph does not satisfy the final feasibility estimator")
            if verbose:
                print(
                    f"[graph {graph_index}] solved depth=0 max_depth=0 "
                    f"edges={start_graph.number_of_edges()} remaining_edges=0 "
                    f"tried=0 elapsed=0m 0.0s eta=0m 0.0s"
                )
            return [start_graph]

        beam = [self._make_state(start_graph, parent=None, score=1.0, depth=0)]
        beam_history = [self._copy_beam(beam)]
        blocked_state_keys_by_depth: dict[int, set] = {}
        tabu_path_signatures = set()
        visited = self._rebuild_visited_from_history(beam_history)
        depth = 0
        fallback_index = -1
        beam_limit = self._beam_limit_for_fallback(fallback_index)
        self.top_k_ = beam_limit
        step_start_time = time.perf_counter()

        if verbose and total_phases > 1:
            print(
                f"[graph {graph_index}] phase=1/{total_phases} "
                f"beam_limit={beam_limit} fallback=0/{n_fallbacks}"
            )

        while beam:
            if depth >= n_edges:
                break

            generated = []
            for state in beam:
                generated.extend(self._expand_state(state))

            self.n_tried_ += len(generated)
            feasible_candidates = []
            infeasible_candidates = []
            repulsion_lambda = 0.0
            if generated:
                generated_graphs = [cand["graph"] for cand in generated]
                partial_feasibility_mask = np.asarray(
                    self.partial_feasibility_estimator.predict(generated_graphs),
                    dtype=bool,
                )
                positive_scores = self._positive_scores(generated_graphs)
                target_scores = self._target_scores(
                    generated_graphs,
                    target=target,
                )
                partial_terminal_candidates = []
                for cand, is_partial_feasible, score, target_score in zip(
                    generated,
                    partial_feasibility_mask,
                    positive_scores,
                    target_scores,
                ):
                    cand["score"] = float(score)
                    cand["target_score"] = float(target_score)
                    cand["selection_score"] = float(
                        cand["score"] + target_lambda * cand["target_score"]
                    )
                    if is_partial_feasible:
                        if cand["graph"].number_of_edges() == n_edges:
                            partial_terminal_candidates.append(cand)
                        else:
                            feasible_candidates.append(cand)
                    else:
                        cand["feasibility_stage"] = "partial"
                        infeasible_candidates.append(cand)

                if partial_terminal_candidates:
                    final_mask = np.asarray(
                        self.final_feasibility_estimator.predict(
                            [cand["graph"] for cand in partial_terminal_candidates]
                        ),
                        dtype=bool,
                    )
                    for cand, is_final_feasible in zip(
                        partial_terminal_candidates,
                        final_mask,
                    ):
                        if is_final_feasible:
                            feasible_candidates.append(cand)
                        else:
                            cand["feasibility_stage"] = "final"
                            infeasible_candidates.append(cand)

            if feasible_candidates:
                repulsions, repulsion_lambda = self._repulsion_values(
                    [cand["graph"] for cand in feasible_candidates],
                    fallback_index=fallback_index,
                )
                for cand, repulsion in zip(feasible_candidates, repulsions):
                    cand["repulsion"] = float(repulsion)
                    cand["selection_score"] = float(
                        cand["selection_score"] - repulsion_lambda * cand["repulsion"]
                    )
                feasible_candidates.sort(
                    key=lambda cand: cand["selection_score"], reverse=True
                )

            if infeasible_candidates:
                self._annotate_infeasible_candidates_with_violations(infeasible_candidates)
                infeasible_candidates.sort(
                    key=lambda cand: (
                        cand.get("selection_score", cand["score"]),
                        -cand.get("violation_count", 0.0),
                    ),
                    reverse=True,
                )

            next_depth = depth + 1
            blocked_state_keys = blocked_state_keys_by_depth.get(next_depth, set())
            unseen_candidates = []
            for cand in feasible_candidates:
                state_key = cand["key"]
                if state_key in visited or state_key in blocked_state_keys:
                    continue
                if cand["path_signature"] in tabu_path_signatures:
                    continue
                if (
                    self.enforce_diversity
                    and cand["graph_hash"] in self.diversity_memory_hash_set_
                ):
                    continue
                unseen_candidates.append(cand)

            retained = self._select_beam_candidates(unseen_candidates, beam_limit=beam_limit)

            if verbose:
                target_active = target is not None
                repulsion_active = repulsion_lambda > 0.0
                best_score = retained[0]["score"] if retained else None
                best_selection_score = (
                    retained[0]["selection_score"] if retained else None
                ) if (target_active or repulsion_active) else None
                best_target_score = (
                    retained[0].get("target_score") if retained and target_active else None
                )
                best_repulsion = retained[0].get("repulsion", 0.0) if retained else 0.0
                step_elapsed = time.perf_counter() - step_start_time
                step_elapsed_str = self._format_minutes_seconds(step_elapsed)
                best_score_str = f"{best_score:.3f}" if best_score is not None else "None"
                best_selection_score_str = (
                    f"{best_selection_score:.3f}"
                    if best_selection_score is not None
                    else "None"
                )
                best_target_score_str = (
                    f"{best_target_score:.3f}"
                    if best_target_score is not None
                    else "None"
                )
                current_edges = (
                    retained[0]["graph"].number_of_edges()
                    if retained
                    else start_graph.number_of_edges() + next_depth
                )
                remaining_edges = max(0, n_edges - current_edges)
                eta = remaining_edges * step_elapsed
                eta_str = self._format_minutes_seconds(eta)
                line1 = (
                    f"[graph {graph_index}] phase={fallback_index + 2}/{total_phases} "
                    f"depth={next_depth} remaining_edges={remaining_edges} "
                    f"step_time={step_elapsed_str} eta={eta_str}"
                )
                line2 = (
                    f"generated={len(generated)} feasible={len(feasible_candidates)} "
                    f"retained={len(retained)} tried={self.n_tried_}"
                )
                line3_parts = [f"best_score={best_score_str}"]
                if target_active:
                    line3_parts.append(f"best_target_score={best_target_score_str}")
                if target_active or repulsion_active:
                    line3_parts.append(f"best_selection_score={best_selection_score_str}")
                if repulsion_active:
                    line3_parts.append(f"best_repulsion={best_repulsion:.3f}")
                line4_parts = []
                if target_active:
                    line4_parts.append(f"target_lambda={target_lambda:.3f}")
                if repulsion_active:
                    line4_parts.append(f"repulsion_lambda={repulsion_lambda:.3f}")
                line4_parts.append(f"beam_limit={beam_limit}")
                print("\n".join([line1, line2, " ".join(line3_parts), " ".join(line4_parts)]))
                if retained:
                    retained_graphs = [cand["graph"] for cand in retained]
                    retained_titles = []
                    for cand in retained:
                        title_line1_parts = []
                        title_line2_parts = [f"clf={cand['score']:.3f}"]
                        if target_active or repulsion_active:
                            title_line1_parts.append(
                                f"sel={cand.get('selection_score', cand['score']):.3f}"
                            )
                        if repulsion_active:
                            title_line1_parts.append(
                                f"rep={cand.get('repulsion', 0.0):.3f}"
                            )
                        if target_active:
                            title_line2_parts.append(
                                f"tgt={cand.get('target_score', 0.0):.3f}"
                            )
                        if title_line1_parts:
                            retained_titles.append(
                                " ".join(title_line1_parts) + "\n" + " ".join(title_line2_parts)
                            )
                        else:
                            retained_titles.append(" ".join(title_line2_parts))
                    self._draw_graphs(
                        draw_graphs_fn,
                        retained_graphs,
                        n_graphs_per_line=min(len(retained_graphs), 7),
                        titles=retained_titles,
                    )

            if retained:
                for cand in retained:
                    visited.add(cand["key"])
                depth = next_depth
                self.max_depth_ = max(self.max_depth_, depth)
                beam = retained
                if len(beam_history) > depth:
                    beam_history[depth] = self._copy_beam(retained)
                    del beam_history[depth + 1 :]
                else:
                    beam_history.append(self._copy_beam(retained))
                step_start_time = time.perf_counter()

                for state in retained:
                    if state["graph"].number_of_edges() == n_edges:
                        path = self._reconstruct_path(state)
                        if verbose:
                            elapsed = time.perf_counter() - start_time
                            elapsed_str = self._format_minutes_seconds(elapsed)
                            print(
                                f"[graph {graph_index}] solved phase={fallback_index + 2}/{total_phases} "
                                f"depth={depth} max_depth={self.max_depth_} "
                                f'edges={state["graph"].number_of_edges()} remaining_edges=0 '
                                f"tried={self.n_tried_} elapsed={elapsed_str} eta=0m 0.0s"
                            )
                        return path
                continue

            blocked_state_keys_by_depth.setdefault(depth, set()).update(
                state["key"] for state in beam
            )
            tabu_path_signatures.update(state["path_signature"] for state in beam)
            self._remember_failed_graphs([state["graph"] for state in beam])

            if fallback_index + 1 >= n_fallbacks:
                break

            fallback_index += 1
            rollback_steps = self._rollback_steps_for_fallback(fallback_index)
            beam_limit = self._beam_limit_for_fallback(fallback_index)
            self.top_k_ = beam_limit
            repaired_beam = self._repair_beam_from_infeasible_candidates(
                beam,
                infeasible_candidates,
                rollback_steps=rollback_steps,
                beam_limit=beam_limit,
            )
            if repaired_beam:
                repaired_depth = repaired_beam[0]["depth"]
                beam_history = beam_history[: repaired_depth + 1]
                beam_history[repaired_depth] = self._copy_beam(repaired_beam)
                beam = repaired_beam
                depth = repaired_depth
                visited = self._rebuild_visited_from_history(beam_history)
                step_start_time = time.perf_counter()

                if verbose:
                    removed_descriptions = [
                        ",".join(str(edge) for edge in state.get("repair_removed_edges", ()))
                        for state in repaired_beam
                    ]
                    print(
                        f"[graph {graph_index}] fallback={fallback_index + 1}/{n_fallbacks} "
                        f"rollback_steps={rollback_steps} surgical_repairs={len(repaired_beam)} "
                        f"to_depth={repaired_depth} beam_limit={beam_limit}"
                    )
                    print(
                        f"[graph {graph_index}] surgical_removed_edges={removed_descriptions}"
                    )

                if verbose and total_phases > 1:
                    print(
                        f"[graph {graph_index}] phase={fallback_index + 2}/{total_phases} "
                        f"beam_limit={beam_limit} fallback={fallback_index + 1}/{n_fallbacks}"
                    )
                continue

            fallback_depth = max(0, depth - rollback_steps)
            beam_history = beam_history[: fallback_depth + 1]
            beam = self._copy_beam(beam_history[fallback_depth])
            depth = fallback_depth
            visited = self._rebuild_visited_from_history(beam_history)
            step_start_time = time.perf_counter()

            if verbose:
                print(
                    f"[graph {graph_index}] fallback={fallback_index + 1}/{n_fallbacks} "
                    f"rollback_steps={rollback_steps} to_depth={fallback_depth} "
                    f"beam_limit={beam_limit}"
                )

            if verbose and total_phases > 1:
                print(
                    f"[graph {graph_index}] phase={fallback_index + 2}/{total_phases} "
                    f"beam_limit={beam_limit} fallback={fallback_index + 1}/{n_fallbacks}"
                )

        raise ValueError("Could not generate a feasible graph with the requested number of edges")

    def _expand_state(self, state):
        candidates = []
        graph = state["graph"]
        for edge in self._missing_edges(graph):
            for edge_attrs in self.edge_attribute_templates_:
                candidate_graph = graph.copy()
                candidate_graph.add_edge(*edge, **edge_attrs)
                candidates.append(
                    self._make_state(
                        candidate_graph,
                        parent=state,
                        score=None,
                        added_edge=edge,
                        depth=state["depth"] + 1,
                    )
                )
        self.rng.shuffle(candidates)
        return candidates

    def _make_state(self, graph, parent, score, *, added_edge=None, depth: int | None = None, edge_order=None):
        state_key = self._state_key(graph)
        graph_hash = hash_graph(graph)
        if parent is None:
            path_signature = (state_key,)
        else:
            path_signature = parent["path_signature"] + (state_key,)
        if edge_order is None:
            if parent is not None and added_edge is not None:
                edge_order = parent["edge_order"] + (
                    _canonicalize_edge(added_edge, graph),
                )
            else:
                edge_order = self._canonical_graph_edges(graph)
        if depth is None:
            depth = 0 if parent is None else parent.get("depth", 0)
            if added_edge is not None:
                depth += 1
        return {
            "graph": graph,
            "graph_hash": graph_hash,
            "parent": parent,
            "score": score,
            "selection_score": score,
            "key": state_key,
            "path_signature": path_signature,
            "added_edge": None if added_edge is None else _canonicalize_edge(added_edge, graph),
            "repair_removed_edges": (),
            "edge_order": tuple(edge_order),
            "depth": int(depth),
        }

    def _repair_beam_from_infeasible_candidates(
        self,
        beam,
        infeasible_candidates,
        *,
        rollback_steps: int,
        beam_limit: int,
    ):
        if not self.surgical_backtracking_ or rollback_steps < 1 or not infeasible_candidates:
            return []

        selected_candidates = self._select_infeasible_candidates_for_repair(
            beam,
            infeasible_candidates,
            beam_limit=beam_limit,
        )
        if not selected_candidates:
            return []

        self._annotate_infeasible_candidates_with_violating_edge_sets(selected_candidates)

        grouped_candidates: dict[tuple, list] = {}
        for cand in selected_candidates:
            grouped_candidates.setdefault(cand["parent"]["key"], []).append(cand)

        repaired_states = []
        for state in beam:
            state_candidates = grouped_candidates.get(state["key"], [])
            removed_edges, repair_score = self._select_edges_for_surgical_repair(
                state,
                state_candidates,
                rollback_steps=rollback_steps,
            )
            if not removed_edges:
                continue
            repaired_states.append(
                self._make_repaired_state(
                    state,
                    removed_edges,
                    score=repair_score,
                )
            )

        if not repaired_states:
            return []

        repaired_states.sort(
            key=lambda state: state.get("selection_score", 0.0),
            reverse=True,
        )
        repaired_states = self._deduplicate_retained_candidates(
            repaired_states,
            fallback_candidates=[],
            target_size=min(beam_limit, len(repaired_states)),
        )
        return repaired_states

    def _select_infeasible_candidates_for_repair(self, beam, infeasible_candidates, *, beam_limit: int):
        if not infeasible_candidates:
            return []
        by_parent = {state["key"]: [] for state in beam}
        for cand in infeasible_candidates:
            parent = cand.get("parent")
            if parent is None:
                continue
            if parent["key"] not in by_parent:
                continue
            by_parent[parent["key"]].append(cand)
        per_parent_limit = max(1, beam_limit)
        selected = []
        for state in beam:
            state_candidates = by_parent.get(state["key"], [])
            state_candidates.sort(
                key=lambda cand: (
                    cand.get("selection_score", cand.get("score", 0.0)),
                    -cand.get("violation_count", 0.0),
                ),
                reverse=True,
            )
            selected.extend(state_candidates[:per_parent_limit])
        return selected

    def _feasibility_estimator_for_stage(self, stage: str):
        if stage == "final":
            return self.final_feasibility_estimator
        return self.partial_feasibility_estimator

    def _annotate_infeasible_candidates_with_violations(self, candidates) -> None:
        if not candidates:
            return
        grouped_candidates = {}
        for cand in candidates:
            stage = cand.get("feasibility_stage", "partial")
            grouped_candidates.setdefault(stage, []).append(cand)
        for stage, stage_candidates in grouped_candidates.items():
            estimator = self._feasibility_estimator_for_stage(stage)
            violation_counts = np.asarray(
                estimator.number_of_violations(
                    [cand["graph"] for cand in stage_candidates]
                ),
                dtype=float,
            ).reshape(-1)
            for cand, violation_count in zip(stage_candidates, violation_counts):
                cand["violation_count"] = float(violation_count)

    def _annotate_infeasible_candidates_with_violating_edge_sets(self, candidates) -> None:
        if not candidates:
            return
        grouped_candidates = {}
        for cand in candidates:
            stage = cand.get("feasibility_stage", "partial")
            grouped_candidates.setdefault(stage, []).append(cand)
        for stage, stage_candidates in grouped_candidates.items():
            estimator = self._feasibility_estimator_for_stage(stage)
            if not hasattr(estimator, "violating_edge_sets"):
                for cand in stage_candidates:
                    cand["violating_edge_sets"] = []
                continue
            violating_edge_sets = estimator.violating_edge_sets(
                [cand["graph"] for cand in stage_candidates]
            )
            for cand, edge_sets in zip(stage_candidates, violating_edge_sets):
                cand["violating_edge_sets"] = edge_sets

    def _select_edges_for_surgical_repair(self, state, candidates, *, rollback_steps: int):
        if rollback_steps < 1:
            return [], 0.0

        parent_edges = set(self._canonical_graph_edges(state["graph"]))
        if not parent_edges:
            return [], 0.0

        edge_counts = {}
        edge_weights = {}
        recency = self._edge_recency_map(state)
        for cand in candidates:
            candidate_score = max(
                0.0,
                float(cand.get("selection_score", cand.get("score", 0.0))),
            )
            for edge_set in cand.get("violating_edge_sets", []):
                relevant_edges = [edge for edge in edge_set if edge in parent_edges]
                if not relevant_edges:
                    continue
                increment = 1.0 / float(len(relevant_edges))
                for edge in relevant_edges:
                    edge_counts[edge] = edge_counts.get(edge, 0.0) + 1.0
                    edge_weights[edge] = edge_weights.get(edge, 0.0) + candidate_score * increment

        ranked_edges = sorted(
            parent_edges,
            key=lambda edge: (
                edge_counts.get(edge, 0.0),
                edge_weights.get(edge, 0.0),
                recency.get(edge, 0),
            ),
            reverse=True,
        )
        if not any(edge_counts.get(edge, 0.0) > 0.0 for edge in ranked_edges):
            return [], 0.0
        selected = [
            edge for edge in ranked_edges if edge_counts.get(edge, 0.0) > 0.0
        ][:rollback_steps]

        if len(selected) < min(rollback_steps, len(parent_edges)):
            for edge in sorted(parent_edges, key=lambda edge: recency.get(edge, 0), reverse=True):
                if edge in selected:
                    continue
                selected.append(edge)
                if len(selected) >= min(rollback_steps, len(parent_edges)):
                    break

        repair_score = float(
            sum(edge_counts.get(edge, 0.0) for edge in selected)
            + sum(edge_weights.get(edge, 0.0) for edge in selected)
        )
        return selected, repair_score

    def _make_repaired_state(self, state, removed_edges, *, score: float):
        repaired_graph = state["graph"].copy()
        for edge in removed_edges:
            if repaired_graph.has_edge(*edge):
                repaired_graph.remove_edge(*edge)
        removed_edge_set = set(removed_edges)
        repaired_edge_order = tuple(
            edge for edge in state["edge_order"] if edge not in removed_edge_set
        )
        repaired_state = self._make_state(
            repaired_graph,
            parent=state,
            score=score,
            depth=max(0, state["depth"] - len(removed_edges)),
            edge_order=repaired_edge_order,
        )
        repaired_state["repair_removed_edges"] = tuple(removed_edges)
        repaired_state["selection_score"] = float(score)
        return repaired_state

    def _select_beam_candidates(self, candidates, *, beam_limit: int | None = None):
        if not candidates:
            return []

        current_beam_limit = self.top_k_ if beam_limit is None else beam_limit
        n_keep = min(current_beam_limit, len(candidates))
        n_top = 1 if n_keep == 1 else max(1, n_keep // 2)
        n_random = n_keep - n_top

        top_candidates = candidates[:n_top]
        remaining_candidates = candidates[n_top:]
        if len(remaining_candidates) <= n_random:
            random_candidates = remaining_candidates
        else:
            random_candidates = self.rng.sample(remaining_candidates, k=n_random)

        retained = top_candidates + random_candidates
        retained = self._deduplicate_retained_candidates(
            retained,
            fallback_candidates=remaining_candidates,
            target_size=n_keep,
        )
        retained.sort(
            key=lambda cand: cand.get("selection_score", cand["score"]),
            reverse=True,
        )
        return retained

    def _deduplicate_retained_candidates(
        self,
        retained,
        *,
        fallback_candidates,
        target_size: int,
    ):
        unique_retained = []
        seen_hashes = set()

        def add_candidate(candidate):
            graph_hash = candidate.get("graph_hash", hash_graph(candidate["graph"]))
            if graph_hash in seen_hashes:
                return False
            seen_hashes.add(graph_hash)
            unique_retained.append(candidate)
            return True

        for candidate in retained:
            add_candidate(candidate)

        if len(unique_retained) >= target_size:
            return unique_retained[:target_size]

        for candidate in fallback_candidates:
            if len(unique_retained) >= target_size:
                break
            add_candidate(candidate)

        return unique_retained

    def _require_stored_dataset(self):
        if self.stored_graphs_ is None or self.stored_distance_matrix_ is None:
            raise ValueError("Call store(graphs, targets=...) before generate_from_pair")

    def _make_retrieval_transformer(self):
        transformer = getattr(self.graph_estimator, "transformer", None)
        if transformer is None:
            raise ValueError("graph_estimator must expose a transformer for store(...)")
        return copy.deepcopy(transformer)

    def _vectorize_retrieval_graphs(self, transformer, graphs, *, fit: bool):
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
        if hasattr(features, "toarray"):
            features = features.toarray()
        return np.asarray(features, dtype=float)

    def _build_pair_query_corpus(self, graph_a, graph_b):
        graphs = [graph.copy() for graph in self.stored_graphs_]
        targets = None if self.stored_targets_ is None else list(self.stored_targets_)
        vectors = np.asarray(self.stored_retrieval_vectors_, dtype=float)
        distance_matrix = np.asarray(self.stored_distance_matrix_, dtype=float).copy()

        source_idx, graphs, targets, vectors, distance_matrix = self._resolve_or_append_query_graph(
            graph_a,
            graphs=graphs,
            targets=targets,
            vectors=vectors,
            distance_matrix=distance_matrix,
        )
        dest_idx, graphs, targets, vectors, distance_matrix = self._resolve_or_append_query_graph(
            graph_b,
            graphs=graphs,
            targets=targets,
            vectors=vectors,
            distance_matrix=distance_matrix,
        )
        if source_idx == dest_idx:
            raise ValueError("graph_a and graph_b resolve to the same stored/query graph")
        return {
            "graphs": graphs,
            "targets": targets,
            "vectors": vectors,
            "distance_matrix": distance_matrix,
            "source_idx": source_idx,
            "dest_idx": dest_idx,
        }

    def _resolve_or_append_query_graph(
        self,
        graph,
        *,
        graphs,
        targets,
        vectors,
        distance_matrix,
    ):
        graph_hash = hash_graph(graph)
        stored_idx = self.stored_graph_hash_to_index_.get(graph_hash)
        if stored_idx is not None:
            return stored_idx, graphs, targets, vectors, distance_matrix

        graph_copy = graph.copy()
        query_vector = self._vectorize_retrieval_graphs(
            self.retrieval_transformer_,
            [graph_copy],
            fit=False,
        )[0]
        idx = len(graphs)
        graphs.append(graph_copy)
        if targets is not None:
            targets.append(None)
        distance_matrix = self._append_distance_row(distance_matrix, vectors, query_vector)
        vectors = np.vstack([vectors, query_vector])
        return idx, graphs, targets, vectors, distance_matrix

    def _append_distance_row(self, distance_matrix, vectors, query_vector):
        if vectors.size == 0:
            return np.zeros((1, 1), dtype=float)
        distances = pairwise_distances(
            np.asarray(query_vector, dtype=float).reshape(1, -1),
            np.asarray(vectors, dtype=float),
        ).ravel()
        old_n = distance_matrix.shape[0]
        expanded = np.zeros((old_n + 1, old_n + 1), dtype=float)
        expanded[:old_n, :old_n] = distance_matrix
        expanded[-1, :-1] = distances
        expanded[:-1, -1] = distances
        expanded[-1, -1] = 0.0
        return expanded

    def _path_matrix_from_distance_matrix(self, distance_matrix, *, k: int):
        if k < 0:
            raise ValueError("path_k must be >= 0")
        distances = np.asarray(distance_matrix, dtype=float)
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError("distance_matrix must be a square matrix")
        if distances.shape[0] <= 1:
            return distances.copy()
        adjacency = _build_adjacency(distances, k=max(1, int(k)), degree_penalty=0.0)
        path_matrix = adjacency.toarray().astype(float, copy=False)
        missing_mask = path_matrix == 0.0
        np.fill_diagonal(missing_mask, False)
        path_matrix[missing_mask] = np.inf
        np.fill_diagonal(path_matrix, 0.0)
        return path_matrix

    def _shortest_paths_from_matrix(self, distance_matrix, source_idx, dest_idx, *, n_paths: int):
        if n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        working_matrix = np.asarray(distance_matrix, dtype=float).copy()
        paths = []
        for _ in range(n_paths):
            path = self._dense_shortest_path(working_matrix, source_idx, dest_idx)
            if not path:
                break
            paths.append(path)
            self._remove_path_edges_from_matrix(working_matrix, path)
        return paths

    def _dense_shortest_path(self, distance_matrix: np.ndarray, source_idx: int, dest_idx: int):
        n_nodes = distance_matrix.shape[0]
        inf = float("inf")
        dist = np.full(n_nodes, inf, dtype=float)
        prev = np.full(n_nodes, -1, dtype=int)
        visited = np.zeros(n_nodes, dtype=bool)
        dist[source_idx] = 0.0

        for _ in range(n_nodes):
            current = -1
            current_dist = inf
            for node_idx in range(n_nodes):
                if visited[node_idx]:
                    continue
                if dist[node_idx] < current_dist:
                    current = node_idx
                    current_dist = dist[node_idx]
            if current < 0 or not np.isfinite(current_dist):
                break
            if current == dest_idx:
                break
            visited[current] = True

            row = distance_matrix[current]
            for neighbor_idx, weight in enumerate(row):
                if visited[neighbor_idx] or neighbor_idx == current:
                    continue
                if not np.isfinite(weight):
                    continue
                candidate_dist = current_dist + weight
                if candidate_dist < dist[neighbor_idx]:
                    dist[neighbor_idx] = candidate_dist
                    prev[neighbor_idx] = current

        if not np.isfinite(dist[dest_idx]):
            return []

        path = [dest_idx]
        cursor = dest_idx
        while cursor != source_idx:
            cursor = prev[cursor]
            if cursor < 0:
                return []
            path.append(int(cursor))
        path.reverse()
        return path

    def _remove_path_edges_from_matrix(self, distance_matrix: np.ndarray, path):
        for left, right in zip(path[:-1], path[1:]):
            distance_matrix[left, right] = np.inf
            distance_matrix[right, left] = np.inf

    def _infer_pair_target(self, target_a, target_b):
        if target_a is None or target_b is None:
            return None
        if not isinstance(target_a, (int, float, np.integer, np.floating)):
            return None
        if not isinstance(target_b, (int, float, np.integer, np.floating)):
            return None
        mean_target = 0.5 * (float(target_a) + float(target_b))
        if self.target_estimator_mode == "regression":
            return mean_target
        return int(round(mean_target))

    def _build_fragment_datasets(self, graphs):
        dataset_seeds = [self.rng.randrange(10**9) for _ in graphs]
        dataset_builder = self._dataset_builder()
        return Parallel(n_jobs=self.fit_n_jobs, backend=self.fit_backend)(
            delayed(dataset_builder)(
                graph,
                **self._dataset_builder_kwargs(dataset_seed),
            )
            for graph, dataset_seed in zip(graphs, dataset_seeds)
        )

    def _dataset_builder(self):
        if self.decomposition_function is None:
            return make_edge_regression_dataset
        return make_edge_regression_dataset_subgraph_ordered

    def _dataset_builder_kwargs(self, dataset_seed: int):
        kwargs = dict(
            n_negative_per_positive=self.n_negative_per_positive,
            n_replicates=self.n_replicates,
            seed=dataset_seed,
            allow_self_loops=self.allow_self_loops,
        )
        if self.decomposition_function is not None:
            kwargs["decomposition_function"] = self.decomposition_function
            kwargs["nbits"] = self._decomposition_nbits()
        return kwargs

    def _decomposition_nbits(self) -> int:
        transformer = getattr(self.graph_estimator, "transformer", None)
        nbits = getattr(transformer, "nbits", None)
        if nbits is None:
            raise ValueError(
                "decomposition_function requires graph_estimator.transformer.nbits"
            )
        return int(nbits)

    def _store_graph_estimator_training_data(self, dataset_parts):
        self.positives_ = []
        self.negatives_ = []
        self.dataset_ = []
        for positives, negatives, dataset in dataset_parts:
            self.positives_.extend(positives)
            self.negatives_.extend(negatives)
            self.dataset_.extend(dataset)

    def _build_target_fragment_dataset(self, dataset_parts, targets):
        target_graphs = []
        target_values = []
        for target_value, (positives, _, _) in zip(targets, dataset_parts):
            target_graphs.extend(positives)
            target_values.extend([target_value] * len(positives))
        return target_graphs, target_values

    def _fit_target_estimator_from_fragments(
        self,
        target_graphs,
        target_values,
        *,
        verbose_prefix: str,
    ):
        if self.target_estimator is None:
            raise ValueError("target_estimator is None; provide one before fitting")
        self.target_graphs_ = list(target_graphs)
        self.target_values_ = list(target_values)
        target_fit_start = time.perf_counter()
        self.target_estimator.fit(self.target_graphs_, self.target_values_)
        target_fit_time = time.perf_counter() - target_fit_start
        if self.verbose:
            target_fit_min = int(target_fit_time // 60)
            target_fit_sec = target_fit_time - 60 * target_fit_min
            target_label = (
                "target_classes"
                if self.target_estimator_mode == "classification"
                else "target_values"
            )
            print(
                f"{verbose_prefix} target_estimator_graphs={len(self.target_graphs_)} "
                f"{target_label}={len(set(self.target_values_))} "
                f"mode={self.target_estimator_mode} "
                f"time={target_fit_min}m {target_fit_sec:.1f}s"
            )

    def _rollback_steps_for_fallback(self, fallback_index: int) -> int:
        steps = self.fallback_base_steps * (self.fallback_growth_factor**fallback_index)
        return max(1, int(math.ceil(steps)))

    def _beam_limit_for_fallback(self, fallback_index: int) -> int:
        if fallback_index < 0:
            beam_limit = self.beam_size
        else:
            beam_limit = int(
                math.ceil(
                    self.beam_size * (self.beam_growth_factor ** (fallback_index + 1))
                )
            )
        beam_limit = max(1, beam_limit)
        if self.max_beam_size is not None:
            beam_limit = min(beam_limit, self.max_beam_size)
        return beam_limit

    def _copy_beam(self, beam):
        return list(beam)

    def _rebuild_visited_from_history(self, beam_history):
        visited = set()
        for beam in beam_history:
            for state in beam:
                visited.add(state["key"])
        return visited

    def _format_minutes_seconds(self, elapsed: float) -> str:
        elapsed_min = int(elapsed // 60)
        elapsed_sec = elapsed - 60 * elapsed_min
        return f"{elapsed_min}m {elapsed_sec:.1f}s"

    def _repulsion_values(self, graphs, *, fallback_index: int):
        memory_hashes = self._repulsion_memory_hashes(fallback_index)
        if not memory_hashes or self.repulsion_weight <= 0:
            return np.zeros(len(graphs), dtype=float), 0.0
        candidate_embeddings = self._graph_embeddings(graphs)
        memory_embeddings = self._graph_embeddings_from_hashes(memory_hashes)
        candidate_norm = self._normalize_rows(candidate_embeddings)
        memory_norm = self._normalize_rows(memory_embeddings)
        similarities = candidate_norm @ memory_norm.T
        repulsion = np.maximum(0.0, np.max(similarities, axis=1))
        if fallback_index < 0:
            lam = float(self.repulsion_weight)
        else:
            lam = float(
                self.repulsion_weight
                * (self.repulsion_growth_factor ** max(0, fallback_index))
            )
        return repulsion, lam

    def _repulsion_memory_hashes(self, fallback_index: int):
        memory_hashes = []
        if self.enforce_diversity and self.diversity_memory_hashes_:
            memory_hashes.extend(self.diversity_memory_hashes_)
        if (
            self.use_similarity_repulsion
            and fallback_index >= 0
            and self.failed_memory_hashes_
        ):
            memory_hashes.extend(self.failed_memory_hashes_)
        deduped_hashes = []
        seen_hashes = set()
        for graph_hash in memory_hashes:
            if graph_hash in seen_hashes:
                continue
            seen_hashes.add(graph_hash)
            deduped_hashes.append(graph_hash)
        return deduped_hashes

    def _remember_failed_graphs(self, graphs) -> None:
        if not self.use_similarity_repulsion or self.max_repulsion_memory <= 0:
            return
        graph_hashes = self._graph_hashes(graphs)
        self._graph_embeddings(graphs)
        for graph_hash in graph_hashes:
            if graph_hash in self.failed_memory_hash_set_:
                continue
            self.failed_memory_hashes_.append(graph_hash)
            self.failed_memory_hash_set_.add(graph_hash)
        while len(self.failed_memory_hashes_) > self.max_repulsion_memory:
            old_hash = self.failed_memory_hashes_.pop(0)
            self.failed_memory_hash_set_.discard(old_hash)

    def _graph_hashes(self, graphs):
        return [hash_graph(graph) for graph in graphs]

    def _unique_graphs(self, graphs):
        unique_graphs = []
        seen_hashes = set()
        for graph in graphs:
            graph_hash = hash_graph(graph)
            if graph_hash in seen_hashes:
                continue
            seen_hashes.add(graph_hash)
            unique_graphs.append(graph)
        return unique_graphs

    def _graph_embeddings(self, graphs):
        graph_hashes = self._graph_hashes(graphs)
        return self._graph_embeddings_from_hashes(graph_hashes, graphs=graphs)

    def _graph_embeddings_from_hashes(self, graph_hashes, *, graphs=None):
        graph_hashes = list(graph_hashes)
        missing_graphs = []
        missing_hashes = []
        if graphs is not None:
            for graph, graph_hash in zip(graphs, graph_hashes):
                if graph_hash in self.embedding_cache_:
                    continue
                missing_graphs.append(graph)
                missing_hashes.append(graph_hash)
        if missing_graphs:
            raw_features = self.graph_estimator._transform_raw(missing_graphs)
            rows = self._matrix_to_rows(raw_features)
            for graph_hash, row in zip(missing_hashes, rows):
                self.embedding_cache_[graph_hash] = row
        if not graph_hashes:
            return np.empty((0, 0), dtype=float)
        return np.vstack([self.embedding_cache_[graph_hash] for graph_hash in graph_hashes])

    def _initialize_diversity_memory(self, graphs):
        self.diversity_memory_hashes_ = []
        self.diversity_memory_hash_set_ = set()
        if not self.enforce_diversity:
            return
        diversity_hashes = self._graph_hashes(graphs)
        self._graph_embeddings(graphs)
        for graph_hash in diversity_hashes:
            if graph_hash in self.diversity_memory_hash_set_:
                continue
            self.diversity_memory_hashes_.append(graph_hash)
            self.diversity_memory_hash_set_.add(graph_hash)

    def _matrix_to_rows(self, features):
        if hasattr(features, "toarray"):
            features = features.toarray()
        rows = []
        for row in features:
            if hasattr(row, "toarray"):
                row = row.toarray()
            rows.append(np.asarray(row, dtype=float).ravel())
        return rows

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms > 0.0, norms, 1.0)
        return matrix / norms

    def _reconstruct_path(self, state):
        path = []
        current = state
        while current is not None:
            path.append(current["graph"])
            current = current["parent"]
        path.reverse()
        return path

    def _positive_scores(self, graphs):
        return self._class_probability(
            self.graph_estimator,
            graphs,
            target=1,
            estimator_name="graph_estimator",
        )

    def _target_scores(self, graphs, *, target):
        if target is None:
            return np.zeros(len(graphs), dtype=float)
        if self.target_estimator is None:
            raise ValueError(
                "generate(..., target=...) requires a fitted target_estimator"
            )
        if self.target_estimator_mode == "classification":
            return self._class_probability(
                self.target_estimator,
                graphs,
                target=target,
                estimator_name="target_estimator",
            )
        return self._regression_target_score(
            self.target_estimator,
            graphs,
            target=target,
            estimator_name="target_estimator",
        )

    def _class_probability(self, estimator, graphs, *, target, estimator_name: str):
        probs = estimator.predict_proba(graphs)
        classes = getattr(estimator.estimator_, "classes_", None)
        if classes is None:
            raise ValueError(f"{estimator_name} does not expose fitted classes_")
        classes = list(classes)
        if target not in classes:
            raise ValueError(
                f"{estimator_name} was not trained with requested class {target!r}; "
                f"available classes: {classes!r}"
            )
        class_idx = classes.index(target)
        return probs[:, class_idx]

    def _regression_target_score(self, estimator, graphs, *, target, estimator_name: str):
        if not hasattr(estimator, "predict"):
            raise ValueError(f"{estimator_name} does not expose predict()")
        predictions = np.asarray(estimator.predict(graphs), dtype=float).reshape(-1)
        target_value = float(target)
        return 1.0 / (1.0 + np.abs(predictions - target_value))

    def _missing_edges(self, graph: nx.Graph):
        nodes = list(graph.nodes())
        if nx.is_directed(graph):
            candidate_edges = list(permutations(nodes, 2))
        else:
            candidate_edges = list(combinations(nodes, 2))
        if self.allow_self_loops:
            candidate_edges += [(node, node) for node in nodes]
        occupied = {tuple(edge) for edge in graph.edges()}
        return [edge for edge in candidate_edges if edge not in occupied]

    def _canonical_graph_edges(self, graph: nx.Graph):
        return tuple(
            sorted(_canonicalize_edge(edge, graph) for edge in graph.edges())
        )

    def _edge_recency_map(self, state):
        return {
            edge: idx + 1
            for idx, edge in enumerate(state.get("edge_order", ()))
        }

    def _collect_edge_attribute_templates(self, graphs):
        templates = []
        seen = set()
        for graph in graphs:
            for _, _, edge_attrs in graph.edges(data=True):
                key = tuple(sorted(edge_attrs.items()))
                if key in seen:
                    continue
                seen.add(key)
                templates.append(dict(edge_attrs))
        return templates or [{}]

    def _as_graph_list(self, graphs):
        return [graphs] if self._is_single_graph_input(graphs) else list(graphs)

    def _coerce_optional_per_graph_argument(self, values, graphs, *, name: str):
        if values is None:
            return None
        if self._is_single_graph_input(graphs):
            return [values]
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be a scalar or a sequence matching graphs")
        try:
            value_list = list(values)
        except TypeError:
            return [values] * len(graphs)
        if len(value_list) != len(graphs):
            raise ValueError(f"{name} and graphs must have the same length")
        return value_list

    def _is_single_graph_input(self, graphs):
        return isinstance(graphs, nx.Graph)

    def _state_key(self, graph: nx.Graph):
        if nx.is_directed(graph):
            edge_key = tuple(
                sorted(
                    (u, v, tuple(sorted(graph.edges[u, v].items())))
                    for u, v in graph.edges()
                )
            )
        else:
            edge_key = tuple(
                sorted(
                    (
                        tuple(sorted((u, v))),
                        tuple(sorted(graph.edges[u, v].items())),
                    )
                    for u, v in graph.edges()
                )
            )
        return (tuple(sorted(graph.nodes())), graph.number_of_edges(), edge_key)

    def _draw_graphs(
        self,
        draw_graphs_fn: DrawGraphsFn | None,
        graphs: list[nx.Graph],
        *,
        n_graphs_per_line: int | None = None,
        titles: list[str] | None = None,
    ) -> None:
        if draw_graphs_fn is not None:
            try:
                kwargs = {}
                if n_graphs_per_line is not None:
                    kwargs["n_graphs_per_line"] = n_graphs_per_line
                if titles is not None:
                    kwargs["titles"] = titles
                draw_graphs_fn(graphs, **kwargs)
            except TypeError:
                draw_graphs_fn(graphs)
