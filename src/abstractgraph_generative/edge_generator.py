from __future__ import annotations

import math
import random
import time
from itertools import combinations, permutations
from typing import Callable

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from abstractgraph.hashing import hash_graph


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
        feasibility_estimator,
        graph_estimator,
        target_estimator=None,
        target_estimator_mode: str = "classification",
        *,
        n_negative_per_positive: int = 3,
        n_replicates: int = 1,
        beam_size: int = 10,
        max_restarts: int = 3,
        fallback_base_steps: int = 2,
        fallback_growth_factor: float = 2.0,
        beam_growth_factor: float = 1.5,
        max_beam_size: int | None = None,
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
        self.feasibility_estimator = feasibility_estimator
        self.graph_estimator = graph_estimator
        self.target_estimator = target_estimator
        if target_estimator_mode not in {"classification", "regression"}:
            raise ValueError(
                "target_estimator_mode must be 'classification' or 'regression'"
            )
        self.target_estimator_mode = target_estimator_mode
        self.n_negative_per_positive = n_negative_per_positive
        self.n_replicates = n_replicates
        self.beam_size = beam_size
        self.max_restarts = max_restarts
        self.fallback_base_steps = fallback_base_steps
        self.fallback_growth_factor = fallback_growth_factor
        self.beam_growth_factor = beam_growth_factor
        self.max_beam_size = max_beam_size
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
        self.failed_memory_hashes_ = []
        self.failed_memory_hash_set_ = set()

    def fit(self, graphs, targets=None):
        graph_list = self._as_graph_list(graphs)
        self.seed_graphs_ = [graph.copy() for graph in graph_list]
        target_list = self._coerce_optional_per_graph_argument(
            targets,
            self.seed_graphs_,
            name="targets",
        )

        dataset_seeds = [self.rng.randrange(10**9) for _ in self.seed_graphs_]
        dataset_parts = Parallel(n_jobs=self.fit_n_jobs, backend=self.fit_backend)(
            delayed(make_edge_regression_dataset)(
                graph,
                n_negative_per_positive=self.n_negative_per_positive,
                n_replicates=self.n_replicates,
                seed=dataset_seed,
                allow_self_loops=self.allow_self_loops,
            )
            for graph, dataset_seed in zip(self.seed_graphs_, dataset_seeds)
        )

        self.positives_ = []
        self.negatives_ = []
        self.dataset_ = []
        self.target_graphs_ = []
        self.target_values_ = []
        for graph_targets, (positives, negatives, dataset) in zip(
            target_list if target_list is not None else [None] * len(dataset_parts),
            dataset_parts,
        ):
            self.positives_.extend(positives)
            self.negatives_.extend(negatives)
            self.dataset_.extend(dataset)
            if graph_targets is not None:
                self.target_graphs_.extend(positives)
                self.target_values_.extend([graph_targets] * len(positives))

        fit_graphs = [graph for graph in self.positives_ if graph.number_of_edges() > 0]
        feasibility_fit_start = time.perf_counter()
        self.feasibility_estimator.fit(fit_graphs)
        feasibility_fit_time = time.perf_counter() - feasibility_fit_start
        if self.verbose:
            feasibility_fit_min = int(feasibility_fit_time // 60)
            feasibility_fit_sec = feasibility_fit_time - 60 * feasibility_fit_min
            print(
                f"[fit] feasibility_graphs={len(fit_graphs)} "
                f"positives={len(self.positives_)} negatives={len(self.negatives_)} "
                f"dataset={len(self.dataset_)} "
                f"time={feasibility_fit_min}m {feasibility_fit_sec:.1f}s"
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

        if target_list is not None:
            if self.target_estimator is None:
                raise ValueError(
                    "targets were provided to fit(), but target_estimator is None"
                )
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
                    f"[fit] target_estimator_graphs={len(self.target_graphs_)} "
                    f"{target_label}={len(set(self.target_values_))} "
                    f"mode={self.target_estimator_mode} "
                    f"time={target_fit_min}m {target_fit_sec:.1f}s"
                )
        else:
            self.target_graphs_ = None
            self.target_values_ = None

        self.edge_attribute_templates_ = self._collect_edge_attribute_templates(
            self.seed_graphs_
        )
        self.embedding_cache_ = {}
        self.failed_memory_hashes_ = []
        self.failed_memory_hash_set_ = set()
        return self

    def generate(
        self,
        graphs,
        n_edges,
        *,
        target=None,
        target_lambda: float = 1.0,
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
            return paths[0] if paths else []
        return paths

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
            print(
                f"[graph {graph_index}] start start_edges={start_graph.number_of_edges()} "
                f"target_edges={n_edges} remaining_edges={remaining_edges} "
                f"target={target} target_lambda={target_lambda:.3f}"
            )
            self._draw_graphs(draw_graphs_fn, [start_graph])

        if start_graph.number_of_edges() == n_edges:
            if verbose:
                print(
                    f"[graph {graph_index}] solved depth=0 max_depth=0 "
                    f"edges={start_graph.number_of_edges()} remaining_edges=0 "
                    f"tried=0 elapsed=0m 0.0s eta=0m 0.0s"
                )
            return [start_graph]

        beam = [self._make_state(start_graph, parent=None, score=1.0)]
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
            feasible_candidates = [
                cand
                for cand in generated
                if bool(self.feasibility_estimator.predict([cand["graph"]])[0])
            ]
            repulsion_lambda = 0.0
            if feasible_candidates:
                positive_scores = self._positive_scores(
                    [cand["graph"] for cand in feasible_candidates]
                )
                target_scores = self._target_scores(
                    [cand["graph"] for cand in feasible_candidates],
                    target=target,
                )
                for cand, score, target_score in zip(
                    feasible_candidates, positive_scores, target_scores
                ):
                    cand["score"] = float(score)
                    cand["target_score"] = float(target_score)
                repulsions, repulsion_lambda = self._repulsion_values(
                    [cand["graph"] for cand in feasible_candidates],
                    fallback_index=fallback_index,
                )
                for cand, repulsion in zip(feasible_candidates, repulsions):
                    cand["repulsion"] = float(repulsion)
                    cand["selection_score"] = float(
                        cand["score"]
                        + target_lambda * cand.get("target_score", 0.0)
                        - repulsion_lambda * cand["repulsion"]
                    )
                feasible_candidates.sort(
                    key=lambda cand: cand["selection_score"], reverse=True
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
                unseen_candidates.append(cand)

            retained = self._select_beam_candidates(unseen_candidates, beam_limit=beam_limit)

            if verbose:
                best_score = retained[0]["score"] if retained else None
                best_selection_score = (
                    retained[0]["selection_score"] if retained else None
                )
                best_target_score = (
                    retained[0].get("target_score") if retained else None
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
                print(
                    f"[graph {graph_index}] phase={fallback_index + 2}/{total_phases} "
                    f"depth={next_depth} beam={len(beam)} generated={len(generated)} "
                    f"feasible={len(feasible_candidates)} retained={len(retained)} "
                    f"tried={self.n_tried_} best_score={best_score_str} "
                    f"best_target_score={best_target_score_str} "
                    f"best_selection_score={best_selection_score_str} "
                    f"best_repulsion={best_repulsion:.3f} "
                    f"target_lambda={target_lambda:.3f} "
                    f"repulsion_lambda={repulsion_lambda:.3f} "
                    f"remaining_edges={remaining_edges} "
                    f"step_time={step_elapsed_str} eta={eta_str} "
                    f"beam_limit={beam_limit}"
                )
                if retained:
                    retained_graphs = [cand["graph"] for cand in retained]
                    retained_titles = [
                        f"sel={cand.get('selection_score', cand['score']):.3f} "
                        f"clf={cand['score']:.3f} "
                        f"tgt={cand.get('target_score', 0.0):.3f} "
                        f"rep={cand.get('repulsion', 0.0):.3f}"
                        for cand in retained
                    ]
                    self._draw_graphs(
                        draw_graphs_fn,
                        retained_graphs,
                        n_graphs_per_line=min(len(retained_graphs), 7),
                        titles=retained_titles,
                    )
                    self._draw_graphs(
                        draw_graphs_fn,
                        [retained[0]["graph"]],
                        n_graphs_per_line=1,
                        titles=[
                            f"selected sel={retained[0].get('selection_score', retained[0]['score']):.3f} "
                            f"clf={retained[0]['score']:.3f} "
                            f"tgt={retained[0].get('target_score', 0.0):.3f} "
                            f"rep={retained[0].get('repulsion', 0.0):.3f}"
                        ],
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
            fallback_depth = max(0, depth - rollback_steps)
            beam_limit = self._beam_limit_for_fallback(fallback_index)
            self.top_k_ = beam_limit
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
                candidates.append(self._make_state(candidate_graph, parent=state, score=None))
        self.rng.shuffle(candidates)
        return candidates

    def _make_state(self, graph, parent, score):
        state_key = self._state_key(graph)
        if parent is None:
            path_signature = (state_key,)
        else:
            path_signature = parent["path_signature"] + (state_key,)
        return {
            "graph": graph,
            "parent": parent,
            "score": score,
            "selection_score": score,
            "key": state_key,
            "path_signature": path_signature,
        }

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
        retained.sort(
            key=lambda cand: cand.get("selection_score", cand["score"]),
            reverse=True,
        )
        return retained

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
        if (
            not self.use_similarity_repulsion
            or fallback_index < 0
            or not self.failed_memory_hashes_
            or self.repulsion_weight <= 0
        ):
            return np.zeros(len(graphs), dtype=float), 0.0
        candidate_embeddings = self._graph_embeddings(graphs)
        memory_embeddings = np.vstack(
            [self.embedding_cache_[graph_hash] for graph_hash in self.failed_memory_hashes_]
        )
        candidate_norm = self._normalize_rows(candidate_embeddings)
        memory_norm = self._normalize_rows(memory_embeddings)
        similarities = candidate_norm @ memory_norm.T
        repulsion = np.maximum(0.0, np.max(similarities, axis=1))
        lam = float(
            self.repulsion_weight
            * (self.repulsion_growth_factor ** max(0, fallback_index))
        )
        return repulsion, lam

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

    def _graph_embeddings(self, graphs):
        graph_hashes = self._graph_hashes(graphs)
        missing_graphs = []
        missing_hashes = []
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
