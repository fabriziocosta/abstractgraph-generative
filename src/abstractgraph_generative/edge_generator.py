from __future__ import annotations

import random
import time
from itertools import combinations, permutations
from typing import Callable

import networkx as nx
import numpy as np
from joblib import Parallel, delayed


DrawGraphsFn = Callable[[list[nx.Graph]], object]


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


class EdgeGenerator:
    def __init__(
        self,
        feasibility_estimator,
        graph_estimator,
        *,
        n_negative_per_positive: int = 3,
        n_replicates: int = 1,
        beam_size: int = 10,
        max_restarts: int = 3,
        allow_self_loops: bool = False,
        fit_n_jobs: int = -1,
        fit_backend: str = "loky",
        verbose: bool = False,
        seed: int | None = None,
    ):
        self.feasibility_estimator = feasibility_estimator
        self.graph_estimator = graph_estimator
        self.n_negative_per_positive = n_negative_per_positive
        self.n_replicates = n_replicates
        self.beam_size = beam_size
        self.max_restarts = max_restarts
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
        self.edge_attribute_templates_ = None
        self.top_k_ = beam_size
        self.n_tried_ = 0
        self.max_depth_ = 0

    def fit(self, graphs):
        graph_list = self._as_graph_list(graphs)
        self.seed_graphs_ = [graph.copy() for graph in graph_list]

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
        for positives, negatives, dataset in dataset_parts:
            self.positives_.extend(positives)
            self.negatives_.extend(negatives)
            self.dataset_.extend(dataset)

        fit_graphs = [graph for graph in self.positives_ if graph.number_of_edges() > 0]
        feasibility_fit_start = time.perf_counter()
        self.feasibility_estimator.fit(fit_graphs)
        feasibility_fit_time = time.perf_counter() - feasibility_fit_start
        if self.verbose:
            print(
                f"[fit] feasibility_graphs={len(fit_graphs)} "
                f"positives={len(self.positives_)} negatives={len(self.negatives_)} "
                f"dataset={len(self.dataset_)} time={feasibility_fit_time:.3f}s"
            )

        self.targets_ = np.array([label for graph, label in self.dataset_], dtype=int)
        train_graphs = [graph for graph, label in self.dataset_]
        graph_estimator_fit_start = time.perf_counter()
        self.graph_estimator.fit(train_graphs, self.targets_)
        graph_estimator_fit_time = time.perf_counter() - graph_estimator_fit_start
        if self.verbose:
            n_positive = int(np.sum(self.targets_ == 1))
            n_negative = int(np.sum(self.targets_ == 0))
            print(
                f"[fit] graph_estimator_graphs={len(train_graphs)} "
                f"positive_labels={n_positive} negative_labels={n_negative} "
                f"time={graph_estimator_fit_time:.3f}s"
            )

        self.edge_attribute_templates_ = self._collect_edge_attribute_templates(
            self.seed_graphs_
        )
        return self

    def generate(
        self,
        graphs,
        n_edges,
        *,
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

        if len(graph_list) != len(edge_counts):
            raise ValueError("graphs and n_edges must have the same length")

        paths = [
            self._generate_one(
                graph,
                target_n_edges,
                draw_graphs_fn=draw_graphs_fn,
                verbose=verbose,
                graph_index=i,
            )
            for i, (graph, target_n_edges) in enumerate(zip(graph_list, edge_counts))
        ]
        return paths[0] if self._is_single_graph_input(graphs) else paths

    def _generate_one(
        self,
        graph: nx.Graph,
        n_edges: int,
        *,
        draw_graphs_fn: DrawGraphsFn | None = None,
        verbose: bool = False,
        graph_index: int = 0,
    ):
        start_graph = graph.copy()
        if start_graph.number_of_edges() > n_edges:
            raise ValueError("Input graph already has more edges than n_edges")

        self.n_tried_ = 0
        self.max_depth_ = 0
        n_attempts = max(1, self.max_restarts)
        start_time = time.perf_counter()
        if verbose:
            remaining_edges = n_edges - start_graph.number_of_edges()
            print(
                f"[graph {graph_index}] start start_edges={start_graph.number_of_edges()} "
                f"target_edges={n_edges} remaining_edges={remaining_edges}"
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

        for attempt in range(1, n_attempts + 1):
            beam = [self._make_state(start_graph, parent=None, score=1.0)]
            visited = {self._state_key(start_graph)}
            depth = 0
            step_start_time = time.perf_counter()

            if verbose and n_attempts > 1:
                print(f"[graph {graph_index}] attempt={attempt}/{n_attempts}")

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
                if feasible_candidates:
                    positive_scores = self._positive_scores(
                        [cand["graph"] for cand in feasible_candidates]
                    )
                    for cand, score in zip(feasible_candidates, positive_scores):
                        cand["score"] = float(score)
                    feasible_candidates.sort(
                        key=lambda cand: cand["score"], reverse=True
                    )

                unseen_candidates = []
                for cand in feasible_candidates:
                    state_key = self._state_key(cand["graph"])
                    if state_key in visited:
                        continue
                    unseen_candidates.append(cand)

                retained = self._select_beam_candidates(unseen_candidates)
                for cand in retained:
                    visited.add(self._state_key(cand["graph"]))

                depth += 1
                self.max_depth_ = max(self.max_depth_, depth)

                if verbose:
                    best_score = retained[0]["score"] if retained else None
                    step_elapsed = time.perf_counter() - step_start_time
                    step_elapsed_min = int(step_elapsed // 60)
                    step_elapsed_sec = step_elapsed - 60 * step_elapsed_min
                    best_score_str = (
                        f"{best_score:.4f}" if best_score is not None else "None"
                    )
                    current_edges = (
                        retained[0]["graph"].number_of_edges()
                        if retained
                        else start_graph.number_of_edges() + depth
                    )
                    remaining_edges = max(0, n_edges - current_edges)
                    eta = remaining_edges * step_elapsed
                    eta_min = int(eta // 60)
                    eta_sec = eta - 60 * eta_min
                    print(
                        f"[graph {graph_index}] attempt={attempt}/{n_attempts} "
                        f"depth={depth} beam={len(beam)} generated={len(generated)} "
                        f"feasible={len(feasible_candidates)} retained={len(retained)} "
                        f"tried={self.n_tried_} best_score={best_score_str} "
                        f"remaining_edges={remaining_edges} "
                        f"step_time={step_elapsed_min}m {step_elapsed_sec:.1f}s "
                        f"eta={eta_min}m {eta_sec:.1f}s"
                    )
                    if retained:
                        self._draw_graphs(draw_graphs_fn, [retained[0]["graph"]])
                    step_start_time = time.perf_counter()

                for state in retained:
                    if state["graph"].number_of_edges() == n_edges:
                        path = self._reconstruct_path(state)
                        if verbose:
                            elapsed = time.perf_counter() - start_time
                            elapsed_min = int(elapsed // 60)
                            elapsed_sec = elapsed - 60 * elapsed_min
                            print(
                                f"[graph {graph_index}] solved attempt={attempt}/{n_attempts} "
                                f"depth={depth} max_depth={self.max_depth_} "
                                f'edges={state["graph"].number_of_edges()} remaining_edges=0 '
                                f"tried={self.n_tried_} elapsed={elapsed_min}m "
                                f"{elapsed_sec:.1f}s eta=0m 0.0s"
                            )
                        return path

                beam = retained

            if verbose and attempt < n_attempts:
                print(f"[graph {graph_index}] restart attempt={attempt + 1}/{n_attempts}")

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
        return {"graph": graph, "parent": parent, "score": score}

    def _select_beam_candidates(self, candidates):
        if not candidates:
            return []

        n_keep = min(self.top_k_, len(candidates))
        n_top = 1 if n_keep == 1 else max(1, n_keep // 2)
        n_random = n_keep - n_top

        top_candidates = candidates[:n_top]
        remaining_candidates = candidates[n_top:]
        if len(remaining_candidates) <= n_random:
            random_candidates = remaining_candidates
        else:
            random_candidates = self.rng.sample(remaining_candidates, k=n_random)

        retained = top_candidates + random_candidates
        retained.sort(key=lambda cand: cand["score"], reverse=True)
        return retained

    def _reconstruct_path(self, state):
        path = []
        current = state
        while current is not None:
            path.append(current["graph"])
            current = current["parent"]
        path.reverse()
        return path

    def _positive_scores(self, graphs):
        probs = self.graph_estimator.predict_proba(graphs)
        classes = getattr(self.graph_estimator.estimator_, "classes_", None)
        if classes is None:
            raise ValueError("graph_estimator does not expose fitted classes_")
        classes = list(classes)
        if 1 not in classes:
            raise ValueError("graph_estimator must be trained with positive label 1")
        pos_idx = classes.index(1)
        return probs[:, pos_idx]

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
        self, draw_graphs_fn: DrawGraphsFn | None, graphs: list[nx.Graph]
    ) -> None:
        if draw_graphs_fn is not None:
            draw_graphs_fn(graphs)
