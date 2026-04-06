from __future__ import annotations

import random

import numpy as np
from sklearn.metrics import pairwise_distances


def select_graphs_via_shortest_paths(
    graphs,
    *,
    vectorizer,
    source_idx: int | None = None,
    dest_idx: int | None = None,
    n_paths: int = 3,
    random_state: int | None = None,
):
    """Select training graphs from edge-disjoint shortest paths in feature space.

    The utility vectorizes the full dataset once, builds a full pairwise distance
    matrix, extracts a shortest path between two endpoint graphs, removes the used
    path edges from the matrix, and repeats until ``n_paths`` have been found or
    no path remains.

    Args:
        graphs: Sequence of NetworkX graphs.
        vectorizer: Graph vectorizer exposing ``fit_transform`` or ``transform``.
        source_idx: Optional source graph index. If omitted, sampled uniformly.
        dest_idx: Optional destination graph index. If omitted, sampled uniformly.
        n_paths: Number of edge-disjoint shortest paths to extract.
        random_state: Optional RNG seed for endpoint sampling.

    Returns:
        Dictionary containing:
            ``selected_indices``: sorted unique graph indices across all paths.
            ``paths``: list of shortest paths as index lists.
            ``source_idx``: chosen source index.
            ``dest_idx``: chosen destination index.
            ``distance_matrix``: dense pairwise distance matrix used for routing.
    """
    graph_list = list(graphs)
    if len(graph_list) < 2:
        raise ValueError("graphs must contain at least two graphs")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    rng = random.Random(random_state)
    if source_idx is None or dest_idx is None:
        sampled = rng.sample(range(len(graph_list)), k=2)
        if source_idx is None:
            source_idx = sampled[0]
        if dest_idx is None:
            dest_idx = sampled[1]

    if source_idx == dest_idx:
        raise ValueError("source_idx and dest_idx must be different")

    features = _vectorize_graphs_once(vectorizer, graph_list)
    distance_matrix = pairwise_distances(features)
    np.fill_diagonal(distance_matrix, 0.0)

    working_matrix = distance_matrix.copy()
    paths = []
    for _ in range(n_paths):
        path = _dense_shortest_path(working_matrix, source_idx, dest_idx)
        if not path:
            break
        paths.append(path)
        _remove_path_edges(working_matrix, path)

    selected_indices = sorted({idx for path in paths for idx in path})
    return {
        "selected_indices": selected_indices,
        "paths": paths,
        "source_idx": source_idx,
        "dest_idx": dest_idx,
        "distance_matrix": distance_matrix,
    }


def _vectorize_graphs_once(vectorizer, graphs):
    if hasattr(vectorizer, "fit_transform"):
        features = vectorizer.fit_transform(graphs)
    else:
        features = vectorizer.transform(graphs)
    if hasattr(features, "toarray"):
        features = features.toarray()
    return np.asarray(features, dtype=float)


def _dense_shortest_path(distance_matrix: np.ndarray, source_idx: int, dest_idx: int):
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
            if weight <= 0.0 or not np.isfinite(weight):
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


def _remove_path_edges(distance_matrix: np.ndarray, path):
    for left, right in zip(path[:-1], path[1:]):
        distance_matrix[left, right] = 0.0
        distance_matrix[right, left] = 0.0
