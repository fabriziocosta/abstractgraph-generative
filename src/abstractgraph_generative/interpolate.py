"""
This module provides graph interpolation and generation utilities built upon the
AbstractGraph formalism, which enables meaningful, structure-preserving mutations
of graphs.

The fundamental operation is the "interpretation-node rewrite." An AbstractGraph is a
two-level graph representation where a high-level interpretation graph summarizes a
more detailed base graph. Each node in the interpretation graph corresponds to a
mapped subgraph in the base graph. A rewrite operation
involves identifying compatible mapped subgraphs between a source graph and a set of
"donor" graphs and swapping them. Compatibility is determined by a configurable
"cut signature" over the boundary between the mapped subgraph and the rest of the graph.
With `cut_radius=None`, the signature uses only the edge label. With
`cut_radius=0`, it includes the edge label and the two endpoint node labels
(internal first, external second). With `cut_radius=n>0`, it includes the edge
label and, for each endpoint, a hash of the induced inside/outside subgraph
within `n` hops (radius-limited). The `cut_scope` flag can restrict the
signature to only the inner endpoint ("inner"), only the outer endpoint
("outer"), or both ("both"). Labels are always hashed first for stability.

Interpolation is framed as a path-finding problem in a high-dimensional
embedding space. First, the source, destination, and all donor graphs are
transformed into numerical vectors (embeddings) via a `GraphTransformer`. This
places each graph in a space where geometric distance reflects structural
similarity. A neighborhood graph is then constructed on these embeddings, with
edges connecting graphs that are close to each other. This adjacency is built
by combining a Minimum Spanning Tree (MST), to ensure full connectivity, with a
mutual k-Nearest Neighbors (kNN) graph, to capture local neighborhood
structure. A shortest-path algorithm (Dijkstra's) finds the most efficient
route from the source to the destination embedding through this neighborhood graph.

The resulting path is a sequence of graphs that guides the interpolation. The
system "walks" this path, iteratively rewriting a candidate graph to move it
closer to the next node on the path. At each step, the current graph is mutated
using rewrites, with the local donors being the next graph on the path and its
k-nearest neighbors. This generates a batch of potential next-step graphs. The
best candidate is selected as the one whose embedding is closest to a local
reference (the mean embedding of the donor set), ensuring a smooth trajectory
through the embedding space.

Finally, the generation loop that builds on these mechanics lives in
`abstractgraph_generative.interpolation` via `InterpolationGenerator`.
"""

import random
from typing import Optional, Sequence, Tuple, Callable

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr



from abstractgraph.graphs import AbstractGraph, graph_to_abstract_graph, graphs_to_abstract_graphs
from abstractgraph.hashing import GraphHashDeduper
from abstractgraph.vectorize import AbstractGraphTransformer
from abstractgraph_generative.rewrite import iterated_rewrite, rewrite

## rewrite is imported from abstractgraph_generative.graph_rewrite


def _precompute_neighbors(distances: np.ndarray, k: int) -> list[list[int]]:
    """Return k nearest neighbor indices for each row in the distance matrix."""
    if distances.size == 0:
        return []
    k = min(k, max(0, distances.shape[0] - 1))
    if k == 0:
        return [[] for _ in range(distances.shape[0])]
    neighbors = []
    for i in range(distances.shape[0]):
        row = distances[i].copy()
        row[i] = np.inf
        neighbors.append([int(j) for j in np.argsort(row)[:k]])
    return neighbors


## iterated_rewrite is imported from abstractgraph_generative.graph_rewrite


class InterpolationEstimator:
    """Estimate interpolation paths between graphs using donor-based rewrites.

    Args:
        graph_transformer: Transformer used to embed graphs and build interpretation nodes.
        rng: Optional random generator for rewrite sampling.
        n_samples: Number of candidate rewrites per step.
        n_iterations: Number of rewrite iterations per step.
        k: Mutual kNN neighborhood size for path adjacency.
        degree_penalty: Edge-weight penalty factor or "auto".
        degree_penalty_mode: "multiplicative" (default) or "additive".
        feasibility_estimator: Optional estimator used to filter infeasible graphs.

    Returns:
        None.
    """
    def __init__(
        self,
        *,
        graph_transformer: AbstractGraphTransformer,
        rng: Optional[random.Random] = None,
        n_samples: int = 1,
        n_iterations: int = 1,
        k: int = 5,
        degree_penalty: float | str = "auto",
        degree_penalty_mode: str = "multiplicative",
        feasibility_estimator: Optional[object] = None,
        cut_radius: Optional[int] = None,
        cut_scope: str = "both",
        distance_metric: str = "euclidean",
        single_replacement: bool = True,
        max_enumerations: Optional[int] = None,
    ) -> None:
        """Initialize the interpolation estimator with transformer and rewrite options.

        Args:
            graph_transformer: Transformer used to embed graphs and build interpretation nodes.
            rng: Optional random generator for rewrite sampling.
            n_samples: Number of candidate rewrites per step.
            n_iterations: Number of rewrite iterations per step.
            k: Mutual kNN neighborhood size for path adjacency.
            degree_penalty: Edge-weight penalty factor or "auto".
            degree_penalty_mode: "multiplicative" (default) or "additive".
            feasibility_estimator: Optional estimator used to filter infeasible graphs.
            cut_radius: Controls cut signature detail. None uses only edge label;
                0 includes endpoint node labels; n>0 includes hashes of induced
                inside/outside subgraphs within n hops.
            cut_scope: Which endpoint neighborhoods to include in the cut
                signature; "both", "inner", or "outer".
            distance_metric: Distance metric for path building (passed to cdist).
            single_replacement: If True, use single donor pairing per step; if False,
                allow multiple pairings per step (possibly capped by max_enumerations).
            max_enumerations: When single_replacement=False, cap the number of pairings
                enumerated/sampled in each replacement.

        Returns:
            None.
        """
        self.rng = rng
        if graph_transformer is None:
            raise ValueError("graph_transformer is required.")
        self.graph_transformer = graph_transformer
        self.decomposition_function = graph_transformer.decomposition_function
        self.nbits = graph_transformer.nbits
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.k = k
        self.degree_penalty = degree_penalty
        self.degree_penalty_mode = degree_penalty_mode
        self.donors: list[nx.Graph] = []
        self.donor_vectors: Optional[np.ndarray] = None
        self.donor_dist: Optional[np.ndarray] = None
        self.donor_ags: Optional[list[AbstractGraph]] = None
        self.donor_adj: Optional[csr_matrix] = None
        self.donor_neighbor_indices: Optional[list[list[int]]] = None
        self.feasibility_estimator = feasibility_estimator
        self.cut_radius = cut_radius
        self.distance_metric = distance_metric
        self.cut_scope = cut_scope
        self.single_replacement = single_replacement
        self.max_enumerations = max_enumerations

    def fit(self, donors: Sequence[nx.Graph]) -> "InterpolationEstimator":
        """Store donor graphs and precompute embeddings and distances.

        Args:
            donors: Donor graphs used to construct interpolation paths.

        Returns:
            Self for chaining.
        """
        if self.decomposition_function is None:
            raise ValueError("graph_transformer.decomposition_function is required to build interpretation nodes.")
        self.donors = list(donors)
        if self.donors:
            self.donor_vectors = self.graph_transformer.transform(self.donors)
            self.donor_dist = cdist(self.donor_vectors, self.donor_vectors, metric=self.distance_metric)
            self.donor_ags = graphs_to_abstract_graphs(self.donors,decomposition_function=self.decomposition_function,nbits=self.nbits)
            self.donor_adj = _build_adjacency(
                self.donor_dist,
                k=self.k,
                degree_penalty=self.degree_penalty,
                degree_penalty_mode=self.degree_penalty_mode,
            )
            self.donor_neighbor_indices = _precompute_neighbors(self.donor_dist, self.k)
        else:
            self.donor_vectors = None
            self.donor_dist = None
            self.donor_ags = None
            self.donor_adj = None
            self.donor_neighbor_indices = None
        return self

    def _closest_graph(
        self,
        reference_graph: nx.Graph,
        candidates: Sequence[nx.Graph],
        *,
        reference_vector: Optional[np.ndarray] = None,
    ) -> Optional[nx.Graph]:
        """Return the candidate closest to the reference in embedding space.

        Args:
            reference_graph: Graph used to anchor the comparison.
            candidates: Candidate graphs to select from.
            reference_vector: Optional precomputed embedding for reference_graph.

        Returns:
            The closest candidate graph, or None if no candidates are provided.
        """
        if not candidates:
            return None
        if self.graph_transformer is None:
            raise ValueError("fit must be called before interpolate.")
        target_vec = reference_vector
        if target_vec is None:
            target_vec = self.graph_transformer.transform([reference_graph])[0]
        batch_vectors = self.graph_transformer.transform(candidates)
        distances = cdist(batch_vectors, target_vec[None, :], metric=self.distance_metric).ravel()
        return candidates[int(np.argmin(distances))]

    @staticmethod
    def _split_forward_backward_indices(path: Sequence[int]) -> tuple[list[int], list[int]]:
        """Split a path into forward and backward index sequences around the midpoint."""
        if len(path) < 2:
            return [], []
        step_count = len(path) - 1
        if step_count <= 1:
            return path[1:], []
        half_steps = step_count // 2
        forward_indices = list(path[1:1 + half_steps])
        backward_indices = list(path[-2:-(half_steps + 2):-1])
        return forward_indices, backward_indices

    def _walk_path_generic(
        self,
        start_graph: nx.Graph,
        indices: Sequence[int],
        graphs: Sequence[nx.Graph],
        vectors: np.ndarray,
        neighbor_indices_by_idx: Optional[Sequence[Sequence[int]]],
        get_ag: Callable[[int], AbstractGraph],
    ) -> list[nx.Graph]:
        """Rewrite along path indices, selecting closest step each time.

        Uses iterated_rewrite with local donor sets (node + neighbors) and picks
        the candidate closest to the mean donor embedding.
        """
        outputs: list[nx.Graph] = []
        current = start_graph
        for idx in indices:
            neighbor_indices = neighbor_indices_by_idx[idx] if neighbor_indices_by_idx else []
            donor_indices = [idx] + list(neighbor_indices)
            donors = [graphs[i] for i in donor_indices]
            donor_ags = [get_ag(i) for i in donor_indices]
            reference_vector = vectors[donor_indices].mean(axis=0)
            batch = iterated_rewrite(
                current,
                donors,
                rng=self.rng,
                decomposition_function=self.decomposition_function,
                nbits=self.nbits,
                n_samples=self.n_samples,
                n_iterations=self.n_iterations,
                feasibility_estimator=self.feasibility_estimator,
                donor_ags=donor_ags,
                cut_radius=self.cut_radius,
                cut_scope=self.cut_scope,
                single_replacement=self.single_replacement,
                max_enumerations=self.max_enumerations,
            )
            if not batch:
                continue
            closest = self._closest_graph(
                graphs[idx],
                batch,
                reference_vector=reference_vector,
            )
            if closest is None:
                continue
            current = closest
            outputs.append(current)
        return outputs

    def interpolate(
        self,
        source: nx.Graph,
        destination: nx.Graph,
    ) -> Sequence[nx.Graph]:
        """
        Interpolate from source to destination via donor graphs using rewiring steps.
        Rewrites are applied from both ends of the shortest-path sequence.

        Args:
            source: Starting graph for interpolation.
            destination: Target graph for interpolation.

        Returns:
            A sequence of intermediate graphs from source toward destination.
        """
        if self.graph_transformer is None:
            raise ValueError("fit must be called before interpolate.")
        if not self.donors:
            return []

        graphs = [source, destination] + list(self.donors)
        base_vectors = self.graph_transformer.transform([source, destination])
        if self.donor_vectors is None:
            donor_vectors = self.graph_transformer.transform(self.donors)
            donor_dist = cdist(donor_vectors, donor_vectors, metric=self.distance_metric)
        else:
            donor_vectors = self.donor_vectors
            donor_dist = self.donor_dist
        vectors = np.vstack([base_vectors, donor_vectors])
        base_dist = cdist(base_vectors, base_vectors, metric=self.distance_metric)
        cross_dist = cdist(base_vectors, donor_vectors, metric=self.distance_metric)
        dist = np.zeros((vectors.shape[0], vectors.shape[0]), dtype=float)
        dist[:2, :2] = base_dist
        dist[:2, 2:] = cross_dist
        dist[2:, :2] = cross_dist.T
        dist[2:, 2:] = donor_dist

        adj = _build_adjacency(
            dist,
            k=self.k,
            degree_penalty=self.degree_penalty,
            degree_penalty_mode=self.degree_penalty_mode,
        )

        path = _shortest_path(adj, source_idx=0, dest_idx=1)
        if len(path) < 2:
            return []

        forward_indices, backward_indices = self._split_forward_backward_indices(path)

        neighbor_indices_by_idx = _precompute_neighbors(dist, self.k)

        ag_cache: dict[int, AbstractGraph] = {}

        def _get_ag(index: int) -> AbstractGraph:
            if index in ag_cache:
                return ag_cache[index]
            if index >= 2 and self.donor_ags is not None:
                ag = self.donor_ags[index - 2]
            else:
                ag = graph_to_abstract_graph(
                    graphs[index],
                    decomposition_function=self.decomposition_function,
                    nbits=self.nbits,
                )
            ag_cache[index] = ag
            return ag

        forward_outputs = self._walk_path_generic(
            source,
            forward_indices,
            graphs,
            vectors,
            neighbor_indices_by_idx,
            _get_ag,
        )
        backward_outputs = self._walk_path_generic(
            destination,
            backward_indices,
            graphs,
            vectors,
            neighbor_indices_by_idx,
            _get_ag,
        )
        outputs = forward_outputs + list(reversed(backward_outputs))
        return outputs

    def interpolate_idxs(
        self,
        source_idx: int,
        destination_idx: int,
    ) -> Sequence[nx.Graph]:
        """Interpolate between donor graphs using donor indices."""
        if self.graph_transformer is None:
            raise ValueError("fit must be called before interpolate_idxs.")
        if not self.donors:
            return []
        if self.donor_vectors is None or self.donor_dist is None:
            raise ValueError("fit must be called before interpolate_idxs.")
        if source_idx == destination_idx:
            return []
        if source_idx < 0 or destination_idx < 0:
            raise ValueError("source_idx and destination_idx must be non-negative.")
        if source_idx >= len(self.donors) or destination_idx >= len(self.donors):
            raise ValueError("source_idx and destination_idx must index into donors.")

        graphs = self.donors
        vectors = self.donor_vectors
        dist = self.donor_dist
        adj = self.donor_adj
        if adj is None:
            adj = _build_adjacency(
                dist,
                k=self.k,
                degree_penalty=self.degree_penalty,
                degree_penalty_mode=self.degree_penalty_mode,
            )
        neighbor_indices_by_idx = self.donor_neighbor_indices
        if neighbor_indices_by_idx is None:
            neighbor_indices_by_idx = _precompute_neighbors(dist, self.k)

        path = _shortest_path(adj, source_idx=source_idx, dest_idx=destination_idx)
        if len(path) < 2:
            return []

        forward_indices, backward_indices = self._split_forward_backward_indices(path)

        ag_cache: dict[int, AbstractGraph] = {}

        def _get_ag(index: int) -> AbstractGraph:
            if index in ag_cache:
                return ag_cache[index]
            if self.donor_ags is None:
                ag = graph_to_abstract_graph(
                    graphs[index],
                    decomposition_function=self.decomposition_function,
                    nbits=self.nbits,
                )
            else:
                ag = self.donor_ags[index]
            ag_cache[index] = ag
            return ag
        forward_outputs = self._walk_path_generic(
            graphs[source_idx],
            forward_indices,
            graphs,
            vectors,
            neighbor_indices_by_idx,
            _get_ag,
        )
        backward_outputs = self._walk_path_generic(
            graphs[destination_idx],
            backward_indices,
            graphs,
            vectors,
            neighbor_indices_by_idx,
            _get_ag,
        )
        outputs = forward_outputs + list(reversed(backward_outputs))
        return outputs

    def iterated_interpolate_idxs(
        self,
        source_idx: int,
        destination_idx: int,
        *,
        n_iterations: int = 3,
        best_of: int = 1,
    ) -> Sequence[nx.Graph]:
        """Iteratively interpolate between donor indices and densify midpoints."""
        total_interpolations_head: list[nx.Graph] = []
        total_interpolations_tail: list[nx.Graph] = []

        def _best_of_interpolate_idxs(start_idx: int, end_idx: int) -> Sequence[nx.Graph]:
            num_tries = max(1, best_of)
            best_graphs: Sequence[nx.Graph] = []
            best_score = None
            for _ in range(num_tries):
                candidate = self.interpolate_idxs(start_idx, end_idx)
                if not candidate:
                    continue
                if self.graph_transformer is None:
                    raise ValueError("fit must be called before iterated_interpolate_idxs.")
                score = interpolation_score(candidate, self.graph_transformer)
                if not np.isfinite(score):
                    score = -np.inf
                if best_score is None or score > best_score:
                    best_score = score
                    best_graphs = candidate
            return best_graphs

        def _best_of_interpolate_graph(start: nx.Graph, end: nx.Graph) -> Sequence[nx.Graph]:
            return self._best_of_interpolate_graphs(start, end, best_of)

        # Initial path via donor indices
        initial_path = _best_of_interpolate_idxs(source_idx, destination_idx)

        # Densify and de-duplicate
        return self._densify_midpoints(
            initial_path,
            interpolate_pair=self.interpolate,
            n_iterations=n_iterations,
            best_of=best_of,
            dedup_bases=[self.donors[source_idx], self.donors[destination_idx]],
        )

    def iterated_interpolate(
        self,
        source_orig: nx.Graph,
        destination_orig: nx.Graph,
        *,
        n_iterations: int = 3,
        best_of: int = 1,
    ) -> Sequence[nx.Graph]:
        """
        Iteratively interpolate by splitting around midpoints and de-duplicating results.

        Starts with a full interpolation between source and destination, then repeatedly
        interpolates between the two midpoint graphs to densify the path. Returns a
        de-duplicated list of intermediate graphs; safeguards stop early when there
        are too few graphs to define a midpoint pair.

        Args:
            source_orig: Starting graph for interpolation.
            destination_orig: Target graph for interpolation.
            n_iterations: Number of midpoint refinement rounds.
            best_of: Number of attempts per interpolation to pick the best path.

        Returns:
            A de-duplicated list of interpolated graphs.
        """
        # Initial path via graphs with best-of selection
        initial_path = self._best_of_interpolate_graphs(source_orig, destination_orig, best_of)

        # Densify and de-duplicate
        return self._densify_midpoints(
            initial_path,
            interpolate_pair=self.interpolate,
            n_iterations=n_iterations,
            best_of=best_of,
            dedup_bases=[source_orig, destination_orig],
        )

    def _best_of_interpolate_graphs(
        self,
        start: nx.Graph,
        end: nx.Graph,
        best_of: int,
    ) -> Sequence[nx.Graph]:
        """Run an interpolation multiple times and return the best-scoring path."""
        num_tries = max(1, best_of)
        best_graphs: Sequence[nx.Graph] = []
        best_score = None
        for _ in range(num_tries):
            candidate = self.interpolate(start, end)
            if not candidate:
                continue
            if self.graph_transformer is None:
                raise ValueError("fit must be called before _best_of_interpolate_graphs.")
            score = interpolation_score(candidate, self.graph_transformer)
            if not np.isfinite(score):
                score = -np.inf
            if best_score is None or score > best_score:
                best_score = score
                best_graphs = candidate
        return best_graphs

    def _densify_midpoints(
        self,
        initial_path: Sequence[nx.Graph],
        *,
        interpolate_pair: Callable[[nx.Graph, nx.Graph], Sequence[nx.Graph]],
        n_iterations: int,
        best_of: int,
        dedup_bases: Sequence[nx.Graph],
    ) -> Sequence[nx.Graph]:
        """Densify a path by repeatedly interpolating midpoints and deduplicate."""
        total_interpolations_head: list[nx.Graph] = []
        total_interpolations_tail: list[nx.Graph] = []

        interpolated_graphs = list(initial_path)
        if not interpolated_graphs:
            return []
        if len(interpolated_graphs) < 2:
            total_interpolations = list(interpolated_graphs)
            total_interpolations_dedup = GraphHashDeduper().fit(list(dedup_bases)).filter(total_interpolations)
            total_interpolations_dedup = GraphHashDeduper().fit_filter(total_interpolations_dedup)
            return total_interpolations_dedup

        mid = len(interpolated_graphs) // 2
        total_interpolations_head += interpolated_graphs[: mid - 1]
        total_interpolations_tail += interpolated_graphs[mid:][::-1]

        for _ in range(n_iterations):
            if len(interpolated_graphs) < 2:
                break
            source = interpolated_graphs[mid - 1]
            destination = interpolated_graphs[mid]
            # best-of via wrapper
            candidate = self._best_of_interpolate_graphs(source, destination, best_of)
            interpolated_graphs = candidate
            if len(interpolated_graphs) < 2:
                break
            mid = len(interpolated_graphs) // 2
            total_interpolations_head += interpolated_graphs[: mid - 1]
            total_interpolations_tail += interpolated_graphs[mid:][::-1]

        total_interpolations = total_interpolations_head + total_interpolations_tail[::-1]
        total_interpolations_dedup = GraphHashDeduper().fit(list(dedup_bases)).filter(total_interpolations)
        total_interpolations_dedup = GraphHashDeduper().fit_filter(total_interpolations_dedup)
        return total_interpolations_dedup


def interpolation_score(
    interpolated_graphs: Sequence[nx.Graph],
    graph_transformer: AbstractGraphTransformer,
) -> float:
    """Return a smoothness score based on Spearman correlation of similarities.

    Args:
        interpolated_graphs: Sequence of graphs along the interpolation path.
        graph_transformer: Transformer used to embed graphs.

    Returns:
        A scalar smoothness score, or NaN if undefined.
    """
    def _is_constant(values: np.ndarray) -> bool:
        """Return True when all values are equal or the array is empty.

        Args:
            values: Array of values to test.

        Returns:
            True if all values are equal or array is empty, otherwise False.
        """
        if values.size == 0:
            return True
        return np.allclose(values, values[0])

    embeddings = graph_transformer.transform(interpolated_graphs)
    gram = embeddings @ embeddings.T
    n = gram.shape[0]
    if n < 2:
        return np.nan
    scores = []
    for i in range(n):
        sim_row = gram[i]
        # forward
        if i + 1 < n:
            forward_idx = np.arange(i + 1, n)
            forward_sims = sim_row[forward_idx]
            forward_dist = np.arange(1, n - i)
            if _is_constant(forward_sims):
                corr = np.nan
            else:
                corr = spearmanr(forward_dist, forward_sims).correlation
            if np.isfinite(corr):
                scores.append(-corr)
        # backward
        if i - 1 >= 0:
            backward_idx = np.arange(0, i)
            backward_sims = sim_row[backward_idx][::-1]
            backward_dist = np.arange(1, i + 1)
            if _is_constant(backward_sims):
                corr = np.nan
            else:
                corr = spearmanr(backward_dist, backward_sims).correlation
            if np.isfinite(corr):
                scores.append(-corr)
    if not scores:
        return np.nan
    return float(np.mean(scores))

def _auto_degree_penalty(adj: csr_matrix, mode: str) -> float:
    """Choose a penalty that modestly downweights hubs given current adj weights.

    Args:
        adj: Adjacency matrix with edge weights.
        mode: Penalty mode, "multiplicative" or "additive".

    Returns:
        Suggested penalty value.
    """
    if adj.nnz == 0:
        return 0.0
    csr = adj.tocsr()
    degrees = np.diff(csr.indptr)
    d75 = float(np.percentile(degrees, 75)) if degrees.size else 0.0
    if d75 <= 0:
        return 0.0
    if mode == "additive":
        # Scale penalty to about half the median edge weight at the 75th percentile degree.
        w50 = float(np.median(csr.data)) if csr.data.size else 0.0
        return 0.25 * w50 / d75
    # Multiplicative: target ~1.5x weight at the 75th percentile degree.
    return 0.25 / d75


def _apply_degree_penalty(adj: csr_matrix, penalty: float, mode: str) -> csr_matrix:
    """Return an adjacency with weights adjusted by node degrees.

    Args:
        adj: Adjacency matrix with edge weights.
        penalty: Penalty factor to apply.
        mode: Penalty mode, "multiplicative" or "additive".

    Returns:
        Adjusted adjacency matrix.
    """
    if penalty <= 0 or adj.nnz == 0:
        return adj
    csr = adj.tocsr()
    degrees = np.diff(csr.indptr)
    coo = csr.tocoo()
    if mode == "additive":
        # Additive penalties discourage hubs regardless of baseline edge length.
        coo.data = coo.data + penalty * (degrees[coo.row] + degrees[coo.col])
    else:
        # Multiplicative penalties preserve relative lengths while nudging away from hubs.
        scale = 1.0 + penalty * (degrees[coo.row] + degrees[coo.col])
        coo.data = coo.data * scale
    return coo.tocsr()


def _build_adjacency(
    dist: np.ndarray,
    k: int,
    degree_penalty: float | str = "auto",
    degree_penalty_mode: str = "multiplicative",
) -> csr_matrix:
    """Build an adjacency matrix via MST + mutual-kNN with zero-distance fallback.

    Args:
        dist: Pairwise distance matrix.
        k: Mutual kNN neighborhood size.
        degree_penalty: Penalty factor or "auto".
        degree_penalty_mode: Penalty mode, "multiplicative" or "additive".

    Returns:
        Adjacency matrix with optional degree penalty applied.
    """
    mst = minimum_spanning_tree(csr_matrix(dist))
    mst = mst + mst.transpose()
    knn = _knn_graph(dist, k=k)
    adj = _merge_graphs(mst, knn)
    if adj.nnz == 0 and dist.shape[0] > 1:
        # Avoid a disconnected adjacency when all pairwise distances are zero.
        dist = dist.copy()
        off_diag = ~np.eye(dist.shape[0], dtype=bool)
        dist[off_diag & (dist == 0)] = 1e-6
        mst = minimum_spanning_tree(csr_matrix(dist))
        mst = mst + mst.transpose()
        knn = _knn_graph(dist, k=k)
        adj = _merge_graphs(mst, knn)
    if degree_penalty == "auto":
        penalty_value = _auto_degree_penalty(adj, degree_penalty_mode)
    else:
        penalty_value = float(degree_penalty)
    return _apply_degree_penalty(adj, penalty_value, degree_penalty_mode)


def _knn_graph(dist: np.ndarray, k: int) -> csr_matrix:
    """Build a mutual kNN adjacency matrix from a distance matrix.

    Args:
        dist: Pairwise distance matrix.
        k: Number of neighbors per node.

    Returns:
        Sparse adjacency matrix for mutual kNN.
    """
    n = dist.shape[0]
    adj = np.zeros_like(dist, dtype=float)
    k = min(k, max(0, n - 1))
    if k == 0 or n == 0:
        return csr_matrix(adj)
    neighbor_sets = []
    for i in range(n):
        order = np.argsort(dist[i])
        neighbors = []
        for j in order:
            if j == i:
                continue
            neighbors.append(j)
            if len(neighbors) == k:
                break
        neighbor_sets.append(set(neighbors))
    for i in range(n):
        for j in neighbor_sets[i]:
            if i in neighbor_sets[j]:
                adj[i, j] = dist[i, j]
                adj[j, i] = dist[i, j]
    return csr_matrix(adj)


def _merge_graphs(mst: csr_matrix, knn: csr_matrix) -> csr_matrix:
    """Combine MST and kNN adjacency matrices with minimum edge weights.

    Args:
        mst: Minimum spanning tree adjacency.
        knn: Mutual kNN adjacency.

    Returns:
        Combined adjacency matrix.
    """
    mst_arr = mst.toarray()
    knn_arr = knn.toarray()
    adj = np.zeros_like(mst_arr)
    mask_mst = mst_arr != 0
    mask_knn = knn_arr != 0
    adj[mask_mst] = mst_arr[mask_mst]
    adj[mask_knn] = np.where(
        adj[mask_knn] == 0, knn_arr[mask_knn], np.minimum(adj[mask_knn], knn_arr[mask_knn])
    )
    return csr_matrix(adj)


def _shortest_path(adj: csr_matrix, source_idx: int, dest_idx: int) -> list[int]:
    """Return the shortest path indices between source and destination.

    Args:
        adj: Adjacency matrix with edge weights.
        source_idx: Source node index.
        dest_idx: Destination node index.

    Returns:
        List of node indices along the shortest path.
    """
    dist, preds = dijkstra(adj, directed=False, indices=source_idx, return_predecessors=True)
    if not np.isfinite(dist[dest_idx]):
        return []
    path = [dest_idx]
    cur = dest_idx
    while cur != source_idx:
        cur = preds[cur]
        if cur < 0:
            return []
        path.append(cur)
    return list(reversed(path))
