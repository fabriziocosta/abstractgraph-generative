"""
Repair utilities that reuse graph_rewrite and transformer embeddings.

Summary
- Caches donor graphs and their embeddings.
- On repair(graph), finds k nearest donors by embedding similarity, rewrites
  the graph using those donors, and picks the candidate whose embedding is most
  similar on average to the k donors. Repeats for n_iterations and returns one
  final graph.

Notes
- Uses cosine similarity on `AbstractGraphTransformer` embeddings.
- Delegates the structural mutation to `abstractgraph_generative.rewrite.rewrite`.
- All `rewrite` parameters can be provided at initialization via **rewrite_kwargs.
"""

from __future__ import annotations

from typing import Optional, Sequence
import random
import numpy as np
import networkx as nx

from abstractgraph.graphs import AbstractGraph, graph_to_abstract_graph, graphs_to_abstract_graphs
from abstractgraph.vectorize import AbstractGraphTransformer
from abstractgraph_generative.rewrite import rewrite


class RepairGenerator:
    """Graph repair via donor-based rewrites and embedding selection.

    Args:
        decomposition_function: Function used to decompose graphs into
            interpretation/base form when constructing AbstractGraphs.
        nbits: Hash bit width used when building AbstractGraphs.
        graph_transformer: Transformer used to embed graphs for neighbor search
            and candidate selection. Typically `AbstractGraphTransformer`.
        rng: Optional random generator for rewrite sampling.
        **rewrite_kwargs: Any keyword parameters accepted by `graph_rewrite.rewrite`.

    Returns:
        None.
    """

    def __init__(
        self,
        *,
        decomposition_function,
        nbits: int,
        graph_transformer: AbstractGraphTransformer,
        rng: Optional[random.Random] = None,
        replace_with_smaller_or_equal_size: bool = True,
        **rewrite_kwargs,
    ) -> None:
        if decomposition_function is None:
            raise ValueError("decomposition_function is required.")
        if graph_transformer is None:
            raise ValueError("graph_transformer is required.")
        self.decomposition_function = decomposition_function
        self.nbits = int(nbits)
        self.graph_transformer = graph_transformer
        self.rng = rng or random.Random()
        self._rewrite_kwargs = dict(rewrite_kwargs)
        # Default constraint: donor mapped-subgraph size <= source mapped-subgraph size
        self._rewrite_kwargs.setdefault(
            "replace_with_smaller_or_equal_size", bool(replace_with_smaller_or_equal_size)
        )

        # Donor cache
        self._donor_graphs: list[nx.Graph] = []
        self._donor_vectors: Optional[np.ndarray] = None
        self._donor_unit_vectors: Optional[np.ndarray] = None
        self._donor_ags: Optional[list[AbstractGraph]] = None

    def fit(self, donors: Sequence[nx.Graph]) -> "RepairGenerator":
        """Cache donor graphs, embeddings, and AbstractGraphs.

        Args:
            donors: Sequence of donor graphs (typically pruned generator graphs).

        Returns:
            Self for chaining.
        """
        self._donor_graphs = list(donors)
        if not self._donor_graphs:
            self._donor_vectors = np.zeros((0, 0), dtype=float)
            self._donor_unit_vectors = np.zeros((0, 0), dtype=float)
            self._donor_ags = []
            return self

        # Embeddings and unit-normalized copies (for cosine similarity)
        vectors = self.graph_transformer.transform(self._donor_graphs)
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._donor_vectors = vectors
        self._donor_unit_vectors = vectors / norms

        # Precompute AbstractGraphs for donors to accelerate rewrite()
        self._donor_ags = graphs_to_abstract_graphs(
            self._donor_graphs,
            decomposition_function=self.decomposition_function,
            nbits=self.nbits,
        )
        return self

    def _top_k_donors(self, graph: nx.Graph, k: int) -> tuple[list[int], np.ndarray]:
        """Return indices of k nearest donors by cosine similarity.

        Args:
            graph: Query graph.
            k: Number of neighbors to return.

        Returns:
            Tuple of (indices, donor_unit_vectors_subset).
        """
        if self._donor_unit_vectors is None:
            raise ValueError("fit(donors) must be called before repair().")
        if len(self._donor_graphs) == 0 or k <= 0:
            return [], np.zeros((0, 0), dtype=float)
        q = self.graph_transformer.transform([graph])
        q = np.asarray(q, dtype=float).reshape(1, -1)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm[q_norm == 0.0] = 1.0
        q_unit = q / q_norm
        sims = self._donor_unit_vectors @ q_unit.ravel()
        k = max(1, min(int(k), len(self._donor_graphs)))
        # Deterministic top-k by similarity
        nn_idx = np.argpartition(-sims, k - 1)[:k]
        nn_idx = nn_idx[np.argsort(-sims[nn_idx])]
        return [int(i) for i in nn_idx], self._donor_unit_vectors[nn_idx]

    def _choose_best_candidate(
        self,
        candidates: Sequence[nx.Graph],
        donor_unit_vectors: np.ndarray,
    ) -> Optional[nx.Graph]:
        """Pick candidate with highest average cosine similarity to donors.

        Args:
            candidates: Candidate graphs to evaluate.
            donor_unit_vectors: Unit-normalized donor embeddings (k x d).

        Returns:
            Selected candidate graph, or None if candidates is empty.
        """
        if not candidates:
            return None
        if donor_unit_vectors.size == 0:
            # No neighbors; return first candidate deterministically
            return candidates[0]
        X = self.graph_transformer.transform(list(candidates))
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        X_unit = X / norms
        # Average cosine similarity to k neighbors
        donor_mean = donor_unit_vectors.mean(axis=0)
        sims = X_unit @ donor_mean
        best = int(np.argmax(sims))
        return candidates[best]

    def _repair_one(
        self,
        graph: nx.Graph,
        *,
        k_neighbors: int = 3,
        n_iterations: int = 1,
    ) -> nx.Graph:
        """Internal helper: repair a single graph."""
        if self._donor_graphs is None:
            raise ValueError("fit(donors) must be called before repair().")
        current = graph
        for _ in range(max(1, int(n_iterations))):
            nn_idx, nn_unit = self._top_k_donors(current, k_neighbors)
            donors_k = [self._donor_graphs[i] for i in nn_idx]
            donor_ags_k = [self._donor_ags[i] for i in nn_idx] if self._donor_ags is not None else None
            source_ag = graph_to_abstract_graph(
                current,
                decomposition_function=self.decomposition_function,
                nbits=self.nbits,
            )
            candidates = rewrite(
                current,
                donors_k,
                rng=self.rng,
                decomposition_function=self.decomposition_function,
                nbits=self.nbits,
                donor_ags=donor_ags_k,
                source_ag=source_ag,
                **self._rewrite_kwargs,
            )
            if not candidates:
                break
            best = self._choose_best_candidate(candidates, nn_unit)
            if best is None:
                break
            current = best
        return current

    def repair(
        self,
        graphs,
        *,
        k_neighbors: int = 3,
        n_iterations: int = 1,
    ):
        """Repair one graph or a list of graphs.

        Process per graph
        - Find k nearest donors by cosine similarity (using transformer embeddings).
        - Rewrite the graph using those donors via `graph_rewrite.rewrite`.
        - Select the candidate most similar on average to the k donors.
        - Repeat for `n_iterations`.

        Args:
            graphs: Either a single NetworkX graph or a list/sequence of graphs.
            k_neighbors: Number of nearest donors (k) to use per iteration.
            n_iterations: Number of rewrite iterations to perform.

        Returns:
            - If input is a single graph: a single repaired graph.
            - If input is a list/sequence: a list of repaired graphs (one per input).
        """
        # Single-graph path
        if isinstance(graphs, nx.Graph):
            return self._repair_one(graphs, k_neighbors=k_neighbors, n_iterations=n_iterations)

        # Sequence path
        seq = list(graphs) if graphs is not None else []
        return [
            self._repair_one(g, k_neighbors=k_neighbors, n_iterations=n_iterations)
            for g in seq
        ]
