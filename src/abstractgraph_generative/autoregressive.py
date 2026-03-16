"""
Autoregressive graph generation utilities.

Components
- Interpretation-node pruning helper: `generate_pruning_sequences` removes
  entire interpretation-node mapped subgraphs from an AbstractGraph decomposition.
- Generator: `AutoregressiveGraphGenerator` grows graphs by repeatedly proposing
  virtual-cut insertions derived from the pruning index and selecting a next
  state based on similarity to the original training graphs (rather than donors),
  reducing pruning bias.

Workflow
1) fit(): Build the donor pool by pruning down to `min_nodes_for_pruning` via interpretation-node
   mapped-subgraph removals, fit the feasibility
   estimator on donors, then embed the original training graphs
   (`self.generator_graphs`) with an `AbstractGraphTransformer` and cache
   unit‑normalized features for similarity (skipped if
   `use_similarity_selection=False`).
2) generate_single(): From a seed subgraph, propose candidates via
   virtual-cut insertions, filter for feasibility and strict growth, and select
   the next graph with probability proportional to the average cosine similarity
   over the top-k nearest training-graph embeddings (k-NN). Repeat until
   reaching a sampled target size.

Notes
- Assumes undirected simple graphs for pruning utilities.
- Uses absolute imports (package name `AbstractGraph`) per project conventions.

Generation modes overview
    The generator supports two selection modes. Context-only mode (set
    `use_context_embedding=True` and `use_similarity_selection=False`) proposes
    a batch of candidates from the cut index, ranks them by outer-context
    similarity using the context vectorizer, and selects the next graph by
    `context_temperature`/`context_top_k`. Context+similarity mode (set both
    `use_context_embedding=True` and `use_similarity_selection=True`) first
    builds candidates using context-aware cut matching, then embeds candidates
    with the similarity vectorizer and selects the next graph by averaging
    top-k cosine similarities to training graphs, with temperature-controlled
    sampling.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence
from collections import defaultdict
import random
import weakref
import warnings
import numpy as np
import networkx as nx
from abstractgraph_generative.rewrite import (
    virtual_cut_entry,
    _precompute_outer_node_keys,
    cut_size_distribution,
    virtual_rewrite_candidates_at_cut_nodes,
    deduplicate_cut_index,
)
from abstractgraph.graphs import graph_to_abstract_graph
from abstractgraph.hashing import GraphHashDeduper, hash_graph
from abstractgraph.vectorize import AbstractGraphTransformer
from joblib import Parallel, delayed


def _warn_deprecated_name(old_name: str, new_name: str) -> None:
    warnings.warn(
        f"`{old_name}` is deprecated and will be removed in a future release; use `{new_name}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _resolve_alias(canonical, deprecated, deprecated_name: str, canonical_name: str):
    if deprecated is None:
        return canonical
    _warn_deprecated_name(deprecated_name, canonical_name)
    if canonical is not None:
        raise ValueError(f"Pass only one of `{canonical_name}` or `{deprecated_name}`.")
    return deprecated


def _top_k_indices(scores: Sequence[float], k: int) -> list[int]:
    """Return indices of the top-k scores in deterministic order.

    Args:
        scores: Sequence of score values.
        k: Number of indices to return.

    Returns:
        List of indices sorted by descending score with index tie-breaks.
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    arr = np.asarray(scores, dtype=float)
    arr = np.nan_to_num(arr, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    order = sorted(range(n), key=lambda i: (-arr[i], i))
    return order[: min(int(k), n)]


def _temperature_weights(scores: Sequence[float], temperature: float) -> np.ndarray:
    """Return selection weights blended with uniform by temperature.

    Args:
        scores: Sequence of score values.
        temperature: Selection temperature in [0, 1].

    Returns:
        Numpy array of non-negative weights.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    if temperature >= 1.0:
        return np.ones_like(arr, dtype=float)
    weights = np.maximum(arr, 0.0)
    if not np.isfinite(weights).any() or weights.sum() <= 0.0:
        weights = np.ones_like(arr, dtype=float)
    if temperature <= 0.0:
        return weights
    return (1.0 - temperature) * weights + temperature * np.ones_like(weights, dtype=float)


def generate_pruning_sequences(
    graph: nx.Graph,
    min_nodes_for_pruning: int,
    *,
    decomposition_function,
    nbits: int,
    cut_radius: Optional[int] = None,
    cut_context_radius: Optional[int] = None,
    context_vectorizer=None,
    use_context_embedding: bool = True,
    association_aware: bool = False,
    fixed_interpretation_graph: Optional[nx.Graph] = None,
    fixed_image_graph: Optional[nx.Graph] = None,
    return_interpretation_steps: bool = False,
    return_image_steps: Optional[bool] = None,
    cut_index: Optional[dict] = None,
    return_cut_index: bool = False,
    seed: Optional[int] = None,
    include_start: bool = False,
) -> list[nx.Graph] | tuple[list[nx.Graph], dict]:
    """
    Produce a sequence by removing interpretation-node mapped subgraphs.

    Summary
        Starting from a copy of `graph`, repeatedly decompose into an
        AbstractGraph and remove one entire mapped subgraph (the base
        subgraph tied to an interpretation node). A removal is accepted only if the
        resulting graph does not drop below `min_nodes_for_pruning`.
        The decomposition is recomputed after each accepted removal.

    Args:
        graph: Input NetworkX graph. Treated as undirected for connectivity checks.
        min_nodes_for_pruning: Minimum number of nodes to keep in the last graph of the sequence.
        decomposition_function: AbstractGraph decomposition function.
        nbits: Hash bit width for the default label function used in decomposition.
        cut_radius: Cut radius used when recording cut signatures.
        cut_context_radius: Neighborhood radius for context embeddings on cuts.
        context_vectorizer: Optional transformer with a .transform(graphs)
            method returning vector embeddings per graph.
        use_context_embedding: If True and context_vectorizer is provided,
            compute and store context embeddings in the cut index.
        association_aware: If True, remove interpretation-node mapped subgraphs while only
            deleting base-graph nodes after their last mapped subgraph is removed.
            This uses a fixed interpretation graph derived from the initial graph (or
            the provided `fixed_interpretation_graph`) and does not recompute
            mapped subgraphs after each step.
        fixed_interpretation_graph: Optional fixed interpretation graph to use
            when association_aware=True. If None, it is built from the initial
            graph using the decomposition_function and nbits.
        fixed_image_graph: Deprecated alias for `fixed_interpretation_graph`.
        return_interpretation_steps: If True and association_aware, also
            return a list of interpretation graphs per pruning step reflecting
            removed mapped subgraphs.
        return_image_steps: Deprecated alias for `return_interpretation_steps`.
        cut_index: Optional dictionary to append cut signatures to.
        return_cut_index: If True, return a tuple (sequence, cut_index).
        seed: Optional random seed for reproducibility.
        include_start: If True, include a copy of the starting graph as the first
            element of the returned sequence.

    Returns:
        List of NetworkX graphs, strictly non-increasing in size; each successive
        element is the result of removing an entire mapped subgraph. If
        return_cut_index=True, returns a tuple (sequence, cut_index). If the
        input already has `<= min_nodes_for_pruning` nodes, the sequence contains the starting
        graph only (if include_start=True), else is empty.

    Notes:
        - Assumes `graph` is an undirected simple NetworkX Graph.
        - Interpretation-node mapped subgraphs are recomputed after each
          accepted removal.
        - Node and edge attributes are preserved for retained parts.
        - Recorded cut signatures use outer subgraph hashes with edge labels omitted.
    """
    fixed_interpretation_graph = _resolve_alias(
        fixed_interpretation_graph,
        fixed_image_graph,
        "fixed_image_graph",
        "fixed_interpretation_graph",
    )
    return_interpretation_steps = _resolve_alias(
        return_interpretation_steps,
        return_image_steps,
        "return_image_steps",
        "return_interpretation_steps",
    )
    if decomposition_function is None:
        raise ValueError("decomposition_function is required for image-node pruning.")
    g = graph.copy()
    out: list[nx.Graph] = []
    interpretation_steps: list[nx.Graph] = []
    if include_start:
        out.append(g.copy())

    rng = random.Random(seed)
    local_cut_index = cut_index
    if local_cut_index is None and return_cut_index:
        local_cut_index = defaultdict(list)

    if not association_aware:
        if min_nodes_for_pruning <= 0 or g.number_of_nodes() <= min_nodes_for_pruning:
            return (out, local_cut_index) if return_cut_index else out

    if not association_aware:
        while g.number_of_nodes() > min_nodes_for_pruning:
            ag = graph_to_abstract_graph(
                g,
                decomposition_function=decomposition_function,
                nbits=nbits,
            )
            candidates = []
            for _node_id, data in ag.interpretation_graph.nodes(data=True):
                mapped_subgraph = data.get("mapped_subgraph", data.get("association"))
                if mapped_subgraph is None:
                    continue
                inner_nodes = set(mapped_subgraph.nodes())
                if not inner_nodes:
                    continue
                if len(inner_nodes) >= g.number_of_nodes():
                    continue
                candidates.append((inner_nodes, mapped_subgraph))

            rng.shuffle(candidates)
            removed = False
            for inner_nodes, mapped_subgraph in candidates:
                if g.number_of_nodes() - len(inner_nodes) < min_nodes_for_pruning:
                    continue
                g2 = g.copy()
                g2.remove_nodes_from(inner_nodes)
                if g2.number_of_nodes() == 0:
                    continue
                if local_cut_index is not None:
                    entry = virtual_cut_entry(
                        ag.base_graph,
                        inner_nodes,
                        cut_radius=cut_radius,
                        cut_context_radius=cut_context_radius,
                        context_vectorizer=context_vectorizer,
                        use_context_embedding=use_context_embedding,
                    )
                    if entry is not None:
                        cut_key, donor_edge_map, donor_cut, outer_ctx, inner_ctx = entry
                        assoc_hash = hash_graph(mapped_subgraph)
                        if use_context_embedding and context_vectorizer is not None:
                            local_cut_index.setdefault(cut_key, []).append(
                                (mapped_subgraph.copy(), donor_cut, donor_edge_map, assoc_hash, outer_ctx, inner_ctx)
                            )
                        else:
                            local_cut_index.setdefault(cut_key, []).append(
                                (mapped_subgraph.copy(), donor_cut, donor_edge_map, assoc_hash)
                            )
                g = g2
                out.append(g.copy())
                removed = True
                break
            if not removed:
                break
    else:
        # Association-aware pruning: only delete base-graph nodes after their last
        # mapped subgraph is removed; keep the interpretation graph fixed.
        if fixed_interpretation_graph is None:
            ag0 = graph_to_abstract_graph(
                g,
                decomposition_function=decomposition_function,
                nbits=nbits,
            )
            fixed_interpretation_graph = ag0.interpretation_graph.copy()
        assoc_nodes_by_image = {}
        assoc_count = defaultdict(int)
        for img_node, data in fixed_interpretation_graph.nodes(data=True):
            mapped_subgraph = data.get("mapped_subgraph", data.get("association"))
            nodes = set(mapped_subgraph.nodes()) if isinstance(mapped_subgraph, nx.Graph) else set()
            assoc_nodes_by_image[img_node] = nodes
            for node in nodes:
                assoc_count[node] += 1
        removed_images = set()
        if include_start and return_interpretation_steps:
            interpretation_steps.append(fixed_interpretation_graph.copy())
        while g.number_of_nodes() > 0:
            candidates = [
                img_node
                for img_node, nodes in assoc_nodes_by_image.items()
                if img_node not in removed_images and nodes
            ]
            if not candidates:
                break
            rng.shuffle(candidates)
            img_node = candidates[0]
            removed_images.add(img_node)
            inner_nodes = set(assoc_nodes_by_image.get(img_node, set()))
            inner_nodes &= set(g.nodes())
            if not inner_nodes:
                continue
            for node in inner_nodes:
                assoc_count[node] = max(0, assoc_count.get(node, 0) - 1)
            remaining_nodes = {n for n, c in assoc_count.items() if c > 0}
            # Auto-remove image nodes whose associations are now empty.
            for inode, nodes in assoc_nodes_by_image.items():
                if inode in removed_images:
                    continue
                if not (nodes & remaining_nodes):
                    removed_images.add(inode)
            g2 = g.subgraph(remaining_nodes).copy()
            if return_interpretation_steps:
                img_step = fixed_interpretation_graph.copy()
                for inode, data in img_step.nodes(data=True):
                    mapped_subgraph = data.get("mapped_subgraph", data.get("association"))
                    if not isinstance(mapped_subgraph, nx.Graph):
                        continue
                    if inode in removed_images:
                        data["mapped_subgraph"] = nx.Graph()
                        data["association"] = nx.Graph()
                    else:
                        next_subgraph = mapped_subgraph.subgraph(remaining_nodes).copy()
                        data["mapped_subgraph"] = next_subgraph
                        data["association"] = next_subgraph
                img_step.graph["removed_images"] = set(removed_images)
                interpretation_steps.append(img_step)
            if local_cut_index is not None:
                entry = virtual_cut_entry(
                    g,
                    inner_nodes,
                    cut_radius=cut_radius,
                    cut_context_radius=cut_context_radius,
                    context_vectorizer=context_vectorizer,
                    use_context_embedding=use_context_embedding,
                )
                if entry is not None:
                    cut_key, donor_edge_map, donor_cut, outer_ctx, inner_ctx = entry
                    mapped_subgraph = fixed_interpretation_graph.nodes[img_node].get("mapped_subgraph")
                    if mapped_subgraph is None:
                        mapped_subgraph = fixed_interpretation_graph.nodes[img_node].get("association")
                    mapped_subgraph = (
                        mapped_subgraph.subgraph(inner_nodes).copy() if isinstance(mapped_subgraph, nx.Graph) else nx.Graph()
                    )
                    assoc_hash = hash_graph(mapped_subgraph)
                    if use_context_embedding and context_vectorizer is not None:
                        local_cut_index.setdefault(cut_key, []).append(
                            (mapped_subgraph.copy(), donor_cut, donor_edge_map, assoc_hash, outer_ctx, inner_ctx)
                        )
                    else:
                        local_cut_index.setdefault(cut_key, []).append(
                            (mapped_subgraph.copy(), donor_cut, donor_edge_map, assoc_hash)
                        )
            g = g2
            out.append(g.copy())

    if return_interpretation_steps and association_aware:
        if return_cut_index:
            return out, interpretation_steps, local_cut_index
        return out, interpretation_steps
    if return_cut_index:
        return out, local_cut_index
    return out


class AutoregressiveGraphGenerator(object):
    """Autoregressive graph generator.

    Summary
        Uses a donor pool constructed from prunings of input graphs by removing
        image-node associations. Starting from a small seed subgraph (with
        `min_nodes_for_pruning`), it grows a graph by repeatedly proposing virtual-cut
        insertions from the pruning-derived cut index and selecting among
        feasible candidates. Optional dataset storage supports sampling a seed
        graph and its nearest neighbors as generator inputs.

        At each step, the generator samples a cut size and cut nodes in the
        current graph, looks up matching donor cuts, and rewrites the graph by
        inserting a donor mapped subgraph. Feasibility filtering and optional
        context/similarity scoring select the next graph before the loop
        repeats until the target size or restart/backtrack limits are reached.

    Args:
        feasibility_estimator: Object with `fit(graphs)` and `filter(graphs)`
            methods used to retain only feasible candidate graphs at each step.
        nbits: Hash bit width used by AbstractGraph operators when building
            image-node associations.
        decomposition_function: AbstractGraph decomposition function used to
            build image-node associations for pruning and virtual-cut indexing.
        cut_radius: Boundary signature radius used for virtual-cut signatures
            when building the pruning index (None/0/positive).
        cut_context_radius: Neighborhood radius for context embeddings attached
            to virtual cuts. None uses the full outer subgraph.
        context_vectorizer: Optional transformer with a .transform(graphs)
            method returning vector embeddings per graph for cut contexts.
        use_context_embedding: If True and context_vectorizer is provided,
            compute and store context embeddings in the cut index.
        min_nodes_for_pruning: Minimum node count for seed subgraphs and the lower bound
            used when constructing the donor pool via pruning.
        max_restarts: Maximum number of restarts if generation fails to reach
            the target size. Also used as the per-step retry budget for
            re-proposing virtual cuts before backtracking.
        n_virtual_candidates: Number of virtual candidates per step (proposal count).
        max_virtual_cut_evaluations: Maximum number of cut-node sets sampled
            per proposal step. Defaults to 200.
        verbose: If True, prints a simple progress ratio per iteration.
        similarity_vectorizer: Optional `AbstractGraphTransformer` used to embed graphs
            for similarity-based selection. If None, a default vectorizer is
            built during `fit` using `nbits` and `decomposition_function`.
        similarity_k_neighbors: Number of training-graph neighbors (k) used to score
            candidates by averaging cosine similarities over their k nearest
            training-graph embeddings. Defaults to 16 if None.
        similarity_temperature: Selection temperature for similarity scoring.
            T=1 selects uniformly at random; T=0 deterministically selects top-k.
        similarity_top_k: Number of top-scoring candidates to select when
            similarity_temperature=0. Defaults to 1.
        context_temperature: Selection temperature for context-aware virtual cuts.
            T=1 selects uniformly at random; T=0 deterministically selects top-k.
        context_top_k: Number of top-scoring virtual-cut candidates to select
            when context_temperature=0. Defaults to n_samples.
        context_dedup_epsilon: Minimum L2 distance between context embeddings
            to keep multiple context variants for a single donor entry.
        initial_virtual_cut_evaluations: Starting budget of cut-node set evaluations
            per propose step. Used only when `adaptive_virtual_cut_evaluations=True`.
        adaptive_virtual_cut_evaluations: If True, start with
            `initial_virtual_cut_evaluations` and double the evaluation budget on
            failure until reaching `max_virtual_cut_evaluations`.
        preserve_node_ids: If True, keep node ids stable during rewrites.
        use_similarity_selection: If True, prefer candidates by similarity to
            training graphs; if False, use the edge-count heuristic and skip
            similarity embeddings.
        max_cut_size: Maximum virtual cut size to sample when proposing
            candidates. If None, no cap is applied.
        min_cut_size: If set, ignore sizes < this value when sampling. Falls
            back to available sizes if none meet the bound.
        size_weights: Optional per-size weights used when sampling sizes; if
            provided, sizes are drawn with probability proportional to
            `size_weights[s]` (after filtering by bounds/caps).
        multicut_sizes: Preferred sizes considered “multi-attachment” (default (2, 3)).
        prob_force_multicut: With this probability per proposal call, restrict
            size sampling to `multicut_sizes` (fallback to all sizes if none available).
        force_multicut_every: Deterministic schedule; when set to k>0, every kth
            `propose_virtual_candidates` call restricts sizes to `multicut_sizes`.
        display_function: Optional callable used to visualize intermediate
            graphs during generation when `verbose=True`.
        display_live: If True, show per-step intermediate graphs during
            generation and force sequential execution. If False, display
            histories after each instance completes.
        n_jobs: Parallelism for donor construction and batch generation. Follows
            joblib semantics (e.g., -1 to use all CPUs; 1 for sequential).
        shrink_target_on_backtrack: If True, each time the walk backtracks
            (i.e., no feasible candidates from current), decrement the current
            target size by `backtrack_shrink_step`. This guarantees eventual
            termination even when backtracking repeatedly.
        backtrack_shrink_step: Amount to reduce the target size on each
            backtrack when `shrink_target_on_backtrack=True`.
        reseed_in_backtrack: When generating without explicit seeds (e.g., `generate()`),
            control whether failed attempts reseed on restart. If True, each restart
            samples a fresh seed (previous behavior). If False, do not reseed and perform
            only a single attempt; if it fails, move on to the next sample.
    """
    def __init__(
            self, 
            feasibility_estimator, 
            nbits, 
            decomposition_function, 
            cut_radius=0, 
            max_cut_size: Optional[int] = 3, 
            cut_context_radius: Optional[int] = None, 
            context_vectorizer: Optional[AbstractGraphTransformer] = None, 
            use_context_embedding: bool = False, 
            min_nodes_for_pruning=3, 
            max_restarts: int = 5, 
            n_virtual_candidates=5, 
            max_virtual_cut_evaluations: int = 10000,
            n_pruning_iterations=5, 
            similarity_vectorizer: Optional[AbstractGraphTransformer] = None, 
            similarity_k_neighbors: Optional[int] = 16, 
            similarity_temperature: float = 1.0,
            similarity_top_k: Optional[int] = 1,
            context_temperature: float = 1.0,
            context_top_k: Optional[int] = None,
            preserve_node_ids: bool = True,
            use_similarity_selection: bool = False, 
            n_jobs: int = -1, 
            display_function=None, 
            display_live: bool = False,
            verbose=False,
            context_dedup_epsilon: float = 1e-6,
            # Size-sampling policy knobs (backward compatible defaults)
            min_cut_size: Optional[int] = None,
            size_weights: Optional[dict[int, float]] = None,
            multicut_sizes: tuple[int, ...] = (2, 3),
            prob_force_multicut: float = 0.0,
            force_multicut_every: Optional[int] = None,
            # Backtrack failsafe: shrink target on backtrack to ensure termination
            shrink_target_on_backtrack: bool = True,
            backtrack_shrink_step: int = 2,
            # Adaptive evaluation budget for proposing cut-node sets
            initial_virtual_cut_evaluations: int = 1000,
            adaptive_virtual_cut_evaluations: bool = True,
            reseed_in_backtrack: bool = False,
            ):
        self.feasibility_estimator=feasibility_estimator
        self.nbits=nbits
        self.decomposition_function=decomposition_function
        self.cut_radius = cut_radius
        self.cut_context_radius = cut_context_radius
        self.context_vectorizer = context_vectorizer
        self.use_context_embedding = bool(use_context_embedding)
        self.min_nodes_for_pruning = min_nodes_for_pruning
        self.max_restarts = int(max_restarts)
        self.n_virtual_candidates = int(n_virtual_candidates)
        self.n_pruning_iterations = n_pruning_iterations
        self.verbose = verbose
        # Optional display hook used when verbose to show intermediate graphs.
        self.display_function = display_function
        # Whether to display intermediate graphs as they are generated.
        self.display_live = bool(display_live)
        # Optional external vectorizer for embeddings; if None, one is built in fit().
        self.similarity_vectorizer: Optional[AbstractGraphTransformer] = similarity_vectorizer
        # Cached reference embeddings (unit-normalized rows) computed from
        # self.generator_graphs for cosine-similarity selection.
        self._reference_unit_features: Optional[np.ndarray] = None
        # Whether to use similarity-based selection for candidate ranking.
        self.use_similarity_selection = bool(use_similarity_selection)
        # Number of training-graph neighbors to average for similarity scoring.
        self.similarity_k_neighbors = (
            16 if similarity_k_neighbors is None else int(similarity_k_neighbors)
        )
        # Selection temperature and top-k for similarity-based choices.
        self.similarity_temperature = float(similarity_temperature)
        self.similarity_top_k = (
            None if similarity_top_k is None else int(similarity_top_k)
        )
        # Selection temperature and top-k for context-based virtual cuts.
        self.context_temperature = float(context_temperature)
        self.context_top_k = None if context_top_k is None else int(context_top_k)
        # Node-id handling for rewrite outputs.
        self.preserve_node_ids = bool(preserve_node_ids)
        # Cap on cut-node sets sampled per step.
        self.max_virtual_cut_evaluations = int(max_virtual_cut_evaluations)
        # Parallelism level for donor construction and batch generation (-1 = all cores)
        self.n_jobs = int(n_jobs)
        # Maximum cut size allowed when sampling virtual cuts (None = no cap).
        self.max_cut_size = None if max_cut_size is None else int(max_cut_size)
        # Minimum distance for distinct context embeddings in deduplication.
        self.context_dedup_epsilon = float(context_dedup_epsilon)
        # Cut dictionary for virtual cut proposals (key -> list of donor tuples)
        self.cut_index = defaultdict(list)
        # Cached per-cut-size sets of per-node cut hashes observed in the donor cut index.
        self._known_cut_node_keys_by_size: dict[int, set] = {}
        # Stored dataset and embeddings used by get_generators().
        self._stored_graphs: Optional[list[nx.Graph]] = None
        self._stored_features: Optional[np.ndarray] = None
        self._stored_unit_features: Optional[np.ndarray] = None
        # Histories from the most recent generation call.
        self._last_generation_histories: list[list[nx.Graph]] = []
        
        # Size-sampling policy state
        self.min_cut_size = None if min_cut_size is None else int(min_cut_size)
        self.size_weights = dict(size_weights) if size_weights is not None else None
        self.multicut_sizes = tuple(int(s) for s in multicut_sizes) if multicut_sizes else tuple()
        self.prob_force_multicut = float(prob_force_multicut)
        self.force_multicut_every = None if force_multicut_every is None else int(force_multicut_every)
        self._proposal_counter = 0  # counts propose_virtual_candidates calls
        # Backtrack failsafe configuration
        self.shrink_target_on_backtrack = bool(shrink_target_on_backtrack)
        self.backtrack_shrink_step = max(1, int(backtrack_shrink_step))
        # Restart/seed policy when a sample fails
        self.reseed_in_backtrack = bool(reseed_in_backtrack)
        # Adaptive proposal evaluation configuration
        self.initial_virtual_cut_evaluations = max(1, int(initial_virtual_cut_evaluations))
        self.adaptive_virtual_cut_evaluations = bool(adaptive_virtual_cut_evaluations)

    def fit(self, generator_graphs):
        """Build the donor pool from training graphs and fit the feasibility estimator.

        Strategy
            For each training graph, generate a decreasing sequence of subgraphs
            using interpretation-node mapped-subgraph pruning down to `min_nodes_for_pruning`. Concatenate
            all sequences to form the donor pool, then fit the feasibility
            estimator on this pool.

        Args:
            generator_graphs: Iterable of NetworkX graphs serving as sources for
                donor extraction and for sampling target sizes during generation.

        Returns:
            self
        """
        self.generator_graphs = generator_graphs
        # Build donor pool (optionally in parallel over repeats × graphs)
        tasks = [(g, idx) for g in generator_graphs for idx in range(self.n_pruning_iterations)]
        if self.n_jobs == 1:
            results = [
                generate_pruning_sequences(
                    g,
                    min_nodes_for_pruning=self.min_nodes_for_pruning,
                    decomposition_function=self.decomposition_function,
                    nbits=self.nbits,
                    cut_radius=self.cut_radius,
                    cut_context_radius=self.cut_context_radius,
                    context_vectorizer=self.context_vectorizer,
                    use_context_embedding=self.use_context_embedding,
                    return_cut_index=True,
                )
                for g, _ in tasks
            ]
        else:
            results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(generate_pruning_sequences)(
                    g,
                    min_nodes_for_pruning=self.min_nodes_for_pruning,
                    decomposition_function=self.decomposition_function,
                    nbits=self.nbits,
                    cut_radius=self.cut_radius,
                    cut_context_radius=self.cut_context_radius,
                    context_vectorizer=self.context_vectorizer,
                    use_context_embedding=self.use_context_embedding,
                    return_cut_index=True,
                )
                for g, _ in tasks
            )
        self.cut_index = defaultdict(list)
        donors_lists = []
        for seq, cut_index in results:
            donors_lists.append(seq)
            if cut_index:
                for key, entries in cut_index.items():
                    self.cut_index[key].extend(entries)
        self.cut_index = deduplicate_cut_index(
            self.cut_index,
            context_dedup_epsilon=self.context_dedup_epsilon,
        )
        self._known_cut_node_keys_by_size = {}
        for key in self.cut_index.keys():
            size = len(key)
            if size <= 0:
                continue
            bucket = self._known_cut_node_keys_by_size.setdefault(size, set())
            for node_key in key:
                bucket.add(node_key)
        self.donors = []
        for lst in donors_lists:
            self.donors.extend(lst)
        #self.donors = GraphHashDeduper().fit_filter(self.donors) # this is an overkill as it almost never happens that two donors are identical by chance
        self.feasibility_estimator.fit(self.donors)

        # Prepare vectorizer and cache reference embeddings (based on training
        # generator graphs) for similarity-based sampling, if enabled.
        if not self.use_similarity_selection:
            self._reference_unit_features = None
        else:
            self._ensure_similarity_vectorizer()
            self._cache_reference_embeddings()
        if self.verbose and self.cut_index:
            size_counts = cut_size_distribution(self.cut_index)
            total_entries = sum(size_counts.values())
            print(
                f"cut_index keys={len(self.cut_index)} total_entries={total_entries} "
                f"size_hist={dict(sorted(size_counts.items()))}"
            )
        return self

    def store(self, graphs) -> "AutoregressiveGraphGenerator":
        """Store graphs and cache unit-normalized embeddings for neighbor sampling.

        If `use_similarity_selection=False`, embeddings are not computed and
        neighbor sampling falls back to uniform draws.

        Args:
            graphs: Iterable of NetworkX graphs to memorize.

        Returns:
            AutoregressiveGraphGenerator: Self.
        """
        stored = list(graphs)
        self._stored_graphs = stored
        if not self.use_similarity_selection:
            self._stored_features = None
            self._stored_unit_features = None
            return self
        self._ensure_similarity_vectorizer()
        if not stored:
            self._stored_features = np.zeros((0, 0))
            self._stored_unit_features = np.zeros((0, 0))
            return self
        features = np.asarray(self.similarity_vectorizer.fit_transform(stored), dtype=float)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._stored_features = features
        self._stored_unit_features = features / norms
        return self

    def get_generators(self, size: int = 1) -> list[nx.Graph]:
        """Sample a seed graph and its nearest neighbors from stored graphs.

        If `use_similarity_selection=False`, seeds are sampled uniformly.

        Args:
            size: Total number of generator graphs to return (seed + neighbors).

        Returns:
            List of generator graphs (length <= size when the store is small).
        """
        if self._stored_graphs is None:
            raise ValueError("store(graphs) must be called before get_generators().")
        graphs = self._stored_graphs
        if not self.use_similarity_selection:
            if not graphs or size <= 0:
                return []
            k = min(max(0, int(size)), len(graphs))
            return random.sample(graphs, k=k)
        if self._stored_unit_features is None:
            raise ValueError("store(graphs) must be called before get_generators().")
        unit_features = self._stored_unit_features
        if not graphs or size <= 0:
            return []
        seed_idx = random.randrange(len(graphs))
        seed_graph = graphs[seed_idx]
        if unit_features.shape[0] != len(graphs):
            raise ValueError("Stored embeddings do not match stored graphs.")
        if unit_features.ndim != 2:
            raise ValueError("Stored embeddings must be a 2D array.")
        sims = unit_features @ unit_features[seed_idx]
        sims[seed_idx] = -np.inf
        n_generator_graphs = int(size)
        k = max(0, min(n_generator_graphs - 1, len(graphs) - 1))
        if k > 0:
            nn_idx = np.argpartition(-sims, k - 1)[:k]
            nn_idx = nn_idx[np.argsort(-sims[nn_idx])]
        else:
            nn_idx = []
        return [seed_graph] + [graphs[i] for i in nn_idx]

    def _generate_from_seed(
        self,
        seed_graph: Optional[nx.Graph],
        *,
        return_history: bool = False,
        resample_seed_on_restart: bool = False,
        include_history: bool = False,
        target_size: Optional[int] = None,
    ):
        """Generate one graph via autoregressive growth from a provided seed.

        Args:
            seed_graph: Seed graph to grow from.
            return_history: If True, return the growth trajectory.
            resample_seed_on_restart: If True, choose a new random seed at each
                restart (ignores seed_graph).
            include_history: If True, return a (graph, history) tuple.

        Returns:
            NetworkX graph if `return_history=False`; otherwise the list of
            intermediate graphs (trajectory).
        """
        if seed_graph is None and not resample_seed_on_restart:
            raise ValueError("seed_graph is required when resample_seed_on_restart=False.")
        graph_hash_cache: dict[int, tuple[weakref.ref, int]] = {}
        can_score_similarity = (
            self.use_similarity_selection
            and self.similarity_vectorizer is not None
            and self._reference_unit_features is not None
        )

        def _graph_hash_cached(graph: nx.Graph) -> int:
            cached = graph_hash_cache.get(id(graph))
            if cached is not None:
                cached_ref, cached_hash = cached
                if cached_ref() is graph:
                    return cached_hash
            current_hash = hash_graph(graph)
            graph_hash_cache[id(graph)] = (weakref.ref(graph), current_hash)
            return current_hash

        best_graph = None
        best_size = -1
        best_history: list[nx.Graph] = []
        last_history: list[nx.Graph] = []
        displayed_hashes = set()
        # If we are not reseeding between restarts, perform only a single attempt
        # to honor the "move on to the next" behavior.
        n_restarts = max(1, self.max_restarts if resample_seed_on_restart else 1)
        reached_target = False
        for _restart in range(n_restarts):
            source = self._choose_seed() if resample_seed_on_restart else seed_graph
            max_nodes = target_size if target_size is not None else self._choose_target_size()
            max_nodes = max(int(len(source)*1.3), max_nodes)  # ensure some potential growth
            path = [source]
            seen_hashes = {_graph_hash_cached(source)}
            candidate_cache: dict[
                int, tuple[list[tuple[nx.Graph, int]], Optional[np.ndarray]]
            ] = {}
            restart_history: list[nx.Graph] = []
            best_restart_history: list[nx.Graph] = []
            best_restart_size = -1
            while path:
                current = path[-1]
                current_size = len(current)
                current_hash = _graph_hash_cached(current)
                if current_size > best_restart_size:
                    best_restart_size = current_size
                    best_restart_history = list(path)
                if self.verbose:
                    ratio = current_size / max_nodes if max_nodes else 1.0
                    print(f'{ratio:.2f}', end=' ')
                if (
                    self.verbose
                    and self.display_function is not None
                    and (self.display_live or not include_history)
                ):
                    if not self.display_live:
                        self._display_intermediate([current])
                    else:
                        if current_hash not in displayed_hashes:
                            displayed_hashes.add(current_hash)
                            self._display_intermediate([current])
                if current_size >= max_nodes:
                    restart_history = list(path)
                    best_restart_history = list(path)
                    if (
                        self.verbose
                        and self.display_function is not None
                        and self.display_live
                    ):
                        self._display_intermediate(restart_history)
                    break
                candidates, candidate_unit = self._context_phase_collect_candidates(
                    current,
                    current_hash,
                    candidate_cache,
                    can_score_similarity,
                    _graph_hash_cached,
                    seen_hashes,
                )
                candidates, candidate_unit = self._context_phase_prune_seen_candidates(
                    candidates,
                    candidate_unit,
                    seen_hashes,
                )
                if not candidates:
                    # Failsafe: if backtracking, optionally shrink the target size
                    # so repeated backtracks eventually satisfy the termination condition.
                    if self.shrink_target_on_backtrack:
                        new_target = max(1, int(max_nodes) - self.backtrack_shrink_step)
                        if new_target != max_nodes and self.verbose:
                            try:
                                print(f"[shrink] target {max_nodes} -> {new_target}", end=' ')
                            except Exception:
                                pass
                        max_nodes = new_target
                    path.pop()
                    continue
                candidate_cache[current_hash] = (candidates, candidate_unit)
                candidate_graphs = [candidate for candidate, _ in candidates]
                next_idx = self._similarity_phase_choose_index(
                    candidate_graphs,
                    current,
                    sample_unit_features=candidate_unit,
                )
                next_graph, next_hash = candidates.pop(next_idx)
                if candidate_unit is not None:
                    candidate_unit = np.delete(candidate_unit, next_idx, axis=0)
                candidate_cache[current_hash] = (candidates, candidate_unit)
                seen_hashes.add(next_hash)
                path.append(next_graph)
            if restart_history:
                candidate = restart_history[-1]
                candidate_size = len(candidate)
                if candidate_size > best_size:
                    best_graph = candidate
                    best_size = candidate_size
                    best_history = list(restart_history)
                last_history = list(restart_history)
            if best_size >= max_nodes:
                reached_target = True
                break
        history_source = best_history if best_history else last_history
        history = list(history_source)
        if return_history:
            if self.verbose and not resample_seed_on_restart and not reached_target:
                try:
                    seed_sz = len(seed_graph) if seed_graph is not None else 0
                    reached_sz = len(history[-1]) if history else seed_sz
                    print(f"[give-up] seed_size={seed_sz} reached_size={reached_sz} reached_target={reached_target}")
                except Exception:
                    pass
            return history
        if best_graph is not None:
            final_graph = best_graph
        else:
            fallback = seed_graph if seed_graph is not None else self._choose_seed()
            final_graph = fallback
        if not history:
            history = [final_graph]
        if include_history:
            if self.verbose and not resample_seed_on_restart and not reached_target:
                try:
                    seed_sz = len(seed_graph) if seed_graph is not None else 0
                    reached_sz = len(history[-1]) if history else seed_sz
                    print(f"[give-up] seed_size={seed_sz} reached_size={reached_sz} reached_target={reached_target}")
                except Exception:
                    pass
            return final_graph, history
        return final_graph

    def _generate_from_seeds(
        self,
        seed_graphs: Sequence[Optional[nx.Graph]],
        *,
        resample_seed_on_restart: bool = False,
        include_history: bool = False,
        target_size: Optional[int] = None,
    ) -> list[nx.Graph]:
        """Generate graphs from provided seeds, optionally in parallel.

        Args:
            seed_graphs: Seed graphs to grow from.
            resample_seed_on_restart: If True, choose a new random seed at each
                restart (ignores seed_graphs).

        Returns:
            List of generated NetworkX graphs. May be shorter than requested if
            outputs are deduplicated against stored/training graphs.
        """
        seeds = list(seed_graphs)
        if not seeds:
            return []
        if self.n_jobs == 1 or len(seeds) <= 1 or self.display_live:
            if resample_seed_on_restart:
                results = [
                    self._generate_from_seed(
                        None,
                        resample_seed_on_restart=True,
                        include_history=include_history,
                        target_size=target_size,
                    )
                    for _ in range(len(seeds))
                ]
            else:
                results = [
                    self._generate_from_seed(
                        seed,
                        include_history=include_history,
                        target_size=target_size,
                    )
                    for seed in seeds
                ]
            if not include_history:
                graphs, _ = self._dedup_outputs(results, seed_graphs=seeds)
                return graphs
            graphs, histories = zip(*results)
            graphs, histories = self._dedup_outputs(graphs, histories, seed_graphs=seeds)
            self._last_generation_histories = histories or []
            if (
                self.verbose
                and self.display_function is not None
                and not self.display_live
            ):
                for history in self._last_generation_histories:
                    self._display_intermediate(history)
            return list(graphs)
        # Avoid nested parallelism in similarity vectorizer inside child processes
        old_n_jobs = getattr(self.similarity_vectorizer, "n_jobs", None)
        if self.similarity_vectorizer is not None:
            try:
                self.similarity_vectorizer.n_jobs = 1
            except Exception:
                pass
        try:
            if resample_seed_on_restart:
                results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                    delayed(self._generate_from_seed)(
                        None,
                        resample_seed_on_restart=True,
                        include_history=include_history,
                        target_size=target_size,
                    )
                    for _ in range(len(seeds))
                )
            else:
                results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                    delayed(self._generate_from_seed)(
                        seed,
                        include_history=include_history,
                        target_size=target_size,
                    )
                    for seed in seeds
                )
        finally:
            if self.similarity_vectorizer is not None and old_n_jobs is not None:
                try:
                    self.similarity_vectorizer.n_jobs = old_n_jobs
                except Exception:
                    pass
        if not include_history:
            graphs, _ = self._dedup_outputs(results, seed_graphs=seeds)
            return graphs
        graphs, histories = zip(*results)
        graphs, histories = self._dedup_outputs(graphs, histories, seed_graphs=seeds)
        self._last_generation_histories = histories or []
        if (
            self.verbose
            and self.display_function is not None
            and not self.display_live
        ):
            for history in self._last_generation_histories:
                self._display_intermediate(history)
        return list(graphs)

    def generate(self, n_samples=1):
        """Generate multiple graphs by calling `generate_single` repeatedly.

        Args:
            n_samples: Number of graphs to generate.

        Returns:
            List of generated NetworkX graphs.
        """
        n = max(0, int(n_samples))
        if n == 0:
            return []
        # If not reseeding on backtrack, pre-select explicit seeds for each sample
        # and disable per-restart reseeding; otherwise, use implicit seeding.
        if self.reseed_in_backtrack:
            seeds = [None] * n
            resample = True
        else:
            seeds = [self._choose_seed() for _ in range(n)]
            resample = False
        self._last_generation_histories = []
        return self._generate_from_seeds(
            seeds,
            resample_seed_on_restart=resample,
            include_history=True,
        )

    def generate_from(self, seed_graphs: Sequence[nx.Graph], *, graph_size: Optional[int] = None) -> list[nx.Graph]:
        """Generate graphs using the provided seed graphs.

        Args:
            seed_graphs: Seed graphs to grow from (one output per seed).
            graph_size: If provided, override the sampled target size and use
                this value as the target node count for growth.

        Returns:
            List of generated NetworkX graphs (length may be less than seed_graphs
            if outputs are deduplicated against stored/training graphs).
        """
        self._last_generation_histories = []
        return self._generate_from_seeds(
            seed_graphs,
            resample_seed_on_restart=False,
            include_history=True,
            target_size=graph_size,
        )

    def generate_single(self, return_history=False):
        """Generate one graph via autoregressive growth.

        Algorithm
            1) Pick a seed subgraph uniformly at random from the donors.
            2) Choose a random target size by sampling one of the training
               graphs and using its node count as an upper bound.
            3) Loop:
               - Propose up to `n_virtual_candidates` virtual candidates via cut insertions.
               - Filter proposals with `feasibility_estimator.filter`.
               - Keep only strictly larger graphs than the current source.
               - Embed training graphs (cached) and candidates; compute cosine
                 similarities to the training set per candidate and average the top-k
                 nearest neighbor similarities. Sample the next source with
                 probability proportional to these averages. If similarity
                 scoring is disabled, fall back to an edge-count heuristic.
               - Stop when reaching/exceeding the target size or when no
                 acceptable proposals remain after retrying virtual-cut
                 proposals up to `max_restarts` times. Backtracking avoids
                 repeating previously visited graphs. If `verbose` and
                 `display_function` are set, visualize intermediate graphs.

        Args:
            return_history: If True, return the list of intermediate sources
                (growth trajectory) instead of only the last graph.

        Returns:
            NetworkX graph if `return_history=False`; otherwise the list of
            intermediate graphs (trajectory). Outputs are reduced to the largest
            connected component at return time.
        """
        # Respect reseed policy: if not reseeding on backtrack, fix the seed and
        # perform a single attempt; otherwise allow reseeding across restarts.
        if self.reseed_in_backtrack:
            result = self._generate_from_seed(
                None,
                return_history=return_history,
                resample_seed_on_restart=True,
            )
        else:
            result = self._generate_from_seed(
                self._choose_seed(),
                return_history=return_history,
                resample_seed_on_restart=False,
            )
        if return_history:
            self._last_generation_histories = [list(result)]
            if (
                self.verbose
                and self.display_function is not None
                and not self.display_live
            ):
                self._display_intermediate(result)
        else:
            self._last_generation_histories = []
        return result

    #--------------------------------------------------------------------------------
    # Auxiliary helpers for modularity/readability
    #--------------------------------------------------------------------------------
    def _ensure_similarity_vectorizer(self) -> None:
        """Ensure a graph vectorizer exists; build a default if not provided."""
        if self.similarity_vectorizer is None:
            self.similarity_vectorizer = AbstractGraphTransformer(
                nbits=self.nbits,
                decomposition_function=self.decomposition_function,
                return_dense=True,
                n_jobs=1,
            )

    def _cache_reference_embeddings(self) -> None:
        """Compute and cache unit-normalized embeddings of generator graphs.

        These serve as the similarity reference set during candidate selection,
        instead of using the donor pool (which may be biased by pruning).
        """
        if not self.use_similarity_selection:
            self._reference_unit_features = None
            return
        if not getattr(self, 'generator_graphs', None):
            self._reference_unit_features = None
            return
        ref_features = self.similarity_vectorizer.fit_transform(self.generator_graphs)
        ref_features = np.asarray(ref_features, dtype=float)
        norms = np.linalg.norm(ref_features, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._reference_unit_features = ref_features / norms

    def _choose_seed(self) -> nx.Graph:
        """Return a random donor graph selected uniformly."""
        return self.donors[random.randrange(len(self.donors))]

    def _choose_target_size(self) -> int:
        """Sample a target size bound from the total node count."""
        graph = self.generator_graphs[random.randrange(len(self.generator_graphs))]
        return len(graph)

    def _propose_candidates(self, source: nx.Graph):
        """Propose candidate next graphs via virtual-cut insertion."""
        return self.propose_virtual_candidates(
            source,
            n_virtual_candidates=self.n_virtual_candidates,
            max_virtual_cut_evaluations=self.max_virtual_cut_evaluations,
        )

    def propose_virtual_candidates(
        self,
        source: nx.Graph,
        *,
        n_virtual_candidates: int = 1,
        max_virtual_cut_evaluations: Optional[int] = None,
        cut_size: Optional[int] = None,
        max_cut_size: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> list[nx.Graph]:
        """Propose candidates by inserting donors at virtual cuts.

        Args:
            source: Source graph to augment.
            n_virtual_candidates: Number of candidate graphs to return.
            max_virtual_cut_evaluations: Optional cap on sampled cut-node sets.
                If None, defaults to `n_virtual_candidates`.
            cut_size: Optional fixed cut size; if None, sample cut sizes
                uniformly at random from available sizes up to `max_cut_size`.
                Each sampled size yields a uniformly sampled cut-node set.
            max_cut_size: Optional per-call cap for virtual cut size. If None,
                uses `self.max_cut_size`.
            rng: Optional random generator for reproducibility.

        Returns:
            List of candidate graphs produced via virtual-cut insertion.
        """
        if not self.cut_index:
            return []
        rng = rng or random.Random()
        # advance proposal counter for deterministic schedules
        self._proposal_counter += 1
        size_counts = cut_size_distribution(self.cut_index)
        effective_max_cut_size = (
            self.max_cut_size if max_cut_size is None else int(max_cut_size)
        )
        if effective_max_cut_size is not None:
            size_counts = {
                s: c for s, c in size_counts.items() if s <= effective_max_cut_size
            }
        sizes = sorted(size_counts.keys())
        if not sizes:
            return []
        if cut_size is not None:
            size = (
                cut_size
                if effective_max_cut_size is None
                else min(cut_size, effective_max_cut_size)
            )
            sizes = [size] if size in size_counts else []
        if not sizes:
            return []
        nodes = list(source.nodes())
        if not nodes:
            return []
        # Apply availability and policy filters
        sizes = [s for s in sizes if s <= len(nodes)]
        if not sizes:
            return []
        base_sizes = list(sizes)
        # Enforce minimum cut size if requested; fallback to base sizes if empty
        if self.min_cut_size is not None:
            sizes = [s for s in sizes if s >= self.min_cut_size]
            if not sizes:
                sizes = base_sizes
        # Occasionally restrict to preferred multicut sizes
        restrict = False
        if self.force_multicut_every is not None and self.force_multicut_every > 0:
            if self._proposal_counter % self.force_multicut_every == 0:
                restrict = True
        if not restrict and self.prob_force_multicut > 0.0:
            try:
                if rng.random() < self.prob_force_multicut:
                    restrict = True
            except Exception:
                pass
        if restrict and self.multicut_sizes:
            preferred = sorted(set(s for s in sizes if s in self.multicut_sizes))
            if preferred:
                sizes = preferred
        if not sizes:
            return []
        node_key_map = _precompute_outer_node_keys(source, self.cut_radius)
        outputs: list[nx.Graph] = []
        max_outputs = max(1, int(n_virtual_candidates))
        budget = max_outputs if max_virtual_cut_evaluations is None else int(max_virtual_cut_evaluations)
        budget = max(0, budget)
        miss_budget = max(1, budget)
        misses = 0
        attempted_sizes: set[int] = set()
        # Verbose accounting: attempts and matches per cut size
        attempts_by_size = defaultdict(int)
        matches_by_size = defaultdict(int)
        for _ in range(budget):
            if len(outputs) >= max_outputs:
                break
            # sample size with optional per-size weights
            if self.size_weights:
                weights = [max(0.0, float(self.size_weights.get(s, 1.0))) for s in sizes]
                if not any(w > 0 for w in weights):
                    weights = [1.0] * len(sizes)
                size = rng.choices(sizes, weights=weights, k=1)[0]
            else:
                size = rng.choice(sizes)
            attempted_sizes.add(size)
            if size > len(nodes):
                misses += 1
                if misses >= miss_budget:
                    break
                continue
            known_keys = self._known_cut_node_keys_by_size.get(size)
            if not known_keys:
                misses += 1
                if misses >= miss_budget:
                    break
                continue
            known_nodes = [
                node for node in nodes if node_key_map.get(node) in known_keys
            ]
            if len(known_nodes) < size:
                misses += 1
                if misses >= miss_budget:
                    break
                continue
            cut_nodes = rng.sample(known_nodes, k=size)
            attempts_by_size[size] += 1
            _cands = virtual_rewrite_candidates_at_cut_nodes(
                source,
                cut_nodes,
                self.cut_index,
                cut_radius=self.cut_radius,
                cut_context_radius=self.cut_context_radius,
                context_vectorizer=self.context_vectorizer,
                use_context_embedding=self.use_context_embedding,
                context_temperature=self.context_temperature,
                context_top_k=self.context_top_k,
                rng=rng,
                n_samples=1,
                node_key_map=node_key_map,
                preserve_node_ids=self.preserve_node_ids,
            )
            if _cands:
                matches_by_size[size] += 1
                outputs.extend(_cands)
        if len(outputs) >= max_outputs:
            if self.verbose and attempts_by_size:
                try:
                    parts = [
                        f"[size={s} tried={attempts_by_size[s]} match={matches_by_size.get(s,0)}]"
                        for s in sorted(attempts_by_size)
                    ]
                    print(f"[cuts] budget={budget} produced={len(outputs)} " + " ".join(parts))
                except Exception:
                    pass
            return outputs
        if misses < miss_budget:
            if self.verbose and attempts_by_size:
                try:
                    parts = [
                        f"[size={s} tried={attempts_by_size[s]} match={matches_by_size.get(s,0)}]"
                        for s in sorted(attempts_by_size)
                    ]
                    print(f"[cuts] budget={budget} produced={len(outputs)} " + " ".join(parts))
                except Exception:
                    pass
            return outputs
        remaining_sizes = [s for s in sizes if s not in attempted_sizes]
        if not remaining_sizes:
            remaining_sizes = list(sizes)
        for size in remaining_sizes:
            if len(outputs) >= max_outputs:
                break
            if size > len(nodes):
                continue
            known_keys = self._known_cut_node_keys_by_size.get(size)
            if not known_keys:
                continue
            known_nodes = [
                node for node in nodes if node_key_map.get(node) in known_keys
            ]
            if len(known_nodes) < size:
                continue
            cut_nodes = rng.sample(known_nodes, k=size)
            attempts_by_size[size] += 1
            _cands = virtual_rewrite_candidates_at_cut_nodes(
                source,
                cut_nodes,
                self.cut_index,
                cut_radius=self.cut_radius,
                cut_context_radius=self.cut_context_radius,
                context_vectorizer=self.context_vectorizer,
                use_context_embedding=self.use_context_embedding,
                context_temperature=self.context_temperature,
                context_top_k=self.context_top_k,
                rng=rng,
                n_samples=1,
                node_key_map=node_key_map,
                preserve_node_ids=self.preserve_node_ids,
            )
            if _cands:
                matches_by_size[size] += 1
                outputs.extend(_cands)
        if self.verbose and attempts_by_size:
            try:
                parts = [
                    f"[size={s} tried={attempts_by_size[s]} match={matches_by_size.get(s,0)}]"
                    for s in sorted(attempts_by_size)
                ]
                print(f"[cuts] budget={budget} produced={len(outputs)} " + " ".join(parts))
            except Exception:
                pass
        return outputs

    def propose_virtual_candidates_at_cut_nodes(
        self,
        source: nx.Graph,
        cut_nodes: Sequence,
        *,
        n_samples: int = 1,
        rng: Optional[random.Random] = None,
        allow_superset: bool = False,
        max_superset_checks: Optional[int] = None,
        single_replacement: bool = True,
        max_enumerations: Optional[int] = None,
    ) -> list[nx.Graph]:
        """Propose candidates by inserting donors at a specified virtual cut.

        Args:
            source: Source graph to augment.
            cut_nodes: Explicit outer nodes defining the virtual cut.
            n_samples: Number of candidate graphs to return.
            rng: Optional random generator for reproducibility.
            allow_superset: If True, fall back to cuts that strictly contain `cut_nodes`.
            max_superset_checks: Optional cap on attempted superset cut evaluations.
            single_replacement: If True, pick one random pairing per label.
            max_enumerations: Cap on enumerated/sampled pairings when single_replacement=False.

        Returns:
            List of candidate graphs produced via virtual-cut insertion.
        """
        if not self.cut_index:
            return []
        return virtual_rewrite_candidates_at_cut_nodes(
            source,
            cut_nodes,
            self.cut_index,
            cut_radius=self.cut_radius,
            cut_context_radius=self.cut_context_radius,
            context_vectorizer=self.context_vectorizer,
            use_context_embedding=self.use_context_embedding,
            context_temperature=self.context_temperature,
            context_top_k=self.context_top_k,
            rng=rng,
            n_samples=n_samples,
            allow_superset=allow_superset,
            max_superset_checks=max_superset_checks,
            single_replacement=single_replacement,
            max_enumerations=max_enumerations,
            preserve_node_ids=self.preserve_node_ids,
        )

    def _filter_growth(self, samples, source: nx.Graph):
        """Filter proposals for feasibility and strict growth over `source`."""
        samples = self.feasibility_estimator.filter(samples)
        return [s for s in samples if len(s) > len(source)]

    def _display_intermediate(self, graphs: Sequence[nx.Graph]) -> None:
        """Display intermediate graphs when a display hook is configured."""
        ipy_display = None
        try:
            from IPython.display import display as ipy_display
        except Exception:
            ipy_display = None
        had_builtin_display = False
        old_builtin_display = None
        if ipy_display is not None:
            try:
                import builtins
                had_builtin_display = hasattr(builtins, "display")
                if had_builtin_display:
                    old_builtin_display = builtins.display
                else:
                    builtins.display = ipy_display
            except Exception:
                had_builtin_display = False
                old_builtin_display = None
        try:
            result = self.display_function(graphs)
        finally:
            if ipy_display is not None:
                try:
                    import builtins
                    if not had_builtin_display:
                        delattr(builtins, "display")
                    else:
                        builtins.display = old_builtin_display
                except Exception:
                    pass
        if result is not None and ipy_display is not None:
            if isinstance(result, (list, tuple)):
                for item in result:
                    ipy_display(item)
            else:
                ipy_display(result)
            return
        if ipy_display is None:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        figs = [plt.figure(num) for num in plt.get_fignums()]
        if not figs:
            return
        for fig in figs:
            ipy_display(fig)
            try:
                plt.close(fig)
            except Exception:
                pass

    def _dataset_for_dedup(self) -> Optional[Sequence[nx.Graph]]:
        """Return the dataset used to deduplicate outputs, if available."""
        if self._stored_graphs:
            return self._stored_graphs
        return getattr(self, "generator_graphs", None)

    def _dedup_outputs(
        self,
        graphs: Sequence[nx.Graph],
        histories: Optional[Sequence[list[nx.Graph]]] = None,
        seed_graphs: Optional[Sequence[nx.Graph]] = None,
    ) -> tuple[list[nx.Graph], Optional[list[list[nx.Graph]]]]:
        """Deduplicate outputs against the dataset and within the batch."""
        dataset = self._dataset_for_dedup()
        if not graphs:
            return list(graphs), list(histories) if histories is not None else None
        dataset_hashes = {hash_graph(g) for g in dataset} if dataset else set()
        seen = set()
        kept_graphs: list[nx.Graph] = []
        kept_histories: list[list[nx.Graph]] = [] if histories is not None else None
        for idx, graph in enumerate(graphs):
            graph_hash = hash_graph(graph)
            is_unmodified_seed = False
            if (
                histories is not None
                and seed_graphs is not None
                and idx < len(histories)
                and idx < len(seed_graphs)
                and seed_graphs[idx] is not None
            ):
                # An unmodified seed graph will have a history of length 1 containing just itself.
                if len(histories[idx]) == 1:
                    # Check if the hash matches the original seed hash.
                    if graph_hash == hash_graph(seed_graphs[idx]):
                        is_unmodified_seed = True
            # Don't dedup unmodified seeds against the training set, just against other outputs.
            if is_unmodified_seed:
                if graph_hash in seen:
                    continue
            elif graph_hash in dataset_hashes or graph_hash in seen:
                continue
            seen.add(graph_hash)
            kept_graphs.append(graph)
            if kept_histories is not None:
                kept_histories.append(list(histories[idx]))
        return kept_graphs, kept_histories

    def _select_next_by_similarity(self, samples, source: nx.Graph):
        """Select the next source by similarity or context preference."""
        if not samples:
            raise ValueError("samples must be non-empty.")
        idx = self._select_next_index_by_similarity(samples, source)
        return samples[idx]

    def _select_next_index_by_similarity(
        self,
        samples,
        source: nx.Graph,
        *,
        sample_unit_features: Optional[np.ndarray] = None,
    ) -> int:
        """Return the index of the chosen sample under similarity/context selection.

        Args:
            samples: Candidate graphs to choose from.
            source: Current graph (unused, kept for signature parity).
            sample_unit_features: Optional unit-normalized embeddings for samples.

        Returns:
            Index of the selected candidate in `samples`.
        """
        if self.use_similarity_selection:
            if self.similarity_vectorizer is not None and self._reference_unit_features is not None:
                if sample_unit_features is None or sample_unit_features.shape[0] != len(samples):
                    sample_features = self.similarity_vectorizer.transform(samples)
                    sample_features = np.asarray(sample_features, dtype=float)
                    s_norms = np.linalg.norm(sample_features, axis=1, keepdims=True)
                    s_norms[s_norms == 0.0] = 1.0
                    sample_unit = sample_features / s_norms
                else:
                    sample_unit = sample_unit_features
                sims = sample_unit @ self._reference_unit_features.T  # (m_samples, n_refs)
                n_refs = sims.shape[1]
                k = min(max(1, int(self.similarity_k_neighbors)), n_refs)
                # Average of top-k cosine similarities per sample
                # Use partition for efficiency; result is not sorted but top-k set is correct
                topk = np.partition(sims, -k, axis=1)[:, -k:]
                scores = topk.mean(axis=1)
                if self.similarity_temperature <= 0.0:
                    k_select = 1 if self.similarity_top_k is None else self.similarity_top_k
                    idx = _top_k_indices(scores, k_select)
                    if idx:
                        return idx[0]
                    return random.randrange(len(samples))
                weights = _temperature_weights(scores, self.similarity_temperature)
                if np.all(np.isfinite(weights)) and weights.sum() > 0:
                    return random.choices(
                        range(len(samples)), weights=weights.tolist(), k=1
                    )[0]
                return random.randrange(len(samples))
        if self.use_context_embedding:
            scores = []
            for sample in samples:
                score = sample.graph.get("context_score")
                if score is None:
                    scores.append(-np.inf)
                else:
                    scores.append(float(score))
            if any(np.isfinite(s) for s in scores):
                scores = np.asarray(scores, dtype=float)
                scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
                if self.context_temperature <= 0.0:
                    idx = _top_k_indices(scores, 1)
                    if idx:
                        return idx[0]
                    return random.randrange(len(samples))
                weights = _temperature_weights(scores, self.context_temperature)
                if np.all(np.isfinite(weights)) and weights.sum() > 0:
                    return random.choices(
                        range(len(samples)), weights=weights.tolist(), k=1
                    )[0]
                return random.randrange(len(samples))
        weights = [max(1, s.number_of_edges()) for s in samples]
        return random.choices(range(len(samples)), weights=weights, k=1)[0]

    def _context_phase_collect_candidates(
        self,
        current: nx.Graph,
        current_hash: int,
        candidate_cache: dict[int, tuple[list[tuple[nx.Graph, int]], Optional[np.ndarray]]],
        can_score_similarity: bool,
        hash_fn: Callable[[nx.Graph], int],
        seen_hashes: set[int],
    ) -> tuple[list[tuple[nx.Graph, int]], Optional[np.ndarray]]:
        """Collect or reuse context-aware candidates around `current`."""
        cached = candidate_cache.get(current_hash)
        if cached is not None:
            return cached
        unique_samples: list[tuple[nx.Graph, int]] = []
        sample_hashes: set[int] = set()
        if self.adaptive_virtual_cut_evaluations:
            # Adaptive doubling of evaluation budget up to the configured maximum
            budget = self.initial_virtual_cut_evaluations
            max_budget = max(1, int(self.max_virtual_cut_evaluations))
            while True:
                proposals = self.propose_virtual_candidates(
                    current,
                    n_virtual_candidates=self.n_virtual_candidates,
                    max_virtual_cut_evaluations=min(budget, max_budget),
                )
                feasible = self.feasibility_estimator.filter(proposals)
                growth_samples = [s for s in feasible if len(s) > len(current)]
                samples = growth_samples
                if self.verbose:
                    try:
                        print(
                            f"[cands] budget={min(budget, max_budget)} proposed={len(proposals)} "
                            f"feasible={len(feasible)} growth={len(growth_samples)}"
                        )
                    except Exception:
                        pass
                for sample in samples:
                    sample_hash = hash_fn(sample)
                    if sample_hash in seen_hashes or sample_hash in sample_hashes:
                        continue
                    unique_samples.append((sample, sample_hash))
                    sample_hashes.add(sample_hash)
                if unique_samples or budget >= max_budget:
                    break
                budget = min(max_budget, budget * 2)
        else:
            # Original behavior: try multiple propose calls up to max_restarts
            max_attempts = max(1, self.max_restarts)
            for _ in range(max_attempts):
                proposals = self._propose_candidates(current)
                feasible = self.feasibility_estimator.filter(proposals)
                growth_samples = [s for s in feasible if len(s) > len(current)]
                samples = growth_samples
                if self.verbose:
                    try:
                        print(
                            f"[cands] proposed={len(proposals)} feasible={len(feasible)} growth={len(growth_samples)}"
                        )
                    except Exception:
                        pass
                for sample in samples:
                    sample_hash = hash_fn(sample)
                    if sample_hash in seen_hashes or sample_hash in sample_hashes:
                        continue
                    unique_samples.append((sample, sample_hash))
                    sample_hashes.add(sample_hash)
                if unique_samples:
                    break
        candidate_unit: Optional[np.ndarray] = None
        if can_score_similarity and unique_samples:
            sample_graphs = [s for s, _ in unique_samples]
            sample_features = self.similarity_vectorizer.transform(sample_graphs)
            sample_features = np.asarray(sample_features, dtype=float)
            s_norms = np.linalg.norm(sample_features, axis=1, keepdims=True)
            s_norms[s_norms == 0.0] = 1.0
            candidate_unit = sample_features / s_norms
        candidate_cache[current_hash] = (unique_samples, candidate_unit)
        return unique_samples, candidate_unit

    def _context_phase_prune_seen_candidates(
        self,
        candidates: list[tuple[nx.Graph, int]],
        candidate_unit: Optional[np.ndarray],
        seen_hashes: set[int],
    ) -> tuple[list[tuple[nx.Graph, int]], Optional[np.ndarray]]:
        """Drop candidates already visited during the current restart walk."""
        if not candidates:
            return candidates, candidate_unit
        keep_indices = [
            idx
            for idx, (_candidate, candidate_hash) in enumerate(candidates)
            if candidate_hash not in seen_hashes
        ]
        if len(keep_indices) == len(candidates):
            return candidates, candidate_unit
        filtered = [candidates[idx] for idx in keep_indices]
        if candidate_unit is not None:
            candidate_unit = candidate_unit[keep_indices, :]
        return filtered, candidate_unit

    def _similarity_phase_choose_index(
        self,
        candidate_graphs: Sequence[nx.Graph],
        source: nx.Graph,
        *,
        sample_unit_features: Optional[np.ndarray] = None,
    ) -> int:
        """Return the index chosen by the similarity/context scoring phase."""
        return self._select_next_index_by_similarity(
            candidate_graphs,
            source,
            sample_unit_features=sample_unit_features,
        )

    # Public utility ----------------------------------------------------------
    def sample_seeds(
        self,
        n: int = 7,
        *,
        with_replacement: bool = False,
        seed: Optional[int] = None,
        strategy: str = "uniform",
    ) -> list[nx.Graph]:
        """Return a list of seed graphs for visualization or initialization.

        Seeds are drawn from donors with exactly `min_nodes_for_pruning` nodes when available;
        if none exist, the smallest donors are used instead.

        Args:
            n: Number of seeds to return.
            with_replacement: If True and strategy="uniform", sample with replacement.
            seed: Optional RNG seed for reproducibility.
            strategy: One of {"uniform", "diverse"}.
                - "uniform": sample uniformly at random from the seed pool.
                - "diverse": farthest-point sampling in vectorizer embedding space
                  (cosine distance), encouraging diversity among chosen seeds.

        Returns:
            List of NetworkX graphs (length <= n if sampling without replacement
            from a smaller pool).
        """
        rng = random.Random(seed)
        # Build seed pool
        pool = [g for g in getattr(self, "donors", []) if len(g) == self.min_nodes_for_pruning]
        if not pool:
            pool = sorted(getattr(self, "donors", []), key=lambda g: len(g))
        if not pool:
            return []

        if strategy == "uniform":
            if with_replacement:
                return [rng.choice(pool) for _ in range(max(0, int(n)))]
            k = min(max(0, int(n)), len(pool))
            return rng.sample(pool, k=k)

        if strategy == "diverse":
            # Ensure vectorizer exists and is fitted; use transform only.
            self._ensure_similarity_vectorizer()
            try:
                feats = self.similarity_vectorizer.transform(pool)
            except Exception:
                # Fallback to uniform if transformation fails
                k = min(max(0, int(n)), len(pool))
                return rng.sample(pool, k=k)
            feats = np.asarray(feats, dtype=float)
            if feats.ndim != 2 or feats.shape[0] == 0:
                k = min(max(0, int(n)), len(pool))
                return rng.sample(pool, k=k)
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            U = feats / norms  # unit-normalized embeddings

            k = min(max(0, int(n)), len(pool))
            if k == 0:
                return []
            # Farthest-point initialization
            first = rng.randrange(len(pool))
            chosen = [first]
            # Cosine distance = 1 - dot(u, v)
            dist = 1.0 - (U @ U[first].reshape(-1, 1)).ravel()
            for _ in range(1, k):
                idx = int(np.argmax(dist))
                chosen.append(idx)
                dist = np.minimum(dist, 1.0 - (U @ U[idx].reshape(-1, 1)).ravel())
            return [pool[i] for i in chosen]

        # Unknown strategy → default to uniform without replacement
        k = min(max(0, int(n)), len(pool))
        return rng.sample(pool, k=k)
