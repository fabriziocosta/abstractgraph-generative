"""
Graph rewriting utilities for AbstractGraph.

Overview
- AbstractGraph represents a graph at two levels: an interpretation graph
  (coarse) and a base graph (fine). Each interpretation node carries a
  mapped-subgraph payload from the base graph.
- Rewiring "swaps" a mapped subgraph in a source graph with a compatible
  mapped subgraph taken from donor graphs, reconnecting the boundary so
  structure is preserved.

Compatibility via cut signatures
- For any mapped subgraph, we compute its boundary cut: all edges crossing from the
  mapped subgraph's inner nodes to the rest of the base graph.
- Each boundary edge is mapped to a per-edge context key that encodes enough
  local information to ensure compatibility:
  - If `cut_radius is None`: key = (edge_label_hash,) or () when
    cut_include_edge_label=False.
  - If `cut_radius >= 0`: key = (edge_label_hash, inner_hash, outer_hash) or
    (inner_hash, outer_hash) when cut_include_edge_label=False, where hashes
    are computed from the inner/outer induced subgraphs within `cut_radius`.
    The `cut_scope` flag can drop either side, yielding:
    - cut_scope="inner": key = (edge_label_hash, inner_hash)
    - cut_scope="outer": key = (edge_label_hash, outer_hash)
    - cut_scope="both": key = (edge_label_hash, inner_hash, outer_hash)
- The cut signature is the order-independent multiset of these per-edge keys
  (represented as a sorted tuple). Two mapped subgraphs are compatible iff their
  cut signatures are identical. This ensures that, for each key, the number of
  boundary edges and their local contexts match.

Context-aware selection (optional)
- When `context_vectorizer` is provided and `use_context_embedding=True`, each
  cut stores aggregated inner/outer context embeddings derived from
  radius-limited neighborhoods. Rewrite selection can then prefer donor/source
  pairs with higher cosine similarity.

Rewrite algorithm (rewrite)
1) Build/cached AGs: Construct AbstractGraph for source and donors (via the
   provided `decomposition_function` and `nbits`) or reuse cached AGs.
2) Index donors: For every donor mapped subgraph, compute its cut signature and
   a map from per-edge keys to the list of inner endpoints (with edge attrs)
   that can accept a boundary edge of that key. Also store the
   mapped-subgraph hash for no-op checks.
3) Index source: For every source mapped subgraph, compute its cut signature and a
   map from per-edge keys to the list of (outer_endpoint, edge_attr) boundary
   edges that must be reconnected when this mapped subgraph is replaced.
4) Match: Intersect source and donor indices by cut signature. For each matching
   signature, pair each source candidate with all donor candidates whose
   mapped-subgraph hash differs (to avoid swapping a mapped subgraph with an identical
   copy).
5) Choose a pair: Randomly pick one (source_assoc, donor_assoc) pair among the
   compatible pairs. The donor and source per-edge key maps guarantee, per key,
   equal cardinality of edges (validated again in replacement).
6) Replace subgraph:
   - Remove the source mapped subgraph's inner nodes from a copy of the source.
- Add donor mapped-subgraph nodes/edges, remapping donor node ids to a fresh,
  contiguous integer block to avoid collisions with heterogeneous node ids.
   - For each per-edge key, connect donor inner endpoints to the source outer
     endpoints. If `single_replacement=True`, one random pairing per key is
     chosen; otherwise, all pairings across keys are either enumerated or
     sampled up to `max_enumerations`.
- Finally, optionally compact node ids to a contiguous integer range to keep ids tidy.
7) Repeat sampling: Steps 5–6 are repeated `n_samples` times if requested.

Iterative rewrites (iterated_rewrite)
- Applies `rewrite` for `n_iterations`, feeding the chosen result graph from
  each iteration into the next. If a `feasibility_estimator` is provided, the
  batch of candidates per iteration is filtered before a random choice is made.

Determinism and sampling
- All random choices (candidate pairing, per-key permutations, sampling under
  `max_enumerations`) are driven by the provided `rng` or an internal
  `random.Random()` instance when `rng=None`.

Complexity notes
- Replacement in non-single mode can explode combinatorially: per-key
  permutations times a Cartesian product across keys. Use
  `single_replacement=True` (default) or set a `max_enumerations` cap to sample
  a manageable subset.

Key functions
- `rewrite`: perform one rewrite attempt with optional multiple samples.
- `iterated_rewrite`: apply `rewrite` iteratively, optionally filtering.

See also
- `cut_radius` and `cut_scope` in `rewrite` for signature semantics.
- `abstractgraph.graphs` and `abstractgraph.hashing` for label hashing
  and mapped-subgraph hashing used in compatibility and no-op avoidance.
"""

import random
import itertools
from collections import defaultdict, Counter
from typing import Optional, Sequence, Tuple, Callable

import numpy as np
import networkx as nx

from abstractgraph.graphs import (
    AbstractGraph,
    get_mapped_subgraph,
    graph_to_abstract_graph,
    graphs_to_abstract_graphs,
)
from abstractgraph.hashing import hash_graph, hash_value


def rewrite(
    source: nx.Graph,
    donors: Sequence[nx.Graph],
    *,
    rng: Optional[random.Random] = None,
    decomposition_function=None,
    nbits: int = 10,
    n_samples: int = 1,
    donor_ags: Optional[Sequence[AbstractGraph]] = None,
    source_ag: Optional[AbstractGraph] = None,
    donors_index: Optional[dict] = None,
    cut_radius: Optional[int] = None,
    cut_scope: str = "both",
    cut_include_edge_label: bool = True,
    cut_context_radius: Optional[int] = None,
    context_vectorizer=None,
    use_context_embedding: bool = False,
    context_temperature: float = 1.0,
    context_top_k: Optional[int] = None,
    single_replacement: bool = True,
    max_enumerations: Optional[int] = None,
    preserve_node_ids: bool = False,
    replace_with_smaller_or_equal_size: bool = False,
) -> Sequence[nx.Graph]:
    """
    Perform a single random swap on source using interpretation nodes from donors.

    Args:
        source: Input graph to mutate via swaps.
        donors: Sequence of input graphs providing candidate interpretation nodes.
        rng: Optional Random instance for deterministic sampling.
        decomposition_function: AbstractGraph decomposition function to build interpretation nodes.
        nbits: Hash bit width used by graph_to_abstract_graph.
        n_samples: Number of rewrites to attempt.
        donor_ags: Optional cached AbstractGraph donors matching the donors sequence.
        source_ag: Optional cached AbstractGraph for the source graph.
        donors_index: Optional cached donor cut-index for donor_ags (must match
            cut_radius, cut_scope, and cut_include_edge_label).
        cut_radius: None for edge-label-only cuts; 0 for endpoint labels; n>0 for
            hashes of inside/outside induced subgraphs within n hops.
        cut_scope: Which endpoint neighborhoods to include in the cut signature.
            "both" includes inside and outside neighborhoods; "inner" uses only
            the inner endpoint; "outer" uses only the outer endpoint. Aliases
            "internal"/"external" and "inside"/"outside" are accepted.
        cut_include_edge_label: If False, omit edge label from the per-edge key.
        cut_context_radius: Neighborhood radius for context embeddings. If None,
            uses the full inner/outer subgraphs for context.
        context_vectorizer: Optional transformer with a .transform(graphs)
            method returning vector embeddings per graph.
        use_context_embedding: If True and context_vectorizer is provided,
            prefer cuts whose inner/outer context embeddings are more similar
            (cosine similarity).
        context_temperature: Selection temperature for context-aware sampling.
            T=1 selects uniformly at random; T=0 deterministically selects top-k.
        context_top_k: Number of top-scoring candidates to return when
            context_temperature=0. Defaults to n_samples.
        single_replacement: If True, pick one random pairing per label. If False,
            enumerate all or sample up to max_enumerations pairings.
        max_enumerations: Cap on enumerated/sampled pairings when single_replacement=False;
            if None, enumerate all combinations.
        preserve_node_ids: If True, do not relabel nodes to a contiguous range.
        replace_with_smaller_or_equal_size: If True, only allow donor associations
            whose node count is less than or equal to the source mapped subgraph's
            node count (prevents growing the replaced part).

    Returns:
        A list of graphs with swaps applied when possible.
    """
    if decomposition_function is None:
        raise ValueError("decomposition_function is required to build interpretation nodes.")
    rng = rng or random.Random()
    if not donors:
        return [source.copy() for _ in range(n_samples)]
    if source_ag is None:
        source_ag = graph_to_abstract_graph(
            source,
            decomposition_function=decomposition_function,
            nbits=nbits,
        )
    if donor_ags is None:
        donor_ags = graphs_to_abstract_graphs(
            donors,
            decomposition_function=decomposition_function,
            nbits=nbits,
        )
    cut_scope = _normalize_cut_scope(cut_scope)
    use_context = bool(use_context_embedding and context_vectorizer is not None)
    if donors_index is None or (use_context and not _cut_index_has_context(donors_index)):
        donors_index = _build_cut_index(
            donor_ags,
            cut_radius=cut_radius,
            cut_scope=cut_scope,
            cut_include_edge_label=cut_include_edge_label,
            cut_context_radius=cut_context_radius,
            context_vectorizer=context_vectorizer,
            use_context_embedding=use_context,
        )
    source_index = _build_source_cut_index(
        source_ag,
        cut_radius=cut_radius,
        cut_scope=cut_scope,
        cut_include_edge_label=cut_include_edge_label,
        cut_context_radius=cut_context_radius,
        context_vectorizer=context_vectorizer,
        use_context_embedding=use_context,
    )
    common_keys = list(set(source_index) & set(donors_index))
    if not common_keys:
        return [source.copy() for _ in range(n_samples)]

    outputs = []
    if use_context:
        candidate_pairs = []
        scores = []
        freqs = []
        freqs = []
        for cut_key in common_keys:
            source_candidates = source_index.get(cut_key)
            donor_candidates = donors_index.get(cut_key)
            if not source_candidates or not donor_candidates:
                continue
            for source_candidate in source_candidates:
                (
                    _source_node,
                    source_assoc,
                    _source_inner,
                    _source_cut,
                    _source_edge_map,
                    source_outer_ctx,
                    source_inner_ctx,
                ) = _unpack_source_entry(source_candidate)
                source_hash = hash_graph(source_assoc)
                filtered_donors = _filter_donor_candidates(donor_candidates, source_hash)
                if not filtered_donors:
                    continue
                for donor_entry in filtered_donors:
                    (
                        _donor_assoc,
                        _donor_cut,
                        _donor_edge_map,
                        _donor_hash,
                        donor_outer_ctx,
                        donor_inner_ctx,
                    ) = _unpack_donor_entry(donor_entry)
                    # Enforce donor size <= source size if requested
                    if replace_with_smaller_or_equal_size:
                        try:
                            donor_size = len(_donor_assoc)
                        except Exception:
                            donor_size = len(set(_donor_assoc.nodes()))
                        source_size = len(set(_source_inner)) if _source_inner is not None else 0
                        if donor_size > source_size:
                            continue
                    sim = _context_similarity(
                        source_outer_ctx,
                        source_inner_ctx,
                        donor_outer_ctx,
                        donor_inner_ctx,
                        cut_scope=cut_scope,
                    )
                    score = -np.inf if sim is None else float(sim)
                    candidate_pairs.append((source_candidate, donor_entry))
                    scores.append(score)
                    freqs.append(_entry_frequency(donor_entry))
        if not candidate_pairs:
            return [source.copy() for _ in range(n_samples)]
        if context_temperature <= 0.0:
            k = n_samples if context_top_k is None else int(context_top_k)
            for idx in _top_k_indices(scores, k):
                source_candidate, donor_entry = candidate_pairs[idx]
                (
                    _source_node,
                    _source_assoc,
                    source_inner,
                    source_cut,
                    source_edge_map,
                    _source_outer_ctx,
                    _source_inner_ctx,
                ) = _unpack_source_entry(source_candidate)
                (
                    donor_assoc,
                    donor_cut,
                    donor_edge_map,
                    _donor_hash,
                    _donor_outer_ctx,
                    _donor_inner_ctx,
                ) = _unpack_donor_entry(donor_entry)
                donor_inner = set(donor_assoc.nodes())
                if not donor_inner:
                    continue
                new_graphs = _replace_subgraph(
                    source,
                    source_inner,
                    source_cut,
                    source_edge_map,
                    donor_assoc,
                    donor_inner,
                    donor_cut,
                    donor_edge_map,
                    rng,
                    single_replacement=single_replacement,
                    max_enumerations=max_enumerations,
                    preserve_node_ids=preserve_node_ids,
                )
                if not new_graphs:
                    continue
                if single_replacement:
                    new_graphs[0].graph["context_score"] = scores[idx]
                    outputs.append(new_graphs[0])
                else:
                    for g in new_graphs:
                        g.graph["context_score"] = scores[idx]
                    outputs.extend(new_graphs)
                if len(outputs) >= max(1, int(n_samples)):
                    break
            return outputs
        weights = _temperature_weights(scores, context_temperature)
        weights = weights * np.asarray(freqs, dtype=float)
        for _ in range(n_samples):
            available_pairs = list(candidate_pairs)
            available_weights = list(weights)
            available_scores = list(scores)
            while available_pairs:
                total_weight = sum(available_weights)
                if total_weight > 0:
                    idx = rng.choices(
                        range(len(available_pairs)),
                        weights=available_weights,
                        k=1,
                    )[0]
                else:
                    idx = rng.randrange(len(available_pairs))
                source_candidate, donor_entry = available_pairs.pop(idx)
                available_weights.pop(idx)
                score = available_scores.pop(idx)
                (
                    _source_node,
                    _source_assoc,
                    source_inner,
                    source_cut,
                    source_edge_map,
                    _source_outer_ctx,
                    _source_inner_ctx,
                ) = _unpack_source_entry(source_candidate)
                (
                    donor_assoc,
                    donor_cut,
                    donor_edge_map,
                    _donor_hash,
                    _donor_outer_ctx,
                    _donor_inner_ctx,
                ) = _unpack_donor_entry(donor_entry)
                donor_inner = set(donor_assoc.nodes())
                if not donor_inner:
                    continue
                new_graphs = _replace_subgraph(
                    source,
                    source_inner,
                    source_cut,
                    source_edge_map,
                    donor_assoc,
                    donor_inner,
                    donor_cut,
                    donor_edge_map,
                    rng,
                    single_replacement=single_replacement,
                    max_enumerations=max_enumerations,
                    preserve_node_ids=preserve_node_ids,
                )
                if not new_graphs:
                    continue
                if single_replacement:
                    new_graphs[0].graph["context_score"] = score
                    outputs.append(new_graphs[0])
                else:
                    for g in new_graphs:
                        g.graph["context_score"] = score
                    outputs.extend(new_graphs)
                break
        return outputs
    for _ in range(n_samples):
        rng.shuffle(common_keys)
        for cut_key in common_keys:
            source_candidates = source_index.get(cut_key)
            donor_candidates = donors_index.get(cut_key)
            if not source_candidates or not donor_candidates:
                continue
            paired_candidates = []
            for source_candidate in source_candidates:
                (
                    _source_node,
                    source_assoc,
                    _source_inner,
                    _source_cut,
                    _source_edge_map,
                    _source_outer_ctx,
                    _source_inner_ctx,
                ) = _unpack_source_entry(source_candidate)
                source_hash = hash_graph(source_assoc)
                filtered_donors = _filter_donor_candidates(donor_candidates, source_hash)
                if filtered_donors:
                    paired_candidates.append((source_candidate, filtered_donors))
            if not paired_candidates:
                continue
            (source_candidate, donor_candidates) = rng.choice(paired_candidates)
            (
                _source_node,
                _source_assoc,
                source_inner,
                source_cut,
                source_edge_map,
                _source_outer_ctx,
                _source_inner_ctx,
            ) = _unpack_source_entry(source_candidate)
            # Optionally filter donors by size relative to source inner
            if replace_with_smaller_or_equal_size:
                filtered = []
                for entry in donor_candidates:
                    donor_assoc, _donor_cut, _donor_edge_map, _donor_hash, _donor_outer_ctx, _donor_inner_ctx = _unpack_donor_entry(entry)
                    try:
                        donor_size = len(donor_assoc)
                    except Exception:
                        donor_size = len(set(donor_assoc.nodes()))
                    if donor_size <= len(set(source_inner)):
                        filtered.append(entry)
                donor_candidates = filtered if filtered else []
            if not donor_candidates:
                continue
            donor_weights = [_entry_frequency(entry) for entry in donor_candidates]
            if sum(donor_weights) > 0:
                donor_entry = rng.choices(
                    donor_candidates,
                    weights=donor_weights,
                    k=1,
                )[0]
            else:
                donor_entry = rng.choice(donor_candidates)
            donor_assoc, donor_cut, donor_edge_map, _donor_hash, _donor_outer_ctx, _donor_inner_ctx = _unpack_donor_entry(donor_entry)
            donor_inner = set(donor_assoc.nodes())
            if not donor_inner:
                continue
            new_graphs = _replace_subgraph(
                source,
                source_inner,
                source_cut,
                source_edge_map,
                donor_assoc,
                donor_inner,
                donor_cut,
                donor_edge_map,
                rng,
                single_replacement=single_replacement,
                max_enumerations=max_enumerations,
                preserve_node_ids=preserve_node_ids,
            )
            if not new_graphs:
                continue
            if single_replacement:
                outputs.append(new_graphs[0])
                break
            else:
                outputs.extend(new_graphs)
    return outputs


def iterated_rewrite(
    source: nx.Graph,
    donors: Sequence[nx.Graph],
    *,
    rng: Optional[random.Random] = None,
    decomposition_function=None,
    nbits: int = 10,
    n_samples: int = 1,
    n_iterations: int = 1,
    feasibility_estimator: Optional[object] = None,
    donor_ags: Optional[Sequence[AbstractGraph]] = None,
    cut_radius: Optional[int] = None,
    cut_scope: str = "both",
    cut_include_edge_label: bool = True,
    cut_context_radius: Optional[int] = None,
    context_vectorizer=None,
    use_context_embedding: bool = True,
    context_temperature: float = 1.0,
    context_top_k: Optional[int] = None,
    single_replacement: bool = True,
    max_enumerations: Optional[int] = None,
    preserve_node_ids: bool = False,
) -> Sequence[nx.Graph]:
    """
    Apply rewrite() iteratively, optionally filtering results.

    Args:
        source: Input graph to rewrite.
        donors: Candidate donor graphs to rewrite from.
        rng: Optional Random instance for deterministic sampling.
        decomposition_function: AbstractGraph decomposition function to build interpretation nodes.
        nbits: Hash bit width used by graph_to_abstract_graph.
        n_samples: Number of candidate rewrites per iteration.
        n_iterations: Number of rewrite iterations to run.
        feasibility_estimator: Optional estimator to filter infeasible graphs.
        donor_ags: Optional cached AbstractGraph donors matching the donors sequence.
        cut_radius: Cut signature radius; see rewrite for semantics.
        cut_scope: Which endpoint neighborhoods to include in the cut signature;
            "both", "inner", or "outer".
        cut_include_edge_label: If False, omit edge label from the per-edge key.
        cut_context_radius: Neighborhood radius for context embeddings. If None,
            uses the full inner/outer subgraphs for context.
        context_vectorizer: Optional transformer with a .transform(graphs)
            method returning vector embeddings per graph.
        use_context_embedding: If True and context_vectorizer is provided,
            prefer cuts whose inner/outer context embeddings are more similar
            (cosine similarity).
        context_temperature: Selection temperature for context-aware sampling.
            T=1 selects uniformly at random; T=0 deterministically selects top-k.
        context_top_k: Number of top-scoring candidates to return when
            context_temperature=0. Defaults to n_samples.
        single_replacement: If True, pick one random pairing per label. If False,
            enumerate all or sample up to max_enumerations pairings.
        max_enumerations: Cap on enumerated/sampled pairings when single_replacement=False;
            if None, enumerate all combinations.
        preserve_node_ids: If True, do not relabel nodes to a contiguous range.

    Returns:
        A list of rewritten graphs, one per iteration.
    """
    rng = rng or random.Random()
    rewritten_graphs = []
    current = source
    if donor_ags is None and donors and decomposition_function is not None:
        donor_ags = graphs_to_abstract_graphs(
            donors,
            decomposition_function=decomposition_function,
            nbits=nbits,
        )
    cut_scope = _normalize_cut_scope(cut_scope)
    donors_index = (
        _build_cut_index(
            donor_ags,
            cut_radius=cut_radius,
            cut_scope=cut_scope,
            cut_include_edge_label=cut_include_edge_label,
            cut_context_radius=cut_context_radius,
            context_vectorizer=context_vectorizer,
            use_context_embedding=use_context_embedding,
        )
        if donor_ags is not None
        else None
    )
    for _ in range(n_iterations):
        batch = rewrite(
            current,
            donors,
            rng=rng,
            decomposition_function=decomposition_function,
            nbits=nbits,
            n_samples=n_samples,
            donor_ags=donor_ags,
            cut_radius=cut_radius,
            cut_scope=cut_scope,
            cut_include_edge_label=cut_include_edge_label,
            cut_context_radius=cut_context_radius,
            context_vectorizer=context_vectorizer,
            use_context_embedding=use_context_embedding,
            context_temperature=context_temperature,
            context_top_k=context_top_k,
            donors_index=donors_index,
            single_replacement=single_replacement,
            max_enumerations=max_enumerations,
            preserve_node_ids=preserve_node_ids,
        )
        if feasibility_estimator is not None:
            batch = feasibility_estimator.filter(batch)
        if batch:
            chosen = rng.choice(batch)
            rewritten_graphs.append(chosen)
            current = chosen
        else:
            rewritten_graphs.append(current)
    return rewritten_graphs


def _cut_edges(graph: nx.Graph, inner_nodes) -> list[Tuple[Tuple[int, int], dict]]:
    """Return cut edges and attributes between inner_nodes and the complement."""
    cut = []
    for u in inner_nodes:
        for v in graph.neighbors(u):
            if v in inner_nodes:
                continue
            cut.append(((u, v), graph.edges[u, v]))
    return cut


def _normalize_cut_scope(cut_scope: Optional[str]) -> str:
    """Normalize cut scope string and validate accepted values."""
    if cut_scope is None:
        return "both"
    normalized = cut_scope.lower()
    if normalized in {"both", "inner", "outer"}:
        return normalized
    if normalized in {"internal", "inside"}:
        return "inner"
    if normalized in {"external", "outside"}:
        return "outer"
    raise ValueError(f"Unsupported cut_scope={cut_scope!r}. Use 'both', 'inner', or 'outer'.")


def _radius_neighborhood_graph(
    graph: nx.Graph,
    nodes_subset: set,
    root,
    radius: Optional[int],
) -> nx.Graph:
    """Return the induced neighborhood subgraph within a radius.

    Args:
        graph: Source graph.
        nodes_subset: Nodes defining the induced subgraph (inner or outer).
        root: Root node for radius-limited neighborhood.
        radius: Hop radius for the neighborhood; None means unlimited.

    Returns:
        Neighborhood subgraph copy restricted to the requested radius.
    """
    sub = graph.subgraph(nodes_subset)
    if radius is None:
        return sub.copy()
    if radius < 0:
        return nx.Graph()
    lengths = nx.single_source_shortest_path_length(sub, root, cutoff=radius)
    nodes = set(lengths.keys())
    return sub.subgraph(nodes).copy()


def _as_dense_matrix(vectors) -> Optional[np.ndarray]:
    """Coerce vector outputs into a 2D dense numpy array.

    Args:
        vectors: Output from a vectorizer transform call.

    Returns:
        Dense 2D numpy array, or None if coercion fails.
    """
    try:
        if hasattr(vectors, "toarray"):
            vectors = vectors.toarray()
        if isinstance(vectors, list):
            rows = []
            for v in vectors:
                if hasattr(v, "toarray"):
                    v = v.toarray()
                v = np.asarray(v, dtype=float).ravel()
                rows.append(v)
            if not rows:
                return None
            return np.vstack(rows)
        arr = np.asarray(vectors, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        return None


def _transform_context_graphs(context_vectorizer, graphs: Sequence[nx.Graph]) -> Optional[np.ndarray]:
    """Transform graphs into a dense embedding matrix.

    Args:
        context_vectorizer: Transformer with a transform method.
        graphs: Graphs to embed.

    Returns:
        2D numpy array of embeddings, or None if transformation fails.
    """
    if context_vectorizer is None or not graphs:
        return None
    try:
        vectors = context_vectorizer.transform(list(graphs))
    except Exception:
        try:
            vectors = [context_vectorizer.transform(g) for g in graphs]
        except Exception:
            return None
    return _as_dense_matrix(vectors)


def select_most_similar_connected_components(
    graph_vectorizer,
    sample_graphs: Sequence[nx.Graph],
    generator_graphs: Sequence[nx.Graph],
) -> list[nx.Graph]:
    """Select connected components most similar to generator graphs.

    Args:
        graph_vectorizer: Transformer with a transform method producing embeddings.
        sample_graphs: Sequence of generated graphs to refine.
        generator_graphs: Sequence of generator graphs used as similarity targets.

    Returns:
        List of connected-component graphs, one per sample, chosen to maximize
        cosine similarity to any generator embedding. Falls back to the largest
        component when embeddings are unavailable.
    """
    if graph_vectorizer is None:
        raise ValueError("graph_vectorizer is required for component selection.")
    if not sample_graphs:
        return []
    generator_features = _transform_context_graphs(graph_vectorizer, generator_graphs)
    if generator_features is None or not generator_graphs:
        outputs = []
        for graph in sample_graphs:
            if graph.number_of_nodes() == 0:
                outputs.append(graph.copy())
                continue
            comps = list(nx.connected_components(graph.to_undirected()))
            largest = max(comps, key=len) if comps else []
            outputs.append(graph.subgraph(largest).copy())
        return outputs
    if generator_features.ndim == 1:
        generator_features = generator_features.reshape(1, -1)
    g_norms = np.linalg.norm(generator_features, axis=1, keepdims=True)
    g_norms[g_norms == 0.0] = 1.0
    generator_unit = generator_features / g_norms

    outputs: list[nx.Graph] = []
    for graph in sample_graphs:
        if graph.number_of_nodes() == 0:
            outputs.append(graph.copy())
            continue
        comps = list(nx.connected_components(graph.to_undirected()))
        if not comps:
            outputs.append(graph.copy())
            continue
        cc_graphs = [graph.subgraph(nodes).copy() for nodes in comps]
        if len(cc_graphs) == 1:
            outputs.append(cc_graphs[0])
            continue
        comp_features = _transform_context_graphs(graph_vectorizer, cc_graphs)
        if comp_features is None:
            largest = max(cc_graphs, key=lambda g: g.number_of_nodes())
            outputs.append(largest)
            continue
        if comp_features.ndim == 1:
            comp_features = comp_features.reshape(1, -1)
        if comp_features.shape[1] != generator_unit.shape[1]:
            largest = max(cc_graphs, key=lambda g: g.number_of_nodes())
            outputs.append(largest)
            continue
        c_norms = np.linalg.norm(comp_features, axis=1, keepdims=True)
        c_norms[c_norms == 0.0] = 1.0
        comp_unit = comp_features / c_norms
        sims = comp_unit @ generator_unit.T
        best_idx = int(np.argmax(sims.max(axis=1)))
        outputs.append(cc_graphs[best_idx])
    return outputs


def _aggregate_context_embeddings(
    graph: nx.Graph,
    nodes_subset: set,
    node_counts: Counter,
    radius: Optional[int],
    context_vectorizer,
) -> Optional[np.ndarray]:
    """Sum context embeddings over nodes, weighted by multiplicity.

    Args:
        graph: Source graph.
        nodes_subset: Nodes allowed in the neighborhood subgraphs.
        node_counts: Counter of how many times each node is used.
        radius: Hop radius for context subgraphs.
        context_vectorizer: Transformer with a transform method.

    Returns:
        Aggregated embedding vector, or None if unavailable.
    """
    if not node_counts:
        return None
    nodes = list(node_counts.keys())
    graphs = [
        _radius_neighborhood_graph(graph, nodes_subset, node, radius)
        for node in nodes
    ]
    vectors = _transform_context_graphs(context_vectorizer, graphs)
    if vectors is None:
        return None
    if vectors.shape[0] != len(nodes):
        return None
    weights = np.asarray([node_counts[node] for node in nodes], dtype=float)
    weighted = vectors * weights.reshape(-1, 1)
    return weighted.sum(axis=0)


def _context_embeddings_for_cut(
    graph: nx.Graph,
    inner_nodes: set,
    cut_edges,
    *,
    cut_context_radius: Optional[int],
    context_vectorizer,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute outer/inner context embeddings for a cut.

    Args:
        graph: Source graph containing inner and outer nodes.
        inner_nodes: Nodes defining the inner subgraph.
        cut_edges: Cut edges between inner and outer nodes.
        cut_context_radius: Hop radius for context subgraphs.
        context_vectorizer: Transformer with a transform method.

    Returns:
        Tuple of (outer_embedding, inner_embedding).
    """
    if context_vectorizer is None:
        return None, None
    inner_counts: Counter = Counter()
    outer_counts: Counter = Counter()
    outer_nodes = set(graph.nodes()) - set(inner_nodes)
    for (u, v), _attr in cut_edges:
        if u in inner_nodes and v in outer_nodes:
            inner_counts[u] += 1
            outer_counts[v] += 1
        elif v in inner_nodes and u in outer_nodes:
            inner_counts[v] += 1
            outer_counts[u] += 1
    outer_embedding = _aggregate_context_embeddings(
        graph,
        outer_nodes,
        outer_counts,
        cut_context_radius,
        context_vectorizer,
    )
    inner_embedding = _aggregate_context_embeddings(
        graph,
        set(inner_nodes),
        inner_counts,
        cut_context_radius,
        context_vectorizer,
    )
    return outer_embedding, inner_embedding


def _context_embeddings_for_virtual_cut(
    graph: nx.Graph,
    cut_nodes: Sequence,
    *,
    cut_context_radius: Optional[int],
    context_vectorizer,
) -> Optional[np.ndarray]:
    """Compute outer context embedding for a virtual cut.

    Args:
        graph: Source graph used for outer neighborhoods.
        cut_nodes: Nodes defining the virtual cut.
        cut_context_radius: Hop radius for context subgraphs.
        context_vectorizer: Transformer with a transform method.

    Returns:
        Aggregated outer context embedding, or None if unavailable.
    """
    if context_vectorizer is None or not cut_nodes:
        return None
    outer_nodes = set(graph.nodes())
    node_counts = Counter(cut_nodes)
    return _aggregate_context_embeddings(
        graph,
        outer_nodes,
        node_counts,
        cut_context_radius,
        context_vectorizer,
    )


def _cosine_similarity(vec_a, vec_b) -> Optional[float]:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity, or None if invalid.
    """
    if vec_a is None or vec_b is None:
        return None
    a = np.asarray(vec_a, dtype=float).ravel()
    b = np.asarray(vec_b, dtype=float).ravel()
    if a.shape != b.shape:
        return None
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _as_vector_list(vecs) -> Optional[list]:
    """Normalize a container into a list of vectors when appropriate.

    Args:
        vecs: Vector or sequence/array of vectors.

    Returns:
        List of vectors if `vecs` is a sequence of vectors, empty list if it is
        an empty sequence, or None if `vecs` is a single vector.
    """
    if vecs is None:
        return None
    if isinstance(vecs, np.ndarray):
        if vecs.ndim > 1:
            return list(vecs)
        return None
    if isinstance(vecs, (list, tuple)):
        if not vecs:
            return []
        first = vecs[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return list(vecs)
    return None


def _max_cosine_similarity(source_vec, donor_vecs) -> Optional[float]:
    """Return the maximum cosine similarity over one or more donor vectors.

    Args:
        source_vec: Source embedding vector.
        donor_vecs: Single vector or a sequence/array of candidate vectors.

    Returns:
        Maximum cosine similarity, or None if not computable.
    """
    candidates = _as_vector_list(donor_vecs)
    if candidates is None:
        return _cosine_similarity(source_vec, donor_vecs)
    if not candidates:
        return None
    sims = []
    for vec in candidates:
        sim = _cosine_similarity(source_vec, vec)
        if sim is not None:
            sims.append(sim)
    if not sims:
        return None
    return max(sims)


def _context_similarity(
    source_outer,
    source_inner,
    donor_outer,
    donor_inner,
    *,
    cut_scope: str,
) -> Optional[float]:
    """Compute average context similarity according to cut_scope.

    Args:
        source_outer: Source outer embedding.
        source_inner: Source inner embedding.
        donor_outer: Donor outer embedding.
        donor_inner: Donor inner embedding.
        cut_scope: Which side(s) to include in the similarity.

    Returns:
        Average cosine similarity, or None if not computable.
    """
    if cut_scope == "both":
        outer_list = _as_vector_list(donor_outer)
        inner_list = _as_vector_list(donor_inner)
        if outer_list is not None and inner_list is not None:
            if not outer_list or not inner_list:
                return None
            if len(outer_list) == len(inner_list):
                best = None
                for outer_vec, inner_vec in zip(outer_list, inner_list):
                    sim_outer = _cosine_similarity(source_outer, outer_vec)
                    sim_inner = _cosine_similarity(source_inner, inner_vec)
                    if sim_outer is None or sim_inner is None:
                        continue
                    sim = 0.5 * (sim_outer + sim_inner)
                    if best is None or sim > best:
                        best = sim
                return best
    sims = []
    if cut_scope in {"outer", "both"}:
        sim = _max_cosine_similarity(source_outer, donor_outer)
        if sim is None:
            return None
        sims.append(sim)
    if cut_scope in {"inner", "both"}:
        sim = _max_cosine_similarity(source_inner, donor_inner)
        if sim is None:
            return None
        sims.append(sim)
    if not sims:
        return None
    return float(sum(sims) / len(sims))


def _unpack_donor_entry(entry):
    """Normalize donor entry tuples to a consistent schema.

    Args:
        entry: Donor entry tuple.

    Returns:
        Tuple of (assoc, donor_cut, donor_edge_map, assoc_hash, outer_ctx, inner_ctx).
    """
    if len(entry) == 4:
        assoc, donor_cut, donor_edge_map, assoc_hash = entry
        outer_ctx = None
        inner_ctx = None
    elif len(entry) == 5:
        assoc, donor_cut, donor_edge_map, assoc_hash, outer_ctx = entry
        inner_ctx = None
    elif len(entry) == 6:
        assoc, donor_cut, donor_edge_map, assoc_hash, outer_ctx, inner_ctx = entry
    else:
        raise ValueError(f"Unsupported donor entry format (len={len(entry)}).")
    return assoc, donor_cut, donor_edge_map, assoc_hash, outer_ctx, inner_ctx


def _entry_frequency(entry) -> float:
    """Return a non-negative frequency weight from a donor entry."""
    assoc, _donor_cut, _donor_edge_map, _assoc_hash, _outer_ctx, _inner_ctx = _unpack_donor_entry(entry)
    freq = assoc.graph.get("frequency", 1)
    try:
        freq_val = float(freq)
    except Exception:
        return 1.0
    if not np.isfinite(freq_val) or freq_val <= 0:
        return 1.0
    return freq_val


def _unpack_source_entry(entry):
    """Normalize source entry tuples to a consistent schema.

    Args:
        entry: Source entry tuple.

    Returns:
        Tuple of (node_id, assoc, inner_nodes, source_cut, source_edge_map, outer_ctx, inner_ctx).
    """
    if len(entry) == 5:
        node_id, assoc, inner_nodes, source_cut, source_edge_map = entry
        outer_ctx = None
        inner_ctx = None
    elif len(entry) == 7:
        node_id, assoc, inner_nodes, source_cut, source_edge_map, outer_ctx, inner_ctx = entry
    else:
        raise ValueError(f"Unsupported source entry format (len={len(entry)}).")
    return node_id, assoc, inner_nodes, source_cut, source_edge_map, outer_ctx, inner_ctx


def _iter_cut_index_entries(cut_index: dict):
    """Yield entry lists from a (possibly nested) cut index.

    Args:
        cut_index: Cut index mapping cut keys to entry lists or nested dicts.

    Returns:
        Generator of entry lists.
    """
    if not cut_index:
        return
    for value in cut_index.values():
        if isinstance(value, dict):
            yield from _iter_cut_index_entries(value)
        else:
            yield value


def _iter_cut_index_items(cut_index: dict):
    """Yield (cut_key, entries) pairs from a (possibly nested) cut index.

    Args:
        cut_index: Cut index mapping cut keys to entry lists or nested dicts.

    Returns:
        Generator of (cut_key, entries) pairs.
    """
    if not cut_index:
        return
    for key, value in cut_index.items():
        if isinstance(value, dict):
            yield from _iter_cut_index_items(value)
        else:
            yield key, value


def _cut_index_has_context(cut_index: Optional[dict]) -> bool:
    """Return True if a cut index entry appears to include context embeddings.

    Args:
        cut_index: Cut dictionary to inspect.

    Returns:
        True if entries include context embeddings.
    """
    if not cut_index:
        return False
    for entries in _iter_cut_index_entries(cut_index):
        if entries:
            return len(entries[0]) >= 6
    return False


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


def _outer_node_key(
    graph: nx.Graph,
    outer_nodes: set,
    node,
    cut_radius: Optional[int],
) -> Tuple:
    """Return the per-node key used for virtual cut signatures.

    Args:
        graph: Source graph used for neighborhood hashing.
        outer_nodes: Nodes defining the outer subgraph.
        node: Node to encode in the cut signature.
        cut_radius: Neighborhood radius for subgraph-hash signatures.

    Returns:
        Tuple key that represents the node in a cut signature.
    """
    if cut_radius is None:
        return tuple()
    return (_radius_subgraph_hash(graph, outer_nodes, node, cut_radius),)


def _precompute_outer_node_keys(graph: nx.Graph, cut_radius: Optional[int]) -> dict:
    """Precompute per-node keys for virtual cut signatures.

    Args:
        graph: Source graph used for outer neighborhoods.
        cut_radius: Neighborhood radius for subgraph-hash signatures.

    Returns:
        Dictionary mapping node id to its cut signature key.
    """
    outer_nodes = set(graph.nodes())
    return {
        node: _outer_node_key(graph, outer_nodes, node, cut_radius)
        for node in graph.nodes()
    }


def _radius_subgraph_hash(graph: nx.Graph, nodes_subset: set, root, radius: int) -> int:
    """Return hash of the induced subgraph within radius from root in subgraph.

    Args:
        graph: Source graph.
        nodes_subset: Nodes defining the induced subgraph (inner or outer).
        root: Root node for radius-limited neighborhood.
        radius: Hop radius for the neighborhood.

    Returns:
        Hash of the induced subgraph within the radius.
    """
    if radius < 0:
        return hash_graph(nx.Graph())
    sub = graph.subgraph(nodes_subset)
    lengths = nx.single_source_shortest_path_length(sub, root, cutoff=radius)
    nodes = set(lengths.keys())
    return hash_graph(sub.subgraph(nodes).copy())


def _per_edge_key(
    graph: nx.Graph,
    inner_nodes: set,
    edge: Tuple[Tuple[int, int], dict],
    cut_radius: Optional[int],
    cut_scope: str,
    cut_include_edge_label: bool,
) -> Tuple:
    """Build per-edge context key per cut rules."""
    (u, v), attr = edge
    # Determine internal/external orientation
    if u in inner_nodes and v not in inner_nodes:
        inner, outer = u, v
    elif v in inner_nodes and u not in inner_nodes:
        inner, outer = v, u
    else:
        inner, outer = (u, v)
    base = ()
    if cut_include_edge_label:
        base = (hash_value(attr.get("label", "")),)
    if cut_radius is None:
        return base
    inner_hash = _radius_subgraph_hash(graph, set(inner_nodes), inner, cut_radius)
    outer_nodes = set(graph.nodes()) - set(inner_nodes)
    outer_hash = _radius_subgraph_hash(graph, outer_nodes, outer, cut_radius)
    if cut_scope == "inner":
        return base + (inner_hash,)
    if cut_scope == "outer":
        return base + (outer_hash,)
    return base + (inner_hash, outer_hash)


def _cut_signature_and_donor_map(
    graph: nx.Graph,
    inner_nodes: set,
    cut_edges,
    *,
    cut_radius: Optional[int],
    cut_scope: str,
    cut_include_edge_label: bool,
):
    """Return order-independent cut signature and donor map by per-edge key."""
    key_to_inners: dict[Tuple, list] = defaultdict(list)
    per_edge_keys = []
    for edge in cut_edges:
        key = _per_edge_key(
            graph,
            inner_nodes,
            edge,
            cut_radius,
            cut_scope,
            cut_include_edge_label,
        )
        per_edge_keys.append(key)
        (u, v), attr = edge
        inner = u if u in inner_nodes else v
        key_to_inners[key].append((inner, dict(attr)))
    cut_key = tuple(sorted(per_edge_keys))
    return cut_key, key_to_inners


def _cut_signature_and_source_map(
    graph: nx.Graph,
    inner_nodes: set,
    cut_edges,
    *,
    cut_radius: Optional[int],
    cut_scope: str,
    cut_include_edge_label: bool,
):
    """Return order-independent cut signature and source map by per-edge key."""
    key_to_outers: dict[Tuple, list] = defaultdict(list)
    per_edge_keys = []
    for (u, v), attr in cut_edges:
        key = _per_edge_key(
            graph,
            inner_nodes,
            ((u, v), attr),
            cut_radius,
            cut_scope,
            cut_include_edge_label,
        )
        per_edge_keys.append(key)
        outer = v if u in inner_nodes else u
        key_to_outers[key].append((outer, attr))
    cut_key = tuple(sorted(per_edge_keys))
    return cut_key, key_to_outers


def _build_cut_index(
    donors: Sequence[AbstractGraph],
    *,
    cut_radius: Optional[int] = None,
    cut_scope: str = "both",
    cut_include_edge_label: bool = True,
    cut_context_radius: Optional[int] = None,
    context_vectorizer=None,
    use_context_embedding: bool = True,
):
    """Index donor associations by cut signature.

    Args:
        donors: Sequence of AbstractGraph donors.
        cut_radius: Neighborhood radius for compatibility signatures.
        cut_scope: Which endpoint neighborhoods to include in the signature.
        cut_include_edge_label: If False, omit edge labels from signatures.
        cut_context_radius: Neighborhood radius for context embeddings.
        context_vectorizer: Transformer with a transform method.
        use_context_embedding: If True and context_vectorizer is provided,
            compute and store context embeddings.

    Returns:
        Dictionary mapping cut keys to donor entries.
    """
    cut_index = defaultdict(list)
    use_context = bool(use_context_embedding and context_vectorizer is not None)
    for donor in donors:
        for node_id, data in donor.interpretation_graph.nodes(data=True):
            assoc = get_mapped_subgraph(data)
            if assoc is None:
                continue
            inner_nodes = set(assoc.nodes())
            if not inner_nodes:
                continue
            donor_cut = _cut_edges(donor.base_graph, inner_nodes)
            cut_key, donor_edge_map = _cut_signature_and_donor_map(
                donor.base_graph,
                inner_nodes,
                donor_cut,
                cut_radius=cut_radius,
                cut_scope=cut_scope,
                cut_include_edge_label=cut_include_edge_label,
            )
            assoc_hash = hash_graph(assoc)
            if use_context:
                outer_ctx, inner_ctx = _context_embeddings_for_cut(
                    donor.base_graph,
                    inner_nodes,
                    donor_cut,
                    cut_context_radius=cut_context_radius,
                    context_vectorizer=context_vectorizer,
                )
                cut_index[cut_key].append(
                    (assoc.copy(), donor_cut, donor_edge_map, assoc_hash, outer_ctx, inner_ctx)
                )
            else:
                cut_index[cut_key].append((assoc.copy(), donor_cut, donor_edge_map, assoc_hash))
    return cut_index


def _build_source_cut_index(
    source: AbstractGraph,
    *,
    cut_radius: Optional[int] = None,
    cut_scope: str = "both",
    cut_include_edge_label: bool = True,
    cut_context_radius: Optional[int] = None,
    context_vectorizer=None,
    use_context_embedding: bool = True,
):
    """Index source mapped subgraphs by cut signature.

    Args:
        source: AbstractGraph source.
        cut_radius: Neighborhood radius for compatibility signatures.
        cut_scope: Which endpoint neighborhoods to include in the signature.
        cut_include_edge_label: If False, omit edge labels from signatures.
        cut_context_radius: Neighborhood radius for context embeddings.
        context_vectorizer: Transformer with a transform method.
        use_context_embedding: If True and context_vectorizer is provided,
            compute and store context embeddings.

    Returns:
        Dictionary mapping cut keys to source entries.
    """
    source_index = defaultdict(list)
    use_context = bool(use_context_embedding and context_vectorizer is not None)
    for node_id, data in source.interpretation_graph.nodes(data=True):
        assoc = get_mapped_subgraph(data)
        if assoc is None:
            continue
        inner_nodes = set(assoc.nodes())
        if not inner_nodes:
            continue
        source_cut = _cut_edges(source.base_graph, inner_nodes)
        cut_key, source_edge_map = _cut_signature_and_source_map(
            source.base_graph,
            inner_nodes,
            source_cut,
            cut_radius=cut_radius,
            cut_scope=cut_scope,
            cut_include_edge_label=cut_include_edge_label,
        )
        if use_context:
            outer_ctx, inner_ctx = _context_embeddings_for_cut(
                source.base_graph,
                inner_nodes,
                source_cut,
                cut_context_radius=cut_context_radius,
                context_vectorizer=context_vectorizer,
            )
            source_index[cut_key].append(
                (node_id, assoc, inner_nodes, source_cut, source_edge_map, outer_ctx, inner_ctx)
            )
        else:
            source_index[cut_key].append((node_id, assoc, inner_nodes, source_cut, source_edge_map))
    return source_index


def cut_size_distribution(cut_index: dict) -> dict[int, float]:
    """Return frequency distribution of cut sizes (by number of outer nodes).

    Args:
        cut_index: Dictionary mapping cut keys to donor entries.

    Returns:
        Dictionary mapping cut size to frequency (weighted by entry frequency).
    """
    counts: dict[int, float] = defaultdict(float)
    for cut_key, entries in _iter_cut_index_items(cut_index):
        counts[len(cut_key)] += sum(_entry_frequency(entry) for entry in entries)
    return dict(counts)


def _edge_map_signature(edge_map: dict) -> tuple:
    """Return a stable signature for a donor edge map.

    Args:
        edge_map: Mapping from per-edge key to donor inner endpoints.

    Returns:
        Tuple signature suitable for equality checks.
    """
    items = []
    for key, entries in edge_map.items():
        entry_sigs = []
        for entry in entries:
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], dict):
                node, attr = entry
            else:
                node, attr = entry, None
            entry_sigs.append(
                (
                    hash_value(node),
                    None if attr is None else hash_value(attr),
                )
            )
        entry_sigs.sort()
        items.append((hash_value(key), tuple(entry_sigs)))
    items.sort()
    return tuple(items)


def deduplicate_cut_index(cut_index: dict, *, context_dedup_epsilon: float = 1e-6) -> dict:
    """Deduplicate cut-index entries and annotate frequency on assoc graphs.

    Args:
        cut_index: Dictionary mapping cut keys to donor entries.
        context_dedup_epsilon: Minimum L2 distance between context embeddings
            to keep multiple context variants for a single donor entry.

    Returns:
        New dictionary with duplicate donor entries collapsed. Each assoc graph
        is annotated with `graph["frequency"]` equal to its occurrence count.
        Context embeddings are retained as lists of distinct variants.
    """
    if not cut_index:
        return {}

    def _iter_context_vectors(ctx):
        """Normalize context inputs into a list of vectors.

        Args:
            ctx: Context embedding or sequence of embeddings.

        Returns:
            List of embeddings (possibly empty).
        """
        if ctx is None:
            return []
        if isinstance(ctx, np.ndarray) and ctx.ndim > 1:
            return list(ctx)
        if isinstance(ctx, (list, tuple)):
            if not ctx:
                return []
            first = ctx[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                return list(ctx)
        return [ctx]

    def _append_distinct_context(target: list, ctx, epsilon: float) -> None:
        """Append a context vector if it is sufficiently different.

        Args:
            target: List of stored context vectors.
            ctx: Candidate context vector.
            epsilon: Minimum L2 distance for distinctness.

        Returns:
            None.
        """
        if ctx is None:
            return
        arr = np.asarray(ctx, dtype=float).ravel()
        if not target:
            target.append(arr)
            return
        for existing in target:
            existing_arr = np.asarray(existing, dtype=float).ravel()
            if existing_arr.shape != arr.shape:
                continue
            if np.linalg.norm(arr - existing_arr) <= epsilon:
                return
        target.append(arr)

    def _append_distinct_context_pair(outer_list: list, inner_list: list, outer_ctx, inner_ctx, epsilon: float) -> None:
        """Append paired context vectors if the pair is sufficiently different.

        Args:
            outer_list: Stored outer context vectors.
            inner_list: Stored inner context vectors.
            outer_ctx: Candidate outer context vector.
            inner_ctx: Candidate inner context vector.
            epsilon: Minimum L2 distance for distinctness.

        Returns:
            None.
        """
        if outer_ctx is None and inner_ctx is None:
            return
        if outer_ctx is None:
            _append_distinct_context(inner_list, inner_ctx, epsilon)
            return
        if inner_ctx is None:
            _append_distinct_context(outer_list, outer_ctx, epsilon)
            return
        outer_arr = np.asarray(outer_ctx, dtype=float).ravel()
        inner_arr = np.asarray(inner_ctx, dtype=float).ravel()
        if not outer_list or not inner_list:
            outer_list.append(outer_arr)
            inner_list.append(inner_arr)
            return
        for existing_outer, existing_inner in zip(outer_list, inner_list):
            existing_outer_arr = np.asarray(existing_outer, dtype=float).ravel()
            existing_inner_arr = np.asarray(existing_inner, dtype=float).ravel()
            if existing_outer_arr.shape != outer_arr.shape or existing_inner_arr.shape != inner_arr.shape:
                continue
            combined = np.concatenate([outer_arr, inner_arr])
            existing_combined = np.concatenate([existing_outer_arr, existing_inner_arr])
            if np.linalg.norm(combined - existing_combined) <= epsilon:
                return
        outer_list.append(outer_arr)
        inner_list.append(inner_arr)

    def _dedup_entries(entries: list) -> list:
        unique = {}
        epsilon = max(0.0, float(context_dedup_epsilon))
        for entry in entries:
            assoc, donor_cut, donor_edge_map, assoc_hash, outer_ctx, inner_ctx = _unpack_donor_entry(entry)
            sig = (assoc_hash, _edge_map_signature(donor_edge_map))
            if sig not in unique:
                unique[sig] = {
                    "assoc": assoc,
                    "donor_cut": donor_cut,
                    "donor_edge_map": donor_edge_map,
                    "assoc_hash": assoc_hash,
                    "outer_ctxs": [],
                    "inner_ctxs": [],
                    "count": 0,
                }
            rec = unique[sig]
            rec["count"] += 1
            outer_list = _iter_context_vectors(outer_ctx)
            inner_list = _iter_context_vectors(inner_ctx)
            if outer_list and inner_list:
                if len(outer_list) == len(inner_list):
                    for outer_vec, inner_vec in zip(outer_list, inner_list):
                        _append_distinct_context_pair(
                            rec["outer_ctxs"],
                            rec["inner_ctxs"],
                            outer_vec,
                            inner_vec,
                            epsilon,
                        )
                else:
                    for outer_vec in outer_list:
                        _append_distinct_context(rec["outer_ctxs"], outer_vec, epsilon)
                    for inner_vec in inner_list:
                        _append_distinct_context(rec["inner_ctxs"], inner_vec, epsilon)
            elif outer_list:
                for outer_vec in outer_list:
                    _append_distinct_context(rec["outer_ctxs"], outer_vec, epsilon)
            elif inner_list:
                for inner_vec in inner_list:
                    _append_distinct_context(rec["inner_ctxs"], inner_vec, epsilon)
        deduped_entries = []
        for rec in unique.values():
            assoc = rec["assoc"]
            assoc.graph["frequency"] = rec["count"]
            outer_ctx = rec["outer_ctxs"] if rec["outer_ctxs"] else None
            inner_ctx = rec["inner_ctxs"] if rec["inner_ctxs"] else None
            if outer_ctx is None and inner_ctx is None:
                entry = (assoc, rec["donor_cut"], rec["donor_edge_map"], rec["assoc_hash"])
            elif inner_ctx is None:
                entry = (
                    assoc,
                    rec["donor_cut"],
                    rec["donor_edge_map"],
                    rec["assoc_hash"],
                    outer_ctx,
                )
            else:
                entry = (
                    assoc,
                    rec["donor_cut"],
                    rec["donor_edge_map"],
                    rec["assoc_hash"],
                    outer_ctx,
                    inner_ctx,
                )
            deduped_entries.append(entry)
        return deduped_entries

    def _dedup_node(node: dict) -> dict:
        out = {}
        for key, value in node.items():
            if isinstance(value, dict):
                child = _dedup_node(value)
                if child:
                    out[key] = child
            else:
                deduped_entries = _dedup_entries(value)
                if deduped_entries:
                    out[key] = deduped_entries
        return out

    return _dedup_node(cut_index)


def merge_cut_index(dest: dict, src: dict) -> dict:
    """Merge cut-index entries in-place, supporting nested indexes.

    Args:
        dest: Destination cut index to update.
        src: Source cut index to merge in.

    Returns:
        Destination cut index with merged entries.
    """
    if not src:
        return dest
    for key, value in src.items():
        if isinstance(value, dict):
            node = dest.get(key)
            if not isinstance(node, dict):
                node = {}
                dest[key] = node
            merge_cut_index(node, value)
        else:
            dest.setdefault(key, []).extend(value)
    return dest


def sample_cut_size(cut_size_counts: dict[int, float], rng: Optional[random.Random] = None) -> Optional[int]:
    """Sample a cut size from a frequency distribution.

    Args:
        cut_size_counts: Mapping from cut size to frequency weights.
        rng: Optional random generator for reproducibility.

    Returns:
        Sampled cut size, or None if distribution is empty.
    """
    if not cut_size_counts:
        return None
    rng = rng or random.Random()
    sizes = list(cut_size_counts.keys())
    weights = [cut_size_counts[s] for s in sizes]
    return rng.choices(sizes, weights=weights, k=1)[0]


def virtual_cut_signature(
    graph: nx.Graph,
    cut_nodes: tuple,
    *,
    cut_radius: Optional[int],
    node_key_map: Optional[dict] = None,
) -> tuple:
    """Return a virtual cut signature based on outer-node neighborhoods.

    Args:
        graph: Input graph providing the outer neighborhood.
        cut_nodes: Nodes defining the virtual cut.
        cut_radius: Neighborhood radius for subgraph-hash signatures.
        node_key_map: Optional precomputed per-node cut keys.

    Returns:
        Sorted tuple of per-node keys defining the cut signature.
    """
    if node_key_map is None:
        outer_nodes = set(graph.nodes())
        keys = [_outer_node_key(graph, outer_nodes, node, cut_radius) for node in cut_nodes]
    else:
        keys = [node_key_map[node] for node in cut_nodes]
    return tuple(sorted(keys))


def virtual_cut_source_map(
    graph: nx.Graph,
    cut_nodes: tuple,
    *,
    cut_radius: Optional[int],
    node_key_map: Optional[dict] = None,
) -> dict:
    """Return source edge map for a virtual cut.

    Args:
        graph: Source graph used for outer neighborhood context.
        cut_nodes: Nodes defining the virtual cut.
        cut_radius: Neighborhood radius for subgraph-hash signatures.
        node_key_map: Optional precomputed per-node cut keys.

    Returns:
        Dictionary mapping per-node keys to lists of outer nodes.
    """
    source_edge_map = defaultdict(list)
    outer_nodes = None
    if node_key_map is None:
        outer_nodes = set(graph.nodes())
    for node in cut_nodes:
        if node_key_map is None:
            key = _outer_node_key(graph, outer_nodes, node, cut_radius)
        else:
            key = node_key_map[node]
        source_edge_map[key].append(node)
    return source_edge_map


def virtual_cut_entry(
    graph: nx.Graph,
    inner_nodes: set,
    *,
    cut_radius: Optional[int],
    cut_context_radius: Optional[int] = None,
    context_vectorizer=None,
    use_context_embedding: bool = True,
    donor_cut=None,
    node_key_map: Optional[dict] = None,
) -> Optional[tuple]:
    """Build a virtual-cut entry from an inner subgraph removal.

    Args:
        graph: Source graph containing inner and outer nodes.
        inner_nodes: Nodes to remove (inner subgraph).
        cut_radius: Neighborhood radius for subgraph-hash signatures.
        cut_context_radius: Neighborhood radius for context embeddings.
        context_vectorizer: Transformer with a transform method.
        use_context_embedding: If True and context_vectorizer is provided,
            compute and return a context embedding for the outer cut nodes.
        donor_cut: Optional precomputed cut edges.
        node_key_map: Optional precomputed per-node cut keys for `cut_radius`.

    Returns:
        Tuple of (cut_key, donor_edge_map, donor_cut, outer_ctx, inner_ctx) or
        None if invalid. Duplicate outer endpoints are ignored (first edge kept).
    """
    if donor_cut is None:
        donor_cut = _cut_edges(graph, inner_nodes)
    outer_nodes = set(graph.nodes()) - set(inner_nodes)
    outer_map = {}
    for (u, v), attr in donor_cut:
        if u in inner_nodes and v not in inner_nodes:
            inner, outer = u, v
        elif v in inner_nodes and u not in inner_nodes:
            inner, outer = v, u
        else:
            continue
        if outer in outer_map:
            continue
        outer_map[outer] = (inner, dict(attr))
    if not outer_map:
        return None
    per_node_keys = []
    donor_edge_map = defaultdict(list)
    for outer, (inner, attr) in outer_map.items():
        if node_key_map is None:
            key = _outer_node_key(graph, outer_nodes, outer, cut_radius)
        else:
            key = node_key_map[outer]
        per_node_keys.append(key)
        donor_edge_map[key].append((inner, attr))
    cut_key = tuple(sorted(per_node_keys))
    outer_ctx = None
    inner_ctx = None
    if use_context_embedding and context_vectorizer is not None:
        outer_ctx = _context_embeddings_for_virtual_cut(
            graph,
            list(outer_map.keys()),
            cut_context_radius=cut_context_radius,
            context_vectorizer=context_vectorizer,
        )
    return cut_key, donor_edge_map, donor_cut, outer_ctx, inner_ctx


def iter_virtual_cut_candidates(
    graph: nx.Graph,
    cut_index: dict,
    *,
    cut_size: int,
    cut_radius: Optional[int],
):
    """Yield (cut_nodes, cut_key, donor_candidates) for matching virtual cuts.

    Args:
        graph: Graph to sample cut node sets from.
        cut_index: Cut dictionary mapping signatures to donor entries.
        cut_size: Number of outer attachments in each virtual cut (distinct nodes).
        cut_radius: Neighborhood radius for subgraph-hash signatures.

    Yields:
        Tuples of (cut_nodes, cut_key, donor_candidates) for matching signatures.
    """
    nodes = list(graph.nodes())
    for cut_nodes in itertools.combinations(nodes, cut_size):
        cut_key = virtual_cut_signature(graph, cut_nodes, cut_radius=cut_radius)
        donor_candidates = cut_index.get(cut_key)
        if donor_candidates:
            yield cut_nodes, cut_key, donor_candidates


def _sample_cut_nodes(
    nodes: Sequence,
    cut_size: int,
    max_checks: int,
    rng: random.Random,
) -> list[tuple]:
    """Reservoir-sample cut node sets before computing cut signatures.

    Args:
        nodes: Nodes to sample from.
        cut_size: Size of each cut-node set.
        max_checks: Maximum number of cut-node sets to keep.
        rng: Random generator.

    Returns:
        List of sampled cut-node tuples.
    """
    if max_checks <= 0 or cut_size <= 0 or cut_size > len(nodes):
        return []
    reservoir: list[tuple] = []
    seen = 0
    for cut_nodes in itertools.combinations(nodes, cut_size):
        seen += 1
        if len(reservoir) < max_checks:
            reservoir.append(cut_nodes)
        else:
            idx = rng.randrange(seen)
            if idx < max_checks:
                reservoir[idx] = cut_nodes
    return reservoir


def _cut_nodes_seen_key(
    graph: nx.Graph,
    cut_nodes: tuple,
    cut_radius: Optional[int],
    node_key_map: Optional[dict] = None,
):
    """Return a per-cut key pairing stable node ids with structural node keys.

    Args:
        graph: Source graph used for node attributes and neighborhoods.
        cut_nodes: Nodes defining the virtual cut.
        cut_radius: Neighborhood radius for subgraph-hash signatures.
        node_key_map: Optional precomputed per-node cut keys.

    Returns:
        Tuple of (stable_id, node_key) pairs for the cut nodes.
    """
    outer_nodes = None
    if node_key_map is None:
        outer_nodes = set(graph.nodes())
    items = []
    for node in cut_nodes:
        data = graph.nodes[node]
        stable_id = data.get("stable_id", node)
        if node_key_map is None:
            node_key = _outer_node_key(graph, outer_nodes, node, cut_radius)
        else:
            node_key = node_key_map[node]
        items.append((stable_id, node_key))
    return tuple(items)


def virtual_rewrite_candidates(
    graph: nx.Graph,
    cut_index: dict,
    *,
    cut_size: Optional[int],
    cut_radius: Optional[int],
    cut_context_radius: Optional[int] = None,
    context_vectorizer=None,
    use_context_embedding: bool = True,
    context_temperature: float = 1.0,
    context_top_k: Optional[int] = None,
    rng: Optional[random.Random] = None,
    n_samples: int = 1,
    max_checks: Optional[int] = None,
    checks_state: Optional[dict] = None,
    cut_nodes_seen: Optional[set] = None,
    single_replacement: bool = True,
    max_enumerations: Optional[int] = None,
    preserve_node_ids: bool = False,
) -> list[nx.Graph]:
    """Attempt rewrites by inserting donors at virtual cuts and return candidates.

    Args:
        graph: Source graph to augment.
        cut_index: Cut dictionary mapping signatures to donor entries.
        cut_size: Number of nodes in the virtual cut.
        cut_radius: Neighborhood radius for subgraph-hash signatures.
        cut_context_radius: Neighborhood radius for context embeddings.
        context_vectorizer: Optional transformer with a .transform(graphs)
            method returning vector embeddings per graph.
        use_context_embedding: If True and context_vectorizer is provided,
            prefer donors by outer context similarity.
        context_temperature: Selection temperature for context-aware sampling.
            T=1 selects uniformly at random; T=0 deterministically selects top-k.
        context_top_k: Number of top-scoring candidates to return when
            context_temperature=0. Defaults to n_samples.
        rng: Optional random generator for reproducibility.
        n_samples: Number of candidate rewrites to return.
        max_checks: Optional cap on the number of cut-node sets sampled before
            computing cut signatures.
        checks_state: Optional dictionary used to track a global remaining
            budget across multiple calls. Uses keys "remaining" and "used".
        cut_nodes_seen: Optional set to record cut_nodes already evaluated.
        single_replacement: If True, pick one random pairing per label.
        max_enumerations: Cap on enumerated/sampled pairings when single_replacement=False.
        preserve_node_ids: If True, do not relabel nodes to a contiguous range.

    Returns:
        List of candidate graphs with donor subgraphs inserted.

    Notes:
        When context embeddings are enabled, a fixed-size pool of matched
        cut/donor pairs is sampled before computing context similarities. This
        avoids embedding every matched cut while preserving context ranking
        among the sampled candidates.
    """
    rng = rng or random.Random()
    outputs: list[nx.Graph] = []
    if cut_size is None:
        return outputs
    use_context = bool(use_context_embedding and context_vectorizer is not None)
    node_key_map = _precompute_outer_node_keys(graph, cut_radius)
    checks = 0
    remaining = None
    if checks_state is not None:
        checks_state.setdefault("used", 0)
        remaining = checks_state.get("remaining")
        if remaining is not None:
            max_checks = remaining if max_checks is None else min(max_checks, remaining)
    nodes = list(graph.nodes())
    cut_nodes_iter: Sequence[tuple]
    if max_checks is not None:
        cut_nodes_iter = _sample_cut_nodes(nodes, cut_size, int(max_checks), rng)
    else:
        cut_nodes_iter = itertools.combinations(nodes, cut_size)
    if use_context:
        candidate_pairs = []
        freqs = []
        seen_pairs = 0
        max_outputs = max(1, int(n_samples))
        pool_size = max_outputs
        if context_top_k is not None:
            pool_size = max(pool_size, int(context_top_k))
        for cut_nodes in cut_nodes_iter:
            seen_key = None
            if cut_nodes_seen is not None:
                seen_key = _cut_nodes_seen_key(
                    graph, cut_nodes, cut_radius, node_key_map=node_key_map
                )
                if seen_key in cut_nodes_seen:
                    continue
            checks += 1
            if cut_nodes_seen is not None:
                cut_nodes_seen.add(seen_key)
            if checks_state is not None:
                checks_state["used"] += 1
                if remaining is not None:
                    remaining = max(0, remaining - 1)
                    checks_state["remaining"] = remaining
            cut_key = virtual_cut_signature(
                graph, cut_nodes, cut_radius=cut_radius, node_key_map=node_key_map
            )
            donor_candidates = cut_index.get(cut_key)
            if not donor_candidates:
                continue
            for donor_entry in donor_candidates:
                seen_pairs += 1
                freq = _entry_frequency(donor_entry)
                if len(candidate_pairs) < pool_size:
                    candidate_pairs.append((cut_nodes, donor_entry))
                    freqs.append(freq)
                else:
                    idx = rng.randrange(seen_pairs)
                    if idx < pool_size:
                        candidate_pairs[idx] = (cut_nodes, donor_entry)
                        freqs[idx] = freq
        if not candidate_pairs:
            return outputs
        scores = []
        outer_ctx_cache: dict[tuple, Optional[np.ndarray]] = {}
        for cut_nodes, donor_entry in candidate_pairs:
            source_outer_ctx = outer_ctx_cache.get(cut_nodes)
            if source_outer_ctx is None and cut_nodes not in outer_ctx_cache:
                source_outer_ctx = _context_embeddings_for_virtual_cut(
                    graph,
                    cut_nodes,
                    cut_context_radius=cut_context_radius,
                    context_vectorizer=context_vectorizer,
                )
                outer_ctx_cache[cut_nodes] = source_outer_ctx
            (
                _donor_assoc,
                _donor_cut,
                _donor_edge_map,
                _assoc_hash,
                donor_outer_ctx,
                _donor_inner_ctx,
            ) = _unpack_donor_entry(donor_entry)
            sim = _max_cosine_similarity(source_outer_ctx, donor_outer_ctx)
            scores.append(-np.inf if sim is None else float(sim))
        if context_temperature <= 0.0:
            k = n_samples if context_top_k is None else int(context_top_k)
            for idx in _top_k_indices(scores, k):
                cut_nodes, donor_entry = candidate_pairs[idx]
                (
                    donor_assoc,
                    donor_cut,
                    donor_edge_map,
                    _assoc_hash,
                    _donor_outer_ctx,
                    _donor_inner_ctx,
                ) = _unpack_donor_entry(donor_entry)
                donor_inner = set(donor_assoc.nodes())
                if not donor_inner:
                    continue
                source_edge_map = virtual_cut_source_map(
                    graph, cut_nodes, cut_radius=cut_radius, node_key_map=node_key_map
                )
                new_graphs = _replace_subgraph(
                    graph,
                    source_inner=set(),
                    source_cut=[],
                    source_edge_map=source_edge_map,
                    donor_assoc=donor_assoc,
                    donor_inner=donor_inner,
                    donor_cut=donor_cut,
                    donor_edge_map=donor_edge_map,
                    rng=rng,
                    single_replacement=single_replacement,
                    max_enumerations=max_enumerations,
                    preserve_node_ids=preserve_node_ids,
                )
                if not new_graphs:
                    continue
                for g in new_graphs:
                    g.graph["context_score"] = scores[idx]
                outputs.extend(new_graphs[: max(1, n_samples - len(outputs))])
                if len(outputs) >= max(1, int(n_samples)):
                    break
            return outputs
        weights = _temperature_weights(scores, context_temperature)
        weights = weights * np.asarray(freqs, dtype=float)
        for _ in range(n_samples):
            available_pairs = list(candidate_pairs)
            available_weights = list(weights)
            available_scores = list(scores)
            while available_pairs:
                total_weight = sum(available_weights)
                if total_weight > 0:
                    idx = rng.choices(
                        range(len(available_pairs)),
                        weights=available_weights,
                        k=1,
                    )[0]
                else:
                    idx = rng.randrange(len(available_pairs))
                cut_nodes, donor_entry = available_pairs.pop(idx)
                available_weights.pop(idx)
                score = available_scores.pop(idx)
                (
                    donor_assoc,
                    donor_cut,
                    donor_edge_map,
                    _assoc_hash,
                    _donor_outer_ctx,
                    _donor_inner_ctx,
                ) = _unpack_donor_entry(donor_entry)
                donor_inner = set(donor_assoc.nodes())
                if not donor_inner:
                    continue
                source_edge_map = virtual_cut_source_map(
                    graph, cut_nodes, cut_radius=cut_radius, node_key_map=node_key_map
                )
                new_graphs = _replace_subgraph(
                    graph,
                    source_inner=set(),
                    source_cut=[],
                    source_edge_map=source_edge_map,
                    donor_assoc=donor_assoc,
                    donor_inner=donor_inner,
                    donor_cut=donor_cut,
                    donor_edge_map=donor_edge_map,
                    rng=rng,
                    single_replacement=single_replacement,
                    max_enumerations=max_enumerations,
                    preserve_node_ids=preserve_node_ids,
                )
                if not new_graphs:
                    continue
                for g in new_graphs:
                    g.graph["context_score"] = score
                outputs.extend(new_graphs[: max(1, n_samples - len(outputs))])
                break
        return outputs
    max_outputs = max(1, int(n_samples))
    for cut_nodes in cut_nodes_iter:
        seen_key = None
        if cut_nodes_seen is not None:
            seen_key = _cut_nodes_seen_key(
                graph, cut_nodes, cut_radius, node_key_map=node_key_map
            )
            if seen_key in cut_nodes_seen:
                continue
        checks += 1
        if cut_nodes_seen is not None:
            cut_nodes_seen.add(seen_key)
        if checks_state is not None:
            checks_state["used"] += 1
            if remaining is not None:
                remaining = max(0, remaining - 1)
                checks_state["remaining"] = remaining
        cut_key = virtual_cut_signature(
            graph, cut_nodes, cut_radius=cut_radius, node_key_map=node_key_map
        )
        donor_candidates = cut_index.get(cut_key)
        if not donor_candidates:
            continue
        donor_weights = [_entry_frequency(entry) for entry in donor_candidates]
        if sum(donor_weights) > 0:
            donor_entry = rng.choices(
                donor_candidates,
                weights=donor_weights,
                k=1,
            )[0]
        else:
            donor_entry = rng.choice(donor_candidates)
        (
            donor_assoc,
            donor_cut,
            donor_edge_map,
            _assoc_hash,
            _donor_outer_ctx,
            _donor_inner_ctx,
        ) = _unpack_donor_entry(donor_entry)
        donor_inner = set(donor_assoc.nodes())
        if not donor_inner:
            continue
        source_edge_map = virtual_cut_source_map(
            graph, cut_nodes, cut_radius=cut_radius, node_key_map=node_key_map
        )
        new_graphs = _replace_subgraph(
            graph,
            source_inner=set(),
            source_cut=[],
            source_edge_map=source_edge_map,
            donor_assoc=donor_assoc,
            donor_inner=donor_inner,
            donor_cut=donor_cut,
            donor_edge_map=donor_edge_map,
            rng=rng,
            single_replacement=single_replacement,
            max_enumerations=max_enumerations,
            preserve_node_ids=preserve_node_ids,
        )
        if not new_graphs:
            continue
        for g in new_graphs:
            g.graph.pop("context_score", None)
            outputs.append(g)
            if len(outputs) >= max_outputs:
                break
        if len(outputs) >= max_outputs:
            break
    return outputs


def virtual_rewrite_candidates_at_cut_nodes(
    graph: nx.Graph,
    cut_nodes: Sequence,
    cut_index: dict,
    *,
    cut_radius: Optional[int],
    cut_context_radius: Optional[int] = None,
    context_vectorizer=None,
    use_context_embedding: bool = True,
    context_temperature: float = 1.0,
    context_top_k: Optional[int] = None,
    rng: Optional[random.Random] = None,
    n_samples: int = 1,
    allow_superset: bool = False,
    max_superset_checks: Optional[int] = None,
    single_replacement: bool = True,
    max_enumerations: Optional[int] = None,
    node_key_map: Optional[dict] = None,
    preserve_node_ids: bool = False,
) -> list[nx.Graph]:
    """Attempt rewrites by inserting donors at a specified virtual cut.

    Args:
        graph: Source graph to augment.
        cut_nodes: Explicit outer nodes defining the virtual cut.
        cut_index: Cut dictionary mapping signatures to donor entries.
        cut_radius: Neighborhood radius for subgraph-hash signatures.
        cut_context_radius: Neighborhood radius for context embeddings.
        context_vectorizer: Optional transformer with a .transform(graphs)
            method returning vector embeddings per graph.
        use_context_embedding: If True and context_vectorizer is provided,
            prefer donors by outer context similarity.
        context_temperature: Selection temperature for context-aware sampling.
            T=1 selects uniformly at random; T=0 deterministically selects top-k.
        context_top_k: Number of top-scoring candidates to return when
            context_temperature=0. Defaults to n_samples.
        rng: Optional random generator for reproducibility.
        n_samples: Number of candidate rewrites to return.
        allow_superset: If True, fall back to cuts that strictly contain `cut_nodes`.
        max_superset_checks: Optional cap on attempted superset cut evaluations.
        single_replacement: If True, pick one random pairing per label.
        max_enumerations: Cap on enumerated/sampled pairings when single_replacement=False.
        node_key_map: Optional precomputed per-node cut keys.
        preserve_node_ids: If True, do not relabel nodes to a contiguous range.

    Returns:
        List of candidate graphs with donor subgraphs inserted.
    """
    # Local import keeps this robust when modules are reloaded in notebooks.
    from collections import Counter

    def _multiset_contains(superset, subset_counts: Counter) -> bool:
        counts = Counter(superset)
        for key, needed in subset_counts.items():
            if counts[key] < needed:
                return False
        return True

    def _iter_superset_cut_nodes():
        base_key = virtual_cut_signature(
            graph, base_nodes, cut_radius=cut_radius, node_key_map=node_key_map
        )
        base_counts = Counter(base_key)
        remaining = [n for n in graph.nodes() if n not in base_nodes]
        keys = [k for k in cut_index.keys() if len(k) > len(base_key)]
        rng.shuffle(keys)
        checks = 0
        for key in keys:
            if not _multiset_contains(key, base_counts):
                continue
            needed = len(key) - len(base_nodes)
            if needed > len(remaining):
                continue
            for extra in itertools.combinations(remaining, needed):
                if max_superset_checks is not None and checks >= max_superset_checks:
                    return
                checks += 1
                candidate_nodes = base_nodes + extra
                if (
                    virtual_cut_signature(
                        graph,
                        candidate_nodes,
                        cut_radius=cut_radius,
                        node_key_map=node_key_map,
                    )
                    == key
                ):
                    yield candidate_nodes

    rng = rng or random.Random()
    outputs: list[nx.Graph] = []
    if not cut_nodes:
        return outputs
    base_nodes = tuple(cut_nodes)
    if node_key_map is None:
        node_key_map = _precompute_outer_node_keys(graph, cut_radius)
    cut_key = virtual_cut_signature(
        graph, base_nodes, cut_radius=cut_radius, node_key_map=node_key_map
    )
    donor_candidates = cut_index.get(cut_key)
    candidate_cuts = []
    if donor_candidates:
        candidate_cuts.append((base_nodes, donor_candidates))
    elif allow_superset:
        for candidate_nodes in _iter_superset_cut_nodes():
            cut_key = virtual_cut_signature(
                graph, candidate_nodes, cut_radius=cut_radius, node_key_map=node_key_map
            )
            donor_candidates = cut_index.get(cut_key)
            if donor_candidates:
                candidate_cuts.append((candidate_nodes, donor_candidates))
            if candidate_cuts and len(candidate_cuts) >= max(1, int(n_samples)):
                break
    if not candidate_cuts:
        return outputs
    use_context = bool(use_context_embedding and context_vectorizer is not None)
    if use_context:
        candidate_pairs = []
        scores = []
        freqs = []
        for cut_nodes_variant, donor_candidates in candidate_cuts:
            source_outer_ctx = _context_embeddings_for_virtual_cut(
                graph,
                cut_nodes_variant,
                cut_context_radius=cut_context_radius,
                context_vectorizer=context_vectorizer,
            )
            for donor_entry in donor_candidates:
                (
                    _donor_assoc,
                    _donor_cut,
                    _donor_edge_map,
                    _assoc_hash,
                    donor_outer_ctx,
                    _donor_inner_ctx,
                ) = _unpack_donor_entry(donor_entry)
                sim = _max_cosine_similarity(source_outer_ctx, donor_outer_ctx)
                score = -np.inf if sim is None else float(sim)
                candidate_pairs.append((cut_nodes_variant, donor_entry))
                scores.append(score)
                freqs.append(_entry_frequency(donor_entry))
        if not candidate_pairs:
            return outputs
        if context_temperature <= 0.0:
            k = n_samples if context_top_k is None else int(context_top_k)
            for idx in _top_k_indices(scores, k):
                cut_nodes_variant, donor_entry = candidate_pairs[idx]
                (
                    donor_assoc,
                    donor_cut,
                    donor_edge_map,
                    _assoc_hash,
                    _donor_outer_ctx,
                    _donor_inner_ctx,
                ) = _unpack_donor_entry(donor_entry)
                donor_inner = set(donor_assoc.nodes())
                if not donor_inner:
                    continue
                source_edge_map = virtual_cut_source_map(
                    graph,
                    cut_nodes_variant,
                    cut_radius=cut_radius,
                    node_key_map=node_key_map,
                )
                new_graphs = _replace_subgraph(
                    graph,
                    source_inner=set(),
                    source_cut=[],
                    source_edge_map=source_edge_map,
                    donor_assoc=donor_assoc,
                    donor_inner=donor_inner,
                    donor_cut=donor_cut,
                    donor_edge_map=donor_edge_map,
                    rng=rng,
                    single_replacement=single_replacement,
                    max_enumerations=max_enumerations,
                    preserve_node_ids=preserve_node_ids,
                )
                if not new_graphs:
                    continue
                for g in new_graphs:
                    g.graph["context_score"] = scores[idx]
                outputs.extend(new_graphs[: max(1, n_samples - len(outputs))])
                if len(outputs) >= max(1, int(n_samples)):
                    break
            return outputs
        weights = _temperature_weights(scores, context_temperature)
        weights = weights * np.asarray(freqs, dtype=float)
        for _ in range(max(1, int(n_samples))):
            available_pairs = list(candidate_pairs)
            available_weights = list(weights)
            available_scores = list(scores)
            while available_pairs:
                total_weight = sum(available_weights)
                if total_weight > 0:
                    idx = rng.choices(
                        range(len(available_pairs)),
                        weights=available_weights,
                        k=1,
                    )[0]
                else:
                    idx = rng.randrange(len(available_pairs))
                cut_nodes_variant, donor_entry = available_pairs.pop(idx)
                available_weights.pop(idx)
                score = available_scores.pop(idx)
                (
                    donor_assoc,
                    donor_cut,
                    donor_edge_map,
                    _assoc_hash,
                    _donor_outer_ctx,
                    _donor_inner_ctx,
                ) = _unpack_donor_entry(donor_entry)
                donor_inner = set(donor_assoc.nodes())
                if not donor_inner:
                    continue
                source_edge_map = virtual_cut_source_map(
                    graph,
                    cut_nodes_variant,
                    cut_radius=cut_radius,
                    node_key_map=node_key_map,
                )
                new_graphs = _replace_subgraph(
                    graph,
                    source_inner=set(),
                    source_cut=[],
                    source_edge_map=source_edge_map,
                    donor_assoc=donor_assoc,
                    donor_inner=donor_inner,
                    donor_cut=donor_cut,
                    donor_edge_map=donor_edge_map,
                    rng=rng,
                    single_replacement=single_replacement,
                    max_enumerations=max_enumerations,
                    preserve_node_ids=preserve_node_ids,
                )
                if not new_graphs:
                    continue
                for g in new_graphs:
                    g.graph["context_score"] = score
                outputs.extend(new_graphs[: max(1, n_samples - len(outputs))])
                break
            if len(outputs) >= n_samples:
                break
        return outputs
    for cut_nodes_variant, donor_candidates in candidate_cuts:
        source_edge_map = virtual_cut_source_map(
            graph,
            cut_nodes_variant,
            cut_radius=cut_radius,
            node_key_map=node_key_map,
        )
        for _ in range(max(1, int(n_samples))):
            donor_weights = [_entry_frequency(entry) for entry in donor_candidates]
            if sum(donor_weights) > 0:
                donor_entry = rng.choices(
                    donor_candidates,
                    weights=donor_weights,
                    k=1,
                )[0]
            else:
                donor_entry = rng.choice(donor_candidates)
            (
                donor_assoc,
                donor_cut,
                donor_edge_map,
                _assoc_hash,
                _donor_outer_ctx,
                _donor_inner_ctx,
            ) = _unpack_donor_entry(donor_entry)
            donor_inner = set(donor_assoc.nodes())
            if not donor_inner:
                continue
            new_graphs = _replace_subgraph(
                graph,
                source_inner=set(),
                source_cut=[],
                source_edge_map=source_edge_map,
                donor_assoc=donor_assoc,
                donor_inner=donor_inner,
                donor_cut=donor_cut,
                donor_edge_map=donor_edge_map,
                rng=rng,
                single_replacement=single_replacement,
                max_enumerations=max_enumerations,
                preserve_node_ids=preserve_node_ids,
            )
        if not new_graphs:
            continue
        for g in new_graphs:
            g.graph.pop("context_score", None)
        outputs.extend(new_graphs[: max(1, n_samples - len(outputs))])
        if len(outputs) >= n_samples:
            break
        if len(outputs) >= n_samples:
            break
    return outputs


def display_cut_index(
    cut_index: dict,
    draw_func: Callable[[Sequence[nx.Graph]], object],
    *,
    max_keys: Optional[int] = None,
    max_per_key: Optional[int] = None,
    sort_by_size: bool = True,
) -> list[tuple[tuple, list[nx.Graph]]]:
    """Display mapped subgraphs per cut key.

    Args:
        cut_index: Dictionary mapping cut keys to donor entries.
        draw_func: Callable that renders a list of graphs.
        max_keys: Optional maximum number of cut keys to display.
        max_per_key: Optional maximum number of graphs per cut key.
        sort_by_size: If True, sort by cut size then by descending count.

    Returns:
        List of (cut_key, assoc_graphs) for displayed keys.
    """
    items = list(_iter_cut_index_items(cut_index))
    if sort_by_size:
        items.sort(key=lambda kv: (len(kv[0]), -len(kv[1])))
    else:
        items.sort(key=lambda kv: -len(kv[1]))
    if max_keys is not None:
        items = items[: max_keys]
    displayed: list[tuple[tuple, list[nx.Graph]]] = []
    for cut_key, entries in items:
        assocs = [_unpack_donor_entry(entry)[0] for entry in entries]
        if max_per_key is not None:
            assocs = assocs[: max_per_key]
        print(f"cut_key_size={len(cut_key)} count={len(entries)} key={cut_key}")
        draw_func(assocs)
        displayed.append((cut_key, assocs))
    return displayed


def _filter_donor_candidates(donor_candidates, source_hash: int):
    """Filter donor associations that match the source mapped subgraph hash.

    Args:
        donor_candidates: Donor entry list for a cut key.
        source_hash: Association hash for the source subgraph.

    Returns:
        Filtered donor entries, or None if none remain.
    """
    if not donor_candidates:
        return None
    filtered = []
    for entry in donor_candidates:
        _assoc, _donor_cut, _donor_edge_map, assoc_hash, _outer_ctx, _inner_ctx = _unpack_donor_entry(entry)
        if assoc_hash != source_hash:
            filtered.append(entry)
    if not filtered:
        return None
    return filtered


def _replace_subgraph(
    graph: nx.Graph,
    source_inner,
    source_cut,
    source_edge_map: dict,
    donor_assoc: nx.Graph,
    donor_inner,
    donor_cut,
    donor_edge_map: dict,
    rng: random.Random,
    *,
    single_replacement: bool = True,
    max_enumerations: Optional[int] = None,
    preserve_node_ids: bool = False,
):
    """Replace source_inner with donor_assoc, rewiring cut edges by label.

    When source edge attributes are unavailable, donor edge attributes are used.
    Newly inserted donor nodes preserve their attributes.

    Args:
        graph: Source graph to rewrite.
        source_inner: Nodes in the source subgraph to replace.
        source_cut: Cut edges for the source subgraph.
        source_edge_map: Mapping from cut keys to source edges.
        donor_assoc: Donor mapped subgraph to insert.
        donor_inner: Nodes in the donor mapped subgraph.
        donor_cut: Cut edges for the donor mapped subgraph.
        donor_edge_map: Mapping from cut keys to donor inner endpoints.
        rng: Random generator used for pairing.
        single_replacement: If True, pick one random pairing per label.
        max_enumerations: Cap on enumerated/sampled pairings when single_replacement=False.
        preserve_node_ids: If True, do not relabel nodes to a contiguous range.

    Returns:
        List of new graphs (possibly empty when replacement fails).
    """
    if source_inner is None:
        source_inner = set()
    else:
        source_inner = set(source_inner)
    base_graph = graph.copy()
    base_graph.remove_nodes_from(source_inner)

    try:
        import numpy as _np
        int_nodes = [n for n in base_graph.nodes if isinstance(n, (int, _np.integer))]
    except Exception:
        int_nodes = [n for n in base_graph.nodes if isinstance(n, int)]
    next_id = (max(int_nodes) if int_nodes else -1) + 1
    id_map = {n: next_id + i for i, n in enumerate(donor_assoc.nodes())}
    for n, attrs in donor_assoc.nodes(data=True):
        node_attrs = dict(attrs)
        base_graph.add_node(id_map[n], **node_attrs)
    for u, v, attrs in donor_assoc.edges(data=True):
        base_graph.add_edge(id_map[u], id_map[v], **dict(attrs))

    source_by_key = source_edge_map
    donor_by_key = donor_edge_map

    def _split_source_edge(entry):
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], dict):
            return entry[0], entry[1]
        return entry, None

    def _split_donor_entry(entry):
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], dict):
            return entry[0], entry[1]
        return entry, None

    def _donor_signature(entry):
        node, attr = _split_donor_entry(entry)
        return (node, hash_value(attr) if attr is not None else None)

    for key, source_edges in source_by_key.items():
        donor_entries = donor_by_key.get(key, [])
        if len(source_edges) != len(donor_entries):
            return []

    def _finalize(mapping_per_key) -> nx.Graph:
        g = base_graph.copy()
        for _key, pairs in mapping_per_key.items():
            for source_edge, donor_entry in pairs:
                outer, source_attr = _split_source_edge(source_edge)
                donor_inner_node, donor_attr = _split_donor_entry(donor_entry)
                edge_attr = source_attr if source_attr is not None else donor_attr
                if edge_attr is None:
                    edge_attr = {}
                g.add_edge(id_map[donor_inner_node], outer, **dict(edge_attr))
        if preserve_node_ids:
            return g
        order = list(g.nodes())
        relabel_map = {old: i for i, old in enumerate(order)}
        g = nx.relabel_nodes(g, relabel_map, copy=True)
        return g

    if single_replacement:
        mapping = {}
        for key, source_edges in source_by_key.items():
            donor_entries = donor_by_key[key][:]
            rng.shuffle(donor_entries)
            mapping[key] = list(zip(source_edges, donor_entries))
        return [_finalize(mapping)]

    import itertools as _it

    def _perm_count(m: int, n: int) -> int:
        out = 1
        for v in range(m, m - n, -1):
            out *= v
        return out

    counts = []
    for key, source_edges in source_by_key.items():
        m = len(donor_by_key[key])
        n = len(source_edges)
        counts.append(_perm_count(m, n))
    total_count = 1
    for c in counts:
        total_count *= c

    if max_enumerations is None or total_count <= max_enumerations:
        per_label_perms = {}
        for key, source_edges in source_by_key.items():
            donor_entries = donor_by_key[key]
            perms = list(_it.permutations(donor_entries, len(source_edges)))
            per_label_perms[key] = [(source_edges, perm) for perm in perms]
        labels = list(per_label_perms.keys())
        all_graphs: list[nx.Graph] = []
        for choice in _it.product(*[per_label_perms[lbl] for lbl in labels]):
            mapping = {}
            for lbl, (src_edges, perm_inners) in zip(labels, choice):
                mapping[lbl] = list(zip(src_edges, list(perm_inners)))
            all_graphs.append(_finalize(mapping))
        return all_graphs

    labels = list(source_by_key.keys())
    seen = set()
    samples: list[nx.Graph] = []
    attempts = 0
    max_attempts = max_enumerations * 4
    while len(samples) < max_enumerations and attempts < max_attempts:
        attempts += 1
        mapping = {}
        signature_parts = []
        for key in labels:
            src_edges = source_by_key[key]
            donors_list = donor_by_key[key][:]
            rng.shuffle(donors_list)
            perm = tuple(donors_list[: len(src_edges)])
            signature_parts.append((key, tuple(_donor_signature(entry) for entry in perm)))
            mapping[key] = list(zip(src_edges, list(perm)))
        sig = tuple(signature_parts)
        if sig in seen:
            continue
        seen.add(sig)
        samples.append(_finalize(mapping))
    return samples


def extract_ball(graph: nx.Graph, center_node, radius: int) -> nx.Graph:
    """
    Extract a radius-limited neighborhood around a center node.

    Args:
        graph: Input graph.
        center_node: Node used as the BFS center.
        radius: Hop radius. Values <= 0 return the center node only.

    Returns:
        nx.Graph: Induced labeled subgraph around the center node.
    """
    if center_node not in graph:
        return nx.Graph()
    if radius is None:
        return graph.copy()
    if int(radius) <= 0:
        return graph.subgraph([center_node]).copy()
    lengths = nx.single_source_shortest_path_length(graph, center_node, cutoff=int(radius))
    nodes = list(lengths.keys())
    return graph.subgraph(nodes).copy()


def anchor_type_train(graph_full: nx.Graph, node, radius: int, *, nbits: int = 19) -> int:
    """
    Compute a training-time anchor hash from a radius-limited neighborhood.

    Args:
        graph_full: Training base graph.
        node: Anchor node in the training graph.
        radius: Radius used for local context extraction.
        nbits: Bit width for bounded graph hashing.

    Returns:
        int: Anchor type hash.
    """
    return hash_graph(extract_ball(graph_full, node, radius), nbits=nbits)


def anchor_type_current(graph_partial: nx.Graph, node, radius: int, *, nbits: int = 19) -> int:
    """
    Compute a generation-time anchor hash from the current partial graph.

    Args:
        graph_partial: Partially materialized graph.
        node: Existing global node id.
        radius: Radius used for local context extraction.
        nbits: Bit width for bounded graph hashing.

    Returns:
        int: Anchor type hash.
    """
    return hash_graph(extract_ball(graph_partial, node, radius), nbits=nbits)


def materialize_component(
    graph_partial: nx.Graph,
    component_subgraph: nx.Graph,
    *,
    start_node_id: Optional[int] = None,
) -> dict:
    """
    Add a component subgraph into a partial graph with fresh integer node ids.

    Args:
        graph_partial: Target graph mutated in place.
        component_subgraph: Local component graph with local node ids.
        start_node_id: Optional first global id for inserted nodes.

    Returns:
        dict: Mapping from local component node ids to global node ids.
    """
    if start_node_id is None:
        int_nodes = [n for n in graph_partial.nodes() if isinstance(n, int)]
        next_id = (max(int_nodes) + 1) if int_nodes else 0
    else:
        next_id = int(start_node_id)
    local_to_global = {}
    for local_node, attrs in component_subgraph.nodes(data=True):
        global_node = next_id
        next_id += 1
        local_to_global[local_node] = global_node
        graph_partial.add_node(global_node, **dict(attrs))
    for u, v, attrs in component_subgraph.edges(data=True):
        graph_partial.add_edge(local_to_global[u], local_to_global[v], **dict(attrs))
    return local_to_global


def unify_anchors(graph_partial: nx.Graph, source_node, target_node):
    """
    Merge source node into target node by rewiring all source edges.

    Args:
        graph_partial: Graph mutated in place.
        source_node: Node to be removed after merge.
        target_node: Node kept as merge representative.

    Returns:
        object: Kept target node id.
    """
    if source_node == target_node:
        return target_node
    if source_node not in graph_partial or target_node not in graph_partial:
        return target_node
    for nbr, attrs in list(graph_partial[source_node].items()):
        if nbr == target_node:
            continue
        if not graph_partial.has_edge(target_node, nbr):
            graph_partial.add_edge(target_node, nbr, **dict(attrs))
    graph_partial.remove_node(source_node)
    return target_node
