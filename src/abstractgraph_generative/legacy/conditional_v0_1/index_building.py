"""Index-building helpers for conditional autoregressive generation."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Callable, Iterable, Optional, Sequence
import itertools
import math
import random

import networkx as nx
import numpy as np

from abstractgraph.hashing import hash_graph, hash_set
from abstractgraph.graphs import AbstractGraph, graph_to_abstract_graph
from abstractgraph_generative.rewrite import (
    _radius_neighborhood_graph,
    _transform_context_graphs,
)
from abstractgraph_generative.legacy.conditional_v0_1.types import CutArchiveEntry


def build_assoc_map_from_image_graph(image_graph: nx.Graph) -> dict:
    """Build a preimage->image association map from image-node subgraphs.

    Args:
        image_graph: Image graph whose nodes may store `association` subgraphs.

    Returns:
        dict: Mapping from preimage node id to a set of image node ids.
    """
    assoc_map: dict = defaultdict(set)
    for img_node, data in image_graph.nodes(data=True):
        assoc = data.get("association")
        if not isinstance(assoc, nx.Graph):
            continue
        for pre_node in assoc.nodes():
            assoc_map[pre_node].add(img_node)
    return assoc_map


def group_graphs_by_image_hash(graphs, decomposition_function, nbits, label_mode="graph_hash"):
    """Group graphs by the hash of their image graph.

    Args:
        graphs: Iterable of NetworkX graphs.
        decomposition_function: Decomposition used to build image graphs.
        nbits: Hash bit width for image labels.
        label_mode: Label mode passed to graph_to_abstract_graph.

    Returns:
        dict: Mapping from image-graph hash to list of graphs.
    """
    buckets = {}
    for g in graphs:
        ag = graph_to_abstract_graph(
            g,
            decomposition_function=decomposition_function,
            nbits=nbits,
            label_mode=label_mode,
        )
        img_hash = hash_graph(ag.image_graph)
        buckets.setdefault(img_hash, []).append(g)
    return buckets


def select_image_group(
    buckets,
    min_group_size,
    *,
    rng: Optional[random.Random] = None,
):
    """Select a graph group sharing the same image hash.

    Args:
        buckets: Mapping from image hash to graph list.
        min_group_size: Minimum group size to prefer.
        rng: Optional random generator for deterministic tie-breaking.

    Returns:
        tuple[list, bool]: (selected_group, used_fallback).
    """
    if not buckets:
        return [], True
    rng = rng or random
    eligible = [g for g in buckets.values() if len(g) >= min_group_size]
    if eligible:
        return rng.choice(eligible), False
    return max(buckets.values(), key=len), True


def _neighbor_subsets(
    neighbors: Sequence,
    *,
    rng: random.Random,
    max_subset_size: int,
    max_subsets: int,
) -> list[tuple]:
    """
    Enumerate or sample non-empty subsets of neighbors.

    Args:
        neighbors: Sequence of neighbor node ids.
        rng: Random generator used for sampling when needed.
        max_subset_size: Maximum subset size to enumerate.
        max_subsets: Cap on number of subsets returned.

    Returns:
        list[tuple]: List of neighbor subsets (as tuples).
    """
    if not neighbors:
        return []
    neighbors = list(neighbors)
    max_subset_size = max(1, int(max_subset_size))
    max_subsets = max(1, int(max_subsets))
    limit = min(max_subset_size, len(neighbors))
    if limit <= 0:
        return []

    # Preserve prior behavior: when truncating subset sizes, include the full
    # neighbor set as an additional candidate.
    include_full_neighbors = len(neighbors) > max_subset_size
    reserve = 1 if include_full_neighbors else 0
    budget = max(0, max_subsets - reserve)

    if budget == 0:
        return [tuple(neighbors)] if include_full_neighbors else []

    total = 0
    for r in range(1, limit + 1):
        total += math.comb(len(neighbors), r)

    if total <= budget:
        sampled_subsets: list[tuple] = []
        for r in range(1, limit + 1):
            for combo in itertools.combinations(neighbors, r):
                sampled_subsets.append(combo)
    else:
        # Reservoir sample combinations from the stream without materializing
        # all subsets in memory.
        sampled_subsets = []
        seen = 0
        for r in range(1, limit + 1):
            for combo in itertools.combinations(neighbors, r):
                seen += 1
                if len(sampled_subsets) < budget:
                    sampled_subsets.append(combo)
                    continue
                pick = rng.randint(1, seen)
                if pick <= budget:
                    sampled_subsets[pick - 1] = combo
        rng.shuffle(sampled_subsets)

    if include_full_neighbors:
        sampled_subsets.append(tuple(neighbors))
    return sampled_subsets


def _degree_guided_image_order(image_graph: nx.Graph, *, rng: random.Random) -> list:
    """
    Produce a degree-guided traversal order over image nodes.

    Args:
        image_graph: Image graph to traverse.
        rng: Random generator used to break ties.

    Returns:
        list: Ordered list of image node ids.
    """
    degrees = dict(image_graph.degree())
    if not degrees:
        return []
    order = []
    visited = set()

    def _pick_start(remaining):
        max_deg = max(degrees[n] for n in remaining)
        starts = [n for n in remaining if degrees[n] == max_deg]
        return rng.choice(starts)

    remaining = set(image_graph.nodes())
    while remaining:
        start = _pick_start(remaining)
        queue = deque([start])
        visited.add(start)
        order.append(start)
        remaining.remove(start)
        while queue:
            node = queue.popleft()
            neighbors = [n for n in image_graph.neighbors(node) if n not in visited]
            rng.shuffle(neighbors)
            neighbors.sort(key=lambda n: degrees.get(n, 0), reverse=True)
            for neigh in neighbors:
                if neigh in visited:
                    continue
                visited.add(neigh)
                remaining.discard(neigh)
                order.append(neigh)
                queue.append(neigh)
    return order


def _default_image_label(image_graph: nx.Graph, node) -> str:
    """
    Return a string label for an image graph node.

    Args:
        image_graph: Image graph containing the node.
        node: Node id to label.

    Returns:
        str: String label for the node.
    """
    label = image_graph.nodes[node].get("label")
    if label is None:
        return "-"
    return str(label)


def _default_preimage_label(preimage_graph: Optional[nx.Graph], node) -> str:
    """
    Return a string label for a preimage graph node.

    Args:
        preimage_graph: Preimage graph containing the node.
        node: Node id to label.

    Returns:
        str: String label for the node (or "-" if missing).
    """
    if preimage_graph is None or node not in preimage_graph:
        return "-"
    label = preimage_graph.nodes[node].get("label")
    if label is None:
        return "-"
    return str(label)


def _anchor_preimage_hash(
    preimage_graph: nx.Graph,
    node,
    radius: Optional[int],
    preimage_label_fn: Callable[[Optional[nx.Graph], object], str],
) -> int:
    """
    Hash a preimage neighborhood around an anchor node.

    Args:
        preimage_graph: Graph containing the anchor node.
        node: Anchor node id.
        radius: Neighborhood radius; None or negative returns 0.
        preimage_label_fn: Label function used for node labels.

    Returns:
        int: Hash of the labeled neighborhood subgraph.
    """
    if radius is None:
        return 0
    if radius < 0:
        return 0
    ego = nx.ego_graph(preimage_graph, node, radius=radius)
    if ego.number_of_nodes() == 0:
        return 0
    sub = ego.copy()
    for n in sub.nodes():
        sub.nodes[n]["label"] = preimage_label_fn(preimage_graph, n)
    return hash_graph(sub)


def _image_context_hash(
    image_graph: nx.Graph,
    node,
    radius: Optional[int],
    image_label_fn: Callable[[nx.Graph, object], str],
) -> Optional[int]:
    """
    Hash an image-graph neighborhood around a node.

    Args:
        image_graph: Image graph containing the node.
        node: Image node id.
        radius: Neighborhood radius; None or negative disables hashing.
        image_label_fn: Label function used for node labels.

    Returns:
        Optional[int]: Hash of the labeled neighborhood subgraph, or None if disabled.
    """
    if radius is None:
        return None
    if radius < 0:
        return None
    ego = nx.ego_graph(image_graph, node, radius=radius)
    if ego.number_of_nodes() == 0:
        return None
    sub = ego.copy()
    for n in sub.nodes():
        sub.nodes[n]["label"] = image_label_fn(image_graph, n)
    return hash_graph(sub)


def _neighbor_overlap_signature(
    image_graph: nx.Graph,
    node,
    radius: Optional[int],
    image_label_fn: Callable[[nx.Graph, object], str],
) -> tuple:
    """
    Build an overlap signature for a node from image-edge overlap attributes.

    Args:
        image_graph: Image graph containing the node.
        node: Image node id.
        radius: Radius used for neighbor role hashing.
        image_label_fn: Label function used for role hashing.

    Returns:
        tuple: Sorted tuple of (neighbor_role_hash, shared_preimage_nodes).
    """
    if node not in image_graph:
        return ()
    signature = []
    for neigh in image_graph.neighbors(node):
        edge_data = image_graph.get_edge_data(node, neigh) or {}
        shared = edge_data.get("shared_preimage_nodes")
        if shared is None:
            continue
        try:
            shared = int(shared)
        except Exception:
            continue
        neigh_role_hash = _image_context_hash(
            image_graph,
            neigh,
            radius,
            image_label_fn,
        )
        signature.append((neigh_role_hash, shared))
    signature.sort()
    return tuple(signature)


def _cut_hash(anchor_hashes: Iterable[int]) -> int:
    """
    Hash a multiset of anchor hashes into a single cut hash.

    Args:
        anchor_hashes: Iterable of anchor hashes.

    Returns:
        int: Hash representing the multiset.
    """
    return hash_set(list(anchor_hashes))


def _cut_hash_with_image(anchor_hashes: Iterable[int], image_hash: Optional[int]) -> int:
    """
    Hash a multiset of anchor hashes plus an optional image-context hash.

    Args:
        anchor_hashes: Iterable of anchor hashes.
        image_hash: Optional image-context hash to include.

    Returns:
        int: Hash representing the multiset (anchors + image context when provided).
    """
    if image_hash is None:
        return _cut_hash(anchor_hashes)
    return hash_set(list(anchor_hashes) + [("image_ctx", image_hash)])


def _image_nodes_for_cut(cut_nodes: Iterable, assoc_map: dict) -> set:
    """
    Return image nodes associated with a set of preimage cut nodes.

    Args:
        cut_nodes: Iterable of preimage node ids.
        assoc_map: Mapping preimage node -> set of image nodes.

    Returns:
        set: Image node ids associated with the cut nodes.
    """
    image_nodes: set = set()
    for node in cut_nodes:
        image_nodes |= set(assoc_map.get(node, set()))
    return image_nodes


def _assoc_map_from_image_associations(image_associations: dict) -> dict:
    """
    Invert image associations into a preimage-to-image mapping.

    Args:
        image_associations: Mapping image_node -> set of preimage nodes.

    Returns:
        dict: Mapping preimage_node -> set of image nodes.
    """
    assoc_map: dict = defaultdict(set)
    for img_node, nodes in image_associations.items():
        for node in nodes:
            assoc_map[node].add(img_node)
    return assoc_map


def _graph_cache_root(graph: nx.Graph) -> dict:
    """
    Return the mutable cache root dictionary stored on a NetworkX graph.

    Args:
        graph: Graph that owns the cache.

    Returns:
        dict: Cache root dictionary.
    """
    return graph.graph.setdefault("__ag_conditional_cache__", {})


def _cache_disabled(graph: nx.Graph) -> bool:
    """
    Return whether conditional per-graph caches are disabled for this graph.

    Args:
        graph: Graph to check.

    Returns:
        bool: True when cache writes should be skipped.
    """
    return bool(graph.graph.get("__ag_disable_conditional_cache__", False))


def _graph_stamp(graph: nx.Graph) -> tuple[int, int]:
    """
    Return a lightweight structural stamp for cache invalidation.

    Args:
        graph: Graph to stamp.

    Returns:
        tuple[int, int]: (number_of_nodes, number_of_edges).
    """
    return graph.number_of_nodes(), graph.number_of_edges()


def _cached_node_hash_map(
    graph: nx.Graph,
    *,
    radius: Optional[int],
    label_fn,
    mode: str,
) -> dict:
    """
    Build or reuse a per-node neighborhood-hash map for a graph.

    Args:
        graph: Graph whose node hashes are requested.
        radius: Neighborhood radius used for hashing.
        label_fn: Label function used by the hash builder.
        mode: Cache namespace key (e.g. "preimage" or "image").

    Returns:
        dict: Mapping node -> neighborhood hash value.
    """
    if _cache_disabled(graph):
        values = {}
        if mode == "preimage":
            for node in graph.nodes():
                values[node] = _anchor_preimage_hash(graph, node, radius, label_fn)
        elif mode == "image":
            for node in graph.nodes():
                values[node] = _image_context_hash(graph, node, radius, label_fn)
        else:
            raise ValueError(f"Unsupported node-hash cache mode: {mode!r}")
        return values

    root = _graph_cache_root(graph)
    bucket = root.setdefault("node_hash_maps", {})
    key = (mode, radius, id(label_fn))
    stamp = _graph_stamp(graph)
    cached = bucket.get(key)
    if cached is not None and cached.get("stamp") == stamp:
        return cached.get("values", {})

    values = {}
    if mode == "preimage":
        for node in graph.nodes():
            values[node] = _anchor_preimage_hash(graph, node, radius, label_fn)
    elif mode == "image":
        for node in graph.nodes():
            values[node] = _image_context_hash(graph, node, radius, label_fn)
    else:
        raise ValueError(f"Unsupported node-hash cache mode: {mode!r}")
    bucket[key] = {"stamp": stamp, "values": values}
    return values


def cached_preimage_node_hash_map(
    graph: nx.Graph,
    *,
    radius: Optional[int],
    preimage_label_fn,
) -> dict:
    """
    Return cached preimage node neighborhood hashes for a graph.

    Args:
        graph: Preimage graph.
        radius: Neighborhood radius.
        preimage_label_fn: Label function for preimage nodes.

    Returns:
        dict: Mapping node -> hash.
    """
    return _cached_node_hash_map(
        graph,
        radius=radius,
        label_fn=preimage_label_fn,
        mode="preimage",
    )


def cached_image_node_hash_map(
    graph: nx.Graph,
    *,
    radius: Optional[int],
    image_label_fn,
) -> dict:
    """
    Return cached image node neighborhood hashes for a graph.

    Args:
        graph: Image graph.
        radius: Neighborhood radius.
        image_label_fn: Label function for image nodes.

    Returns:
        dict: Mapping node -> hash (or None when radius disables hashing).
    """
    return _cached_node_hash_map(
        graph,
        radius=radius,
        label_fn=image_label_fn,
        mode="image",
    )


def cached_context_embedding_sum_for_nodes(
    graph: nx.Graph,
    *,
    nodes: Sequence,
    radius: Optional[int],
    context_vectorizer,
) -> Optional[np.ndarray]:
    """
    Sum cached per-node context embeddings over a node multiset.

    Args:
        graph: Graph used to build neighborhoods.
        nodes: Node sequence; multiplicity contributes linearly.
        radius: Neighborhood radius for each node context.
        context_vectorizer: Graph vectorizer with transform support.

    Returns:
        Optional[np.ndarray]: Summed context embedding, or None if unavailable.
    """
    if context_vectorizer is None or not nodes:
        return None
    counts = {}
    for node in nodes:
        if node in graph:
            counts[node] = counts.get(node, 0) + 1
    if not counts:
        return None

    cache_enabled = not _cache_disabled(graph)
    emb_by_node = None
    if cache_enabled:
        root = _graph_cache_root(graph)
        bucket = root.setdefault("node_context_embeddings", {})
        key = (radius, id(context_vectorizer))
        stamp = _graph_stamp(graph)
        cached = bucket.get(key)
        if cached is not None and cached.get("stamp") == stamp:
            emb_by_node = cached.get("values", {})
    if emb_by_node is None:
        node_list = list(graph.nodes())
        if not node_list:
            return None
        if radius is None:
            # All nodes share the whole-graph context; compute once.
            vectors = _transform_context_graphs(context_vectorizer, [graph.copy()])
            if vectors is None or vectors.shape[0] == 0:
                return None
            vec = np.asarray(vectors[0], dtype=float).ravel()
            emb_by_node = {node: vec for node in node_list}
        else:
            nodes_subset = set(graph.nodes())
            neighborhood_graphs = [
                _radius_neighborhood_graph(graph, nodes_subset, node, radius)
                for node in node_list
            ]
            vectors = _transform_context_graphs(context_vectorizer, neighborhood_graphs)
            if vectors is None or vectors.shape[0] != len(node_list):
                return None
            emb_by_node = {
                node: np.asarray(vectors[idx], dtype=float).ravel()
                for idx, node in enumerate(node_list)
            }
        if cache_enabled:
            bucket[key] = {"stamp": stamp, "values": emb_by_node}

    first = next(iter(emb_by_node.values()), None)
    if first is None:
        return None
    out = np.zeros_like(np.asarray(first, dtype=float).ravel())
    for node, mult in counts.items():
        vec = emb_by_node.get(node)
        if vec is None:
            continue
        out += np.asarray(vec, dtype=float).ravel() * float(mult)
    return out




def generate_image_conditioned_pruning_sequences(
    graph: nx.Graph,
    *,
    decomposition_function,
    nbits: int,
    label_mode: str = "graph_hash",
    preimage_cut_radius: Optional[int],
    image_cut_radius: Optional[int],
    preimage_context_radius: Optional[int] = None,
    image_context_radius: Optional[int] = None,
    context_vectorizer=None,
    use_context_embedding: bool = False,
    fixed_image_graph: Optional[nx.Graph] = None,
    assoc_map: Optional[dict] = None,
    return_image_steps: bool = False,
    return_cut_index: bool = False,
    seed: Optional[int] = None,
    include_start: bool = False,
    preimage_label_fn: Optional[Callable[[Optional[nx.Graph], object], str]] = None,
    image_label_fn: Optional[Callable[[nx.Graph, object], str]] = None,
):
    """
    Prune by removing image-node associations while recording anchor-aware cuts.

    Args:
        graph: Source preimage graph.
        decomposition_function: Decomposition used to build the fixed image graph.
        nbits: Hash bit width for image labels.
        label_mode: Label mode passed to graph_to_abstract_graph.
        preimage_cut_radius: Radius used for preimage anchor hashes.
        image_cut_radius: Radius used for image-context hashes.
        preimage_context_radius: Context radius for preimage embeddings.
        image_context_radius: Context radius for image embeddings.
        context_vectorizer: Vectorizer used for context embeddings.
        use_context_embedding: If True, compute context embeddings for cuts.
        fixed_image_graph: Optional precomputed image graph to reuse.
        assoc_map: Optional mapping preimage node -> image nodes.
        return_image_steps: If True, include image-graph snapshots per step.
        return_cut_index: If True, include the cut index entries.
        seed: RNG seed for pruning order.
        include_start: If True, include the initial graph in the output.
        preimage_label_fn: Optional label function for preimage nodes.
        image_label_fn: Optional label function for image nodes.

    Returns:
        list[nx.Graph] | tuple: Sequence of pruned graphs. If return_image_steps
        or return_cut_index is True, returns a tuple containing those extras.
    """
    if decomposition_function is None:
        raise ValueError("decomposition_function is required for image-conditioned pruning.")
    g = graph.copy()
    out: list[nx.Graph] = []
    image_steps: list[nx.Graph] = []
    if include_start:
        out.append(g.copy())

    rng = random.Random(seed)
    if fixed_image_graph is None:
        ag0 = graph_to_abstract_graph(
            g,
            decomposition_function=decomposition_function,
            nbits=nbits,
            label_mode=label_mode,
        )
        fixed_image_graph = ag0.image_graph.copy()
    if assoc_map is None:
        assoc_map = build_assoc_map_from_image_graph(fixed_image_graph)

    preimage_label_fn = preimage_label_fn or _default_preimage_label
    image_label_fn = image_label_fn or _default_image_label

    assoc_nodes_by_image = {}
    assoc_count = defaultdict(int)
    for img_node, data in fixed_image_graph.nodes(data=True):
        assoc = data.get("association")
        nodes = set(assoc.nodes()) if isinstance(assoc, nx.Graph) else set()
        assoc_nodes_by_image[img_node] = nodes
        for node in nodes:
            assoc_count[node] += 1

    removed_images = set()
    if include_start and return_image_steps:
        image_steps.append(fixed_image_graph.copy())

    local_cut_index: dict = defaultdict(list) if return_cut_index else None
    # Prune in reverse of the construction order used by the generator:
    # pick highest-degree image node, take a RNG-shuffled BFS order, then remove in reverse.
    def _bfs_order_rng(G: nx.Graph, start, rng: random.Random) -> list:
        if start is None or start not in G:
            return list(G.nodes())
        visited = set([start])
        order = [start]
        queue = deque([start])
        while queue:
            node = queue.popleft()
            neighbors = [n for n in G.neighbors(node) if n not in visited]
            rng.shuffle(neighbors)
            for neigh in neighbors:
                if neigh in visited:
                    continue
                visited.add(neigh)
                order.append(neigh)
                queue.append(neigh)
        return order

    if fixed_image_graph.number_of_nodes() > 0:
        degrees = dict(fixed_image_graph.degree())
        max_deg = max(degrees.values()) if degrees else 0
        candidates = [n for n, d in degrees.items() if d == max_deg]
        start = rng.choice(candidates) if candidates else None
        if start is not None and start in fixed_image_graph:
            prune_order = list(reversed(_bfs_order_rng(fixed_image_graph, start, rng)))
        else:
            prune_order = list(reversed(list(fixed_image_graph.nodes())))
    else:
        prune_order = []
    while g.number_of_nodes() > 0:
        img_node = None
        for candidate in prune_order:
            if candidate in removed_images:
                continue
            if assoc_nodes_by_image.get(candidate):
                img_node = candidate
                break
        if img_node is None:
            break
        removed_images.add(img_node)
        inner_nodes = set(assoc_nodes_by_image.get(img_node, set()))
        inner_nodes &= set(g.nodes())
        if not inner_nodes:
            continue
        for node in inner_nodes:
            assoc_count[node] = max(0, assoc_count.get(node, 0) - 1)
        remaining_nodes = {n for n, c in assoc_count.items() if c > 0}
        for inode, nodes in assoc_nodes_by_image.items():
            if inode in removed_images:
                continue
            if not (nodes & remaining_nodes):
                removed_images.add(inode)
        g2 = g.subgraph(remaining_nodes).copy()

        if return_image_steps:
            img_step = fixed_image_graph.copy()
            for inode, data in img_step.nodes(data=True):
                assoc = data.get("association")
                if not isinstance(assoc, nx.Graph):
                    continue
                if inode in removed_images:
                    data["association"] = nx.Graph()
                else:
                    data["association"] = assoc.subgraph(remaining_nodes).copy()
            img_step.graph["removed_images"] = set(removed_images)
            image_steps.append(img_step)

        if local_cut_index is not None:
            assoc_full = fixed_image_graph.nodes[img_node].get("association")
            if isinstance(assoc_full, nx.Graph):
                assoc_full = assoc_full.copy()
            else:
                assoc_full = nx.Graph()
            anchor_candidates = [
                node for node in assoc_full.nodes() if node in remaining_nodes
            ]
            anchor_nodes = tuple(sorted(anchor_candidates))
            outer_hash_map = cached_preimage_node_hash_map(
                g2,
                radius=preimage_cut_radius,
                preimage_label_fn=preimage_label_fn,
            )
            inner_hash_map = cached_preimage_node_hash_map(
                assoc_full,
                radius=preimage_cut_radius,
                preimage_label_fn=preimage_label_fn,
            )
            anchor_outer_hashes = tuple(
                outer_hash_map.get(node, 0)
                for node in anchor_nodes
            )
            anchor_inner_hashes = tuple(
                inner_hash_map.get(node, 0)
                for node in anchor_nodes
            )
            anchor_pairs = tuple(
                (outer_h, inner_h, node)
                for node, outer_h, inner_h in zip(
                    anchor_nodes, anchor_outer_hashes, anchor_inner_hashes
                )
            )
            image_cut_hash = _image_context_hash(
                fixed_image_graph,
                img_node,
                image_cut_radius,
                image_label_fn,
            )
            role_key = image_cut_hash
            cut_hash = _cut_hash_with_image(anchor_outer_hashes, image_cut_hash)
            pre_ctx = None
            img_ctx = None
            if use_context_embedding and context_vectorizer is not None:
                pre_ctx = cached_context_embedding_sum_for_nodes(
                    g2,
                    nodes=[n for n in anchor_nodes if n in g2],
                    radius=preimage_context_radius,
                    context_vectorizer=context_vectorizer,
                )
                img_ctx = cached_context_embedding_sum_for_nodes(
                    fixed_image_graph,
                    nodes=[img_node],
                    radius=image_context_radius,
                    context_vectorizer=context_vectorizer,
                )
            entry = CutArchiveEntry(
                assoc=assoc_full,
                assoc_hash=hash_graph(assoc_full),
                image_node=img_node,
                image_label=image_label_fn(fixed_image_graph, img_node),
                anchor_nodes=anchor_nodes,
                anchor_outer_hashes=anchor_outer_hashes,
                anchor_inner_hashes=anchor_inner_hashes,
                anchor_pairs=anchor_pairs,
                neighbor_overlap_signature=_neighbor_overlap_signature(
                    fixed_image_graph,
                    img_node,
                    image_cut_radius,
                    image_label_fn,
                ),
                cut_hash=cut_hash,
                preimage_ctx=pre_ctx,
                image_ctx=img_ctx,
                source="pruning",
            )
            local_cut_index[role_key].append(entry)

        g = g2
        out.append(g.copy())

    if return_image_steps:
        if return_cut_index:
            return out, image_steps, local_cut_index
        return out, image_steps
    if return_cut_index:
        return out, local_cut_index
    return out


def _deterministic_cut_entries_for_graph(
    graph: nx.Graph,
    *,
    decomposition_function,
    nbits: int,
    label_mode: str = "graph_hash",
    preimage_cut_radius: Optional[int],
    image_cut_radius: Optional[int],
    preimage_context_radius: Optional[int],
    image_context_radius: Optional[int],
    context_vectorizer=None,
    use_context_embedding: bool,
    include_subsets: bool,
    max_subset_size: int = 3,
    max_subsets: int = 128,
    seed: Optional[int] = None,
    preimage_label_fn: Optional[Callable[[Optional[nx.Graph], object], str]] = None,
    image_label_fn: Optional[Callable[[nx.Graph, object], str]] = None,
    preimage_graph: Optional[nx.Graph] = None,
    image_graph: Optional[nx.Graph] = None,
    assoc_map: Optional[dict] = None,
) -> tuple[dict, nx.Graph, dict]:
    """
    Build deterministic cut entries from a full graph.

    Args:
        graph: Source preimage graph.
        decomposition_function: Decomposition used to build the image graph.
        nbits: Hash bit width for image labels.
        label_mode: Label mode passed to graph_to_abstract_graph.
        preimage_cut_radius: Radius for preimage anchor hashes.
        image_cut_radius: Radius for image-context hashes.
        preimage_context_radius: Context radius for preimage embeddings.
        image_context_radius: Context radius for image embeddings.
        context_vectorizer: Vectorizer used for context embeddings.
        use_context_embedding: If True, compute context embeddings for cuts.
        include_subsets: If True, add subset-based cut entries.
        max_subset_size: Maximum subset size for neighbor subsets.
        max_subsets: Cap on number of subsets.
        seed: RNG seed for subset sampling.
        preimage_label_fn: Optional label function for preimage nodes.
        image_label_fn: Optional label function for image nodes.
        preimage_graph: Optional precomputed preimage graph.
        image_graph: Optional precomputed image graph.
        assoc_map: Optional precomputed association map.

    Returns:
        tuple[dict, nx.Graph, dict]: (cut_index, image_graph, assoc_map).
    """
    if decomposition_function is None:
        raise ValueError("decomposition_function is required for deterministic cut building.")
    rng = random.Random(seed)
    if preimage_graph is None or image_graph is None or assoc_map is None:
        ag = graph_to_abstract_graph(
            graph,
            decomposition_function=decomposition_function,
            nbits=nbits,
            label_mode=label_mode,
        )
        preimage_graph = ag.preimage_graph.copy()
        image_graph = ag.image_graph.copy()
        assoc_map = build_assoc_map_from_image_graph(image_graph)

    preimage_label_fn = preimage_label_fn or _default_preimage_label
    image_label_fn = image_label_fn or _default_image_label

    index: dict = defaultdict(list)

    def _add_entry(img_node, assoc_full, anchor_nodes, pre_graph, *, source: str) -> None:
        anchor_nodes = tuple(sorted(anchor_nodes))
        outer_hash_map = cached_preimage_node_hash_map(
            pre_graph,
            radius=preimage_cut_radius,
            preimage_label_fn=preimage_label_fn,
        )
        inner_hash_map = cached_preimage_node_hash_map(
            assoc_full,
            radius=preimage_cut_radius,
            preimage_label_fn=preimage_label_fn,
        )
        anchor_outer_hashes = tuple(
            outer_hash_map.get(node, 0)
            for node in anchor_nodes
        )
        anchor_inner_hashes = tuple(
            inner_hash_map.get(node, 0)
            for node in anchor_nodes
        )
        anchor_pairs = tuple(
            (outer_h, inner_h, node)
            for node, outer_h, inner_h in zip(
                anchor_nodes, anchor_outer_hashes, anchor_inner_hashes
            )
        )
        image_cut_hash = _image_context_hash(
            image_graph,
            img_node,
            image_cut_radius,
            image_label_fn,
        )
        role_key = image_cut_hash
        cut_hash = _cut_hash_with_image(anchor_outer_hashes, image_cut_hash)
        pre_ctx = None
        img_ctx = None
        if use_context_embedding and context_vectorizer is not None:
            pre_ctx = cached_context_embedding_sum_for_nodes(
                pre_graph,
                nodes=[n for n in anchor_nodes if n in pre_graph],
                radius=preimage_context_radius,
                context_vectorizer=context_vectorizer,
            )
            img_ctx = cached_context_embedding_sum_for_nodes(
                image_graph,
                nodes=[img_node],
                radius=image_context_radius,
                context_vectorizer=context_vectorizer,
            )
        entry = CutArchiveEntry(
            assoc=assoc_full,
            assoc_hash=hash_graph(assoc_full),
            image_node=img_node,
            image_label=image_label_fn(image_graph, img_node),
            anchor_nodes=anchor_nodes,
            anchor_outer_hashes=anchor_outer_hashes,
            anchor_inner_hashes=anchor_inner_hashes,
            anchor_pairs=anchor_pairs,
            neighbor_overlap_signature=_neighbor_overlap_signature(
                image_graph,
                img_node,
                image_cut_radius,
                image_label_fn,
            ),
            cut_hash=cut_hash,
            preimage_ctx=pre_ctx,
            image_ctx=img_ctx,
            source=source,
        )
        index[role_key].append(entry)

    for img_node, data in image_graph.nodes(data=True):
        assoc_full = data.get("association")
        if isinstance(assoc_full, nx.Graph):
            assoc_full = assoc_full.copy()
        else:
            assoc_full = nx.Graph()
        anchor_nodes = [
            node
            for node in assoc_full.nodes()
            if assoc_map.get(node, set()) - {img_node}
        ]
        remaining_nodes = {
            node
            for node, images in assoc_map.items()
            if images - {img_node}
        }
        pre_graph = preimage_graph.subgraph(remaining_nodes).copy()
        _add_entry(img_node, assoc_full, anchor_nodes, pre_graph, source="deterministic")

        if include_subsets:
            related_images = set()
            for node in assoc_full.nodes():
                related_images |= (assoc_map.get(node, set()) - {img_node})
            if not related_images:
                continue
            neighbors = list(related_images)
            subsets = _neighbor_subsets(
                neighbors,
                rng=rng,
                max_subset_size=max_subset_size,
                max_subsets=max_subsets,
            )
            for subset in subsets:
                subset_set = set(subset)
                if not subset_set:
                    continue
                subset_anchor_nodes = [
                    node
                    for node in assoc_full.nodes()
                    if assoc_map.get(node, set()) & subset_set
                ]
                if not subset_anchor_nodes:
                    continue
                remaining_nodes = {
                    node
                    for node, images in assoc_map.items()
                    if images & subset_set
                }
                pre_graph = preimage_graph.subgraph(remaining_nodes).copy()
                _add_entry(img_node, assoc_full, subset_anchor_nodes, pre_graph, source="subset")

    return index, image_graph, assoc_map


def build_image_conditioned_cut_index_from_pruning(
    generator_graphs: Sequence[nx.Graph],
    decomposition_function,
    nbits: int,
    n_pruning_iterations: int,
    preimage_cut_radius: Optional[int],
    image_cut_radius: Optional[int],
    preimage_context_radius: Optional[int] = None,
    image_context_radius: Optional[int] = None,
    context_vectorizer=None,
    use_context_embedding: bool = False,
    max_num_anchors: Optional[int] = None,
    max_num_anchor_sets_retry: Optional[int] = None,
    seed: Optional[int] = None,
    label_mode: str = "graph_hash",
):
    """
    Build a cut index by image-conditioned pruning of generator graphs.

    Args:
        generator_graphs: Source graphs used to build the cut index.
        decomposition_function: Decomposition used to build the image graph.
        nbits: Hash bit width for image labels.
        n_pruning_iterations: Number of pruning iterations per graph.
        preimage_cut_radius: Radius for preimage anchor hashes.
        image_cut_radius: Radius for image-context hashes.
        preimage_context_radius: Context radius for preimage embeddings.
        image_context_radius: Context radius for image embeddings.
        context_vectorizer: Vectorizer used for context embeddings.
        use_context_embedding: If True, compute context embeddings for cuts.
        max_num_anchors: Maximum subset size for neighbor subsets.
        max_num_anchor_sets_retry: Cap on number of subsets.
        seed: RNG seed for pruning order.
        label_mode: Label mode passed to graph_to_abstract_graph.

    Returns:
        tuple: (cut_index, donors, fixed_image_graph, assoc_map).
    """
    rng = random.Random(seed)
    if not generator_graphs:
        return {}, [], None, {}
    cut_index: dict = defaultdict(list)
    donors = []
    fixed_image_graph = None
    assoc_map = None
    n_iters = max(1, int(n_pruning_iterations))
    max_num_anchors = 3 if max_num_anchors is None else int(max_num_anchors)
    max_num_anchor_sets_retry = (
        128 if max_num_anchor_sets_retry is None else int(max_num_anchor_sets_retry)
    )
    for source_graph in generator_graphs:
        ag0 = graph_to_abstract_graph(
            source_graph,
            decomposition_function=decomposition_function,
            nbits=nbits,
            label_mode=label_mode,
        )
        image_graph = ag0.image_graph.copy()
        preimage_graph = ag0.preimage_graph.copy()
        local_assoc_map = build_assoc_map_from_image_graph(image_graph)

        for _ in range(n_iters):
            seq, local_index = generate_image_conditioned_pruning_sequences(
                source_graph,
                decomposition_function=decomposition_function,
                nbits=nbits,
                label_mode=label_mode,
                preimage_cut_radius=preimage_cut_radius,
                image_cut_radius=image_cut_radius,
                preimage_context_radius=preimage_context_radius,
                image_context_radius=image_context_radius,
                context_vectorizer=context_vectorizer,
                use_context_embedding=use_context_embedding,
                fixed_image_graph=image_graph,
                assoc_map=local_assoc_map,
                return_cut_index=True,
                seed=rng.randint(0, 10**9),
            )
            donors.extend(seq)
            if local_index:
                for key, entries in local_index.items():
                    cut_index[key].extend(entries)

        include_subsets = (preimage_cut_radius or 0) > 0 or (image_cut_radius or 0) > 0
        det_index, det_img_graph, det_assoc_map = _deterministic_cut_entries_for_graph(
            source_graph,
            decomposition_function=decomposition_function,
            nbits=nbits,
            label_mode=label_mode,
            preimage_cut_radius=preimage_cut_radius,
            image_cut_radius=image_cut_radius,
            preimage_context_radius=preimage_context_radius,
            image_context_radius=image_context_radius,
            context_vectorizer=context_vectorizer,
            use_context_embedding=use_context_embedding,
            include_subsets=include_subsets,
            max_subset_size=max_num_anchors,
            max_subsets=max_num_anchor_sets_retry,
            seed=rng.randint(0, 10**9),
            preimage_graph=preimage_graph,
            image_graph=image_graph,
            assoc_map=local_assoc_map,
        )
        for key, entries in det_index.items():
            cut_index[key].extend(entries)
        if fixed_image_graph is None:
            fixed_image_graph = det_img_graph
            assoc_map = det_assoc_map
    return cut_index, donors, fixed_image_graph, assoc_map


# =================================================================================================
# Generator
# =================================================================================================
