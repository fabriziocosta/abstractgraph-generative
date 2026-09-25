"""Small component and anchor operations shared by graph workflows."""

from __future__ import annotations

from typing import Optional

import networkx as nx

from abstractgraph.hashing import hash_graph


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
    lengths = nx.single_source_shortest_path_length(
        graph, center_node, cutoff=int(radius)
    )
    nodes = list(lengths.keys())
    return graph.subgraph(nodes).copy()


def anchor_type_train(
    graph_full: nx.Graph, node, radius: int, *, nbits: int = 19
) -> int:
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


def anchor_type_current(
    graph_partial: nx.Graph, node, radius: int, *, nbits: int = 19
) -> int:
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
