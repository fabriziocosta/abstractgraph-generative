"""Materialization helpers for conditional autoregressive generation."""

from __future__ import annotations

import networkx as nx


def _next_node_id(graph: nx.Graph) -> int:
    """
    Return the next integer node id not present in the graph.

    Args:
        graph: Graph whose integer node ids are inspected.

    Returns:
        int: Next available integer node id.
    """
    try:
        import numpy as _np
        int_nodes = [n for n in graph.nodes if isinstance(n, (int, _np.integer))]
    except Exception:
        int_nodes = [n for n in graph.nodes if isinstance(n, int)]
    return (max(int_nodes) if int_nodes else -1) + 1


def _merge_with_anchors(
    current: nx.Graph,
    assoc: nx.Graph,
    anchor_map: dict,
) -> tuple[nx.Graph, dict]:
    """
    Merge an association subgraph into the current graph with anchors.

    Args:
        current: Current preimage graph to merge into.
        assoc: Association subgraph to merge.
        anchor_map: Mapping assoc node -> existing node id in current.

    Returns:
        tuple[nx.Graph, dict]: (new_graph, id_map) where id_map maps assoc node
        to its node id in new_graph.
    """
    def _normalized_node_attrs(attrs: dict) -> dict:
        node_attrs = dict(attrs)
        label = node_attrs.get("label")
        # Restore original labels when a placeholder is present.
        if label in (None, "-") and "original_label" in node_attrs:
            restored = node_attrs.get("original_label")
            if restored is not None:
                node_attrs["label"] = restored
        return node_attrs
    def _normalized_edge_attrs(attrs: dict) -> dict:
        edge_attrs = dict(attrs)
        label = edge_attrs.get("label")
        if label in (None, "-") and "original_label" in edge_attrs:
            restored = edge_attrs.get("original_label")
            if restored is not None:
                edge_attrs["label"] = restored
        return edge_attrs

    new_graph = current.copy()
    id_map: dict = {}
    next_id = _next_node_id(new_graph)

    for node, data in assoc.nodes(data=True):
        if node in anchor_map:
            target = anchor_map[node]
            id_map[node] = target
            # Merge node attributes (preserve existing keys)
            for k, v in _normalized_node_attrs(data).items():
                if k not in new_graph.nodes[target]:
                    new_graph.nodes[target][k] = v
        else:
            node_attrs = _normalized_node_attrs(data)
            new_graph.add_node(next_id, **node_attrs)
            id_map[node] = next_id
            next_id += 1

    for u, v, data in assoc.edges(data=True):
        mu = id_map[u]
        mv = id_map[v]
        if new_graph.has_edge(mu, mv):
            edge_attrs = _normalized_edge_attrs(data)
            for k, v in edge_attrs.items():
                if k not in new_graph.edges[mu, mv]:
                    new_graph.edges[mu, mv][k] = v
                elif k == "label" and new_graph.edges[mu, mv].get("label") in (None, "-"):
                    new_graph.edges[mu, mv][k] = v
        else:
            new_graph.add_edge(mu, mv, **_normalized_edge_attrs(data))

    return new_graph, id_map


# =================================================================================================
# Display helpers
# =================================================================================================

