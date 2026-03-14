"""Visualization utilities for discrete story graphs.

This module renders validated story-graph payloads using NetworkX and Matplotlib
for quick structural inspection during notebook workflows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from abstractgraph_generative.story.validation import validate_story_graph

NODE_TYPE_COLORS = {
    "ENT": "#4E79A7",
    "EVT": "#F28E2B",
    "REL": "#76B7B2",
    "GOAL": "#59A14F",
    "INT": "#E15759",
    "MORAL": "#B07AA1",
}


def _summary_excerpt(
    text: str,
    *,
    max_chars: int | None = 120,
    max_sentences: int | None = 2,
) -> str:
    """Create a configurable short excerpt from summary text.

    Args:
        text: Input text.
        max_chars: Max characters in output excerpt.
        max_sentences: Max sentence count in output excerpt.

    Returns:
        Excerpt text.
    """

    value = (text or "").strip()
    if not value:
        return ""
    excerpt = value
    if max_sentences is not None and max_sentences > 0:
        parts = [p.strip() for p in value.replace("!", ".").replace("?", ".").split(".") if p.strip()]
        if parts:
            excerpt = ". ".join(parts[:max_sentences]).strip()
            if value.endswith((".", "!", "?")) and not excerpt.endswith("."):
                excerpt += "."
    if max_chars is not None and max_chars > 0 and len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 3].rstrip() + "..."
    return excerpt


def to_networkx_story_graph(
    graph: dict,
    *,
    include_summary_excerpt: bool = False,
    summary_excerpt_chars: int | None = 120,
    summary_excerpt_sentences: int | None = 2,
    dictionary_dir: str | None = None,
) -> nx.DiGraph:
    """Convert a story-graph dictionary to a labeled NetworkX graph.

    Args:
        graph: Story graph payload.
        include_summary_excerpt: If True, append summary excerpt to node label.
        summary_excerpt_chars: Maximum chars for summary excerpt.
        summary_excerpt_sentences: Maximum sentences for summary excerpt.
        dictionary_dir: Optional dictionary directory for induced vocab validation.

    Returns:
        Directed graph with copied node and edge attributes.
    """

    if dictionary_dir is None:
        meta = graph.get("_dictionary_dir")
        if isinstance(meta, str) and meta.strip():
            dictionary_dir = meta
    valid = validate_story_graph(graph, dictionary_dir=dictionary_dir)
    g = nx.DiGraph()

    for node in valid["nodes"]:
        node_id = node["id"]
        node_type = node.get("type", "?")
        if node_type == "ENT":
            entity_label = (
                str(node.get("label", "")).strip()
                or str(node.get("dictionary_name", "")).strip()
                or str(node.get("canonical_name", "")).strip()
            )
            entity_name = (
                str(node.get("name", "")).strip()
                or str(node.get("canonical_name", "")).strip()
            )
            if entity_name:
                label = f"{node_id}\n{entity_label} | {entity_name}"
            else:
                label = f"{node_id}\n{entity_label}"
        elif node_type == "EVT":
            event_label = str(node.get("label", node.get("event_type", ""))).strip()
            modality = str(node.get("modality", "ASSERTED")).strip()
            label = f"{node_id}\n{event_label} [{modality}]"
        elif node_type == "REL":
            label = f"{node_id}\n{node.get('relation_type', '')}"
        elif node_type == "GOAL":
            label = f"{node_id}\n{node.get('goal_type', '')}"
        elif node_type == "INT":
            intention_label = str(node.get("label", node.get("intention_type", ""))).strip()
            status = str(node.get("status", "")).strip()
            label = f"{node_id}\n{intention_label} [{status}]"
        else:
            label = node_id

        if include_summary_excerpt:
            summary = str(node.get("summary_text", "")).strip()
            if summary:
                excerpt = _summary_excerpt(
                    summary,
                    max_chars=summary_excerpt_chars,
                    max_sentences=summary_excerpt_sentences,
                )
                if excerpt:
                    label = f"{label}\n---\n{excerpt}"

        g.add_node(node_id, **node, viz_label=label)

    for edge in valid["edges"]:
        g.add_edge(edge["source"], edge["target"], **edge)

    return g


def plot_story_graph(
    graph: dict,
    *,
    figsize: tuple[float, float] = (13.0, 8.0),
    layout_seed: int = 7,
    show_edge_labels: bool = True,
    dictionary_dir: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a story graph with node colors by type and edge labels.

    Args:
        graph: Story graph payload.
        figsize: Figure size in inches.
        layout_seed: Seed used by spring layout.
        show_edge_labels: Whether to draw edge label text.
        dictionary_dir: Optional dictionary directory for induced vocab validation.

    Returns:
        Tuple `(figure, axes)` containing the rendered plot.
    """

    g = to_networkx_story_graph(graph, dictionary_dir=dictionary_dir)
    fig, ax = plt.subplots(figsize=figsize)

    pos = nx.spring_layout(g, seed=layout_seed)
    node_colors = [NODE_TYPE_COLORS.get(g.nodes[n].get("type", ""), "#9C9C9C") for n in g.nodes]
    node_labels = {n: g.nodes[n].get("viz_label", n) for n in g.nodes}

    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=1450, alpha=0.95, ax=ax)
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=8, ax=ax)
    nx.draw_networkx_edges(g, pos, arrows=True, arrowstyle="-|>", width=1.3, alpha=0.75, ax=ax)

    if show_edge_labels:
        edge_labels = {(u, v): d.get("label", "") for u, v, d in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8, rotate=False, ax=ax)

    ax.set_title("Story Graph Visualization")
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def story_graph_to_dot_string(graph: dict, *, dictionary_dir: str | None = None) -> str:
    """Serialize a story graph to DOT text.

    Args:
        graph: Story graph payload.
        dictionary_dir: Optional dictionary directory for induced vocab validation.

    Returns:
        DOT language string.
    """

    g = to_networkx_story_graph(
        graph,
        include_summary_excerpt=True,
        summary_excerpt_chars=120,
        summary_excerpt_sentences=2,
        dictionary_dir=dictionary_dir,
    )
    try:
        agraph = nx.nx_agraph.to_agraph(g)
    except Exception as exc:
        raise RuntimeError(
            "DOT export requires pygraphviz. Install graphviz + pygraphviz."
        ) from exc

    for node_id in g.nodes:
        node = g.nodes[node_id]
        n = agraph.get_node(node_id)
        n.attr["label"] = node.get("viz_label", node_id)
        n.attr["shape"] = "box" if node.get("type") == "EVT" else "ellipse"
        n.attr["style"] = "filled"
        n.attr["fillcolor"] = NODE_TYPE_COLORS.get(node.get("type", ""), "#9C9C9C")
        n.attr["fontname"] = "Helvetica"

    for u, v, data in g.edges(data=True):
        e = agraph.get_edge(u, v)
        e.attr["label"] = data.get("label", "")
        e.attr["fontname"] = "Helvetica"
        e.attr["fontsize"] = "10"

    agraph.graph_attr.update(rankdir="LR", splines="true", overlap="false")
    return agraph.to_string()


def render_story_graph_dot(
    graph: dict,
    output_path: str | Path,
    *,
    format: str = "svg",
    prog: str = "dot",
    include_summary_excerpt: bool = True,
    summary_excerpt_chars: int | None = 120,
    summary_excerpt_sentences: int | None = 2,
    dictionary_dir: str | None = None,
) -> str:
    """Render a story graph using Graphviz DOT.

    Args:
        graph: Story graph payload.
        output_path: Output image path.
        format: Graphviz output format (e.g., svg, png, pdf).
        prog: Graphviz program name; defaults to dot.
        include_summary_excerpt: If True, append summary excerpt in node labels.
        summary_excerpt_chars: Maximum characters for summary excerpt.
        summary_excerpt_sentences: Maximum sentences for summary excerpt.
        dictionary_dir: Optional dictionary directory for induced vocab validation.

    Returns:
        String path to rendered output file.
    """

    g = to_networkx_story_graph(
        graph,
        include_summary_excerpt=include_summary_excerpt,
        summary_excerpt_chars=summary_excerpt_chars,
        summary_excerpt_sentences=summary_excerpt_sentences,
        dictionary_dir=dictionary_dir,
    )
    try:
        agraph = nx.nx_agraph.to_agraph(g)
    except Exception as exc:
        raise RuntimeError(
            "DOT rendering requires pygraphviz. Install graphviz + pygraphviz."
        ) from exc

    for node_id in g.nodes:
        node = g.nodes[node_id]
        n = agraph.get_node(node_id)
        n.attr["label"] = node.get("viz_label", node_id)
        n.attr["shape"] = "box" if node.get("type") == "EVT" else "ellipse"
        n.attr["style"] = "filled"
        n.attr["fillcolor"] = NODE_TYPE_COLORS.get(node.get("type", ""), "#9C9C9C")
        n.attr["fontname"] = "Helvetica"
        n.attr["fontsize"] = "11"

    for u, v, data in g.edges(data=True):
        e = agraph.get_edge(u, v)
        e.attr["label"] = data.get("label", "")
        e.attr["fontname"] = "Helvetica"
        e.attr["fontsize"] = "10"

    agraph.graph_attr.update(rankdir="LR", splines="true", overlap="false")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    agraph.draw(path=str(output), format=format, prog=prog)
    return str(output)
