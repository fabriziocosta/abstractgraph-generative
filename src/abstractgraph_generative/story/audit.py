"""Quality gates and provenance audit reporting for story graphs."""

from __future__ import annotations

import html
from pathlib import Path

from abstractgraph_generative.story.validation import validate_story_graph


def graph_validity_stats(graph: dict) -> dict:
    """Compute structural validation statistics for a graph.

    Args:
        graph: Input story graph.

    Returns:
        Dictionary with counts and basic diagnostics.
    """

    valid = validate_story_graph(graph)
    node_types = {}
    for node in valid["nodes"]:
        node_types[node["type"]] = node_types.get(node["type"], 0) + 1

    event_nodes = [node for node in valid["nodes"] if node["type"] == "EVT"]
    event_ids = {node["id"] for node in event_nodes}

    roles_per_event = {event_id: 0 for event_id in event_ids}
    for edge in valid["edges"]:
        if edge["source"] in event_ids and edge["label"] in {
            "AGENT",
            "PATIENT",
            "TARGET",
            "RECIPIENT",
            "INSTRUMENT",
            "LOCATION",
            "TIME",
        }:
            roles_per_event[edge["source"]] += 1

    disconnected_events = [event_id for event_id, n_roles in roles_per_event.items() if n_roles == 0]

    return {
        "doc_id": valid["doc_id"],
        "n_nodes": len(valid["nodes"]),
        "n_edges": len(valid["edges"]),
        "node_type_counts": node_types,
        "n_events": len(event_nodes),
        "avg_roles_per_event": (sum(roles_per_event.values()) / len(roles_per_event)) if roles_per_event else 0.0,
        "events_without_roles": disconnected_events,
    }


def write_provenance_audit_html(graph: dict, out_path: str | Path) -> str:
    """Write a minimal provenance audit report.

    Args:
        graph: Input story graph.
        out_path: Output HTML path.

    Returns:
        String path to generated report.
    """

    valid = validate_story_graph(graph)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    sentence_by_id = {s["sentence_id"]: s for s in valid.get("sentences", [])}
    nodes_by_id = {node["id"]: node for node in valid["nodes"]}

    lines = [
        "<html><head><meta charset='utf-8'><title>Story Graph Audit</title></head><body>",
        f"<h1>Story Graph Audit: {html.escape(valid['doc_id'])}</h1>",
        "<h2>Sentences</h2><ol>",
    ]

    for sentence in sorted(valid.get("sentences", []), key=lambda x: x["sentence_id"]):
        lines.append(
            "<li>"
            f"<b>{html.escape(sentence['sentence_id'])}</b>: "
            f"{html.escape(sentence['text'])}"
            "</li>"
        )
    lines.append("</ol>")

    lines.append("<h2>Events</h2><ul>")
    for node in [n for n in valid["nodes"] if n["type"] == "EVT"]:
        prov = node["provenance"]
        sentence = sentence_by_id.get(prov["sentence_id"], {})
        lines.append(
            "<li>"
            f"<b>{html.escape(node['id'])}</b> [{html.escape(node['event_type'])}] "
            f"sentence={html.escape(prov['sentence_id'])}, span=({prov['char_start']},{prov['char_end']})"
            "<br/>"
            f"{html.escape(sentence.get('text', ''))}"
            "</li>"
        )
    lines.append("</ul>")

    lines.append("<h2>Role Edges</h2><ul>")
    for edge in valid["edges"]:
        if edge["label"] not in {"AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT", "LOCATION", "TIME"}:
            continue
        source = nodes_by_id.get(edge["source"], {})
        target = nodes_by_id.get(edge["target"], {})
        lines.append(
            "<li>"
            f"{html.escape(edge['source'])} --{html.escape(edge['label'])}--> {html.escape(edge['target'])} "
            f"({html.escape(source.get('event_type', '?'))} -> {html.escape(target.get('canonical_name', '?'))})"
            "</li>"
        )
    lines.append("</ul>")

    lines.append("</body></html>")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    return str(out_file)
