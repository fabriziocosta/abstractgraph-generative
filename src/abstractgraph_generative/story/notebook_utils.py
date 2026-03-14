"""Notebook-oriented helpers for story-graph roundtrip workflows.

These utilities keep notebooks concise by wrapping common extraction,
rendering, generation, and validation steps.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from abstractgraph_generative.story.graph_to_text import (
    build_story_plan,
    realize_story_from_graph_with_repair,
    render_story_with_style,
)
from abstractgraph_generative.story.text_to_graph import build_story_graph_from_text
from abstractgraph_generative.story.validation import validate_story_graph
from abstractgraph_generative.story.visualization import render_story_graph_dot
from abstractgraph_generative.story.vocab import load_vocab, resolve_vocab_path


def resolve_model_name(model: str | None = None) -> str:
    """Resolve model name from argument or environment.

    Args:
        model: Optional explicit model name.

    Returns:
        Model name string.
    """

    return model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def select_story(fables: list[str], story_index: int) -> tuple[str, str]:
    """Return `(doc_id, story_text)` for a selected story index.

    Args:
        fables: Story list.
        story_index: Index into the list.

    Returns:
        Tuple containing document id and story text.
    """

    if story_index < 0 or story_index >= len(fables):
        raise IndexError(f"story_index {story_index} out of range [0, {len(fables)-1}]")
    doc_id = f"aesop_{story_index:04d}"
    return doc_id, fables[story_index]


def extract_story_graph(
    story_text: str,
    doc_id: str,
    *,
    ask_llm_fn,
    model: str | None = None,
    add_node_summaries: bool = True,
    dictionary_dir: str | None = None,
    verbose: bool = False,
) -> tuple[dict, dict]:
    """Build a story graph and return validity stats.

    Args:
        story_text: Story text.
        doc_id: Document id.
        ask_llm_fn: LLM callable.
        model: Optional model name.
        add_node_summaries: Whether to add node summaries.
        dictionary_dir: Optional dictionary directory.
        verbose: If True, print dictionary usage stats.

    Returns:
        Tuple `(graph, stats)`.
    """

    graph = build_story_graph_from_text(
        text=story_text,
        doc_id=doc_id,
        ask_llm_fn=ask_llm_fn,
        model=resolve_model_name(model),
        add_node_summaries=add_node_summaries,
        dictionary_dir=dictionary_dir,
    )
    valid = validate_story_graph(graph, dictionary_dir=dictionary_dir)
    if dictionary_dir is not None:
        valid["_dictionary_dir"] = str(dictionary_dir)
    if verbose:
        print("Dictionary stats")
        print("-" * 100)
        print(dump_json(dictionary_usage_stats(dictionary_dir=dictionary_dir)))
    stats = {
        "doc_id": valid["doc_id"],
        "n_nodes": len(valid["nodes"]),
        "n_edges": len(valid["edges"]),
        "n_events": sum(1 for n in valid["nodes"] if n.get("type") == "EVT"),
        "n_entities": sum(1 for n in valid["nodes"] if n.get("type") == "ENT"),
        "n_goals": sum(1 for n in valid["nodes"] if n.get("type") == "GOAL"),
        "n_intentions": sum(1 for n in valid["nodes"] if n.get("type") == "INT"),
    }
    return valid, stats


def dictionary_usage_stats(dictionary_dir: str | None = None) -> dict:
    """Return stats about dictionary files used by story-graph functions.

    Args:
        dictionary_dir: Optional dictionary folder.

    Returns:
        Dictionary containing vocab path and per-category sizes.
    """

    vocab_path = resolve_vocab_path(dictionary_dir=dictionary_dir)
    vocab = load_vocab(dictionary_dir=dictionary_dir)
    out = {
        "dictionary_dir": str(dictionary_dir) if dictionary_dir is not None else None,
        "vocab_path": str(vocab_path),
        "sizes": {
            "entity_types": len(vocab.get("entity_types", [])),
            "relation_types": len(vocab.get("relation_types", [])),
            "event_types": len(vocab.get("event_types", [])),
            "goal_types": len(vocab.get("goal_types", [])),
            "intention_types": len(vocab.get("intention_types", [])),
            "outcome_types": len(vocab.get("outcome_types", [])),
        },
    }
    if dictionary_dir is not None:
        induced_path = Path(dictionary_dir) / "induced_closed_alphabet.json"
        out["induced_path"] = str(induced_path)
        if induced_path.exists():
            try:
                payload = json.loads(induced_path.read_text(encoding="utf-8"))
                consolidated = payload.get("consolidated", {}) if isinstance(payload, dict) else {}
                out["induced_sizes"] = {
                    "entities": len(consolidated.get("entities", {}).get("canonical_terms", [])),
                    "relations": len(consolidated.get("relations", {}).get("canonical_terms", [])),
                    "goals": len(consolidated.get("goals", {}).get("canonical_terms", [])),
                    "intentions": len(consolidated.get("intentions", {}).get("canonical_terms", [])),
                    "events": len(consolidated.get("events", {}).get("canonical_terms", [])),
                }
            except Exception:
                out["induced_sizes"] = {}
    return out


def intention_summary_rows(graph: dict) -> list[dict]:
    """Return structured intention-summary rows for printing.

    Args:
        graph: Story graph.

    Returns:
        List of summary rows.
    """

    node_by_id = {n["id"]: n for n in graph.get("nodes", [])}
    rows = []
    for intention in [n for n in graph.get("nodes", []) if n.get("type") == "INT"]:
        int_id = intention["id"]
        owners = [
            edge["source"]
            for edge in graph.get("edges", [])
            if edge.get("label") == "INTENDS" and edge.get("target") == int_id
        ]
        owner_names = [node_by_id.get(o, {}).get("canonical_name", o) for o in owners]
        effects = [
            edge
            for edge in graph.get("edges", [])
            if edge.get("target") == int_id and edge.get("label") in {"ADVANCES", "THWARTS", "FULFILLS", "FAILS"}
        ]
        rows.append(
            {
                "id": int_id,
                "owners": owner_names,
                "label": intention.get("label", intention.get("intention_type", "")),
                "name": intention.get("name", intention.get("description", "")),
                "intention_type": intention.get("intention_type"),
                "status": intention.get("status"),
                "outcome": intention.get("outcome_type"),
                "effects": [
                    {
                        "event_id": edge.get("source"),
                        "event_type": node_by_id.get(edge.get("source"), {}).get("event_type", "?"),
                        "effect": edge.get("label"),
                        "confidence": edge.get("confidence"),
                    }
                    for edge in effects
                ],
            }
        )
    return rows


def entity_summary_rows(graph: dict) -> list[dict]:
    """Return structured entity-summary rows for printing.

    Args:
        graph: Story graph.

    Returns:
        List of entity rows.
    """

    role_labels = {"AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT", "LOCATION", "TIME"}
    role_counts: dict[str, int] = {}
    for edge in graph.get("edges", []):
        if edge.get("label") not in role_labels:
            continue
        ent_id = edge.get("target")
        if not isinstance(ent_id, str):
            continue
        role_counts[ent_id] = role_counts.get(ent_id, 0) + 1

    rows = []
    for node in [n for n in graph.get("nodes", []) if n.get("type") == "ENT"]:
        ent_id = node["id"]
        entity_label = node.get("label", "") or node.get("dictionary_name", "")
        entity_name = node.get("name", "") or node.get("canonical_name", "")
        canonical_name = node.get("canonical_name", "")
        rows.append(
            {
                "id": ent_id,
                "label": entity_label or canonical_name,
                "name": entity_name or canonical_name,
                "canonical_name": canonical_name,
                "entity_type": node.get("entity_type", ""),
                "role_links": role_counts.get(ent_id, 0),
                "summary_text": node.get("summary_text", ""),
            }
        )
    rows.sort(key=lambda r: r["id"])
    return rows


def event_summary_rows(graph: dict, *, dictionary_dir: str | None = None) -> list[dict]:
    """Return structured event-summary rows in plan order.

    Args:
        graph: Story graph.
        dictionary_dir: Optional dictionary directory.

    Returns:
        List of event rows.
    """

    if dictionary_dir is None:
        meta = graph.get("_dictionary_dir")
        if isinstance(meta, str) and meta.strip():
            dictionary_dir = meta
    plan = build_story_plan(graph, dictionary_dir=dictionary_dir)
    node_by_id = {n["id"]: n for n in graph.get("nodes", []) if "id" in n}
    rows = []
    for row in plan:
        event_id = row.get("event_id", "")
        evt_node = node_by_id.get(event_id, {})
        roles = row.get("roles", {})
        role_pairs = []
        for role in ["AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT", "LOCATION", "TIME"]:
            ent_id = roles.get(role)
            if not ent_id:
                continue
            ent_node = node_by_id.get(ent_id, {})
            ent_name = (
                ent_node.get("label", "")
                or ent_node.get("dictionary_name", "")
                or ent_node.get("canonical_name", ent_id)
            )
            role_pairs.append(f"{role}={ent_name}")
        rows.append(
            {
                "order": row.get("order"),
                "id": event_id,
                "label": evt_node.get("label", row.get("event_type", "")),
                "name": evt_node.get("name", evt_node.get("summary_text", "")),
                "event_type": row.get("event_type", ""),
                "modality": row.get("modality", ""),
                "roles": role_pairs,
                "summary_text": evt_node.get("summary_text", ""),
                "evidence_text": evt_node.get("evidence_text", ""),
            }
        )
    return rows


def relation_summary_rows(graph: dict) -> list[dict]:
    """Return structured relation-summary rows.

    Args:
        graph: Story graph.

    Returns:
        List of relation rows.
    """

    node_by_id = {n["id"]: n for n in graph.get("nodes", []) if "id" in n}
    rows = []
    for rel in [n for n in graph.get("nodes", []) if n.get("type") == "REL"]:
        rel_id = rel["id"]
        subj = next(
            (
                e.get("target")
                for e in graph.get("edges", [])
                if e.get("source") == rel_id and e.get("label") == "REL_SUBJECT"
            ),
            "",
        )
        obj = next(
            (
                e.get("target")
                for e in graph.get("edges", [])
                if e.get("source") == rel_id and e.get("label") == "REL_OBJECT"
            ),
            "",
        )
        rows.append(
            {
                "id": rel_id,
                "label": rel.get("label", rel.get("relation_type", "")),
                "name": rel.get("name", rel.get("relation_type", "")),
                "relation_type": rel.get("relation_type", ""),
                "subject": node_by_id.get(subj, {}).get("label", "")
                or node_by_id.get(subj, {}).get("dictionary_name", "")
                or node_by_id.get(subj, {}).get("canonical_name", subj),
                "object": node_by_id.get(obj, {}).get("label", "")
                or node_by_id.get(obj, {}).get("dictionary_name", "")
                or node_by_id.get(obj, {}).get("canonical_name", obj),
            }
        )
    rows.sort(key=lambda r: r["id"])
    return rows


def goal_summary_rows(graph: dict) -> list[dict]:
    """Return structured goal-summary rows.

    Args:
        graph: Story graph.

    Returns:
        List of goal rows.
    """

    node_by_id = {n["id"]: n for n in graph.get("nodes", []) if "id" in n}
    rows = []
    for goal in [n for n in graph.get("nodes", []) if n.get("type") == "GOAL"]:
        goal_id = goal["id"]
        owners = [
            edge.get("source")
            for edge in graph.get("edges", [])
            if edge.get("label") == "HAS_GOAL" and edge.get("target") == goal_id
        ]
        owner_names = [
            node_by_id.get(owner, {}).get("label", "")
            or node_by_id.get(owner, {}).get("dictionary_name", "")
            or node_by_id.get(owner, {}).get("canonical_name", owner)
            for owner in owners
        ]
        motiv_ints = [
            edge.get("target")
            for edge in graph.get("edges", [])
            if edge.get("label") == "MOTIVATES" and edge.get("source") == goal_id
        ]
        motiv_types = []
        seen_motiv_types = set()
        for iid in motiv_ints:
            intention_type = str(node_by_id.get(iid, {}).get("intention_type", iid))
            if intention_type in seen_motiv_types:
                continue
            seen_motiv_types.add(intention_type)
            motiv_types.append(intention_type)
        rows.append(
            {
                "id": goal_id,
                "owners": owner_names,
                "label": goal.get("label", goal.get("goal_type", "")),
                "name": goal.get("name", goal.get("goal_type", "")),
                "goal_type": goal.get("goal_type", ""),
                "status": goal.get("status", ""),
                "outcome": goal.get("outcome_type", ""),
                "motivates": motiv_types,
            }
        )
    rows.sort(key=lambda r: r["id"])
    return rows


def print_graph_semantic_layers(graph: dict, *, dictionary_dir: str | None = None) -> None:
    """Print graph semantic layers: entities, intentions, then events.

    Args:
        graph: Story graph.
        dictionary_dir: Optional dictionary directory.

    Returns:
        None.
    """

    print("Entities")
    print("-" * 100)
    entities = entity_summary_rows(graph)
    if not entities:
        print("No entity nodes extracted.")
    else:
        for row in entities:
            print(
                f"{row['id']}: label={row['label']}, name={row['name']}, type={row['entity_type']}, role_links={row['role_links']}"
            )
            if row.get("canonical_name") and row.get("canonical_name") != row.get("name"):
                print(f"  canonical: {row['canonical_name']}")
            if row.get("summary_text"):
                print(f"  summary: {row['summary_text']}")

    print("\nRelations")
    print("-" * 100)
    relations = relation_summary_rows(graph)
    if not relations:
        print("No relation nodes extracted.")
    else:
        for row in relations:
            print(
                f"{row['id']}: label={row['label']}, name={row['name']}, "
                f"relation={row['relation_type']}({row['subject']}, {row['object']})"
            )

    print("\nGoals")
    print("-" * 100)
    goals = goal_summary_rows(graph)
    if not goals:
        print("No goal nodes extracted.")
    else:
        for row in goals:
            print(
                f"{row['id']}: label={row['label']}, name={row['name']}, "
                f"owner={row['owners']}, goal={row['goal_type']}, "
                f"status={row['status']}, outcome={row['outcome']}"
            )
            if row["motivates"]:
                print(f"  motivates intentions: {row['motivates']}")

    print("\nIntentions")
    print("-" * 100)
    intentions = intention_summary_rows(graph)
    if not intentions:
        print("No intention nodes extracted.")
    else:
        for row in intentions:
            print(
                f"{row['id']}: label={row['label']}, name={row['name']}, "
                f"owner={row['owners']}, intention={row['intention_type']}, "
                f"status={row['status']}, outcome={row['outcome']}"
            )
            for effect in row["effects"]:
                print(
                    f"  - {effect['event_id']} ({effect['event_type']}): "
                    f"{effect['effect']} conf={effect['confidence']}"
                )

    print("\nEvents")
    print("-" * 100)
    if dictionary_dir is None:
        meta = graph.get("_dictionary_dir")
        if isinstance(meta, str) and meta.strip():
            dictionary_dir = meta
    events = event_summary_rows(graph, dictionary_dir=dictionary_dir)
    if not events:
        print("No event nodes extracted.")
    else:
        for row in events:
            role_txt = ", ".join(row["roles"]) if row["roles"] else "none"
            print(
                f"{row['order']}. {row['id']} label={row['label']}, name={row['name']}, "
                f"{row['event_type']} [{row['modality']}] roles: {role_txt}"
            )
            if row.get("summary_text"):
                print(f"  summary: {row['summary_text']}")
            if row.get("evidence_text"):
                print(f"  evidence: {row['evidence_text']}")


def render_story_graph_dot_svg(
    graph: dict,
    *,
    output_path: str = "notebooks/examples/data/story_graph_roundtrip.dot.svg",
    use_summary: bool = True,
    summary_excerpt_sentences: int = 2,
    summary_excerpt_chars: int = 180,
    dictionary_dir: str | None = None,
) -> str:
    """Render graph as DOT SVG and return output path.

    Args:
        graph: Story graph payload.
        output_path: Output SVG path.
        use_summary: If True, include summary excerpts in node labels.
        summary_excerpt_sentences: Max sentence count in node summary excerpt.
        summary_excerpt_chars: Max chars in node summary excerpt.
        dictionary_dir: Optional dictionary directory.

    Returns:
        Output path string.
    """

    if dictionary_dir is None:
        meta = graph.get("_dictionary_dir")
        if isinstance(meta, str) and meta.strip():
            dictionary_dir = meta
    out = render_story_graph_dot(
        graph,
        output_path=output_path,
        format="svg",
        prog="dot",
        include_summary_excerpt=use_summary,
        summary_excerpt_chars=summary_excerpt_chars,
        summary_excerpt_sentences=summary_excerpt_sentences,
        dictionary_dir=dictionary_dir,
    )
    return out


def generate_story_with_repair(
    graph: dict,
    *,
    ask_llm_fn,
    model: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
    dictionary_dir: str | None = None,
) -> tuple[str, list[dict], dict]:
    """Generate story from graph with sequencing repair.

    Args:
        graph: Story graph.
        ask_llm_fn: LLM callable.
        model: Optional model name.
        temperature: Sampling temperature.
        max_output_tokens: Output token budget.
        dictionary_dir: Optional dictionary directory.

    Returns:
        Tuple `(generated_story, plan, repair_info)`.
    """

    return realize_story_from_graph_with_repair(
        graph,
        ask_llm_fn=ask_llm_fn,
        model=resolve_model_name(model),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        dictionary_dir=dictionary_dir,
    )


def generate_styled_story(
    generated_story: str,
    style_prompt: str,
    *,
    ask_llm_fn,
    model: str | None = None,
    temperature: float = 0.3,
    max_output_tokens: int = 1400,
) -> str:
    """Generate style-conditioned detailed rendering from baseline story.

    Args:
        generated_story: Baseline story text.
        style_prompt: Style/tone instruction.
        ask_llm_fn: LLM callable.
        model: Optional model name.
        temperature: Sampling temperature.
        max_output_tokens: Output token budget.

    Returns:
        Styled story text.
    """

    return render_story_with_style(
        generated_story=generated_story,
        style_prompt=style_prompt,
        ask_llm_fn=ask_llm_fn,
        model=resolve_model_name(model),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def roundtrip_check(
    original_graph: dict,
    generated_story: str,
    *,
    ask_llm_fn,
    model: str | None = None,
    add_node_summaries: bool = True,
    dictionary_dir: str | None = None,
) -> tuple[dict, dict]:
    """Re-extract graph from generated story and compare plan signatures.

    Args:
        original_graph: Original story graph.
        generated_story: Generated story text.
        ask_llm_fn: LLM callable.
        model: Optional model name.
        add_node_summaries: Whether to add summaries on regenerated graph.
        dictionary_dir: Optional dictionary directory.

    Returns:
        Tuple `(check_dict, regenerated_graph)`.
    """

    model_name = resolve_model_name(model)
    doc_id = str(original_graph.get("doc_id", "story"))
    regenerated_graph = build_story_graph_from_text(
        text=generated_story,
        doc_id=f"{doc_id}_regen",
        ask_llm_fn=ask_llm_fn,
        model=model_name,
        add_node_summaries=add_node_summaries,
        dictionary_dir=dictionary_dir,
    )

    original_plan = build_story_plan(original_graph, dictionary_dir=dictionary_dir)
    regen_plan = build_story_plan(regenerated_graph, dictionary_dir=dictionary_dir)

    def _signature(records: list[dict], graph: dict) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        sig = []
        for row in records:
            role_pairs = []
            for role, ent_id in sorted(row.get("roles", {}).items()):
                ent_node = node_by_id.get(ent_id, {})
                role_pairs.append((role, str(ent_node.get("canonical_name", ent_id)).lower()))
            sig.append((row.get("event_type"), tuple(role_pairs)))
        return sig

    sig_original = _signature(original_plan, original_graph)
    sig_regen = _signature(regen_plan, regenerated_graph)

    mismatch = None
    if sig_original != sig_regen:
        max_len = max(len(sig_original), len(sig_regen))
        for i in range(max_len):
            left = sig_original[i] if i < len(sig_original) else None
            right = sig_regen[i] if i < len(sig_regen) else None
            if left != right:
                mismatch = {"index": i, "original": left, "regenerated": right}
                break

    check = {
        "n_events_original": len(original_plan),
        "n_events_regenerated": len(regen_plan),
        "event_count_match": len(original_plan) == len(regen_plan),
        "event_type_and_role_sequence_match": sig_original == sig_regen,
        "first_mismatch": mismatch,
    }
    return check, regenerated_graph


def dump_json(data: dict) -> str:
    """Pretty JSON dump convenience for notebook printing.

    Args:
        data: Dictionary payload.

    Returns:
        Formatted JSON string.
    """

    return json.dumps(data, indent=2)


def ensure_output_dir(path: str) -> str:
    """Ensure parent directory exists for a target output path.

    Args:
        path: Target file path.

    Returns:
        Original path string.
    """

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path
