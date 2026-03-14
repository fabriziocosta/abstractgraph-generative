"""Validation and canonicalization utilities for story graphs.

Validation enforces bounded schema/vocabulary constraints, provenance presence,
role requirements, and optional DAG constraints over BEFORE edges.
"""

from __future__ import annotations

from collections import defaultdict

from abstractgraph_generative.story.schema import (
    EDGE_LABELS,
    EVENT_ALLOWED_ROLES,
    EVENT_REQUIRED_ROLE_MINIMUM,
    INTENTION_STATUS_VALUES,
    MENTAL_EDGE_LABELS,
    MODALITY_VALUES,
    NODE_TYPES,
    OUTCOME_VALUES,
    POLARITY_VALUES,
    PROVENANCE_KEYS,
    ROLE_EDGE_LABELS,
    RELATION_EDGE_LABELS,
    SCHEMA_VERSION,
    TENSE_VALUES,
)
from abstractgraph_generative.story.vocab import induced_sets, vocab_sets


class StoryGraphValidationError(ValueError):
    """Raised when a story graph violates schema or vocabulary constraints."""


def _validate_provenance(payload: dict, context: str) -> None:
    """Validate a provenance dictionary.

    Args:
        payload: Provenance dictionary.
        context: Human-readable object context.

    Returns:
        None.
    """

    if not isinstance(payload, dict):
        raise StoryGraphValidationError(f"{context} provenance must be a dictionary")
    for key in PROVENANCE_KEYS:
        if key not in payload:
            raise StoryGraphValidationError(f"{context} missing provenance key '{key}'")

    if not isinstance(payload["doc_id"], str) or not payload["doc_id"]:
        raise StoryGraphValidationError(f"{context} provenance doc_id must be a non-empty string")

    for key in ("char_start", "char_end"):
        if not isinstance(payload[key], int) or payload[key] < 0:
            raise StoryGraphValidationError(f"{context} provenance {key} must be a non-negative integer")

    if payload["char_end"] < payload["char_start"]:
        raise StoryGraphValidationError(f"{context} provenance char_end must be >= char_start")

    if not isinstance(payload["sentence_id"], str) or not payload["sentence_id"]:
        raise StoryGraphValidationError(f"{context} provenance sentence_id must be a non-empty string")


def _assert_acyclic_before(graph: dict) -> None:
    """Verify that BEFORE edges form an acyclic relation.

    Args:
        graph: Story graph.

    Returns:
        None.
    """

    adjacency: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = defaultdict(int)

    for edge in graph.get("edges", []):
        if edge.get("label") != "BEFORE":
            continue
        src = edge["source"]
        dst = edge["target"]
        if dst not in adjacency[src]:
            adjacency[src].add(dst)
            indegree[dst] += 1
            indegree.setdefault(src, 0)

    queue = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0

    while queue:
        current = queue.pop()
        visited += 1
        for nxt in adjacency.get(current, set()):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if visited != len(indegree):
        raise StoryGraphValidationError("BEFORE edges contain at least one directed cycle")


def canonicalize_graph(graph: dict) -> dict:
    """Return a deterministically ordered copy of the graph.

    Args:
        graph: Story graph dictionary.

    Returns:
        Canonicalized dictionary with sorted nodes and edges.
    """

    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    sentences = list(graph.get("sentences", []))

    nodes.sort(key=lambda x: str(x.get("id", "")))
    edges.sort(key=lambda x: (str(x.get("source", "")), str(x.get("label", "")), str(x.get("target", ""))))
    sentences.sort(key=lambda x: str(x.get("sentence_id", "")))

    canonical = dict(graph)
    canonical["sentences"] = sentences
    canonical["nodes"] = nodes
    canonical["edges"] = edges
    return canonical


def validate_story_graph(
    graph: dict,
    *,
    vocab_path: str | None = None,
    dictionary_dir: str | None = None,
    require_before_dag: bool = True,
    require_event_agent: bool = True,
    require_connected_entities: bool = False,
) -> dict:
    """Validate story graph payload and return canonicalized output.

    Args:
        graph: Story graph dictionary.
        vocab_path: Optional custom vocabulary file path.
        dictionary_dir: Optional dictionary directory containing `vocab_v0.json`.
        require_before_dag: If True, reject cycles in BEFORE edges.
        require_event_agent: If True, each event must have an AGENT role edge.
        require_connected_entities: If True, reject ENT nodes disconnected from all
            role/intent/reference edges.

    Returns:
        Canonicalized and validated graph dictionary.
    """

    if not isinstance(graph, dict):
        raise StoryGraphValidationError("Graph must be a dictionary")

    schema_version = graph.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise StoryGraphValidationError(
            f"Unsupported schema_version '{schema_version}', expected '{SCHEMA_VERSION}'"
        )

    doc_id = graph.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id:
        raise StoryGraphValidationError("doc_id must be a non-empty string")

    if "nodes" not in graph or "edges" not in graph:
        raise StoryGraphValidationError("Graph must include 'nodes' and 'edges'")

    vocab = vocab_sets(vocab_path=vocab_path, dictionary_dir=dictionary_dir)
    induced = induced_sets(dictionary_dir=dictionary_dir)
    allowed_entity_labels = set(induced.get("entities", set()))
    allowed_relation_types = set(induced.get("relations", set())) or set(vocab["relation_types"])
    allowed_event_types = set(induced.get("events", set())) or set(vocab["event_types"])
    allowed_goal_types = set(induced.get("goals", set())) or set(vocab["goal_types"])
    allowed_intention_types = set(induced.get("intentions", set())) or set(vocab["intention_types"])

    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))

    node_ids = set()
    node_by_id: dict[str, dict] = {}
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise StoryGraphValidationError("Each node must have a non-empty string id")
        if node_id in node_ids:
            raise StoryGraphValidationError(f"Duplicate node id '{node_id}'")
        node_ids.add(node_id)
        node_by_id[node_id] = node

        node_type = node.get("type")
        if node_type not in NODE_TYPES:
            raise StoryGraphValidationError(f"Node '{node_id}' has invalid type '{node_type}'")

        label = node.get("label")
        name = node.get("name")
        if not isinstance(label, str) or not label.strip():
            raise StoryGraphValidationError(f"Node '{node_id}' must include non-empty string field 'label'")
        if not isinstance(name, str) or not name.strip():
            raise StoryGraphValidationError(f"Node '{node_id}' must include non-empty string field 'name'")

        _validate_provenance(node.get("provenance", {}), context=f"node '{node_id}'")

        polarity = node.get("polarity", "POS")
        if polarity not in POLARITY_VALUES:
            raise StoryGraphValidationError(f"Node '{node_id}' has invalid polarity '{polarity}'")

        modality = node.get("modality", "ASSERTED")
        if modality not in MODALITY_VALUES:
            raise StoryGraphValidationError(f"Node '{node_id}' has invalid modality '{modality}'")

        tense = node.get("tense", "PAST")
        if tense not in TENSE_VALUES:
            raise StoryGraphValidationError(f"Node '{node_id}' has invalid tense '{tense}'")

        if node_type == "ENT":
            entity_type = node.get("entity_type")
            if entity_type not in vocab["entity_types"]:
                raise StoryGraphValidationError(
                    f"Entity node '{node_id}' has invalid entity_type '{entity_type}'"
                )
            if "canonical_name" not in node or "surface_name" not in node:
                raise StoryGraphValidationError(
                    f"Entity node '{node_id}' must include canonical_name and surface_name"
                )
            if allowed_entity_labels and "label" in node:
                label = str(node.get("label", ""))
                if label not in allowed_entity_labels:
                    raise StoryGraphValidationError(
                        f"Entity node '{node_id}' has label '{label}' outside induced entity vocabulary"
                    )

        if node_type == "EVT":
            event_type = node.get("event_type")
            if event_type not in allowed_event_types:
                raise StoryGraphValidationError(
                    f"Event node '{node_id}' has invalid event_type '{event_type}'"
                )

        if node_type == "GOAL":
            goal_type = node.get("goal_type")
            if goal_type not in allowed_goal_types:
                raise StoryGraphValidationError(
                    f"Mental node '{node_id}' has invalid goal_type '{goal_type}'"
                )
        if node_type == "REL":
            relation_type = node.get("relation_type")
            if relation_type not in allowed_relation_types:
                raise StoryGraphValidationError(
                    f"Relation node '{node_id}' has invalid relation_type '{relation_type}'"
                )

    event_role_map: dict[str, set[str]] = defaultdict(set)
    incoming_intends: dict[str, int] = defaultdict(int)
    fulfills_pairs: set[tuple[str, str]] = set()
    fails_pairs: set[tuple[str, str]] = set()
    intention_effect_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for idx, edge in enumerate(edges, start=1):
        source = edge.get("source")
        target = edge.get("target")
        label = edge.get("label")
        if source not in node_ids or target not in node_ids:
            raise StoryGraphValidationError(f"Edge #{idx} references unknown node(s)")
        if label not in EDGE_LABELS:
            raise StoryGraphValidationError(f"Edge #{idx} has invalid label '{label}'")

        _validate_provenance(edge.get("provenance", {}), context=f"edge #{idx}")

        if label in ROLE_EDGE_LABELS:
            if node_by_id[source]["type"] != "EVT" or node_by_id[target]["type"] != "ENT":
                raise StoryGraphValidationError(
                    f"Role edge #{idx} '{label}' must connect EVT -> ENT"
                )
            if label not in EVENT_ALLOWED_ROLES:
                raise StoryGraphValidationError(f"Role edge #{idx} uses disallowed role '{label}'")
            event_role_map[source].add(label)

        if label in RELATION_EDGE_LABELS:
            if node_by_id[source]["type"] != "REL" or node_by_id[target]["type"] != "ENT":
                raise StoryGraphValidationError(
                    f"Relation edge #{idx} '{label}' must connect REL -> ENT"
                )

        if label in MENTAL_EDGE_LABELS:
            confidence = edge.get("confidence")
            if confidence is None:
                raise StoryGraphValidationError(
                    f"Mental edge #{idx} '{label}' must include confidence in [0,1]"
                )
            if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
                raise StoryGraphValidationError(
                    f"Mental edge #{idx} has invalid confidence '{confidence}'"
                )

        if label == "INTENDS":
            if node_by_id[source]["type"] != "ENT" or node_by_id[target]["type"] != "INT":
                raise StoryGraphValidationError("INTENDS edges must connect ENT -> INT")
            incoming_intends[target] += 1

        if label == "INTENTION_OF":
            if node_by_id[source]["type"] != "INT" or node_by_id[target]["type"] != "EVT":
                raise StoryGraphValidationError("INTENTION_OF edges must connect INT -> EVT")

        if label in {"EXECUTES", "ADVANCES", "THWARTS", "FULFILLS", "FAILS"}:
            if node_by_id[source]["type"] != "EVT" or node_by_id[target]["type"] != "INT":
                raise StoryGraphValidationError(f"{label} edges must connect EVT -> INT")
            intention_effect_counts[target][label] += 1
            if label == "FULFILLS":
                fulfills_pairs.add((source, target))
            if label == "FAILS":
                fails_pairs.add((source, target))

    if require_event_agent:
        for node in nodes:
            if node.get("type") != "EVT":
                continue
            event_id = node["id"]
            present_roles = event_role_map.get(event_id, set())
            missing = EVENT_REQUIRED_ROLE_MINIMUM - present_roles
            if missing:
                missing_txt = ", ".join(sorted(missing))
                raise StoryGraphValidationError(f"Event '{event_id}' missing required role(s): {missing_txt}")

    if require_connected_entities:
        connected_entities: set[str] = set()
        for edge in edges:
            label = edge["label"]
            src = edge["source"]
            dst = edge["target"]
            if label in ROLE_EDGE_LABELS and node_by_id.get(dst, {}).get("type") == "ENT":
                connected_entities.add(dst)
            if label == "INTENDS" and node_by_id.get(src, {}).get("type") == "ENT":
                connected_entities.add(src)
            if label == "COREF":
                if node_by_id.get(src, {}).get("type") == "ENT":
                    connected_entities.add(src)
                if node_by_id.get(dst, {}).get("type") == "ENT":
                    connected_entities.add(dst)
        for node in nodes:
            if node.get("type") == "ENT" and node["id"] not in connected_entities:
                raise StoryGraphValidationError(f"Entity '{node['id']}' is disconnected from the graph")

    for node in nodes:
        if node.get("type") != "INT":
            continue
        node_id = node["id"]
        intention_type = node.get("intention_type")
        if intention_type not in allowed_intention_types:
            raise StoryGraphValidationError(
                f"Intention node '{node_id}' has invalid intention_type '{intention_type}'"
            )
        status = node.get("status")
        if status not in INTENTION_STATUS_VALUES:
            raise StoryGraphValidationError(f"Intention node '{node_id}' has invalid status '{status}'")
        outcome_type = node.get("outcome_type")
        if outcome_type is not None and outcome_type not in OUTCOME_VALUES:
            raise StoryGraphValidationError(
                f"Intention node '{node_id}' has invalid outcome_type '{outcome_type}'"
            )
        if incoming_intends.get(node_id, 0) < 1:
            raise StoryGraphValidationError(f"Intention node '{node_id}' must have at least one incoming INTENDS edge")

        effects = intention_effect_counts.get(node_id, {})
        if status == "ACHIEVED" and effects.get("FULFILLS", 0) < 1:
            raise StoryGraphValidationError(
                f"Intention node '{node_id}' status ACHIEVED requires at least one FULFILLS edge"
            )
        if status == "FAILED" and (effects.get("FAILS", 0) + effects.get("THWARTS", 0) < 1):
            raise StoryGraphValidationError(
                f"Intention node '{node_id}' status FAILED requires FAILS or THWARTS evidence"
            )

    overlap = fulfills_pairs & fails_pairs
    if overlap:
        one_src, one_dst = sorted(overlap)[0]
        raise StoryGraphValidationError(
            f"Event '{one_src}' cannot both FULFILLS and FAILS intention '{one_dst}'"
        )

    if require_before_dag:
        _assert_acyclic_before(graph)

    return canonicalize_graph(graph)
