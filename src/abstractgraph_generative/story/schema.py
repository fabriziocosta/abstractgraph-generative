"""Canonical schema constants for discrete story graphs (v0).

This module defines node types, edge labels, constrained attributes, and common
metadata keys used by text-to-graph and graph-to-text pipelines.
"""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final[str] = "story_graph_v0"
VOCAB_VERSION: Final[str] = "v0"

NODE_TYPES: Final[set[str]] = {"ENT", "EVT", "REL", "GOAL", "INT", "MORAL"}

ROLE_EDGE_LABELS: Final[set[str]] = {
    "AGENT",
    "PATIENT",
    "TARGET",
    "RECIPIENT",
    "INSTRUMENT",
    "LOCATION",
    "TIME",
}

DYNAMIC_EDGE_LABELS: Final[set[str]] = {"BEFORE", "CAUSES", "ENABLES", "PREVENTS"}
MENTAL_EDGE_LABELS: Final[set[str]] = {
    "HAS_GOAL",
    "FORMS_INTENTION",
    "INTENDS",
    "MOTIVATES",
    "INTENTION_OF",
    "EXECUTES",
    "ADVANCES",
    "THWARTS",
    "FULFILLS",
    "FAILS",
}
REFERENCE_EDGE_LABELS: Final[set[str]] = {"COREF"}
RELATION_EDGE_LABELS: Final[set[str]] = {"REL_SUBJECT", "REL_OBJECT"}
EDGE_LABELS: Final[set[str]] = (
    ROLE_EDGE_LABELS | DYNAMIC_EDGE_LABELS | MENTAL_EDGE_LABELS | REFERENCE_EDGE_LABELS | RELATION_EDGE_LABELS
)

POLARITY_VALUES: Final[set[str]] = {"POS", "NEG"}
MODALITY_VALUES: Final[set[str]] = {"ASSERTED", "POSSIBLE", "OBLIGATED"}
TENSE_VALUES: Final[set[str]] = {"PAST", "PRESENT"}
INTENTION_STATUS_VALUES: Final[set[str]] = {"PENDING", "IN_PROGRESS", "ACHIEVED", "FAILED", "ABANDONED"}
OUTCOME_VALUES: Final[set[str]] = {"SUCCESS", "PARTIAL_SUCCESS", "FAILURE", "BACKFIRE", "BLOCKED"}

EVENT_REQUIRED_ROLE_MINIMUM: Final[set[str]] = {"AGENT"}
EVENT_ALLOWED_ROLES: Final[set[str]] = ROLE_EDGE_LABELS

PROVENANCE_KEYS: Final[tuple[str, ...]] = ("doc_id", "char_start", "char_end", "sentence_id")


def make_empty_graph(doc_id: str, text: str) -> dict:
    """Create an empty story graph payload with canonical top-level keys.

    Args:
        doc_id: Document identifier.
        text: Full story text.

    Returns:
        Empty graph dictionary.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "vocab_version": VOCAB_VERSION,
        "doc_id": doc_id,
        "text": text,
        "sentences": [],
        "nodes": [],
        "edges": [],
    }
