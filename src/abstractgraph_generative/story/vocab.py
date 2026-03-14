"""Versioned vocabulary registry utilities for story graphs.

The registry pins bounded discrete vocabularies for entity/event/goal typing.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_VOCAB_PATH = Path(__file__).resolve().parent / "resources" / "vocab_v0.json"

UNIVERSAL_FILE_BY_CATEGORY = {
    "entities": "universal_entities.json",
    "relations": "universal_relations.json",
    "events": "universal_events.json",
    "goals": "universal_goals.json",
    "intentions": "universal_intentions.json",
}


def _load_universal_terms(category: str) -> set[str]:
    """Load one universal fallback dictionary category from resources files."""

    filename = UNIVERSAL_FILE_BY_CATEGORY.get(category)
    if not filename:
        return set()
    path = Path(__file__).resolve().parent / "resources" / filename
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(payload, list):
        return set()
    return {str(v).strip() for v in payload if str(v).strip()}


def resolve_induced_path(
    dictionary_dir: str | Path | None = None,
    filename: str = "induced_closed_alphabet.json",
) -> Path | None:
    """Resolve induced-lexicon file path from a dictionary directory.

    Args:
        dictionary_dir: Optional dictionary directory.
        filename: Induced dictionary filename.

    Returns:
        Resolved path if existing, otherwise None.
    """

    if dictionary_dir is None:
        return None
    raw_dir = Path(dictionary_dir)
    candidates = []
    if raw_dir.is_absolute():
        candidates.append(raw_dir)
    else:
        candidates.append((Path.cwd() / raw_dir).resolve())
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append((repo_root / raw_dir).resolve())
        candidates.append(raw_dir.resolve())
    for directory in candidates:
        path = directory / filename
        if path.exists():
            return path
    return None


def resolve_vocab_path(
    vocab_path: str | Path | None = None,
    dictionary_dir: str | Path | None = None,
) -> Path:
    """Resolve vocabulary file path from explicit path or dictionary directory.

    Args:
        vocab_path: Optional explicit path to vocabulary JSON.
        dictionary_dir: Optional directory containing dictionary files.

    Returns:
        Resolved vocabulary file path.
    """

    if vocab_path is not None:
        path = Path(vocab_path)
        if path.exists():
            return path
        if not path.is_absolute():
            cwd_path = (Path.cwd() / path).resolve()
            if cwd_path.exists():
                return cwd_path
            repo_root = Path(__file__).resolve().parents[2]
            root_path = (repo_root / path).resolve()
            if root_path.exists():
                return root_path
        return path
    if dictionary_dir is not None:
        raw_dir = Path(dictionary_dir)
        candidates = []
        if raw_dir.is_absolute():
            candidates.append(raw_dir)
        else:
            candidates.append((Path.cwd() / raw_dir).resolve())
            repo_root = Path(__file__).resolve().parents[2]
            candidates.append((repo_root / raw_dir).resolve())
            candidates.append(raw_dir.resolve())
        for directory in candidates:
            candidate = directory / "vocab_v0.json"
            if candidate.exists():
                return candidate
        # Fall back to cwd-resolved path for a clear error message.
        return candidates[0] / "vocab_v0.json"
    return DEFAULT_VOCAB_PATH


def load_vocab(
    vocab_path: str | Path | None = None,
    *,
    dictionary_dir: str | Path | None = None,
) -> dict:
    """Load the story-graph vocabulary registry.

    Args:
        vocab_path: Optional path override for the vocabulary JSON file.

    Returns:
        Registry dictionary with `entity_types`, `event_types`, `goal_types`,
        `intention_types`, and `outcome_types`.
    """

    path = resolve_vocab_path(vocab_path=vocab_path, dictionary_dir=dictionary_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))

    required_keys = {
        "version",
        "entity_types",
        "relation_types",
        "event_types",
        "goal_types",
        "intention_types",
        "outcome_types",
    }
    missing = required_keys - set(payload.keys())
    if missing:
        missing_txt = ", ".join(sorted(missing))
        raise ValueError(f"Vocabulary file missing required keys: {missing_txt}")

    return payload


def vocab_sets(
    vocab_path: str | Path | None = None,
    *,
    dictionary_dir: str | Path | None = None,
) -> dict[str, set[str]]:
    """Return vocabulary values as sets for fast membership checks.

    Args:
        vocab_path: Optional path override.

    Returns:
        Dictionary containing entity/event/goal/intention/outcome sets.
    """

    payload = load_vocab(vocab_path=vocab_path, dictionary_dir=dictionary_dir)
    return {
        "entity_types": set(payload["entity_types"]),
        "relation_types": set(payload["relation_types"]),
        "event_types": set(payload["event_types"]),
        "goal_types": set(payload["goal_types"]),
        "intention_types": set(payload["intention_types"]),
        "outcome_types": set(payload["outcome_types"]),
    }


def load_induced_alphabet(
    *,
    dictionary_dir: str | Path | None = None,
    filename: str = "induced_closed_alphabet.json",
) -> dict:
    """Load induced closed alphabet payload from dictionary directory.

    Args:
        dictionary_dir: Dictionary directory.
        filename: Induced payload filename.

    Returns:
        Payload dictionary, or empty dict when unavailable/invalid.
    """

    path = resolve_induced_path(dictionary_dir=dictionary_dir, filename=filename)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def induced_sets(
    *,
    dictionary_dir: str | Path | None = None,
    filename: str = "induced_closed_alphabet.json",
    include_universal: bool = True,
) -> dict[str, set[str]]:
    """Return induced closed-alphabet values as sets.

    Args:
        dictionary_dir: Dictionary directory.
        filename: Induced payload filename.
        include_universal: If True, add universal generic fallback terms.

    Returns:
        Set dictionary with keys: entities, relations, goals, intentions, events.
    """

    payload = load_induced_alphabet(dictionary_dir=dictionary_dir, filename=filename)
    consolidated = payload.get("consolidated", {}) if isinstance(payload, dict) else {}

    def _terms(key: str) -> set[str]:
        block = consolidated.get(key, {}) if isinstance(consolidated, dict) else {}
        values = block.get("canonical_terms", []) if isinstance(block, dict) else []
        if not isinstance(values, list):
            return set()
        return {str(v).strip() for v in values if str(v).strip()}

    out = {
        "entities": _terms("entities"),
        "relations": _terms("relations"),
        "goals": _terms("goals"),
        "intentions": _terms("intentions"),
        "events": _terms("events"),
    }

    def _norm_key(value: str) -> str:
        return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())

    if include_universal:
        for key in UNIVERSAL_FILE_BY_CATEGORY:
            existing = out.setdefault(key, set())
            existing_norm = {_norm_key(v) for v in existing}
            for candidate in _load_universal_terms(key):
                nkey = _norm_key(candidate)
                if nkey in existing_norm:
                    continue
                existing.add(candidate)
                existing_norm.add(nkey)
    return out
