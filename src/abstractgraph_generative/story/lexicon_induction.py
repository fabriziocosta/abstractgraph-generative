"""LLM-assisted lexicon induction for closed-alphabet story graph vocabularies.

This module builds candidate dictionaries for entities, relations, goals,
intentions, and events from a corpus of stories, then consolidates terms into bounded
closed alphabets with frequency tracking.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable


def _parse_json_object(raw_text: str) -> dict:
    """Parse a JSON object from raw model output.

    Args:
        raw_text: Raw text returned by an LLM.

    Returns:
        Parsed dictionary or empty dict.
    """

    try:
        payload = json.loads(raw_text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {}
        try:
            payload = json.loads(raw_text[first : last + 1])
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


def _normalize_label(term: str, *, upper_snake: bool) -> str:
    """Normalize one term for stable counting.

    Args:
        term: Raw term.
        upper_snake: Whether to convert to `UPPER_SNAKE_CASE`.

    Returns:
        Normalized term.
    """

    text = re.sub(r"\s+", " ", str(term or "").strip())
    if not text:
        return ""
    if upper_snake:
        text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()
        return text
    return text.title()


def term_extraction_prompt(story_text: str) -> str:
    """Build first-pass extraction prompt for lexical candidates.

    Args:
        story_text: Input story text.

    Returns:
        Prompt text.
    """

    return (
        "Extract lexical candidates from the story as strict JSON only.\n"
        "Goal: collect reusable dictionary terms for a closed symbolic alphabet.\n"
        "Schema:\n"
        "{"
        '"entities":["..."],'
        '"relations":["..."],'
        '"goals":["..."],'
        '"intentions":["..."],'
        '"events":["..."]'
        "}\n"
        "Rules:\n"
        "- entities: abstract category-like names (e.g., PERSON, ANIMAL, ARTIFACT), not full phrases.\n"
        "- relations: relation labels between entities (e.g., PARENT_OF, ENEMY_OF).\n"
        "- goals: story-wide objectives (e.g., SURVIVE, RESTORE_ORDER, OBTAIN_RESOURCE).\n"
        "- intentions: motivation labels in verb-noun style (e.g., PROTECT_OTHER).\n"
        "- events: event/action labels (e.g., GIVE, ATTACK, FULFILL_TASK).\n"
        "- Keep terms short, reusable, and discrete.\n"
        "- Prefer canonical uppercase snake_case for goals/relations/intentions/events.\n"
        "- Return empty arrays when absent.\n\n"
        f"Story:\n{story_text}"
    )


def extract_story_term_candidates(
    story_text: str,
    *,
    ask_llm_fn: Callable[..., str],
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, list[str]]:
    """Extract candidate terms from one story.

    Args:
        story_text: Input story text.
        ask_llm_fn: LLM callable.
        model: Optional model name.
        api_key: Optional API key.

    Returns:
        Dictionary with keys: entities, relations, goals, intentions, events.
    """

    raw = ask_llm_fn(
        question=term_extraction_prompt(story_text),
        model=model,
        temperature=0.0,
        max_output_tokens=900,
        api_key=api_key,
        system_prompt="Return strict JSON only.",
    )
    payload = _parse_json_object(raw)
    out: dict[str, list[str]] = {"entities": [], "relations": [], "goals": [], "intentions": [], "events": []}
    for key in out.keys():
        rows = payload.get(key, []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            continue
        seen = set()
        vals = []
        upper_snake = key in {"goals", "relations", "intentions", "events"}
        for row in rows:
            norm = _normalize_label(str(row), upper_snake=upper_snake)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            vals.append(norm)
        out[key] = vals
    return out


def build_raw_term_inventory(
    stories: list[str],
    *,
    ask_llm_fn: Callable[..., str],
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Counter]:
    """Build raw frequency counters for all categories across stories.

    Args:
        stories: Story corpus.
        ask_llm_fn: LLM callable.
        model: Optional model name.
        api_key: Optional API key.

    Returns:
        Dict of counters for entities/relations/goals/intentions/events.
    """

    counters: dict[str, Counter] = {
        "entities": Counter(),
        "relations": Counter(),
        "goals": Counter(),
        "intentions": Counter(),
        "events": Counter(),
    }
    for story in stories:
        extracted = extract_story_term_candidates(
            story,
            ask_llm_fn=ask_llm_fn,
            model=model,
            api_key=api_key,
        )
        for key in counters:
            counters[key].update(extracted.get(key, []))
    return counters


def consolidation_prompt(
    category: str,
    term_counts: list[tuple[str, int]],
    target_size: int,
) -> str:
    """Build second-pass consolidation prompt for one category.

    Args:
        category: One of entities, relations, goals, intentions, events.
        term_counts: Terms with frequencies.
        target_size: Maximum alphabet size for this category.

    Returns:
        Prompt text.
    """

    lines = [f"- {term}: {count}" for term, count in term_counts]
    return (
        f"Consolidate {category} terms into a closed alphabet.\n"
        "Return strict JSON only with schema:\n"
        "{"
        '"canonical_terms":["..."],'
        '"mappings":[{"term":"...","canonical":"..."}]'
        "}\n"
        f"Constraints:\n- Keep canonical_terms <= {target_size}.\n"
        "- Every input term must appear in mappings exactly once.\n"
        "- Canonical terms must be reusable, abstract, and discrete.\n"
        "- Prefer UPPER_SNAKE_CASE for relations/intentions/events; title case for entities.\n"
        "- Merge close synonyms/variants into one canonical term.\n\n"
        "Input terms with frequencies:\n"
        + "\n".join(lines)
    )


def consolidate_category_terms(
    category: str,
    counts: Counter,
    *,
    target_size: int,
    ask_llm_fn: Callable[..., str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Consolidate one category to a bounded canonical alphabet.

    Args:
        category: entities, relations, goals, intentions, or events.
        counts: Raw term counter.
        target_size: Target maximum vocabulary size.
        ask_llm_fn: Optional LLM callable. If None, falls back to top-k.
        model: Optional model name.
        api_key: Optional API key.

    Returns:
        Dict with canonical_terms, mappings, canonical_counts.
    """

    term_counts = counts.most_common()
    if target_size <= 0:
        return {"canonical_terms": [], "mappings": [], "canonical_counts": {}}

    upper_snake = category in {"goals", "relations", "intentions", "events"}
    if ask_llm_fn is None or not term_counts:
        canonical_terms = [_normalize_label(t, upper_snake=upper_snake) for t, _ in term_counts[:target_size]]
        mappings = [{"term": t, "canonical": _normalize_label(t, upper_snake=upper_snake)} for t, _ in term_counts]
    else:
        raw = ask_llm_fn(
            question=consolidation_prompt(category, term_counts, target_size),
            model=model,
            temperature=0.0,
            max_output_tokens=1400,
            api_key=api_key,
            system_prompt="Return strict JSON only.",
        )
        payload = _parse_json_object(raw)
        candidate_terms = payload.get("canonical_terms", []) if isinstance(payload, dict) else []
        candidate_map = payload.get("mappings", []) if isinstance(payload, dict) else []

        canonical_terms = []
        seen = set()
        for row in candidate_terms if isinstance(candidate_terms, list) else []:
            norm = _normalize_label(str(row), upper_snake=upper_snake)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            canonical_terms.append(norm)
            if len(canonical_terms) >= target_size:
                break
        if not canonical_terms:
            canonical_terms = [_normalize_label(t, upper_snake=upper_snake) for t, _ in term_counts[:target_size]]

        valid = set(canonical_terms)
        mappings = []
        seen_terms = set()
        if isinstance(candidate_map, list):
            for row in candidate_map:
                if not isinstance(row, dict):
                    continue
                term = _normalize_label(str(row.get("term", "")), upper_snake=upper_snake)
                canonical = _normalize_label(str(row.get("canonical", "")), upper_snake=upper_snake)
                if not term or term in seen_terms:
                    continue
                if canonical not in valid:
                    canonical = canonical_terms[0]
                mappings.append({"term": term, "canonical": canonical})
                seen_terms.add(term)

        for term, _ in term_counts:
            norm = _normalize_label(term, upper_snake=upper_snake)
            if norm in seen_terms:
                continue
            fallback = canonical_terms[0] if canonical_terms else norm
            mappings.append({"term": norm, "canonical": fallback})
            seen_terms.add(norm)

    canonical_counts: Counter = Counter()
    raw_counts = {k: int(v) for k, v in counts.items()}
    for row in mappings:
        term = row["term"]
        canonical = row["canonical"]
        canonical_counts[canonical] += raw_counts.get(term, 0)

    # Keep only canonical labels that are actually used by mappings/counts.
    # This avoids zero-count orphan labels suggested by consolidation output.
    used_canonicals = sorted({str(row["canonical"]) for row in mappings if str(row.get("canonical", ""))})
    if canonical_counts:
        used_canonicals = sorted(set(used_canonicals) & set(canonical_counts.keys()))
    else:
        used_canonicals = []

    return {
        "canonical_terms": used_canonicals,
        "mappings": mappings,
        "canonical_counts": dict(canonical_counts),
    }


def induce_closed_alphabets(
    stories: list[str],
    *,
    ask_llm_fn: Callable[..., str],
    model: str | None = None,
    api_key: str | None = None,
    n_entities: int = 100,
    n_relations: int = 40,
    n_goals: int = 40,
    n_intentions: int = 80,
    n_events: int = 160,
    min_freq: int = 2,
    dictionary_dir: str | None = None,
    save_filename: str | None = None,
    **kwargs,
) -> dict:
    """Run end-to-end lexicon induction and consolidation.

    Args:
        stories: Story corpus.
        ask_llm_fn: LLM callable.
        model: Optional model name.
        api_key: Optional API key.
        n_entities: Target entity alphabet size.
        n_relations: Target relation alphabet size.
        n_goals: Target goal alphabet size.
        n_intentions: Target intention alphabet size.
        n_events: Target event alphabet size.
        min_freq: Minimum corpus frequency required for a term to be considered.
        dictionary_dir: Optional dictionary directory for saving outputs.
        save_filename: Optional filename to save result JSON in dictionary_dir.
        **kwargs: Compatibility aliases:
            `N_ENTITIES`, `N_RELATIONS`, `N_GOALS`, `N_INTENTIONS`, `N_EVENTS`, `MIN_FREQ`.

    Returns:
        Dictionary with raw counters and consolidated alphabets/mappings.
    """

    n_entities_eff = int(kwargs.pop("N_ENTITIES", n_entities))
    n_relations_eff = int(kwargs.pop("N_RELATIONS", n_relations))
    n_goals_eff = int(kwargs.pop("N_GOALS", n_goals))
    n_intentions_eff = int(kwargs.pop("N_INTENTIONS", n_intentions))
    n_events_eff = int(kwargs.pop("N_EVENTS", n_events))
    min_freq_eff = int(kwargs.pop("MIN_FREQ", min_freq))
    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    raw = build_raw_term_inventory(
        stories,
        ask_llm_fn=ask_llm_fn,
        model=model,
        api_key=api_key,
    )
    filtered = {
        key: Counter({term: count for term, count in counter.items() if int(count) >= min_freq_eff})
        for key, counter in raw.items()
    }
    consolidated = {
        "entities": consolidate_category_terms(
            "entities",
            filtered["entities"],
            target_size=n_entities_eff,
            ask_llm_fn=ask_llm_fn,
            model=model,
            api_key=api_key,
        ),
        "relations": consolidate_category_terms(
            "relations",
            filtered["relations"],
            target_size=n_relations_eff,
            ask_llm_fn=ask_llm_fn,
            model=model,
            api_key=api_key,
        ),
        "goals": consolidate_category_terms(
            "goals",
            filtered["goals"],
            target_size=n_goals_eff,
            ask_llm_fn=ask_llm_fn,
            model=model,
            api_key=api_key,
        ),
        "intentions": consolidate_category_terms(
            "intentions",
            filtered["intentions"],
            target_size=n_intentions_eff,
            ask_llm_fn=ask_llm_fn,
            model=model,
            api_key=api_key,
        ),
        "events": consolidate_category_terms(
            "events",
            filtered["events"],
            target_size=n_events_eff,
            ask_llm_fn=ask_llm_fn,
            model=model,
            api_key=api_key,
        ),
    }
    result = {
        "raw_counts": {k: dict(v) for k, v in raw.items()},
        "targets": {
            "entities": n_entities_eff,
            "relations": n_relations_eff,
            "goals": n_goals_eff,
            "intentions": n_intentions_eff,
            "events": n_events_eff,
            "min_freq": min_freq_eff,
        },
        "filtered_counts": {k: dict(v) for k, v in filtered.items()},
        "consolidated": consolidated,
    }
    if dictionary_dir is not None and save_filename:
        save_induced_alphabets(result, dictionary_dir=dictionary_dir, filename=save_filename)
    return result


def save_induced_alphabets(
    induced: dict,
    *,
    dictionary_dir: str,
    filename: str = "induced_closed_alphabet.json",
) -> str:
    """Save induced dictionaries to a folder.

    Args:
        induced: Induction payload.
        dictionary_dir: Target dictionary folder.
        filename: Output filename.

    Returns:
        Written file path as string.
    """

    out = Path(dictionary_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(induced, indent=2), encoding="utf-8")
    return str(out)


def load_induced_alphabets(
    *,
    dictionary_dir: str,
    filename: str = "induced_closed_alphabet.json",
) -> dict:
    """Load induced dictionaries from a folder.

    Args:
        dictionary_dir: Dictionary folder.
        filename: JSON filename.

    Returns:
        Loaded dictionary payload.
    """

    path = Path(dictionary_dir) / filename
    return json.loads(path.read_text(encoding="utf-8"))
