"""Deterministic text-to-story-graph extraction pipeline (v0).

The extractor targets compact, auditable graphs with strict vocabularies,
provenance on each semantic item, and deterministic IDs.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

from abstractgraph_generative.story.schema import VOCAB_VERSION, make_empty_graph
from abstractgraph_generative.story.validation import validate_story_graph
from abstractgraph_generative.story.vocab import induced_sets, vocab_sets

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?", flags=re.M)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

DETERMINERS = {"a", "an", "the", "this", "that", "these", "those"}
STOPWORDS = {
    "and",
    "or",
    "but",
    "if",
    "then",
    "because",
    "so",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "once",
    "then",
    "thus",
    "shortly",
    "afterwards",
    "however",
    "therefore",
    "meanwhile",
    "again",
    "upon",
    "who",
    "whom",
    "whose",
    "which",
    "that",
    "this",
    "those",
    "these",
    "him",
    "her",
    "his",
    "their",
    "its",
    "they",
    "them",
    "he",
    "she",
    "it",
}
PRONOUNS = {
    "i",
    "we",
    "you",
    "he",
    "she",
    "it",
    "they",
    "me",
    "us",
    "him",
    "her",
    "them",
    "my",
    "our",
    "your",
    "his",
    "its",
    "their",
    "mine",
    "ours",
    "yours",
    "hers",
    "theirs",
}

VERB_TO_EVENT: dict[str, str] = {}


@dataclass(frozen=True)
class SentenceSpan:
    """Sentence text and character offsets."""

    sentence_id: str
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class EntityMention:
    """Entity mention candidate with local metadata."""

    text: str
    canonical_name: str
    sentence_id: str
    char_start: int
    char_end: int


def normalize_text(text: str) -> str:
    """Normalize whitespace and quote characters.

    Args:
        text: Raw text.

    Returns:
        Normalized text.
    """

    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
    cleaned = re.sub(r"[\t ]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_sentences_with_offsets(text: str) -> list[SentenceSpan]:
    """Split text into sentence spans with offsets.

    Args:
        text: Normalized text.

    Returns:
        Ordered sentence span list.
    """

    out: list[SentenceSpan] = []
    for idx, match in enumerate(SENTENCE_RE.finditer(text), start=1):
        segment = match.group(0)
        stripped = segment.strip()
        if not stripped:
            continue

        start = match.start() + (len(segment) - len(segment.lstrip()))
        end = start + len(stripped)
        out.append(
            SentenceSpan(
                sentence_id=f"S{idx:04d}",
                text=stripped,
                char_start=start,
                char_end=end,
            )
        )
    return out


def _canonicalize_name(raw: str) -> str:
    """Convert a surface form to canonical entity name.

    Args:
        raw: Raw mention string.

    Returns:
        Canonicalized entity name.
    """

    tokens = [t for t in WORD_RE.findall(raw) if t]
    while tokens and tokens[0].lower() in DETERMINERS:
        tokens = tokens[1:]
    return " ".join(tokens).strip().title()


def _abstract_entity_name(canonical_name: str) -> str:
    """Map overly specific surface mentions into compact canonical categories.

    Args:
        canonical_name: Canonicalized surface name.

    Returns:
        Possibly abstracted canonical name.
    """

    return canonical_name


def _load_induced_entity_terms(dictionary_dir: str | None) -> set[str]:
    """Load induced entity dictionary terms when available."""

    terms = induced_sets(dictionary_dir=dictionary_dir).get("entities", set())
    return {re.sub(r"\s+", " ", str(term).strip()) for term in terms if str(term).strip()}


def _entity_dictionary_label(canonical_name: str, entity_type: str, induced_terms: set[str]) -> str:
    """Map entity mention to a constrained induced dictionary label."""

    if not induced_terms:
        return canonical_name

    def _key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    term_by_key = {_key(term): term for term in induced_terms}
    name_key = _key(canonical_name)
    if name_key in term_by_key:
        return term_by_key[name_key]
    if name_key.endswith("s") and name_key[:-1] in term_by_key:
        return term_by_key[name_key[:-1]]

    type_key = _key(str(entity_type).replace("_", " ").title())
    if type_key in term_by_key:
        return term_by_key[type_key]

    type_first_fallbacks = {
        "ANIMAL": ["Animal", "Object"],
        "PERSON": ["Person", "Object"],
        "YOUNG_PERSON": ["Young Person", "Person", "Object"],
        "PEOPLE": ["People", "Group", "Object"],
        "GROUP": ["Group", "People", "Object"],
        "ARTIFACT": ["Artifact", "Object"],
        "NATURAL_OBJECT": ["Natural Object", "Object"],
        "RESOURCE": ["Resource", "Money", "Object"],
        "MONEY": ["Money", "Resource", "Object"],
        "ABSTRACT": ["Abstract", "Object"],
        "TIME_UNIT": ["Time Unit", "Object"],
        "LOCATION": ["Location", "Natural Object", "Object"],
        "OBJECT": ["Object"],
    }
    for generic in type_first_fallbacks.get(str(entity_type or "").upper(), ["Object"]):
        gkey = _key(generic)
        if gkey in term_by_key:
            return term_by_key[gkey]
    # Strict closed-alphabet fallback: always return a dictionary entry.
    return sorted(induced_terms)[0]


def _entity_dictionary_label_with_llm(
    canonical_name: str,
    entity_type: str,
    induced_terms: set[str],
    *,
    ask_llm_fn: Callable[..., str] | None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Select a dictionary label constrained to induced terms, with LLM assist."""

    base = _entity_dictionary_label(canonical_name, entity_type, induced_terms)
    if not induced_terms or ask_llm_fn is None:
        return base

    terms_sorted = sorted(induced_terms)
    prompt = (
        "Choose exactly one label from the allowed dictionary labels.\n"
        "Return strict JSON only with schema:\n"
        '{"label":"..."}\n'
        f"Entity name: {canonical_name}\n"
        f"Entity type: {entity_type}\n"
        f"Allowed labels: {', '.join(terms_sorted)}\n"
    )
    try:
        raw = ask_llm_fn(
            question=prompt,
            model=model,
            temperature=0.0,
            max_output_tokens=120,
            api_key=api_key,
            system_prompt="Return strict JSON only.",
        )
        payload = _parse_llm_json_object(raw)
        label = str(payload.get("label", "")).strip()
        if label in induced_terms:
            return label
    except Exception:
        pass
    return base


def extract_entity_mentions(sentences: list[SentenceSpan]) -> list[EntityMention]:
    """Extract simple entity mentions from sentence text.

    Args:
        sentences: Sentence spans.

    Returns:
        Mention candidates sorted by first appearance.
    """

    mentions: list[EntityMention] = []
    for sentence in sentences:
        for match in WORD_RE.finditer(sentence.text):
            token = match.group(0)
            lower = token.lower()
            if lower in STOPWORDS or lower in DETERMINERS:
                continue
            if len(lower) <= 2:
                continue

            is_candidate = token[0].isupper()
            if not is_candidate:
                continue

            canonical = _canonicalize_name(token)
            if not canonical:
                continue

            mentions.append(
                EntityMention(
                    text=token,
                    canonical_name=canonical,
                    sentence_id=sentence.sentence_id,
                    char_start=sentence.char_start + match.start(),
                    char_end=sentence.char_start + match.end(),
                )
            )
    return mentions


def infer_entity_type(canonical_name: str, allowed_entity_types: set[str]) -> str:
    """Map canonical entity name to bounded `ENTITY_TYPES`.

    Args:
        canonical_name: Canonical entity name.
        allowed_entity_types: Allowed entity type set.

    Returns:
        One entity type from the allowed set.
    """

    name_key = canonical_name.strip().upper().replace(" ", "_")
    if name_key in allowed_entity_types:
        return name_key
    return "OBJECT" if "OBJECT" in allowed_entity_types else sorted(allowed_entity_types)[0]


def _infer_entity_type_with_llm(
    canonical_name: str,
    *,
    allowed_entity_types: set[str],
    ask_llm_fn: Callable[..., str] | None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Infer entity type using only allowed closed-vocabulary labels."""

    fallback = infer_entity_type(canonical_name, allowed_entity_types)
    if ask_llm_fn is None:
        return fallback
    allowed_sorted = sorted(allowed_entity_types)
    prompt = (
        "Choose exactly one entity type from the allowed list.\n"
        "Return strict JSON only with schema:\n"
        '{"entity_type":"..."}\n'
        f"Allowed entity types: {', '.join(allowed_sorted)}\n"
        f"Entity: {canonical_name}\n"
    )
    try:
        raw = ask_llm_fn(
            question=prompt,
            model=model,
            temperature=0.0,
            max_output_tokens=120,
            api_key=api_key,
            system_prompt="Return strict JSON only.",
        )
        payload = _parse_llm_json_object(raw)
        candidate = str(payload.get("entity_type", "")).strip().upper()
        if candidate in allowed_entity_types:
            return candidate
    except Exception:
        pass
    return fallback


def _extract_event_type(sentence_text: str, allowed_event_types: set[str]) -> str:
    """Infer event type from sentence with bounded vocabulary.

    Args:
        sentence_text: Sentence text.
        allowed_event_types: Allowed event type set.

    Returns:
        Chosen event type.
    """

    text_upper = str(sentence_text).upper()
    for event_type in sorted(allowed_event_types):
        if event_type == "OTHER_EVENT":
            continue
        phrase = event_type.replace("_", " ")
        if phrase and phrase in text_upper:
            return event_type
    return "OTHER_EVENT" if "OTHER_EVENT" in allowed_event_types else sorted(allowed_event_types)[0]


def _candidate_event_types_from_sentence(sentence_text: str, allowed_event_types: set[str]) -> set[str]:
    """Collect all lexically supported event types present in a sentence.

    Args:
        sentence_text: Sentence text.
        allowed_event_types: Allowed event type set.

    Returns:
        Set of event types supported by explicit trigger tokens.
    """

    out: set[str] = set()
    text_upper = str(sentence_text).upper()
    for event_type in sorted(allowed_event_types):
        if event_type == "OTHER_EVENT":
            continue
        phrase = event_type.replace("_", " ")
        if phrase and phrase in text_upper:
            out.add(event_type)
    return out


def _event_extraction_prompt(
    sentence_text: str,
    event_types: list[str],
    role_labels: list[str],
    entity_names: list[str],
) -> str:
    """Build strict JSON extraction prompt for optional LLM event extraction.

    Args:
        sentence_text: Sentence text.
        event_types: Allowed event type list.
        role_labels: Allowed role labels.
        entity_names: Allowed entity names for role fillers.

    Returns:
        Prompt text.
    """

    event_types_txt = ", ".join(event_types)
    role_labels_txt = ", ".join(role_labels)
    entity_names_txt = ", ".join(entity_names)
    return (
        "Extract all explicit events from the sentence as strict JSON only. Schema: "
        "{\"events\":[{\"event_type\":...,\"roles\":{\"AGENT\":...,\"PATIENT\":...,\"TARGET\":...,"
        "\"RECIPIENT\":...,\"INSTRUMENT\":...,\"LOCATION\":...,\"TIME\":...},"
        "\"polarity\":\"POS|NEG\",\"modality\":\"ASSERTED|POSSIBLE|OBLIGATED\","
        "\"tense\":\"PAST|PRESENT\",\"summary_text\":\"...\",\"evidence_text\":\"...\","
        "\"span\":{\"char_start\":int,\"char_end\":int}}]}. "
        f"Allowed event_type values: {event_types_txt}. Allowed roles: {role_labels_txt}. "
        "Role filler strings must be selected only from this entity list when possible: "
        f"{entity_names_txt}. "
        "Do not invent entities. If a role is absent, omit that role key. "
        "Use only explicit sentence evidence. Return 1 to 3 events ordered as they appear in text."
        f"\nSentence: {sentence_text}"
    )


def _label_to_name(label: str) -> str:
    """Convert an uppercase label token to a readable name."""

    value = str(label or "").strip().replace("_", " ").title()
    return value or "Unknown"


def parse_llm_events(raw_text: str) -> list[dict]:
    """Parse event extraction JSON from LLM output.

    Args:
        raw_text: Raw LLM output.

    Returns:
        Parsed event list.
    """

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return []
        try:
            payload = json.loads(raw_text[first : last + 1])
        except json.JSONDecodeError:
            return []

    events = payload.get("events", []) if isinstance(payload, dict) else []
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _normalize_entity_name_for_match(text: str) -> str:
    """Normalize entity name text for fuzzy matching.

    Args:
        text: Candidate entity text.

    Returns:
        Normalized lowercase string without leading determiners.
    """

    canonical = _canonicalize_name(text).lower()
    canonical = canonical.replace("'s", "").strip()
    parts = [part for part in canonical.split() if part]
    while parts and parts[0] in DETERMINERS:
        parts = parts[1:]
    return " ".join(parts)


def _canonicalize_role_filler_entity_name(text: str) -> str:
    """Canonicalize a role filler into a noun-like entity name.

    Args:
        text: Raw role filler text.

    Returns:
        Canonical entity name or empty string when unsuitable.
    """

    tokens = [tok.lower() for tok in WORD_RE.findall(text)]
    if not tokens:
        return ""

    # Remove leading determiners and pronoun-only fillers.
    while tokens and tokens[0] in DETERMINERS:
        tokens = tokens[1:]
    if not tokens:
        return ""
    if all(tok in PRONOUNS for tok in tokens):
        return ""

    # Keep noun-like content and drop obvious predicate/action tokens.
    noun_like = [tok for tok in tokens if tok not in STOPWORDS and tok not in PRONOUNS and tok not in VERB_TO_EVENT]
    if not noun_like:
        return ""

    head = noun_like[-1]
    if head in {"piece", "pieces", "them", "him", "her"}:
        return ""
    return _canonicalize_name(head)


def _resolve_role_filler_to_entity_id(
    filler: str,
    entity_id_by_canonical: dict[str, str],
) -> str | None:
    """Resolve a role filler string to an existing entity ID.

    Args:
        filler: Role filler text from extraction.
        entity_id_by_canonical: Mapping canonical-name->entity-id.

    Returns:
        Entity ID when matched, otherwise None.
    """

    if not filler:
        return None
    normalized = _normalize_entity_name_for_match(filler)
    if not normalized:
        return None
    if normalized in entity_id_by_canonical:
        return entity_id_by_canonical[normalized]

    canonical_filler = _canonicalize_role_filler_entity_name(filler).lower()
    if canonical_filler and canonical_filler in entity_id_by_canonical:
        return entity_id_by_canonical[canonical_filler]

    filler_tokens = {tok for tok in normalized.split() if tok}
    for canonical_name, entity_id in entity_id_by_canonical.items():
        canonical_tokens = {tok for tok in canonical_name.split() if tok}
        if filler_tokens and canonical_tokens and (filler_tokens & canonical_tokens):
            return entity_id
    return None


def _resolve_role_filler_to_entity_id_with_llm(
    filler: str,
    entity_id_by_canonical: dict[str, str],
    ask_llm_fn: Callable[..., str],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> str | None:
    """Resolve role filler to an existing entity using LLM disambiguation.

    Args:
        filler: Role filler text.
        entity_id_by_canonical: Mapping canonical name -> entity id.
        ask_llm_fn: LLM callable.
        model: Optional model name.
        api_key: Optional API key.

    Returns:
        Chosen entity id or None.
    """

    if not filler or not entity_id_by_canonical:
        return None

    choices = sorted((eid, cname) for cname, eid in entity_id_by_canonical.items())
    choices_txt = "\n".join(f"- {eid}: {cname}" for eid, cname in choices)
    prompt = (
        "Map the role filler to one existing entity id.\n"
        "Return strict JSON only with schema: {\"entity_id\":\"ENT_0001\"} or {\"entity_id\":null}.\n"
        "Choose null if no entity matches semantically.\n\n"
        f"Role filler: {filler}\n"
        f"Candidates:\n{choices_txt}"
    )
    raw = ask_llm_fn(
        question=prompt,
        model=model,
        temperature=0.0,
        max_output_tokens=120,
        api_key=api_key,
        system_prompt="You are a strict entity linker. Return JSON only.",
    )
    payload = _parse_llm_json_object(raw)
    candidate = payload.get("entity_id") if isinstance(payload, dict) else None
    if candidate is None:
        return None
    candidate_id = str(candidate).strip()
    valid_ids = {eid for _, eid in entity_id_by_canonical.items()}
    return candidate_id if candidate_id in valid_ids else None


def _is_valid_new_entity_filler(text: str) -> bool:
    """Return True if role filler text is suitable as a new entity.

    Args:
        text: Candidate role filler string.

    Returns:
        True if suitable as entity mention.
    """

    canonical_name = _canonicalize_role_filler_entity_name(text)
    normalized = _normalize_entity_name_for_match(canonical_name)
    if not normalized:
        return False
    if normalized in STOPWORDS or normalized in DETERMINERS or normalized in PRONOUNS:
        return False
    if len(normalized) <= 2:
        return False
    if len(normalized.split()) > 3:
        return False
    banned_generic = {
        "past",
        "present",
        "future",
        "idea",
        "sleep",
        "wake",
        "laugh",
        "hunt",
        "kill",
        "exclaim",
        "thought",
        "plan",
    }
    if normalized in banned_generic:
        return False
    return True


def _span_for_filler_in_sentence(sentence: SentenceSpan, filler: str) -> tuple[int, int]:
    """Estimate character span for a role filler inside a sentence.

    Args:
        sentence: Sentence span.
        filler: Role filler text.

    Returns:
        Tuple of absolute `(char_start, char_end)`.
    """

    sentence_lower = sentence.text.lower()
    filler_norm = _normalize_entity_name_for_match(filler)
    if filler_norm:
        idx = sentence_lower.find(filler_norm.lower())
        if idx != -1:
            return sentence.char_start + idx, sentence.char_start + idx + len(filler_norm)
    return sentence.char_start, sentence.char_end


def _main_characters_prompt(story_text: str) -> str:
    """Build prompt for extracting main characters as strict JSON.

    Args:
        story_text: Full story text.

    Returns:
        Prompt text.
    """

    return (
        "Identify the main characters and key supporting actors in this story. "
        "Return STRICT JSON only with schema: "
        '{"characters": ["Character Name", "..."]}. '
        "Include protagonists, antagonists, and important groups (e.g., hunters, villagers) when present. "
        "Exclude narrator words, adverbs, pronouns, and generic terms. "
        "If uncertain, return your best compact list.\n\n"
        f"Story:\n{story_text}"
    )


def parse_llm_main_characters(raw_text: str) -> list[str]:
    """Parse main-character JSON response from LLM output.

    Args:
        raw_text: Raw LLM output text.

    Returns:
        Character names in canonical title case.
    """

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return []
        try:
            payload = json.loads(raw_text[first : last + 1])
        except json.JSONDecodeError:
            return []

    candidates = payload.get("characters", []) if isinstance(payload, dict) else []
    if not isinstance(candidates, list):
        return []

    out: list[str] = []
    seen = set()
    for item in candidates:
        text = _canonicalize_name(str(item))
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def infer_main_characters_with_llm(
    story_text: str,
    ask_llm_fn: Callable[..., str],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> set[str]:
    """Infer main characters using an LLM with strict JSON output.

    Args:
        story_text: Full story text.
        ask_llm_fn: Callable that performs LLM query.
        model: Optional model name.
        api_key: Optional API key.

    Returns:
        Lowercased set of canonical character names.
    """

    raw = ask_llm_fn(
        question=_main_characters_prompt(story_text),
        model=model,
        temperature=0.0,
        max_output_tokens=250,
        api_key=api_key,
        system_prompt="Return strict JSON only.",
    )
    parsed = parse_llm_main_characters(raw)
    return {name.lower() for name in parsed}


def _matches_main_character(canonical_name: str, main_character_keys: set[str]) -> bool:
    """Return True if canonical name matches one inferred main character.

    Args:
        canonical_name: Candidate canonical name.
        main_character_keys: Main-character lowercased names.

    Returns:
        True on exact or token-overlap match.
    """

    key = canonical_name.lower()
    if key in main_character_keys:
        return True
    tokens = {tok for tok in key.split() if tok}
    if not tokens:
        return False
    for main_name in main_character_keys:
        main_tokens = {tok for tok in main_name.split() if tok}
        if tokens & main_tokens:
            return True
    return False


def _intention_extraction_prompt(
    story_text: str,
    entity_names: list[str],
    event_ids: list[str],
    intention_types: list[str],
) -> str:
    """Build prompt for extracting intention nodes and event-level effects.

    Args:
        story_text: Full story text.
        entity_names: Allowed entity names.
        event_ids: Ordered event node IDs.
        intention_types: Allowed intention types.

    Returns:
        Prompt text.
    """

    return (
        "Extract intention dynamics as strict JSON only. Schema: "
        '{"intentions":[{"entity":"...","intention_type":"...","confidence":0.0,"evidence":"..."}],'
        '"effects":[{"event_id":"EVT_0001","entity":"...","intention_type":"...","effect":"ADVANCES|THWARTS|FULFILLS|FAILS","confidence":0.0}]}. '
        "Constraints: use only provided entity names, event IDs, and intention types. "
        "Only include intentions and effects explicitly supported by text. "
        "If none, return empty arrays.\n"
        f"Allowed entities: {', '.join(entity_names)}\n"
        f"Allowed event_ids: {', '.join(event_ids)}\n"
        f"Allowed intention_types: {', '.join(intention_types)}\n\n"
        f"Story:\n{story_text}"
    )


def _premise_sentence_prompt(story_text: str, sentences: list[SentenceSpan]) -> str:
    """Build prompt to identify premise/setup sentences before event labeling.

    Args:
        story_text: Full story text.
        sentences: Sentence spans with IDs.

    Returns:
        Prompt text.
    """

    lines = [f"- {s.sentence_id}: {s.text}" for s in sentences]
    return (
        "Identify premise sentences in this story. A premise sentence establishes background conditions, "
        "initial state, constraints, or context needed to understand later events. "
        "It is setup, not the main procedural action chain.\n"
        "Return STRICT JSON only with schema: "
        '{"premise_sentence_ids":["S0001","S0002"]}. '
        "Use only provided sentence IDs.\n\n"
        "Sentences:\n"
        + "\n".join(lines)
        + "\n\nStory:\n"
        + story_text
    )


def _infer_premise_sentence_ids_with_llm(
    story_text: str,
    sentences: list[SentenceSpan],
    ask_llm_fn: Callable[..., str] | None,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> set[str]:
    """Infer premise sentence IDs using an LLM.

    Args:
        story_text: Full story text.
        sentences: Sentence spans.
        ask_llm_fn: Optional LLM callable.
        model: Optional model name.
        api_key: Optional API key.

    Returns:
        Set of sentence IDs classified as premise.
    """

    if ask_llm_fn is None or not sentences:
        return set()
    valid_ids = {s.sentence_id for s in sentences}
    try:
        raw = ask_llm_fn(
            question=_premise_sentence_prompt(story_text=story_text, sentences=sentences),
            model=model,
            temperature=0.0,
            max_output_tokens=500,
            api_key=api_key,
            system_prompt="Return strict JSON only.",
        )
        payload = _parse_llm_json_object(raw)
    except Exception:
        return set()
    rows = payload.get("premise_sentence_ids", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return set()
    out = set()
    for sid in rows:
        value = str(sid).strip()
        if value in valid_ids:
            out.add(value)
    return out


def _associate_premise_events_with_llm(
    graph: dict,
    premise_sentence_ids: set[str],
    ask_llm_fn: Callable[..., str] | None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    min_confidence: float = 0.7,
) -> dict:
    """Associate events to premise layer using LLM after sentence-level premise detection.

    Args:
        graph: Story graph dictionary.
        premise_sentence_ids: Sentence IDs marked as premise candidates.
        ask_llm_fn: Optional LLM callable.
        model: Optional model name.
        api_key: Optional API key.
        min_confidence: Minimum confidence to assign PREMISE.

    Returns:
        Graph with selected event nodes tagged as PREMISE.
    """

    if ask_llm_fn is None:
        return graph
    node_by_id = {n.get("id"): n for n in graph.get("nodes", []) if isinstance(n, dict)}
    sentence_by_id = {str(s.get("sentence_id", "")): s for s in graph.get("sentences", []) if isinstance(s, dict)}

    candidates = []
    for node in graph.get("nodes", []):
        if node.get("type") != "EVT":
            continue
        if str(node.get("event_type", "")).upper() not in {"OTHER_EVENT", "PREMISE"}:
            continue
        sid = str(node.get("provenance", {}).get("sentence_id", ""))
        if sid not in premise_sentence_ids:
            continue
        evt_id = str(node.get("id", ""))
        roles = []
        for edge in graph.get("edges", []):
            if edge.get("source") != evt_id:
                continue
            label = str(edge.get("label", ""))
            if label not in {"AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT", "LOCATION", "TIME"}:
                continue
            ent = node_by_id.get(edge.get("target"), {})
            name = str(ent.get("canonical_name", edge.get("target", ""))).strip()
            roles.append(f"{label}={name}")
        candidates.append(
            {
                "id": evt_id,
                "sentence_id": sid,
                "sentence_text": str(sentence_by_id.get(sid, {}).get("text", "")),
                "event_type": str(node.get("event_type", "")),
                "summary_text": str(node.get("summary_text", "")),
                "evidence_text": str(node.get("evidence_text", "")),
                "roles": "; ".join(roles),
            }
        )

    if not candidates:
        return graph

    prompt = (
        "Associate premise sentences to event nodes.\n"
        "A PREMISE event encodes setup/background state required for the story, not a main action step.\n"
        "Return STRICT JSON only with schema:\n"
        '{"mappings":[{"id":"EVT_0001","is_premise":true,"confidence":0.0}]}\n'
        "Use only event IDs provided. If unsure, set is_premise=false.\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=True)}"
    )
    try:
        raw = ask_llm_fn(
            question=prompt,
            model=model,
            temperature=0.0,
            max_output_tokens=700,
            api_key=api_key,
            system_prompt="Return strict JSON only.",
        )
        payload = _parse_llm_json_object(raw)
    except Exception:
        return graph

    mappings = payload.get("mappings", []) if isinstance(payload, dict) else []
    if not isinstance(mappings, list):
        return graph
    cand_ids = {c["id"] for c in candidates}
    for item in mappings:
        if not isinstance(item, dict):
            continue
        evt_id = str(item.get("id", "")).strip()
        if evt_id not in cand_ids:
            continue
        is_premise = bool(item.get("is_premise", False))
        try:
            conf = float(item.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if is_premise and conf >= min_confidence:
            node = next((n for n in graph.get("nodes", []) if n.get("id") == evt_id), None)
            if node is not None:
                node["event_type"] = "PREMISE"
    return graph


def _parse_llm_json_object(raw_text: str) -> dict:
    """Parse a JSON object from raw LLM text with fenced/snippet tolerance.

    Args:
        raw_text: Raw LLM output.

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


def _compute_intention_status(
    fulfills: int,
    fails: int,
    advances: int,
    thwarts: int,
) -> tuple[str, str]:
    """Compute intention status and coarse outcome from effect counts.

    Args:
        fulfills: Number of fulfilling events.
        fails: Number of failing events.
        advances: Number of advancing events.
        thwarts: Number of thwarting events.

    Returns:
        `(status, outcome_type)` tuple.
    """

    if fulfills > 0:
        return "ACHIEVED", "SUCCESS"
    if fails > 0:
        return "FAILED", "FAILURE"
    if thwarts > advances:
        return "FAILED", "BLOCKED"
    if advances > 0:
        return "IN_PROGRESS", "PARTIAL_SUCCESS"
    return "PENDING", "BLOCKED"


def _goal_type_from_intention(intention_type: str, allowed_goal_types: set[str]) -> str:
    """Map one intention type to a broader goal type.

    Args:
        intention_type: Intention label.
        allowed_goal_types: Allowed goal type set.

    Returns:
        Goal type from vocabulary.
    """

    value = str(intention_type or "").strip().upper()
    if value in allowed_goal_types:
        return value

    # Data-driven fallback: pick goal label with maximal token overlap.
    src_tokens = {tok for tok in value.split("_") if tok}
    best_goal = None
    best_score = -1
    for goal in sorted(allowed_goal_types):
        g_tokens = {tok for tok in str(goal).upper().split("_") if tok}
        score = len(src_tokens & g_tokens)
        if score > best_score:
            best_goal = goal
            best_score = score
    if best_goal is not None and best_score > 0:
        return best_goal
    if "OTHER_GOAL" in allowed_goal_types:
        return "OTHER_GOAL"
    return sorted(allowed_goal_types)[0]


def _ensure_entity_goals(
    graph: dict,
    *,
    allowed_goal_types: set[str],
) -> dict:
    """Create explicit story-wide GOAL nodes and link them to entities/intentions.

    Strategy:
    - For each entity owning one or more INT nodes, choose one primary goal.
    - Add `ENT -> GOAL` via HAS_GOAL.
    - Add `GOAL -> INT` via MOTIVATES for that entity's intentions.

    Args:
        graph: Story graph dictionary.
        allowed_goal_types: Allowed goal vocabulary.

    Returns:
        Graph with explicit GOAL nodes/edges.
    """

    node_by_id = {n.get("id"): n for n in graph.get("nodes", []) if isinstance(n, dict)}
    intent_nodes = [n for n in graph.get("nodes", []) if n.get("type") == "INT"]
    if not intent_nodes:
        return graph

    owner_to_ints: dict[str, list[dict]] = defaultdict(list)
    for edge in graph.get("edges", []):
        if edge.get("label") != "INTENDS":
            continue
        ent_id = str(edge.get("source", ""))
        int_id = str(edge.get("target", ""))
        int_node = node_by_id.get(int_id, {})
        if int_node.get("type") != "INT":
            continue
        owner_to_ints[ent_id].append(int_node)

    if not owner_to_ints:
        return graph

    existing_goal_by_owner: dict[str, str] = {}
    for edge in graph.get("edges", []):
        if edge.get("label") != "HAS_GOAL":
            continue
        src = str(edge.get("source", ""))
        dst = str(edge.get("target", ""))
        if node_by_id.get(src, {}).get("type") == "ENT" and node_by_id.get(dst, {}).get("type") == "GOAL":
            existing_goal_by_owner[src] = dst

    goal_counter = 0
    for node in graph.get("nodes", []):
        nid = str(node.get("id", ""))
        if nid.startswith("GOAL_"):
            try:
                goal_counter = max(goal_counter, int(nid.split("_", 1)[1]))
            except Exception:
                pass

    def _aggregate_goal_status(ints: list[dict]) -> tuple[str, str]:
        statuses = [str(n.get("status", "PENDING")).upper() for n in ints]
        outcomes = [str(n.get("outcome_type", "BLOCKED")).upper() for n in ints]
        if "SUCCESS" in outcomes:
            return "ACHIEVED", "SUCCESS"
        if "FAILURE" in outcomes:
            return "FAILED", "FAILURE"
        if "PARTIAL_SUCCESS" in outcomes:
            return "IN_PROGRESS", "PARTIAL_SUCCESS"
        if "BACKFIRE" in outcomes:
            return "FAILED", "BACKFIRE"
        if "IN_PROGRESS" in statuses:
            return "IN_PROGRESS", "PARTIAL_SUCCESS"
        return "PENDING", "BLOCKED"

    for ent_id, int_list in owner_to_ints.items():
        if ent_id in existing_goal_by_owner:
            continue
        goal_votes: Counter = Counter()
        for int_node in int_list:
            goal_type = _goal_type_from_intention(str(int_node.get("intention_type", "")), allowed_goal_types)
            goal_votes[goal_type] += 1
        if not goal_votes:
            continue
        chosen_goal_type, _ = goal_votes.most_common(1)[0]
        status, outcome = _aggregate_goal_status(int_list)
        ent_node = node_by_id.get(ent_id, {})
        prov = dict(ent_node.get("provenance", {}))
        if not prov:
            first_int = int_list[0]
            prov = dict(first_int.get("provenance", {}))
        goal_counter += 1
        goal_id = f"GOAL_{goal_counter:04d}"
        graph["nodes"].append(
            {
                "id": goal_id,
                "type": "GOAL",
                "goal_type": chosen_goal_type,
                "label": chosen_goal_type,
                "name": _label_to_name(chosen_goal_type),
                "status": status,
                "outcome_type": outcome,
                "polarity": "POS",
                "modality": "ASSERTED",
                "tense": "PAST",
                "provenance": prov,
            }
        )
        graph["edges"].append(
            {
                "source": ent_id,
                "target": goal_id,
                "label": "HAS_GOAL",
                "confidence": 0.75,
                "provenance": prov,
            }
        )
        for int_node in int_list:
            graph["edges"].append(
                {
                    "source": goal_id,
                    "target": int_node["id"],
                    "label": "MOTIVATES",
                    "confidence": 0.65,
                    "provenance": dict(int_node.get("provenance", prov)),
                }
            )
    return graph


def _event_order_index(graph: dict) -> dict[str, int]:
    """Build deterministic order index for event nodes.

    Args:
        graph: Story graph dictionary.

    Returns:
        Mapping `event_id -> order_index`.
    """

    events = [n for n in graph.get("nodes", []) if n.get("type") == "EVT"]
    events_sorted = sorted(
        events,
        key=lambda n: (
            str(n.get("provenance", {}).get("sentence_id", "")),
            int(n.get("provenance", {}).get("char_start", 0)),
            str(n.get("id", "")),
        ),
    )
    return {evt["id"]: i for i, evt in enumerate(events_sorted)}


def _attach_intention_of_edges(graph: dict) -> dict:
    """Attach INTENTION_OF edges linking intentions to relevant events.

    Strategy:
    1) If an intention has effect edges, link to earliest affected event.
    2) Otherwise fallback to earliest event where owner is AGENT.

    Args:
        graph: Story graph dictionary.

    Returns:
        Graph with added `INTENTION_OF` edges when possible.
    """

    event_idx = _event_order_index(graph)
    if not event_idx:
        return graph

    edges = graph.get("edges", [])
    node_by_id = {n["id"]: n for n in graph.get("nodes", []) if "id" in n}

    owners_by_int: dict[str, list[str]] = defaultdict(list)
    effect_events_by_int: dict[str, list[str]] = defaultdict(list)
    agent_events_by_ent: dict[str, list[str]] = defaultdict(list)
    existing_pairs = {
        (e.get("source"), e.get("target"))
        for e in edges
        if e.get("label") == "INTENTION_OF"
    }

    for edge in edges:
        label = edge.get("label")
        src = edge.get("source")
        dst = edge.get("target")
        if label == "INTENDS":
            owners_by_int[dst].append(src)
        if label in {"ADVANCES", "THWARTS", "FULFILLS", "FAILS"}:
            effect_events_by_int[dst].append(src)
        if label == "AGENT":
            agent_events_by_ent[dst].append(src)

    for int_node in [n for n in graph.get("nodes", []) if n.get("type") == "INT"]:
        int_id = int_node["id"]
        candidate_events = [eid for eid in effect_events_by_int.get(int_id, []) if eid in event_idx]

        if not candidate_events:
            owner_ids = owners_by_int.get(int_id, [])
            for owner_id in owner_ids:
                candidate_events.extend([eid for eid in agent_events_by_ent.get(owner_id, []) if eid in event_idx])

        if not candidate_events:
            continue

        chosen_event = sorted(set(candidate_events), key=lambda eid: event_idx[eid])[0]
        if (int_id, chosen_event) in existing_pairs:
            continue

        evt_node = node_by_id.get(chosen_event, {})
        prov = dict(evt_node.get("provenance", {}))
        if not prov:
            prov = {
                "doc_id": str(graph.get("doc_id", "")),
                "char_start": 0,
                "char_end": 0,
                "sentence_id": "S0001",
            }
        edges.append(
            {
                "source": int_id,
                "target": chosen_event,
                "label": "INTENTION_OF",
                "confidence": 0.7,
                "provenance": prov,
            }
        )
        existing_pairs.add((int_id, chosen_event))

    graph["edges"] = edges
    return graph


def _node_summary_prompt(graph: dict, story_text: str) -> str:
    """Build prompt for node-level summaries with textual grounding.

    Args:
        graph: Story graph dictionary.
        story_text: Original story text.

    Returns:
        Prompt string.
    """

    lines = []
    for node in graph.get("nodes", []):
        node_id = node.get("id", "")
        node_type = node.get("type", "")
        if node_type == "ENT":
            descriptor = f"character/entity: {node.get('canonical_name', '')}"
        elif node_type == "EVT":
            descriptor = f"event: {node.get('event_type', '')}"
        elif node_type == "INT":
            descriptor = f"intention: {node.get('intention_type', '')}"
        elif node_type == "GOAL":
            descriptor = f"goal: {node.get('goal_type', '')}"
        else:
            descriptor = "node"
        lines.append(f"- {node_id} [{node_type}] {descriptor}")

    node_block = "\n".join(lines)
    return (
        "Generate grounded node summaries as strict JSON only. "
        "Schema: {\"nodes\":[{\"id\":\"...\",\"summary_text\":\"...\",\"evidence_text\":\"...\"}]}. "
        "For each listed node id, provide:\n"
        "- summary_text: short factual summary of that node's role in the story.\n"
        "- evidence_text: short text excerpt or paraphrase from the story supporting the summary.\n"
        "Do not invent facts. Keep each field concise.\n\n"
        f"Nodes:\n{node_block}\n\n"
        f"Story:\n{story_text}"
    )


def _annotate_node_summaries_with_llm(
    graph: dict,
    story_text: str,
    ask_llm_fn: Callable[..., str],
    *,
    model: str | None = None,
    api_key: str | None = None,
    max_output_tokens: int = 1400,
) -> dict:
    """Populate node-level summary fields using LLM structured output.

    Args:
        graph: Story graph dictionary.
        story_text: Original story text.
        ask_llm_fn: LLM callable.
        model: Optional model name.
        api_key: Optional API key.
        max_output_tokens: Token budget for summary response.

    Returns:
        Graph with `summary_text` and `evidence_text` fields set on nodes when available.
    """

    prompt = _node_summary_prompt(graph=graph, story_text=story_text)
    raw = ask_llm_fn(
        question=prompt,
        model=model,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        api_key=api_key,
        system_prompt="Return strict JSON only.",
    )
    payload = _parse_llm_json_object(raw)
    rows = payload.get("nodes", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return graph

    by_id = {node["id"]: node for node in graph.get("nodes", []) if "id" in node}
    for row in rows:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id", "")).strip()
        node = by_id.get(node_id)
        if node is None:
            continue
        summary = str(row.get("summary_text", "")).strip()
        evidence = str(row.get("evidence_text", "")).strip()
        if summary:
            node["summary_text"] = summary
        if evidence:
            node["evidence_text"] = evidence
    return graph


def _apply_node_summary_fallbacks(graph: dict) -> dict:
    """Populate missing summary/evidence fields from provenance and node attrs.

    Args:
        graph: Story graph dictionary.

    Returns:
        Graph with fallback `summary_text` and `evidence_text` values.
    """

    sentence_by_id = {s["sentence_id"]: s for s in graph.get("sentences", []) if "sentence_id" in s}
    for node in graph.get("nodes", []):
        if node.get("summary_text"):
            pass
        else:
            if node.get("type") == "EVT":
                evt = node.get("event_type", "OTHER_EVENT")
                sent = sentence_by_id.get(node.get("provenance", {}).get("sentence_id", ""), {})
                sent_text = str(sent.get("text", "")).strip()
                node["summary_text"] = sent_text or f"Event {evt.lower()}."
            elif node.get("type") == "ENT":
                name = node.get("canonical_name", node.get("id", "entity"))
                node["summary_text"] = f"{name} is a story entity."
            elif node.get("type") == "INT":
                intent = node.get("intention_type", "OTHER_INTENTION").lower().replace("_", " ")
                status = str(node.get("status", "")).lower().replace("_", " ")
                node["summary_text"] = f"Intention to {intent}; status: {status}."
            elif node.get("type") == "GOAL":
                goal = node.get("goal_type", "OTHER_GOAL").lower().replace("_", " ")
                node["summary_text"] = f"Goal: {goal}."
            else:
                node["summary_text"] = f"Node {node.get('id', '')}."

        if node.get("evidence_text"):
            continue
        sent = sentence_by_id.get(node.get("provenance", {}).get("sentence_id", ""), {})
        sent_text = str(sent.get("text", "")).strip()
        if sent_text:
            node["evidence_text"] = sent_text
        else:
            node["evidence_text"] = str(node.get("summary_text", ""))
    return graph


def _align_event_summaries_with_modality(graph: dict) -> dict:
    """Ensure event summaries are consistent with modality.

    Args:
        graph: Story graph dictionary.

    Returns:
        Graph with modality-aligned event summaries.
    """

    sentence_by_id = {s["sentence_id"]: s for s in graph.get("sentences", []) if "sentence_id" in s}
    for node in graph.get("nodes", []):
        if node.get("type") != "EVT":
            continue
        modality = str(node.get("modality", "ASSERTED")).upper()
        summary = str(node.get("summary_text", "")).strip()
        evidence = str(node.get("evidence_text", "")).strip()
        sent = sentence_by_id.get(node.get("provenance", {}).get("sentence_id", ""), {})
        sent_text = str(sent.get("text", "")).strip()
        anchor = evidence or sent_text or summary

        if modality == "POSSIBLE":
            lowered = summary.lower()
            cue_words = {"intend", "intended", "planned", "wants", "wanted", "hopes", "resolved"}
            if not any(cue in lowered for cue in cue_words):
                if anchor:
                    node["summary_text"] = f"Intended action (not yet completed): {anchor}"
                else:
                    node["summary_text"] = "Intended action (not yet completed)."
        elif modality == "OBLIGATED":
            lowered = summary.lower()
            cue_words = {"must", "should", "obligated", "expected"}
            if not any(cue in lowered for cue in cue_words):
                if anchor:
                    node["summary_text"] = f"Obligated/expected action: {anchor}"
                else:
                    node["summary_text"] = "Obligated/expected action."
    return graph


def _retag_event_types_from_summary(graph: dict, allowed_event_types: set[str]) -> dict:
    """Retag selected event types when summary/evidence implies finer semantics.

    Args:
        graph: Story graph dictionary.
        allowed_event_types: Allowed event type set.

    Returns:
        Graph with adjusted event types.
    """

    def _has_any(haystack: str, tokens: list[str]) -> bool:
        return any(tok in haystack for tok in tokens)

    refuse_cues = [
        "refus",
        "refuses",
        "deny",
        "denied",
        "no reward",
        "not pay",
        "did not pay",
        "does not pay",
        "doesn't pay",
        "would not pay",
        "without paying",
        "not fulfill",
        "not fulfil",
        "did not fulfill",
        "did not fulfil",
        "does not fulfill",
        "does not fulfil",
        "failed to pay",
        "reneged",
        "sufficient recompense",
        "already had a sufficient recompense",
    ]
    demand_cues = [
        "demand",
        "asked for payment",
        "ask for payment",
        "asked payment",
        "claim reward",
        "claimed reward",
        "requested payment",
        "requested reward",
        "asked for reward",
        "ask for reward",
    ]
    promise_cues = [
        "promise",
        "promised payment",
        "offered payment",
        "reward you",
        "pay you",
    ]
    extract_cues = ["extract", "pulled out", "remove the bone", "removed the bone"]
    task_cues = [
        "help",
        "service",
        "task",
        "quest",
        "mission",
        "favor",
        "favour",
        "job",
        "for her service",
        "for his service",
        "after helping",
        "after help",
        "did the job",
        "completed",
        "fulfill",
        "fulfil",
    ]
    demand_task_cues = ["demanded", "demand", "asked", "request", "requested", "required"]
    assign_task_cues = ["ordered", "commanded", "assigned", "sent to", "tasked"]
    accept_task_cues = ["accepted", "agreed", "undertook", "set out", "set off"]
    refuse_task_cues = ["refused", "declined", "would not", "wouldn't"]
    deal_cues = ["deal", "bargain", "negotiat", "pact"]
    break_promise_cues = ["broke promise", "break promise", "betrayed promise", "did not keep"]
    curse_cues = ["cursed", "curse", "hexed", "spell upon"]
    bless_cues = ["blessed", "bless", "boon"]
    transform_cues = ["transformed", "transform", "turned into", "changed into"]
    disguise_cues = ["disguised", "disguise", "in disguise"]
    reveal_identity_cues = ["revealed himself", "revealed herself", "revealed identity", "true identity"]
    test_cues = ["test", "tested", "trial", "prove"]
    pass_test_cues = ["passed", "succeeded", "proved worthy"]
    fail_test_cues = ["failed", "could not", "unworthy"]
    banish_cues = ["banished", "banish", "exiled", "cast out"]
    return_home_cues = ["returned home", "went home", "came home"]
    marry_cues = ["married", "wed", "wedded"]
    premise_cues = [
        "stuck in",
        "in his throat",
        "in her throat",
        "had a bone",
        "had a family",
        "family of",
        "were perpetually",
        "perpetually quarrel",
        "quarreling among",
        "quarrelling among",
        "were quarreling",
        "were quarrelling",
        "was hungry",
        "were hungry",
        "needed help",
        "could not",
        "unable to",
        "in trouble",
    ]
    action_like_cues = (
        refuse_cues
        + demand_cues
        + promise_cues
        + extract_cues
        + task_cues
        + demand_task_cues
        + assign_task_cues
        + accept_task_cues
        + refuse_task_cues
        + deal_cues
        + break_promise_cues
        + curse_cues
        + bless_cues
        + transform_cues
        + disguise_cues
        + reveal_identity_cues
        + test_cues
        + pass_test_cues
        + fail_test_cues
        + banish_cues
        + return_home_cues
        + marry_cues
    )

    sentence_by_id = {str(s.get("sentence_id", "")): s for s in graph.get("sentences", []) if isinstance(s, dict)}
    events_by_sentence: dict[str, list[dict]] = defaultdict(list)
    for evt in graph.get("nodes", []):
        if evt.get("type") != "EVT":
            continue
        sid = str(evt.get("provenance", {}).get("sentence_id", ""))
        events_by_sentence[sid].append(evt)
    for sid in list(events_by_sentence.keys()):
        events_by_sentence[sid].sort(
            key=lambda n: (
                int(n.get("provenance", {}).get("char_start", 10**9)),
                int(n.get("provenance", {}).get("char_end", 10**9)),
                str(n.get("id", "")),
            )
        )

    for node in graph.get("nodes", []):
        if node.get("type") != "EVT":
            continue
        event_type = str(node.get("event_type", "OTHER_EVENT")).upper()
        text = f"{node.get('summary_text', '')} {node.get('evidence_text', '')}".lower()
        sent_id = str(node.get("provenance", {}).get("sentence_id", ""))
        sent_text = str(sentence_by_id.get(sent_id, {}).get("text", "")).lower()
        has_sent_text = bool(sent_text.strip())
        sentence_events = events_by_sentence.get(sent_id, [])
        is_first_event = bool(sentence_events) and str(sentence_events[0].get("id", "")) == str(node.get("id", ""))

        def _grounded(cues: list[str]) -> bool:
            if not _has_any(text, cues):
                return False
            if not has_sent_text:
                return True
            return _has_any(sent_text, cues)

        # Payment semantics should override generic or mismatched action labels.
        if _grounded(refuse_cues) and "DENY_PAYMENT" in allowed_event_types:
            node["event_type"] = "DENY_PAYMENT"
            continue
        if _grounded(demand_cues) and "DEMAND_PAYMENT" in allowed_event_types:
            node["event_type"] = "DEMAND_PAYMENT"
            continue
        if _grounded(promise_cues) and "PROMISE_PAYMENT" in allowed_event_types:
            node["event_type"] = "PROMISE_PAYMENT"
            continue

        if _grounded(break_promise_cues):
            if "BREAK_PROMISE" in allowed_event_types:
                node["event_type"] = "BREAK_PROMISE"
                continue

        has_task_context = _grounded(task_cues)
        if has_task_context:
            if _grounded(assign_task_cues):
                if "ASSIGN_TASK" in allowed_event_types:
                    node["event_type"] = "ASSIGN_TASK"
                    continue
            if _grounded(demand_task_cues):
                if "DEMAND_TASK" in allowed_event_types:
                    node["event_type"] = "DEMAND_TASK"
                    continue
            if _grounded(accept_task_cues):
                if "ACCEPT_TASK" in allowed_event_types:
                    node["event_type"] = "ACCEPT_TASK"
                    continue
            if _grounded(refuse_task_cues):
                if "REFUSE_TASK" in allowed_event_types:
                    node["event_type"] = "REFUSE_TASK"
                    continue

        if _grounded(deal_cues):
            if "MAKE_DEAL" in allowed_event_types:
                node["event_type"] = "MAKE_DEAL"
                continue
        if _grounded(curse_cues):
            if "CURSE" in allowed_event_types:
                node["event_type"] = "CURSE"
                continue
        if _grounded(bless_cues):
            if "BLESS" in allowed_event_types:
                node["event_type"] = "BLESS"
                continue
        if _grounded(transform_cues):
            if "TRANSFORM" in allowed_event_types:
                node["event_type"] = "TRANSFORM"
                continue
        if _grounded(disguise_cues):
            if "DISGUISE" in allowed_event_types:
                node["event_type"] = "DISGUISE"
                continue
        if _grounded(reveal_identity_cues):
            if "REVEAL_IDENTITY" in allowed_event_types:
                node["event_type"] = "REVEAL_IDENTITY"
                continue
        if _grounded(test_cues):
            if _grounded(pass_test_cues) and "PASS_TEST" in allowed_event_types:
                node["event_type"] = "PASS_TEST"
                continue
            if _grounded(fail_test_cues) and "FAIL_TEST" in allowed_event_types:
                node["event_type"] = "FAIL_TEST"
                continue
            if "TEST" in allowed_event_types:
                node["event_type"] = "TEST"
                continue
        if _grounded(banish_cues):
            if "BANISH" in allowed_event_types:
                node["event_type"] = "BANISH"
                continue
        if _grounded(return_home_cues):
            if "RETURN_HOME" in allowed_event_types:
                node["event_type"] = "RETURN_HOME"
                continue
        if _grounded(marry_cues):
            if "MARRY" in allowed_event_types:
                node["event_type"] = "MARRY"
                continue
        if (
            event_type in {"OTHER_EVENT", "PREMISE"}
            and is_first_event
            and _grounded(premise_cues)
            and not _grounded(action_like_cues)
        ):
            if "PREMISE" in allowed_event_types:
                node["event_type"] = "PREMISE"
                continue

        if event_type == "OTHER_EVENT":
            if _grounded(extract_cues):
                if "FULFILL_TASK" in allowed_event_types:
                    node["event_type"] = "FULFILL_TASK"
                elif "OTHER_EVENT" in allowed_event_types:
                    node["event_type"] = "OTHER_EVENT"
    return graph


def _demote_extra_premise_events(graph: dict, allowed_event_types: set[str]) -> dict:
    """Keep at most one PREMISE node and remap remaining ones to concrete labels.

    Args:
        graph: Story graph dictionary.
        allowed_event_types: Allowed event type set.

    Returns:
        Graph with non-initial PREMISE events demoted/remapped.
    """

    sentence_by_id = {str(s.get("sentence_id", "")): s for s in graph.get("sentences", []) if isinstance(s, dict)}
    events = [n for n in graph.get("nodes", []) if n.get("type") == "EVT"]
    events_sorted = sorted(
        events,
        key=lambda n: (
            str(n.get("provenance", {}).get("sentence_id", "")),
            int(n.get("provenance", {}).get("char_start", 10**9)),
            int(n.get("provenance", {}).get("char_end", 10**9)),
            str(n.get("id", "")),
        ),
    )
    premise_events = [n for n in events_sorted if str(n.get("event_type", "")).upper() == "PREMISE"]
    if len(premise_events) <= 1:
        return graph

    refuse_cues = ["refus", "deny", "denied", "not pay", "would not pay", "sufficient recompense"]
    demand_cues = ["demand", "asked for payment", "requested payment", "requested reward", "promised payment"]
    assign_task_cues = ["hired", "hire", "ordered", "commanded", "assigned", "tasked", "sent to"]
    fulfill_task_cues = [
        "extract",
        "remove the bone",
        "removed the bone",
        "draw out the bone",
        "completed the task",
        "fulfilled the task",
        "service",
    ]

    def _has_any(text: str, cues: list[str]) -> bool:
        return any(c in text for c in cues)

    # keep the earliest setup node as PREMISE
    keep_id = str(premise_events[0].get("id", ""))
    for node in premise_events[1:]:
        text = f"{node.get('summary_text', '')} {node.get('evidence_text', '')}".lower()
        sid = str(node.get("provenance", {}).get("sentence_id", ""))
        sent_text = str(sentence_by_id.get(sid, {}).get("text", "")).lower()
        all_text = f"{text} {sent_text}"
        if _has_any(all_text, refuse_cues) and "DENY_PAYMENT" in allowed_event_types:
            node["event_type"] = "DENY_PAYMENT"
        elif _has_any(all_text, demand_cues) and "DEMAND_PAYMENT" in allowed_event_types:
            node["event_type"] = "DEMAND_PAYMENT"
        elif _has_any(all_text, assign_task_cues) and "ASSIGN_TASK" in allowed_event_types:
            node["event_type"] = "ASSIGN_TASK"
        elif _has_any(all_text, fulfill_task_cues) and "FULFILL_TASK" in allowed_event_types:
            node["event_type"] = "FULFILL_TASK"
        else:
            node["event_type"] = "OTHER_EVENT" if "OTHER_EVENT" in allowed_event_types else sorted(allowed_event_types)[0]

    for node in graph.get("nodes", []):
        if node.get("type") == "EVT" and str(node.get("event_type", "")).upper() == "PREMISE" and str(node.get("id", "")) != keep_id:
            # safety net in case a node escaped remap above
            node["event_type"] = "OTHER_EVENT" if "OTHER_EVENT" in allowed_event_types else sorted(allowed_event_types)[0]
    return graph


def _semantic_remap_other_events_with_llm(
    graph: dict,
    *,
    allowed_event_types: set[str],
    ask_llm_fn: Callable[..., str] | None,
    model: str | None = None,
    api_key: str | None = None,
    min_confidence: float = 0.72,
) -> dict:
    """Use LLM to remap OTHER_EVENT nodes to a specific label when confident.

    Args:
        graph: Story graph dictionary.
        allowed_event_types: Allowed event type set.
        ask_llm_fn: Optional LLM callable.
        model: Optional model name.
        api_key: Optional API key.
        min_confidence: Minimum confidence to accept remap.

    Returns:
        Graph with selected OTHER_EVENT labels remapped.
    """

    if ask_llm_fn is None:
        return graph

    other_nodes = [n for n in graph.get("nodes", []) if n.get("type") == "EVT" and n.get("event_type") == "OTHER_EVENT"]
    if not other_nodes:
        return graph

    sentence_by_id = {str(s.get("sentence_id", "")): s for s in graph.get("sentences", []) if isinstance(s, dict)}
    role_edges = [e for e in graph.get("edges", []) if e.get("label") in {"AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT", "LOCATION", "TIME"}]
    node_by_id = {n.get("id"): n for n in graph.get("nodes", []) if isinstance(n, dict)}
    roles_by_event: dict[str, list[str]] = defaultdict(list)
    for edge in role_edges:
        evt_id = str(edge.get("source", ""))
        ent_id = str(edge.get("target", ""))
        ent_name = str(node_by_id.get(ent_id, {}).get("canonical_name", "")).strip()
        if ent_name:
            roles_by_event[evt_id].append(f"{edge.get('label')}={ent_name}")

    allowed = sorted(t for t in allowed_event_types if t != "OTHER_EVENT")
    rows = []
    for node in other_nodes:
        evt_id = str(node.get("id", ""))
        sent_id = str(node.get("provenance", {}).get("sentence_id", ""))
        sent_text = str(sentence_by_id.get(sent_id, {}).get("text", "")).strip()
        summary = str(node.get("summary_text", "")).strip()
        evidence = str(node.get("evidence_text", "")).strip()
        roles_txt = "; ".join(roles_by_event.get(evt_id, []))
        rows.append(
            {
                "id": evt_id,
                "sentence_text": sent_text,
                "summary_text": summary,
                "evidence_text": evidence,
                "roles": roles_txt,
            }
        )

    prompt = (
        "Remap only OTHER_EVENT nodes to more specific event types using explicit evidence.\n"
        "Return strict JSON only with schema:\n"
        '{"mappings":[{"id":"EVT_0001","event_type":"...","confidence":0.0,"reason":"..."}]}\n'
        "Constraints:\n"
        f"- Allowed event_type values: {', '.join(allowed)}\n"
        "- Use only provided sentence_text/summary_text/evidence_text/roles.\n"
        "- If uncertain, omit that id entirely.\n"
        "- Do not output OTHER_EVENT.\n\n"
        f"Nodes:\n{json.dumps(rows, ensure_ascii=True)}"
    )
    try:
        raw = ask_llm_fn(
            question=prompt,
            model=model,
            temperature=0.0,
            max_output_tokens=900,
            api_key=api_key,
            system_prompt="Return strict JSON only.",
        )
        payload = _parse_llm_json_object(raw)
    except Exception:
        return graph

    mappings = payload.get("mappings", []) if isinstance(payload, dict) else []
    if not isinstance(mappings, list):
        return graph

    by_id = {str(n.get("id", "")): n for n in other_nodes}
    events_by_sentence: dict[str, list[dict]] = defaultdict(list)
    for evt in graph.get("nodes", []):
        if evt.get("type") != "EVT":
            continue
        sid = str(evt.get("provenance", {}).get("sentence_id", ""))
        events_by_sentence[sid].append(evt)
    for sid in list(events_by_sentence.keys()):
        events_by_sentence[sid].sort(
            key=lambda n: (
                int(n.get("provenance", {}).get("char_start", 10**9)),
                int(n.get("provenance", {}).get("char_end", 10**9)),
                str(n.get("id", "")),
            )
        )
    for item in mappings:
        if not isinstance(item, dict):
            continue
        evt_id = str(item.get("id", "")).strip()
        event_type = str(item.get("event_type", "")).strip().upper()
        try:
            confidence = float(item.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        if evt_id not in by_id:
            continue
        if event_type not in allowed_event_types or event_type == "OTHER_EVENT":
            continue
        if confidence < min_confidence:
            continue
        if event_type == "PREMISE":
            node = by_id.get(evt_id, {})
            sent_id = str(node.get("provenance", {}).get("sentence_id", ""))
            sent_events = events_by_sentence.get(sent_id, [])
            is_first_event = bool(sent_events) and str(sent_events[0].get("id", "")) == evt_id
            text = f"{node.get('summary_text', '')} {node.get('evidence_text', '')}".lower()
            premise_like = any(
                tok in text
                for tok in ["stuck in", "in his throat", "in her throat", "had a bone", "was hungry", "in trouble"]
            )
            if not (is_first_event and premise_like):
                continue
        by_id[evt_id]["event_type"] = event_type
    return graph


def _canonicalize_forbidden_event_types(graph: dict, allowed_event_types: set[str]) -> dict:
    """Rewrite forbidden legacy event labels to canonical abstractions.

    Args:
        graph: Story graph dictionary.
        allowed_event_types: Allowed event type set.

    Returns:
        Graph with forbidden labels rewritten.
    """

    _ = allowed_event_types
    return graph


def _finalize_payment_event_polarity(graph: dict, allowed_event_types: set[str]) -> dict:
    """Force payment event polarity from node-local text cues.

    This is a final deterministic guard against lingering DEMAND/PROMISE mislabels
    when summary/evidence clearly expresses refusal/denial.

    Args:
        graph: Story graph dictionary.
        allowed_event_types: Allowed event type set.

    Returns:
        Graph with payment labels normalized.
    """

    refuse_cues = [
        "refus",
        "deny",
        "denied",
        "not pay",
        "would not pay",
        "does not pay",
        "doesn't pay",
        "without paying",
        "failed to pay",
        "reneged",
        "sufficient recompense",
    ]
    demand_cues = [
        "demand",
        "asked for payment",
        "request payment",
        "requested payment",
        "claim reward",
        "requested reward",
    ]
    promise_cues = ["promised payment", "promise to pay", "offer payment", "will pay"]

    def _has_any(text: str, cues: list[str]) -> bool:
        return any(c in text for c in cues)

    for node in graph.get("nodes", []):
        if node.get("type") != "EVT":
            continue
        text = f"{node.get('summary_text', '')} {node.get('evidence_text', '')}".lower()
        if not text.strip():
            continue
        if _has_any(text, refuse_cues) and "DENY_PAYMENT" in allowed_event_types:
            node["event_type"] = "DENY_PAYMENT"
            continue
        if _has_any(text, demand_cues) and "DEMAND_PAYMENT" in allowed_event_types:
            node["event_type"] = "DEMAND_PAYMENT"
            continue
        if _has_any(text, promise_cues) and "PROMISE_PAYMENT" in allowed_event_types:
            node["event_type"] = "PROMISE_PAYMENT"
            continue
    return graph


def _drop_generic_other_intentions(graph: dict) -> dict:
    """Remove generic OTHER_INTENTION nodes and incident edges.

    Args:
        graph: Story graph dictionary.

    Returns:
        Graph with OTHER_INTENTION nodes removed.
    """

    remove_ids = {
        str(node.get("id", ""))
        for node in graph.get("nodes", [])
        if node.get("type") == "INT" and str(node.get("intention_type", "")).upper() == "OTHER_INTENTION"
    }
    if not remove_ids:
        return graph

    graph["nodes"] = [node for node in graph.get("nodes", []) if str(node.get("id", "")) not in remove_ids]
    graph["edges"] = [
        edge
        for edge in graph.get("edges", [])
        if str(edge.get("source", "")) not in remove_ids and str(edge.get("target", "")) not in remove_ids
    ]
    return graph


def _deduplicate_redundant_events(graph: dict) -> dict:
    """Drop duplicate event nodes that represent the same sentence-local action.

    Deduplication is intentionally conservative and only merges events when all of
    the following match:
    - same sentence id
    - same event type/modality/polarity
    - same participant role bindings
    - same normalized summary/evidence hint text

    Args:
        graph: Story graph dictionary.

    Returns:
        Graph with duplicate EVT nodes removed.
    """

    role_labels = ["AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT", "LOCATION", "TIME"]
    roles_by_event: dict[str, dict[str, str]] = defaultdict(dict)
    for edge in graph.get("edges", []):
        label = str(edge.get("label", ""))
        if label not in role_labels:
            continue
        src = str(edge.get("source", ""))
        dst = str(edge.get("target", ""))
        roles_by_event[src][label] = dst

    def _norm_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    events = [n for n in graph.get("nodes", []) if n.get("type") == "EVT"]
    if len(events) <= 1:
        return graph
    events_sorted = sorted(
        events,
        key=lambda n: (
            str(n.get("provenance", {}).get("sentence_id", "")),
            int(n.get("provenance", {}).get("char_start", 10**9)),
            int(n.get("provenance", {}).get("char_end", 10**9)),
            str(n.get("id", "")),
        ),
    )

    seen_keys: set[tuple] = set()
    drop_ids: set[str] = set()
    for evt in events_sorted:
        evt_id = str(evt.get("id", ""))
        sid = str(evt.get("provenance", {}).get("sentence_id", ""))
        text_hint = _norm_text(str(evt.get("summary_text", "")) or str(evt.get("evidence_text", "")))
        role_sig = tuple((label, roles_by_event.get(evt_id, {}).get(label, "")) for label in role_labels)
        key = (
            sid,
            str(evt.get("event_type", "OTHER_EVENT")),
            str(evt.get("modality", "ASSERTED")),
            str(evt.get("polarity", "POS")),
            role_sig,
            text_hint,
        )
        if key in seen_keys:
            drop_ids.add(evt_id)
        else:
            seen_keys.add(key)

    if not drop_ids:
        return graph

    graph["nodes"] = [n for n in graph.get("nodes", []) if str(n.get("id", "")) not in drop_ids]
    graph["edges"] = [
        e
        for e in graph.get("edges", [])
        if str(e.get("source", "")) not in drop_ids and str(e.get("target", "")) not in drop_ids
    ]
    return graph


def _rebuild_before_edges_from_provenance(graph: dict) -> dict:
    """Rebuild a strict linear BEFORE chain from event provenance ordering.

    Args:
        graph: Story graph dictionary.

    Returns:
        Graph with deterministic BEFORE edges.
    """

    events = [n for n in graph.get("nodes", []) if n.get("type") == "EVT"]
    if len(events) <= 1:
        graph["edges"] = [e for e in graph.get("edges", []) if str(e.get("label", "")) != "BEFORE"]
        return graph

    sentences = [s for s in graph.get("sentences", []) if isinstance(s, dict)]
    sentence_rank = {
        str(s.get("sentence_id", "")): i
        for i, s in enumerate(sorted(sentences, key=lambda x: str(x.get("sentence_id", ""))))
    }
    events_sorted = sorted(
        events,
        key=lambda n: (
            sentence_rank.get(str(n.get("provenance", {}).get("sentence_id", "")), 10**9),
            int(n.get("provenance", {}).get("char_start", 10**9)),
            int(n.get("provenance", {}).get("char_end", 10**9)),
            str(n.get("id", "")),
        ),
    )
    by_id = {str(n.get("id", "")): n for n in events_sorted}

    graph["edges"] = [e for e in graph.get("edges", []) if str(e.get("label", "")) != "BEFORE"]
    for left, right in zip(events_sorted[:-1], events_sorted[1:]):
        rid = str(right.get("id", ""))
        prov = dict(by_id.get(rid, {}).get("provenance", {}))
        graph["edges"].append(
            {
                "source": str(left.get("id", "")),
                "target": rid,
                "label": "BEFORE",
                "provenance": prov,
            }
        )
    return graph


def _renumber_event_ids(graph: dict) -> dict:
    """Renumber EVT ids deterministically and rewrite all incident edges.

    Args:
        graph: Story graph dictionary.

    Returns:
        Graph with contiguous EVT ids starting from EVT_0001.
    """

    sentences = [s for s in graph.get("sentences", []) if isinstance(s, dict)]
    sentence_rank = {str(s.get("sentence_id", "")): i for i, s in enumerate(sorted(sentences, key=lambda x: str(x.get("sentence_id", ""))))}

    events = [n for n in graph.get("nodes", []) if n.get("type") == "EVT"]
    events_sorted = sorted(
        events,
        key=lambda n: (
            sentence_rank.get(str(n.get("provenance", {}).get("sentence_id", "")), 10**9),
            int(n.get("provenance", {}).get("char_start", 10**9)),
            int(n.get("provenance", {}).get("char_end", 10**9)),
            str(n.get("id", "")),
        ),
    )

    id_map: dict[str, str] = {}
    for idx, node in enumerate(events_sorted, start=1):
        old_id = str(node.get("id", ""))
        new_id = f"EVT_{idx:04d}"
        id_map[old_id] = new_id
        node["id"] = new_id

    if not id_map:
        return graph

    for edge in graph.get("edges", []):
        src = str(edge.get("source", ""))
        dst = str(edge.get("target", ""))
        if src in id_map:
            edge["source"] = id_map[src]
        if dst in id_map:
            edge["target"] = id_map[dst]
    return graph


def _infer_modality_from_sentence(sentence_text: str, default_modality: str) -> str:
    """Infer modality from textual cues to avoid asserting planned actions as facts.

    Args:
        sentence_text: Sentence text.
        default_modality: Existing modality value.

    Returns:
        Refined modality value.
    """

    if default_modality != "ASSERTED":
        return default_modality
    text = sentence_text.lower()
    cue_tokens = {
        "intend",
        "intended",
        "intention",
        "plan",
        "planned",
        "resolved",
        "wanted",
        "wished",
        "hoped",
        "pretext",
        "in order to",
        "so that",
    }
    if any(cue in text for cue in cue_tokens) and " to " in text:
        return "POSSIBLE"
    return default_modality


def _extract_relationship_nodes(
    graph: dict,
    *,
    allowed_relation_types: set[str],
) -> dict:
    """Extract relation nodes with discrete relation vocabulary.

    Args:
        graph: Story graph dictionary.
        allowed_relation_types: Allowed relation-type set.

    Returns:
        Graph with REL nodes/edges appended.
    """

    ents = [n for n in graph.get("nodes", []) if n.get("type") == "ENT"]
    if not ents:
        return graph
    by_id = {n["id"]: n for n in ents}

    rel_counter = 0
    for n in graph.get("nodes", []):
        nid = str(n.get("id", ""))
        if nid.startswith("REL_"):
            try:
                rel_counter = max(rel_counter, int(nid.split("_", 1)[1]))
            except Exception:
                pass

    existing_rel_pairs = {
        (
            str(node.get("relation_type", "")),
            str(sub.get("target", "")),
            str(obj.get("target", "")),
        )
        for node in graph.get("nodes", [])
        if node.get("type") == "REL"
        for sub in graph.get("edges", [])
        for obj in graph.get("edges", [])
        if sub.get("source") == node.get("id")
        and obj.get("source") == node.get("id")
        and sub.get("label") == "REL_SUBJECT"
        and obj.get("label") == "REL_OBJECT"
    }

    def _add_rel(relation_type: str, subj_id: str, obj_id: str, provenance: dict) -> None:
        nonlocal rel_counter
        if relation_type not in allowed_relation_types:
            return
        key = (relation_type, subj_id, obj_id)
        if key in existing_rel_pairs:
            return
        rel_counter += 1
        rel_id = f"REL_{rel_counter:04d}"
        graph["nodes"].append(
            {
                "id": rel_id,
                "type": "REL",
                "relation_type": relation_type,
                "label": relation_type,
                "name": _label_to_name(relation_type),
                "polarity": "POS",
                "modality": "ASSERTED",
                "tense": "PAST",
                "provenance": provenance,
            }
        )
        graph["edges"].append({"source": rel_id, "target": subj_id, "label": "REL_SUBJECT", "provenance": provenance})
        graph["edges"].append({"source": rel_id, "target": obj_id, "label": "REL_OBJECT", "provenance": provenance})
        existing_rel_pairs.add(key)

    for sent in graph.get("sentences", []):
        text = str(sent.get("text", "")).lower()
        prov = {
            "doc_id": str(graph.get("doc_id", "")),
            "char_start": int(sent.get("char_start", 0)),
            "char_end": int(sent.get("char_end", 0)),
            "sentence_id": str(sent.get("sentence_id", "S0001")),
        }
        ent_ids = [n["id"] for n in ents if n.get("provenance", {}).get("sentence_id") == sent.get("sentence_id")]
        if len(ent_ids) < 2:
            continue
        person_ids = [eid for eid in ent_ids if by_id.get(eid, {}).get("entity_type") in {"PERSON", "YOUNG_PERSON", "PEOPLE", "GROUP"}]
        if "father" in text or "mother" in text or "parent" in text:
            for sid in person_ids:
                sname = str(by_id.get(sid, {}).get("canonical_name", "")).lower()
                if sname in {"father", "mother", "parent"}:
                    for oid in person_ids:
                        if oid == sid:
                            continue
                        oname = str(by_id.get(oid, {}).get("canonical_name", "")).lower()
                        if oname in {"sons", "daughters", "children", "son", "daughter", "child", "people"}:
                            _add_rel("PARENT_OF", sid, oid, prov)
    return graph


def _prune_orphan_entities(graph: dict) -> dict:
    """Remove entity nodes that are disconnected from all semantic edges.

    Args:
        graph: Story graph dictionary.

    Returns:
        Graph with orphan `ENT` nodes removed.
    """

    edges = list(graph.get("edges", []))
    ent_ids = [node["id"] for node in graph.get("nodes", []) if node.get("type") == "ENT"]
    connected = set()
    for edge in edges:
        label = edge.get("label")
        src = edge.get("source")
        dst = edge.get("target")
        if label in {"AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT", "LOCATION", "TIME"}:
            if dst in ent_ids:
                connected.add(dst)
        if label == "INTENDS":
            if src in ent_ids:
                connected.add(src)
        if label == "COREF":
            if src in ent_ids:
                connected.add(src)
            if dst in ent_ids:
                connected.add(dst)

    pruned_nodes = []
    for node in graph.get("nodes", []):
        if node.get("type") == "ENT" and node["id"] not in connected:
            continue
        pruned_nodes.append(node)
    graph["nodes"] = pruned_nodes
    return graph


def build_story_graph_from_text(
    text: str,
    doc_id: str,
    *,
    ask_llm_fn: Callable[..., str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    enforce_known_vocab: bool = False,
    vocab_path: str | None = None,
    dictionary_dir: str | None = None,
    add_node_summaries: bool = False,
) -> dict:
    """Extract a discrete story graph from raw story text.

    Args:
        text: Raw story text.
        doc_id: Story identifier.
        ask_llm_fn: Optional LLM callable for main-character and event extraction refinement.
        model: Optional model name for ask_llm_fn.
        api_key: Optional API key for ask_llm_fn.
        enforce_known_vocab: If True, fail when unknown type would be needed.
        vocab_path: Optional custom vocabulary path.
        dictionary_dir: Optional directory containing `vocab_v0.json`.
        add_node_summaries: If True and ask_llm_fn is provided, annotate each node
            with `summary_text` and `evidence_text`.

    Returns:
        Validated story graph dictionary.
    """

    normalized = normalize_text(text)
    graph = make_empty_graph(doc_id=doc_id, text=normalized)
    graph["vocab_version"] = VOCAB_VERSION

    vocab = vocab_sets(vocab_path=vocab_path, dictionary_dir=dictionary_dir)
    allowed_entity_types = vocab["entity_types"]
    allowed_relation_types = vocab["relation_types"]
    allowed_event_types = vocab["event_types"]
    allowed_goal_types = vocab["goal_types"]
    allowed_intention_types = vocab["intention_types"]
    induced = induced_sets(dictionary_dir=dictionary_dir)
    if induced.get("relations"):
        allowed_relation_types = set(induced["relations"])
    if induced.get("events"):
        allowed_event_types = set(induced["events"])
    if induced.get("goals"):
        allowed_goal_types = set(induced["goals"])
    if induced.get("intentions"):
        allowed_intention_types = set(induced["intentions"])
    induced_entity_terms = _load_induced_entity_terms(dictionary_dir)

    sentences = split_sentences_with_offsets(normalized)
    for sentence in sentences:
        graph["sentences"].append(
            {
                "sentence_id": sentence.sentence_id,
                "text": sentence.text,
                "char_start": sentence.char_start,
                "char_end": sentence.char_end,
            }
        )
    premise_sentence_ids = _infer_premise_sentence_ids_with_llm(
        story_text=normalized,
        sentences=sentences,
        ask_llm_fn=ask_llm_fn,
        model=model,
        api_key=api_key,
    )

    mentions = extract_entity_mentions(sentences)
    main_character_keys: set[str] | None = None
    if ask_llm_fn is not None:
        try:
            main_character_keys = infer_main_characters_with_llm(
                story_text=normalized,
                ask_llm_fn=ask_llm_fn,
                model=model,
                api_key=api_key,
            )
        except Exception:
            main_character_keys = None

    entity_id_by_canonical: dict[str, str] = {}
    entity_type_cache: dict[str, str] = {}
    entity_label_cache: dict[tuple[str, str], str] = {}
    ent_counter = 0
    for mention in mentions:
        canonical_name = _abstract_entity_name(mention.canonical_name)
        canonical_key = canonical_name.lower()
        if main_character_keys:
            if not _matches_main_character(canonical_name, main_character_keys):
                continue
        if canonical_key in entity_id_by_canonical:
            continue
        ent_counter += 1
        entity_id = f"ENT_{ent_counter:04d}"
        if canonical_key in entity_type_cache:
            entity_type = entity_type_cache[canonical_key]
        else:
            entity_type = _infer_entity_type_with_llm(
                canonical_name,
                allowed_entity_types=allowed_entity_types,
                ask_llm_fn=ask_llm_fn,
                model=model,
                api_key=api_key,
            )
            entity_type_cache[canonical_key] = entity_type
        label_key = (canonical_key, entity_type)
        if label_key in entity_label_cache:
            dictionary_name = entity_label_cache[label_key]
        else:
            dictionary_name = _entity_dictionary_label_with_llm(
                canonical_name,
                entity_type,
                induced_entity_terms,
                ask_llm_fn=ask_llm_fn,
                model=model,
                api_key=api_key,
            )
            entity_label_cache[label_key] = dictionary_name

        if enforce_known_vocab and entity_type not in allowed_entity_types:
            raise ValueError(f"Unknown entity type '{entity_type}' for '{canonical_name}'")

        graph["nodes"].append(
            {
                "id": entity_id,
                "type": "ENT",
                "entity_type": entity_type,
                "surface_name": mention.text,
                "name": canonical_name,
                "canonical_name": canonical_name,
                "label": dictionary_name,
                "dictionary_name": dictionary_name,
                "polarity": "POS",
                "modality": "ASSERTED",
                "tense": "PAST",
                "provenance": {
                    "doc_id": doc_id,
                    "char_start": mention.char_start,
                    "char_end": mention.char_end,
                    "sentence_id": mention.sentence_id,
                },
            }
        )
        entity_id_by_canonical[canonical_key] = entity_id

    mention_ids_by_sentence: dict[str, list[str]] = {}
    for mention in mentions:
        canonical_key = _abstract_entity_name(mention.canonical_name).lower()
        if canonical_key not in entity_id_by_canonical:
            continue
        entity_id = entity_id_by_canonical[canonical_key]
        mention_ids_by_sentence.setdefault(mention.sentence_id, []).append(entity_id)

    evt_counter = 0
    previous_event_id: str | None = None
    event_ids_in_order: list[str] = []

    for sentence in sentences:
        sentence_entity_ids = []
        seen = set()
        for ent_id in mention_ids_by_sentence.get(sentence.sentence_id, []):
            if ent_id not in seen:
                sentence_entity_ids.append(ent_id)
                seen.add(ent_id)

        llm_events: list[dict] = []
        if ask_llm_fn is not None:
            prompt = _event_extraction_prompt(
                sentence_text=sentence.text,
                event_types=sorted(allowed_event_types),
                role_labels=["AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT", "LOCATION", "TIME"],
                entity_names=sorted(name.title() for name in entity_id_by_canonical.keys()),
            )
            raw = ask_llm_fn(
                question=prompt,
                model=model,
                temperature=0.0,
                max_output_tokens=700,
                api_key=api_key,
                system_prompt="Return strict JSON only.",
            )
            llm_events = parse_llm_events(raw)

        events_to_emit = llm_events if llm_events else [{}]
        lexical_candidates = _candidate_event_types_from_sentence(sentence.text, allowed_event_types)

        for chosen in events_to_emit:
            roles: list[tuple[str, str]] = []
            llm_role_pairs: list[tuple[str, str]] = []

            event_type = _extract_event_type(sentence.text, allowed_event_types)
            polarity = "POS"
            modality = "ASSERTED"
            tense = "PAST"
            summary_text = ""
            evidence_text = ""
            span_start = sentence.char_start
            span_end = sentence.char_end

            if isinstance(chosen, dict) and chosen:
                summary_text = str(chosen.get("summary_text", "")).strip()
                evidence_text = str(chosen.get("evidence_text", "")).strip()

                span_payload = chosen.get("span", {})
                if isinstance(span_payload, dict):
                    cstart = span_payload.get("char_start")
                    cend = span_payload.get("char_end")
                    if isinstance(cstart, int) and isinstance(cend, int) and cstart >= 0 and cend >= cstart:
                        span_start = sentence.char_start + cstart
                        span_end = sentence.char_start + cend

                # Determine lexical support from event-local anchor text first.
                local_anchor = ""
                if span_start >= sentence.char_start and span_end <= sentence.char_end and span_end > span_start:
                    rel_start = max(0, span_start - sentence.char_start)
                    rel_end = min(len(sentence.text), span_end - sentence.char_start)
                    local_anchor = sentence.text[rel_start:rel_end].strip()
                if not local_anchor:
                    local_anchor = sentence.text

                local_candidates = _candidate_event_types_from_sentence(local_anchor, allowed_event_types)
                if local_candidates:
                    event_type = _extract_event_type(local_anchor, allowed_event_types)

                candidate_type = str(chosen.get("event_type", "")).strip().upper()
                acceptance_candidates = local_candidates if local_candidates else lexical_candidates
                if candidate_type in allowed_event_types and (
                    candidate_type in acceptance_candidates
                    or (not acceptance_candidates and candidate_type == event_type)
                ):
                    event_type = candidate_type

                candidate_polarity = str(chosen.get("polarity", "")).strip().upper()
                if candidate_polarity in {"POS", "NEG"}:
                    polarity = candidate_polarity

                candidate_modality = str(chosen.get("modality", "")).strip().upper()
                if candidate_modality in {"ASSERTED", "POSSIBLE", "OBLIGATED"}:
                    modality = candidate_modality

                candidate_tense = str(chosen.get("tense", "")).strip().upper()
                if candidate_tense in {"PAST", "PRESENT"}:
                    tense = candidate_tense

                raw_roles = chosen.get("roles", {})
                if isinstance(raw_roles, dict):
                    for label in ["AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT", "LOCATION", "TIME"]:
                        filler = raw_roles.get(label)
                        if filler is None:
                            continue
                        entity_id = _resolve_role_filler_to_entity_id(
                            filler=str(filler),
                            entity_id_by_canonical=entity_id_by_canonical,
                        )
                        if entity_id is None and ask_llm_fn is not None:
                            try:
                                entity_id = _resolve_role_filler_to_entity_id_with_llm(
                                    filler=str(filler),
                                    entity_id_by_canonical=entity_id_by_canonical,
                                    ask_llm_fn=ask_llm_fn,
                                    model=model,
                                    api_key=api_key,
                                )
                            except Exception:
                                entity_id = None
                        if entity_id is None and _is_valid_new_entity_filler(str(filler)):
                            canonical_base = _canonicalize_role_filler_entity_name(str(filler))
                            if not canonical_base:
                                continue
                            canonical_name = _abstract_entity_name(canonical_base)
                            canonical_key = canonical_name.lower()
                            if canonical_key not in entity_id_by_canonical:
                                ent_counter += 1
                                new_entity_id = f"ENT_{ent_counter:04d}"
                                f_start, f_end = _span_for_filler_in_sentence(sentence, str(filler))
                                if canonical_key in entity_type_cache:
                                    entity_type = entity_type_cache[canonical_key]
                                else:
                                    entity_type = _infer_entity_type_with_llm(
                                        canonical_name,
                                        allowed_entity_types=allowed_entity_types,
                                        ask_llm_fn=ask_llm_fn,
                                        model=model,
                                        api_key=api_key,
                                    )
                                    entity_type_cache[canonical_key] = entity_type
                                label_key = (canonical_key, entity_type)
                                if label_key in entity_label_cache:
                                    dictionary_name = entity_label_cache[label_key]
                                else:
                                    dictionary_name = _entity_dictionary_label_with_llm(
                                        canonical_name,
                                        entity_type,
                                        induced_entity_terms,
                                        ask_llm_fn=ask_llm_fn,
                                        model=model,
                                        api_key=api_key,
                                    )
                                    entity_label_cache[label_key] = dictionary_name
                                graph["nodes"].append(
                                    {
                                        "id": new_entity_id,
                                        "type": "ENT",
                                        "entity_type": entity_type,
                                        "surface_name": str(filler),
                                        "name": canonical_name,
                                        "canonical_name": canonical_name,
                                        "label": dictionary_name,
                                        "dictionary_name": dictionary_name,
                                        "polarity": "POS",
                                        "modality": "ASSERTED",
                                        "tense": "PAST",
                                        "provenance": {
                                            "doc_id": doc_id,
                                            "char_start": f_start,
                                            "char_end": f_end,
                                            "sentence_id": sentence.sentence_id,
                                        },
                                    }
                                )
                                entity_id_by_canonical[canonical_key] = new_entity_id
                            entity_id = entity_id_by_canonical.get(canonical_key)
                        if entity_id is None:
                            continue
                        llm_role_pairs.append((label, entity_id))

            modality = _infer_modality_from_sentence(sentence.text, modality)
            if (
                sentence.sentence_id in premise_sentence_ids
                and event_type in {"OTHER_EVENT", "PREMISE"}
                and "PREMISE" in allowed_event_types
            ):
                event_type = "PREMISE"
            if enforce_known_vocab and event_type not in allowed_event_types:
                raise ValueError(f"Unknown event type '{event_type}' for sentence '{sentence.sentence_id}'")

            if llm_role_pairs:
                seen_roles = set()
                for role_label, role_ent_id in llm_role_pairs:
                    key = (role_label, role_ent_id)
                    if key in seen_roles:
                        continue
                    seen_roles.add(key)
                    roles.append((role_label, role_ent_id))
            else:
                if not sentence_entity_ids:
                    continue
                roles.append(("AGENT", sentence_entity_ids[0]))
                if len(sentence_entity_ids) > 1:
                    roles.append(("PATIENT", sentence_entity_ids[1]))

            if not any(label == "AGENT" for label, _ in roles):
                if sentence_entity_ids:
                    roles.insert(0, ("AGENT", sentence_entity_ids[0]))
                else:
                    continue

            agent_ids = [entity_id for label, entity_id in roles if label == "AGENT"]
            if agent_ids:
                agent_id = agent_ids[0]
                reflexive_markers = {"myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves"}
                is_reflexive = any(marker in sentence.text.lower() for marker in reflexive_markers)
                if not is_reflexive:
                    roles = [
                        (label, entity_id)
                        for label, entity_id in roles
                        if not (label == "PATIENT" and entity_id == agent_id)
                    ]

            evt_counter += 1
            event_id = f"EVT_{evt_counter:04d}"
            event_ids_in_order.append(event_id)
            graph["nodes"].append(
                {
                    "id": event_id,
                    "type": "EVT",
                    "event_type": event_type,
                    "label": event_type,
                    "name": (summary_text or evidence_text or _label_to_name(event_type)),
                    "summary_text": summary_text,
                    "evidence_text": evidence_text,
                    "polarity": polarity,
                    "modality": modality,
                    "tense": tense,
                    "provenance": {
                        "doc_id": doc_id,
                        "char_start": span_start,
                        "char_end": span_end,
                        "sentence_id": sentence.sentence_id,
                    },
                }
            )

            for label, target_id in roles:
                graph["edges"].append(
                    {
                        "source": event_id,
                        "target": target_id,
                        "label": label,
                        "provenance": {
                            "doc_id": doc_id,
                            "char_start": span_start,
                            "char_end": span_end,
                            "sentence_id": sentence.sentence_id,
                        },
                    }
                )

            if previous_event_id is not None:
                graph["edges"].append(
                    {
                        "source": previous_event_id,
                        "target": event_id,
                        "label": "BEFORE",
                        "provenance": {
                            "doc_id": doc_id,
                            "char_start": span_start,
                            "char_end": span_end,
                            "sentence_id": sentence.sentence_id,
                        },
                    }
                )
            previous_event_id = event_id

    if ask_llm_fn is not None and entity_id_by_canonical and event_ids_in_order:
        try:
            intention_prompt = _intention_extraction_prompt(
                story_text=normalized,
                entity_names=sorted(name.title() for name in entity_id_by_canonical.keys()),
                event_ids=event_ids_in_order,
                intention_types=sorted(allowed_intention_types),
            )
            raw_int = ask_llm_fn(
                question=intention_prompt,
                model=model,
                temperature=0.0,
                max_output_tokens=700,
                api_key=api_key,
                system_prompt="Return strict JSON only.",
            )
            payload = _parse_llm_json_object(raw_int)
        except Exception:
            payload = {}
    else:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("intentions", [])
    payload.setdefault("effects", [])

    existing_int_keys = {
        (
            str(item.get("entity", "")).strip().lower(),
            str(item.get("intention_type", "")).strip().upper(),
        )
        for item in payload.get("intentions", [])
        if isinstance(item, dict) and str(item.get("intention_type", "")).strip().upper() != "OTHER_INTENTION"
    }

    existing_eff_keys = {
        (
            str(item.get("event_id", "")).strip(),
            str(item.get("entity", "")).strip().lower(),
            str(item.get("intention_type", "")).strip().upper(),
            str(item.get("effect", "")).strip().upper(),
        )
        for item in payload.get("effects", [])
        if isinstance(item, dict)
    }
    intention_nodes_by_key: dict[tuple[str, str], str] = {}
    int_counter = 0
    effects_by_intention: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    effects_payload = payload.get("effects", []) if isinstance(payload, dict) else []

    for item in payload.get("intentions", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        entity_name = str(item.get("entity", "")).strip()
        intention_type = str(item.get("intention_type", "")).strip().upper()
        if intention_type == "OTHER_INTENTION":
            continue
        if intention_type not in allowed_intention_types:
            continue
        entity_id = _resolve_role_filler_to_entity_id(entity_name, entity_id_by_canonical)
        if entity_id is None and ask_llm_fn is not None:
            try:
                entity_id = _resolve_role_filler_to_entity_id_with_llm(
                    filler=entity_name,
                    entity_id_by_canonical=entity_id_by_canonical,
                    ask_llm_fn=ask_llm_fn,
                    model=model,
                    api_key=api_key,
                )
            except Exception:
                entity_id = None
        if entity_id is None:
            continue

        key = (entity_id, intention_type)
        if key in intention_nodes_by_key:
            continue

        int_counter += 1
        int_id = f"INT_{int_counter:04d}"
        confidence = float(item.get("confidence", 0.6))
        confidence = max(0.0, min(1.0, confidence))
        evidence = str(item.get("evidence", "")).strip()
        ev_start, ev_end = _span_for_filler_in_sentence(
            sentences[0] if sentences else SentenceSpan("S0001", normalized, 0, len(normalized)),
            evidence or entity_name,
        )
        graph["nodes"].append(
            {
                "id": int_id,
                "type": "INT",
                "intention_type": intention_type,
                "label": intention_type,
                "name": (evidence or f"{entity_name} intends {intention_type.lower().replace('_', ' ')}"),
                "status": "PENDING",
                "outcome_type": "BLOCKED",
                "description": evidence or f"{entity_name} intends {intention_type.lower().replace('_', ' ')}",
                "polarity": "POS",
                "modality": "ASSERTED",
                "tense": "PAST",
                "provenance": {
                    "doc_id": doc_id,
                    "char_start": ev_start,
                    "char_end": ev_end,
                    "sentence_id": "S0001",
                },
            }
        )
        graph["edges"].append(
            {
                "source": entity_id,
                "target": int_id,
                "label": "INTENDS",
                "confidence": confidence,
                "provenance": {
                    "doc_id": doc_id,
                    "char_start": ev_start,
                    "char_end": ev_end,
                    "sentence_id": "S0001",
                },
            }
        )
        intention_nodes_by_key[key] = int_id

    for item in effects_payload if isinstance(effects_payload, list) else []:
        if not isinstance(item, dict):
            continue
        event_id = str(item.get("event_id", "")).strip()
        entity_name = str(item.get("entity", "")).strip()
        intention_type = str(item.get("intention_type", "")).strip().upper()
        effect = str(item.get("effect", "")).strip().upper()
        if event_id not in event_ids_in_order:
            continue
        if intention_type not in allowed_intention_types:
            continue
        if effect not in {"ADVANCES", "THWARTS", "FULFILLS", "FAILS"}:
            continue
        entity_id = _resolve_role_filler_to_entity_id(entity_name, entity_id_by_canonical)
        if entity_id is None and ask_llm_fn is not None:
            try:
                entity_id = _resolve_role_filler_to_entity_id_with_llm(
                    filler=entity_name,
                    entity_id_by_canonical=entity_id_by_canonical,
                    ask_llm_fn=ask_llm_fn,
                    model=model,
                    api_key=api_key,
                )
            except Exception:
                entity_id = None
        if entity_id is None:
            continue
        int_id = intention_nodes_by_key.get((entity_id, intention_type))
        if int_id is None:
            continue

        confidence = float(item.get("confidence", 0.6))
        confidence = max(0.0, min(1.0, confidence))
        # Event provenance is reused for intention-effect edge traceability.
        evt_node = next((n for n in graph["nodes"] if n.get("id") == event_id), None)
        if not evt_node:
            continue
        prov = dict(evt_node["provenance"])
        graph["edges"].append(
            {
                "source": event_id,
                "target": int_id,
                "label": effect,
                "confidence": confidence,
                "provenance": prov,
            }
        )
        effects_by_intention[int_id][effect] += 1

    for node in graph["nodes"]:
        if node.get("type") != "INT":
            continue
        int_id = node["id"]
        counts = effects_by_intention.get(int_id, {})
        status, outcome_type = _compute_intention_status(
            fulfills=counts.get("FULFILLS", 0),
            fails=counts.get("FAILS", 0),
            advances=counts.get("ADVANCES", 0),
            thwarts=counts.get("THWARTS", 0),
        )
        node["status"] = status
        node["outcome_type"] = outcome_type
    graph = _attach_intention_of_edges(graph)
    graph = _ensure_entity_goals(graph, allowed_goal_types=allowed_goal_types)

    graph = _prune_orphan_entities(graph)
    if add_node_summaries and ask_llm_fn is not None:
        try:
            graph = _annotate_node_summaries_with_llm(
                graph=graph,
                story_text=normalized,
                ask_llm_fn=ask_llm_fn,
                model=model,
                api_key=api_key,
            )
        except Exception:
            pass
    graph = _apply_node_summary_fallbacks(graph)
    graph = _align_event_summaries_with_modality(graph)
    graph = _retag_event_types_from_summary(graph, allowed_event_types=allowed_event_types)
    graph = _semantic_remap_other_events_with_llm(
        graph,
        allowed_event_types=allowed_event_types,
        ask_llm_fn=ask_llm_fn,
        model=model,
        api_key=api_key,
    )
    graph = _associate_premise_events_with_llm(
        graph,
        premise_sentence_ids=premise_sentence_ids,
        ask_llm_fn=ask_llm_fn,
        model=model,
        api_key=api_key,
    )
    graph = _retag_event_types_from_summary(graph, allowed_event_types=allowed_event_types)
    graph = _canonicalize_forbidden_event_types(graph, allowed_event_types=allowed_event_types)
    graph = _finalize_payment_event_polarity(graph, allowed_event_types=allowed_event_types)
    graph = _drop_generic_other_intentions(graph)
    graph = _deduplicate_redundant_events(graph)
    graph = _rebuild_before_edges_from_provenance(graph)
    graph = _extract_relationship_nodes(graph, allowed_relation_types=allowed_relation_types)
    graph = _renumber_event_ids(graph)
    return validate_story_graph(
        graph,
        vocab_path=vocab_path,
        dictionary_dir=dictionary_dir,
        require_before_dag=True,
        require_event_agent=True,
    )
