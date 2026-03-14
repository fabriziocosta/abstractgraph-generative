"""Rendering and reconstruction utilities for hierarchical story outputs."""

from __future__ import annotations

import textwrap


def normalize_output_level(level: str) -> str:
    """Normalize user-facing level aliases into canonical level names.

    Args:
        level: Requested output level.

    Returns:
        Canonical level name.
    """

    normalized = (level or "paragraph").strip().lower()
    if normalized in {"sentencen", "sentences"}:
        normalized = "sentence"
    if normalized in {"events", "original_events"}:
        normalized = "event"
    if normalized in {"simplify", "simplified_events"}:
        normalized = "simplified"
    return normalized


def pretty_print_story_hierarchy(hierarchy: dict) -> None:
    """Pretty-print all hierarchy levels: paragraphs, sentences, and events.

    Args:
        hierarchy: Nested hierarchy dictionary.

    Returns:
        None.
    """

    cfg = hierarchy.get("config", {})
    if cfg:
        print("Hierarchy config:")
        print(f"- paragraphs: {cfg.get('n_paragraphs')}")
        print(f"- sentences/paragraph: {cfg.get('n_sentences_per_paragraph')}")
        print(f"- events/sentence: {cfg.get('n_events_per_sentence')}")
        print(f"- simplify with wordnet: {cfg.get('simplify_with_wordnet')}")
        print(f"- n_abstraction_levels: {cfg.get('n_abstraction_levels')}")
        print()

    paragraphs = hierarchy.get("paragraphs", [])
    for p in paragraphs:
        p_idx = p.get("paragraph_index", "?")
        print(f"Paragraph {p_idx}:")
        print(p.get("text", ""))
        print()

        for s in p.get("sentences", []):
            s_idx = s.get("sentence_index", "?")
            print(f"  Sentence {p_idx}.{s_idx}:")
            print(f"  {s.get('text', '')}")
            print("  Events:")
            original_events = s.get("original_events", s.get("events", []))
            abstraction_levels = s.get("abstraction_levels", {})
            for e_idx, orig in enumerate(original_events, start=1):
                print(f"  - {p_idx}.{s_idx}.{e_idx} original:   {orig}")
                for level_id in sorted(int(k) for k in abstraction_levels.keys()):
                    level_events = abstraction_levels.get(str(level_id), [])
                    if e_idx - 1 < len(level_events):
                        print(f"    {p_idx}.{s_idx}.{e_idx}.{level_id} simplified: {level_events[e_idx - 1]}")
            print()

    print(f"Total events: {len(hierarchy.get('flat_events', []))}")


def build_text_from_hierarchy(hierarchy: dict, level: str = "paragraph", abstraction_level: int | None = None) -> str:
    """Build reconstructed story text from a selected hierarchy level.

    Args:
        hierarchy: Nested hierarchy dictionary.
        level: Output level. One of paragraph, sentence, event, simplified.
        abstraction_level: Abstraction level used for simplified text.
            Defaults to the maximum available level in each sentence.

    Returns:
        Reconstructed story text.
    """

    normalized = normalize_output_level(level)
    allowed = {"paragraph", "sentence", "event", "simplified"}
    if normalized not in allowed:
        raise ValueError(f"level must be one of {sorted(allowed)}")

    paragraphs = hierarchy.get("paragraphs", [])
    lines: list[str] = []

    for p in paragraphs:
        if normalized == "paragraph":
            text = (p.get("text", "") or "").strip()
            if text:
                lines.append(text)
            continue

        sentences = p.get("sentences", [])
        if normalized == "sentence":
            sentence_texts = [(s.get("text", "") or "").strip() for s in sentences]
            sentence_texts = [x for x in sentence_texts if x]
            if sentence_texts:
                lines.append(" ".join(sentence_texts))
            continue

        event_texts: list[str] = []
        for s in sentences:
            if normalized == "event":
                items = s.get("original_events", s.get("events", []))
            else:
                level_map = s.get("abstraction_levels", {})
                if abstraction_level is None:
                    level_ids = sorted(int(k) for k in level_map.keys())
                    use_level = str(level_ids[-1]) if level_ids else None
                else:
                    use_level = str(abstraction_level)

                if use_level and use_level in level_map:
                    items = level_map[use_level]
                else:
                    items = s.get("simplified_events", s.get("events", []))
            event_texts.extend([str(x).strip() for x in items if str(x).strip()])

        if event_texts:
            lines.append(". ".join(event_texts) + ".")

    return "\n\n".join(lines).strip()


def format_text_max_columns(text: str, max_columns: int = 80) -> str:
    """Wrap a text to a maximum line width while preserving paragraphs.

    Args:
        text: Input text text.
        max_columns: Maximum number of columns per line.

    Returns:
        Formatted text with wrapped lines and preserved blank-line paragraph breaks.
    """

    paragraphs = [p.strip() for p in text.strip().split("\n\n")]
    wrapped = [
        textwrap.fill(
            paragraph,
            width=max_columns,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for paragraph in paragraphs
        if paragraph
    ]
    return "\n\n".join(wrapped)


def pretty_print_text(text: str, max_columns: int = 88) -> None:
    """Print a text wrapped to a maximum number of columns.

    Args:
        text: Text to print.
        max_columns: Maximum columns in each line.

    Returns:
        None.
    """

    print(format_text_max_columns(text=text, max_columns=max_columns))


def print_story_from_hierarchy(
    hierarchy: dict,
    level: str = "paragraph",
    max_columns: int = 88,
    abstraction_level: int | None = None,
) -> str:
    """Build story text from hierarchy and pretty print it.

    Args:
        hierarchy: Nested hierarchy dictionary.
        level: Output level. One of paragraph, sentence, event, simplified.
        max_columns: Maximum columns for wrapped output.
        abstraction_level: Abstraction level for simplified outputs.

    Returns:
        Reconstructed story text.
    """

    story = build_text_from_hierarchy(
        hierarchy=hierarchy,
        level=level,
        abstraction_level=abstraction_level,
    )

    normalized = normalize_output_level(level)
    cfg = hierarchy.get("config", {})
    resolved_abstraction = abstraction_level
    if resolved_abstraction is None and normalized == "simplified":
        resolved_abstraction = cfg.get("n_abstraction_levels")

    print(f"level: {normalized}")
    if normalized == "simplified":
        print(f"abstraction_level: {resolved_abstraction}")

    pretty_print_text(text=story, max_columns=max_columns)
    return story
