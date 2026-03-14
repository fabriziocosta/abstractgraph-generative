"""Hierarchy assembly pipeline for paragraph/sentence/event decomposition."""

from __future__ import annotations

from typing import Callable

from abstractgraph_generative.story.text_hierarchy_abstraction import simplify_events_with_wordnet
from abstractgraph_generative.story.text_hierarchy_llm import enforce_explicit_subjects, llm_list_of_n


def hierarchical_story_events(
    story_text: str,
    ask_llm_fn: Callable[..., str],
    model: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 700,
    api_key: str | None = None,
    n_paragraphs: int = 3,
    n_sentences_per_paragraph: int = 3,
    n_events_per_sentence: int = 3,
    enforce_explicit_subjects_flag: bool = True,
    simplify_with_wordnet: bool = False,
    n_abstraction_levels: int = 2,
) -> dict:
    """Decompose story into paragraphs, then sentences, then events.

    Args:
        story_text: Original story text.
        ask_llm_fn: Callable that sends prompts to the LLM.
        model: Optional OpenAI model name.
        temperature: Sampling temperature.
        max_output_tokens: Max output tokens per LLM call.
        api_key: Optional OpenAI API key override.
        n_paragraphs: Number of summary paragraphs.
        n_sentences_per_paragraph: Number of summary sentences per paragraph.
        n_events_per_sentence: Number of atomic events per sentence.
        enforce_explicit_subjects_flag: If True, rewrite outputs to remove pronouns.
        simplify_with_wordnet: If True, simplify event words by replacing content
            words with WordNet hypernyms.
        n_abstraction_levels: Maximum abstraction levels to generate.
            Level k applies k hypernym hops to each event token.

    Returns:
        Nested dictionary with paragraphs, sentences, and events.
    """

    if n_paragraphs < 1 or n_sentences_per_paragraph < 1 or n_events_per_sentence < 1:
        raise ValueError("All hierarchy sizes must be >= 1")
    if n_abstraction_levels < 1:
        raise ValueError("n_abstraction_levels must be >= 1")

    paragraphs = llm_list_of_n(
        input_text=story_text,
        instruction=(
            f"Summarize the story into exactly {n_paragraphs} short paragraphs. "
            "Use explicit subject nouns; avoid pronouns."
        ),
        n_items=n_paragraphs,
        ask_llm_fn=ask_llm_fn,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        api_key=api_key,
    )

    if enforce_explicit_subjects_flag:
        paragraphs = enforce_explicit_subjects(
            items=paragraphs,
            context_text=story_text,
            ask_llm_fn=ask_llm_fn,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
        )

    structured = {
        "config": {
            "n_paragraphs": n_paragraphs,
            "n_sentences_per_paragraph": n_sentences_per_paragraph,
            "n_events_per_sentence": n_events_per_sentence,
            "simplify_with_wordnet": simplify_with_wordnet,
            "n_abstraction_levels": n_abstraction_levels,
        },
        "paragraphs": [],
        "flat_events": [],
    }

    for p_idx, paragraph in enumerate(paragraphs, start=1):
        sentences = llm_list_of_n(
            input_text=paragraph,
            instruction=(
                f"Summarize this paragraph into exactly {n_sentences_per_paragraph} sentences. "
                "Each sentence should capture one key sub-point. Use explicit subjects; avoid pronouns."
            ),
            n_items=n_sentences_per_paragraph,
            ask_llm_fn=ask_llm_fn,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
        )

        if enforce_explicit_subjects_flag:
            sentences = enforce_explicit_subjects(
                items=sentences,
                context_text=paragraph,
                ask_llm_fn=ask_llm_fn,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                api_key=api_key,
            )

        paragraph_block = {
            "paragraph_index": p_idx,
            "text": paragraph,
            "sentences": [],
        }

        for s_idx, sentence in enumerate(sentences, start=1):
            events = llm_list_of_n(
                input_text=sentence,
                instruction=(
                    f"Decompose this sentence into exactly {n_events_per_sentence} atomic events. "
                    "Each event should be short, concrete, action-oriented, and use explicit subject nouns (no pronouns)."
                ),
                n_items=n_events_per_sentence,
                ask_llm_fn=ask_llm_fn,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                api_key=api_key,
            )

            if enforce_explicit_subjects_flag:
                events = enforce_explicit_subjects(
                    items=events,
                    context_text=sentence,
                    ask_llm_fn=ask_llm_fn,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    api_key=api_key,
                )

            original_events = list(events)
            abstraction_levels: dict[int, list[str]] = {}
            if simplify_with_wordnet:
                for abstraction_level in range(1, n_abstraction_levels + 1):
                    abstraction_levels[abstraction_level] = simplify_events_with_wordnet(
                        items=original_events,
                        n_up_levels=abstraction_level,
                    )

            simplified_events = (
                abstraction_levels[n_abstraction_levels] if abstraction_levels else list(original_events)
            )

            sentence_block = {
                "sentence_index": s_idx,
                "text": sentence,
                "events": simplified_events,
                "original_events": original_events,
                "simplified_events": simplified_events,
                "abstraction_levels": {str(k): v for k, v in abstraction_levels.items()},
            }
            paragraph_block["sentences"].append(sentence_block)

            for e_idx, original_event in enumerate(original_events, start=1):
                abstraction_by_level = {
                    str(level_idx): level_events[e_idx - 1]
                    for level_idx, level_events in abstraction_levels.items()
                    if e_idx - 1 < len(level_events)
                }
                simplified_event = abstraction_by_level.get(str(n_abstraction_levels), original_event)
                structured["flat_events"].append(
                    {
                        "paragraph_index": p_idx,
                        "sentence_index": s_idx,
                        "event_index": e_idx,
                        "event": simplified_event,
                        "original_event": original_event,
                        "simplified_event": simplified_event,
                        "abstraction_levels": abstraction_by_level,
                    }
                )

        structured["paragraphs"].append(paragraph_block)

    return structured
