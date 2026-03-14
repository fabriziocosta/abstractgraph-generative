"""Utility helpers for story-graph demos and datasets.

This module provides reusable text formatting and Aesop corpus loading with
local caching so notebooks stay focused on extraction/realization workflows.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path
from urllib.request import urlopen

from openai import NOT_GIVEN, OpenAI


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def format_text_max_columns(text: str, max_columns: int = 88) -> str:
    """Wrap text to a maximum line width while preserving paragraphs.

    Args:
        text: Input text.
        max_columns: Maximum number of columns per line.

    Returns:
        Wrapped text string.
    """

    paragraphs = [p.strip() for p in (text or "").strip().split("\n\n")]
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
    """Print wrapped text.

    Args:
        text: Input text.
        max_columns: Maximum number of columns per line.

    Returns:
        None.
    """

    print(format_text_max_columns(text=text, max_columns=max_columns))


def load_aesop_fables_from_gutenberg(
    url: str = "https://www.gutenberg.org/ebooks/21.txt.utf-8",
    timeout: int = 30,
    cache_path: str = "notebooks/examples/data/aesop_fables.json",
    force_refresh: bool = False,
) -> list[str]:
    """Load Aesop fables from local cache or Project Gutenberg.

    Args:
        url: Gutenberg plain-text URL for Aesop's fables.
        timeout: Timeout in seconds for HTTP requests.
        cache_path: JSON file path for parsed-fables cache.
        force_refresh: If True, ignore cache and re-download.

    Returns:
        List of strings where each item is one full fable (title + body).
    """

    cache_file = Path(cache_path)
    if cache_file.exists() and not force_refresh:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if isinstance(cached, list) and all(isinstance(item, str) for item in cached):
            return cached

    raw_text = urlopen(url, timeout=timeout).read().decode("utf-8")
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    if start_idx != -1:
        first_newline_after_start = text.find("\n", start_idx)
        text = text[first_newline_after_start + 1 :]
    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]

    heading_candidates = ["AESOP'S FABLES", "AESOP’S FABLES"]
    fables_start = -1
    for heading in heading_candidates:
        first = text.find(heading)
        if first == -1:
            continue
        second = text.find(heading, first + len(heading))
        if second != -1:
            fables_start = second + len(heading)
            break

    if fables_start == -1:
        raise ValueError("Could not locate the Aesop fables section in Gutenberg text.")

    body = text[fables_start:]

    end_markers = ["\n\nFOOTNOTES\n", "\n\nINDEX\n"]
    stop_positions = [pos for pos in (body.find(marker) for marker in end_markers) if pos != -1]
    if stop_positions:
        body = body[: min(stop_positions)]

    title_pattern = re.compile(r"\n\n([A-ZÆ][A-Za-zÆæ’'.,;:!?\- ]{2,90})\n\n")
    title_matches = list(title_pattern.finditer(body))

    def is_probable_fable_title(title: str) -> bool:
        if title.upper() == title:
            return False
        return len(title.split()) >= 2

    fables: list[str] = []
    for i, match in enumerate(title_matches):
        title = match.group(1).strip()
        if not is_probable_fable_title(title):
            continue

        chunk_start = match.end()
        chunk_end = title_matches[i + 1].start() if i + 1 < len(title_matches) else len(body)
        story = body[chunk_start:chunk_end].strip()
        if not story:
            continue

        story = re.sub(r"\n{3,}", "\n\n", story).strip()
        fables.append(f"{title}\n\n{story}")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(fables, ensure_ascii=False, indent=2), encoding="utf-8")
    return fables


def call_openai_llm(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 900,
    api_key: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Call the OpenAI Responses API and return plain text output.

    Args:
        prompt: User prompt text.
        model: OpenAI model name. Falls back to env OPENAI_MODEL or default.
        temperature: Sampling temperature for generation.
        max_output_tokens: Maximum number of output tokens.
        api_key: API key override. Falls back to env OPENAI_API_KEY.
        system_prompt: Optional system instruction.

    Returns:
        Model response text.
    """

    resolved_model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=resolved_api_key)

    if system_prompt:
        input_payload = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ]
    else:
        input_payload = prompt

    response = client.responses.create(
        model=resolved_model,
        input=input_payload,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        text=NOT_GIVEN,
    )
    return response.output_text.strip()


def ask_llm(
    question: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 900,
    api_key: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Ask a question to an OpenAI model and return answer text.

    Args:
        question: Question or instruction.
        model: OpenAI model name.
        temperature: Sampling temperature.
        max_output_tokens: Maximum generated tokens.
        api_key: API key override.
        system_prompt: Optional system instruction.

    Returns:
        Model response text.
    """

    return call_openai_llm(
        prompt=question,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        api_key=api_key,
        system_prompt=system_prompt,
    )
