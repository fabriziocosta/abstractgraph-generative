"""LLM parsing and controlled list-generation helpers for text hierarchy tasks."""

from __future__ import annotations

import json
import re
from typing import Callable

PRONOUN_PATTERN = re.compile(
    r"\b(i|we|you|he|she|it|they|me|us|him|her|them|my|our|your|his|its|their|who|which|that)\b",
    flags=re.IGNORECASE,
)


def extract_items_fallback(raw_text: str) -> list[str]:
    """Best-effort extraction of list items from non-JSON LLM output.

    Args:
        raw_text: Raw model output text.

    Returns:
        Parsed candidate items.
    """

    text = (raw_text or "").strip()
    if not text:
        return []

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    m = re.search(r'"items"\s*:\s*\[(.*)\]', text, flags=re.S)
    if m:
        arr_body = m.group(1)
        quoted = re.findall(r'"([^"\n][^"\n]*)"', arr_body)
        cleaned = [q.strip() for q in quoted if q.strip()]
        if cleaned:
            return cleaned

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items: list[str] = []
    for ln in lines:
        ln = re.sub(r"^[-*]\s+", "", ln)
        ln = re.sub(r"^\d+[.)]\s+", "", ln)
        if ln in {"{", "}", "[", "]", '"items": ['}:
            continue
        if ln.endswith("]") and ln.startswith('"items"'):
            continue
        if ln:
            items.append(ln)

    if len(items) <= 1:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        parts = [p for p in parts if not re.match(r'^[\{\}\[\]"]+$', p)]
        if len(parts) > len(items):
            items = parts

    return items


def parse_items_from_raw(raw: str) -> list[str]:
    """Parse candidate list items from raw model output.

    Args:
        raw: Raw model output text.

    Returns:
        Parsed list items.
    """

    parsed: list[str] = []

    try:
        payload = json.loads(raw)
        cand = payload.get("items", []) if isinstance(payload, dict) else []
        if isinstance(cand, list):
            parsed = [str(x).strip() for x in cand if str(x).strip()]
    except Exception:
        parsed = []

    if not parsed:
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last != -1 and last > first:
            snippet = raw[first : last + 1]
            try:
                payload = json.loads(snippet)
                cand = payload.get("items", []) if isinstance(payload, dict) else []
                if isinstance(cand, list):
                    parsed = [str(x).strip() for x in cand if str(x).strip()]
            except Exception:
                parsed = []

    if not parsed:
        parsed = extract_items_fallback(raw)

    cleaned: list[str] = []
    for x in parsed:
        xx = x.strip()
        if not xx:
            continue
        if xx in {"{", "}", "[", "]", '"items": ['}:
            continue
        if xx.startswith('{"items"') or xx.startswith('"items"'):
            continue
        cleaned.append(xx)

    return cleaned


def llm_list_of_n(
    input_text: str,
    instruction: str,
    n_items: int,
    ask_llm_fn: Callable[..., str],
    model: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 700,
    api_key: str | None = None,
) -> list[str]:
    """Ask an LLM for exactly n strings and parse robustly.

    Args:
        input_text: Input context text for generation.
        instruction: Task instruction.
        n_items: Required number of output items.
        ask_llm_fn: Callable that sends a prompt to an LLM.
        model: Optional model name.
        temperature: Sampling temperature.
        max_output_tokens: Maximum generated tokens.
        api_key: Optional API key override.

    Returns:
        List with exactly n_items strings.
    """

    system_prompt = (
        "You are a precise text-structuring assistant. "
        "Return strict JSON only with no extra commentary."
    )

    prompt = f"""
{instruction}

Constraints:
1. Return JSON only, with schema: {{"items": ["...", "...", ...]}}.
2. `items` must contain exactly {n_items} non-empty strings.
3. Keep language concise and faithful to the input.

Input text:
{input_text}
"""

    raw = ask_llm_fn(
        question=prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        api_key=api_key,
        system_prompt=system_prompt,
    )
    items = parse_items_from_raw(raw)

    if len(items) != n_items:
        repair_prompt = f"""
Return STRICT JSON ONLY with exactly {n_items} non-empty strings.
Schema: {{"items": ["...", "...", ...]}}
Do not include markdown fences.

Previous invalid output:
{raw}
"""
        repaired = ask_llm_fn(
            question=repair_prompt,
            model=model,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            system_prompt=system_prompt,
        )
        items = parse_items_from_raw(repaired)

    if len(items) != n_items:
        base = [x for x in items if x] or ["(missing)"]
        while len(base) < n_items:
            base.append(base[-1])
        items = base[:n_items]

    if not items:
        items = ["(missing)"]

    if len(items) < n_items:
        last = items[-1]
        items.extend([last] * (n_items - len(items)))

    return [it if it.strip() else "(missing)" for it in items[:n_items]]


def contains_pronouns(items: list[str]) -> bool:
    """Return True if any item contains pronouns.

    Args:
        items: Candidate text items.

    Returns:
        True if at least one item includes a pronoun.
    """

    return any(PRONOUN_PATTERN.search(text or "") for text in items)


def enforce_explicit_subjects(
    items: list[str],
    context_text: str,
    ask_llm_fn: Callable[..., str],
    model: str | None,
    temperature: float,
    max_output_tokens: int,
    api_key: str | None,
) -> list[str]:
    """Rewrite list items to replace pronouns with explicit entities.

    Args:
        items: Candidate items.
        context_text: Context used for rewriting.
        ask_llm_fn: Callable that sends prompts to an LLM.
        model: Optional model name.
        temperature: Sampling temperature.
        max_output_tokens: Maximum generated tokens.
        api_key: Optional API key override.

    Returns:
        Rewritten items without pronouns when possible.
    """

    if not items or not contains_pronouns(items):
        return items

    instruction = (
        "Rewrite each item to eliminate pronouns by using explicit entity nouns from context. "
        "Do not use pronouns. Keep same meaning and ordering."
    )
    packed = "\n".join(f"- {it}" for it in items)

    rewritten = llm_list_of_n(
        input_text=f"Context:\n{context_text}\n\nItems to rewrite:\n{packed}",
        instruction=instruction,
        n_items=len(items),
        ask_llm_fn=ask_llm_fn,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        api_key=api_key,
    )

    out: list[str] = []
    for old, new in zip(items, rewritten):
        out.append(old if PRONOUN_PATTERN.search(new or "") else new)
    return out
