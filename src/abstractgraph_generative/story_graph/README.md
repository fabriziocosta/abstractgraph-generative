# Story Graph (Text <-> Graph Roundtrip)

This module implements a compact, auditable, bidirectional representation for narrative stories.

## Goals

- Convert story text into a constrained story graph (`text -> graph`)
- Convert graph back to coherent text while preserving encoded dynamics (`graph -> text`)
- Keep structure discrete and inspectable:
  - bounded vocabularies
  - explicit node/edge types
  - provenance on semantic objects
- Model not just events, but also **intentions** and whether plans succeed/fail

---

## Core idea

A story graph is a causal/motivational structure, not a syntactic parse.

- `ENT` nodes: persistent entities (characters/objects)
- `EVT` nodes: actions/state changes with semantic roles
- `INT` nodes: intentions and their lifecycle (`PENDING`, `IN_PROGRESS`, `ACHIEVED`, `FAILED`, ...)
- edges encode roles, chronology, causal/motivational links, and plan outcomes

Every semantic node/edge includes provenance fields:
`doc_id`, `char_start`, `char_end`, `sentence_id`.

---

## Module map

- `schema.py`
  - canonical node/edge labels
  - constrained attribute enums (polarity/modality/tense/intention status/outcome)
- `resources/vocab_v0.json`
  - bounded vocabularies for entity/event/goal/intention/outcome types
- `vocab.py`
  - vocabulary loading and set conversion
- `validation.py`
  - strict graph validation and canonical ordering
- `text_to_graph.py`
  - extraction pipeline
  - LLM-assisted event/role extraction
  - entity grounding + fallback disambiguation
  - intention extraction + intention effect edges
- `graph_to_text.py`
  - ordered planner
  - deterministic realization
  - sequence-repair pass (LLM-assisted)
  - style-conditioned rendering helper
- `visualization.py`
  - matplotlib and Graphviz/DOT renderers
  - configurable node summary excerpts in DOT labels
- `utils.py`
  - OpenAI helpers (`ask_llm`, `call_openai_llm`)
  - Aesop corpus loader
  - text pretty-print helpers
- `notebook_utils.py`
  - high-level wrappers to keep notebooks concise
- `dataset.py`, `audit.py`
  - packaging and quality-report utilities

---

## Extraction pipeline (`text -> graph`)

High-level flow in `build_story_graph_from_text(...)`:

1. Normalize and sentence-split text
2. Extract candidate entities (plus optional LLM main-character support)
3. Extract events per sentence (LLM structured JSON, 1..N events in order)
4. Ground role fillers to entity IDs:
   - deterministic string/token matching first
   - LLM disambiguation fallback if needed
5. Build `BEFORE` event chain
6. Extract intentions (`INT`) and event-level intent effects (`ADVANCES/THWARTS/FULFILLS/FAILS`)
7. Derive intention status/outcome
8. Add `INTENTION_OF` links (intent -> relevant/planned event)
9. Add node summaries/evidence (LLM + deterministic fallback)
10. Prune orphan entities and validate

---

## Realization pipeline (`graph -> text`)

High-level flow:

1. Validate graph and build ordered plan from `BEFORE`
2. Render each event in sequence with role-aware templates
3. Prefer grounded event summary/evidence text for vague events
4. Verbalize intention effects when present
5. Run optional repair loop (`realize_story_from_graph_with_repair`) that:
   - re-extracts graph from generated text
   - compares sequence compatibility
   - does up to two strict repair passes if needed

---

## Quickstart

```python
from abstractgraph_generative.story import (
    ask_llm,
    build_story_graph_from_text,
    realize_story_from_graph_with_repair,
    render_story_graph_dot,
)

graph = build_story_graph_from_text(
    text=story_text,
    doc_id="aesop_0003",
    ask_llm_fn=ask_llm,
    model="gpt-4o-mini",
    add_node_summaries=True,
)

generated_story, plan, repair_info = realize_story_from_graph_with_repair(
    graph,
    ask_llm_fn=ask_llm,
    model="gpt-4o-mini",
)

render_story_graph_dot(
    graph,
    output_path="notebooks/examples/data/story_graph.dot.svg",
    format="svg",
    include_summary_excerpt=True,
    summary_excerpt_sentences=2,
    summary_excerpt_chars=180,
)
```

Notebook entry point:
`notebooks/examples/example_story_graph_roundtrip_aesop.ipynb`

---

## Why outputs can still be imperfect

Current limitations are mostly extraction fidelity issues:

- lexical event mapping can still miss rare paraphrases
- role assignment can drift when text is highly implicit
- LLM abstractions may still over/under-generalize in edge cases

Mitigations already implemented:

- bounded vocab + strict validation
- modality handling (`POSSIBLE`/`OBLIGATED` vs asserted facts)
- grounded summary/evidence fallback on nodes
- sequence repair with compatibility check
- entity abstraction rules (e.g., monetary mentions -> `Money`)

---

## Extension ideas

- event-type-specific role constraints (stronger than generic role set)
- richer temporal/causal extraction (`CAUSES/ENABLES/PREVENTS` with confidence)
- stronger coreference for pronouns/aliases
- learned calibration for intention-effect confidence
- train/evaluate extraction quality with annotated subsets

---

## Practical notes

- Requires `OPENAI_API_KEY` for LLM-backed paths.
- DOT rendering requires Graphviz (`dot`) and `pygraphviz`.
- If notebook imports fail after edits, restart kernel (module cache).
