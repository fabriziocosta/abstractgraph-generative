# Conditional Generative Submodule

This folder contains the modular implementation of image-conditioned autoregressive generation for `abstractgraph`.

## Purpose

The conditional generator grows preimage graphs while keeping an image graph as fixed context.

Core mechanics:
- Image nodes hold association subgraphs in preimage space.
- Generation materializes one image-node association at a time.
- Anchor nodes shared with current preimage graph are merged.
- Cut-index entries built from pruning sequences drive insertion choices.

## Package Layout

- `types.py`
  - Shared datatypes.
  - `CutArchiveEntry`: archive row used by cut-index and insertion logic.

- `utils.py`
  - Generic helpers reused across modules.
  - `_available_cpu_count`, `_fmt_elapsed`.

- `index_building.py`
  - Functions that build candidate archives/cut index from data.
  - Includes pruning-sequence generation and deterministic index extraction.
  - Key public helpers:
    - `build_assoc_map_from_image_graph`
    - `generate_image_conditioned_pruning_sequences`
    - `build_image_conditioned_cut_index_from_pruning`

- `materialization.py`
  - Low-level graph merge primitives.
  - Key helper:
    - `_merge_with_anchors`

- `generator_core.py`
  - Main model class and runtime generation loop.
  - Key public class:
    - `ConditionalAutoregressiveGenerator`
  - Visualization helper:
    - `display_conditioned_graphs`

- `dataset_generator.py`
  - Dataset-level orchestration around `ConditionalAutoregressiveGenerator`.
  - Key public class:
    - `ConditionalAutoregressiveGraphsGenerator`

- `__init__.py`
  - Public exports for the subpackage.

## Public Import Paths

Preferred:

```python
from abstractgraph_generative.conditional import (
    ConditionalAutoregressiveGenerator,
)
from abstractgraph_generative.conditional_batch import (
    ConditionalAutoregressiveGraphsGenerator,
)
```

Also supported:

```python
from abstractgraph_generative.conditional import (
    ConditionalAutoregressiveGenerator,
    ConditionalAutoregressiveGraphsGenerator,
)
```

## Execution Flow

1. Build cut index from generator graphs:
   - `build_image_conditioned_cut_index_from_pruning(...)`
2. Fit generator:
   - `ConditionalAutoregressiveGenerator.fit(...)`
3. Generate samples:
   - `ConditionalAutoregressiveGenerator.generate(...)`
   - or `generate_from_graphs(...)` for prepare+generate in one call.
4. Optional dataset workflow:
   - `ConditionalAutoregressiveGraphsGenerator.fit(graphs, targets)`
   - `ConditionalAutoregressiveGraphsGenerator.generate(n_samples=None)`

## Feasibility During Iteration

Construction-time feasibility is controlled by the
`apply_feasibility_during_construction` flag on
`ConditionalAutoregressiveGenerator`.

- Default: `False` (final feasibility filtering only).
- Set to `True` to enforce feasibility at each iterative construction step.

Notebook usage:

```python
from abstractgraph_generative.conditional import (
    ConditionalAutoregressiveGenerator,
)

generator = ConditionalAutoregressiveGenerator(
    decomposition_function=df,
    nbits=19,
    random_seed=42,  # deterministic fit/prepare/generate defaults
    feasibility_estimator=feasibility_estimator,
    apply_feasibility_during_construction=True,
)

# Or toggle later
generator.apply_feasibility_during_construction = True
```

## Design Boundaries

- `index_building.py` should remain pure data/index construction logic.
- `materialization.py` should only contain merge/materialization primitives.
- `generator_core.py` should orchestrate generation, not reimplement low-level helpers.
- `dataset_generator.py` should handle class partitioning, clustering, progress reporting, and estimator-based target assignment.

## Maintenance Notes

- Preserve backward-compatible exports in:
  - `abstractgraph_generative.conditional`
  - any explicit compatibility shim you choose to maintain outside this repo
- If adding new public API, export it from the concrete module that owns it.
- Keep notebook-facing APIs stable when possible.
- For notebook edits, always run:
  - `bash scripts/backup_ipynb.sh <notebook.ipynb>`

## Quick Smoke Check

```bash
python -m py_compile \
  src/abstractgraph_generative/conditional_v0_1/types.py \
  src/abstractgraph_generative/conditional_v0_1/utils.py \
  src/abstractgraph_generative/conditional_v0_1/materialization.py \
  src/abstractgraph_generative/conditional_v0_1/index_building.py \
  src/abstractgraph_generative/conditional_v0_1/generator_core.py \
  src/abstractgraph_generative/conditional_v0_1/dataset_generator.py \
  src/abstractgraph_generative/conditional_v0_1/__init__.py
```
