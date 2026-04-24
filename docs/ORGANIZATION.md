# abstractgraph-generative Organization

This document covers code organization, local setup, validation, and supporting
documentation for `abstractgraph-generative`.

For the semantic role of this repository, see [../README.md](../README.md).

## Package Layout

- `src/abstractgraph_generative/rewrite.py`
- `src/abstractgraph_generative/autoregressive.py`
- `src/abstractgraph_generative/conditional.py`
- `src/abstractgraph_generative/conditional_batch.py`
- `src/abstractgraph_generative/conditional_attributed.py`
- `src/abstractgraph_generative/edge_generator.py`
- `src/abstractgraph_generative/interpolate.py`
- `src/abstractgraph_generative/interpolation.py`
- `src/abstractgraph_generative/optimize.py`
- `src/abstractgraph_generative/repair.py`
- `src/abstractgraph_generative/legacy/conditional_v0_1/`

## Documentation

- [README.md](README.md)
- [Autoregressive Generator and Rewrite](guides/autoregressive-generator-and-rewrite.md)
- [Conditional Autoregressive Generation](guides/conditional-autoregressive-generation.md)
- [Edge Generator](guides/edge-generator.md)

## Notebooks

- `notebooks/examples/` contains the remaining core generative workflows.
- Text-oriented notebooks now live in the sibling `abstractgraph-text` repository.
- Backend-generator notebooks now live in the sibling
  `abstractgraph-generative-backends` repository.
- `notebooks/research/` contains exploratory generation notebooks.
- Example and research notebooks bootstrap imports and normalize the working
  directory automatically for the standard ecosystem layout.

## Dependencies

- `abstractgraph`
- `abstractgraph-ml`

## Local Validation

```bash
python -m pip install -e ../abstractgraph --no-deps
python -m pip install -e ../abstractgraph-ml --no-deps
python -m pip install -e . --no-deps
python scripts/smoke_test.py
```
