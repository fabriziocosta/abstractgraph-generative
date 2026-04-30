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

## Install

Standalone editable install, after `abstractgraph` and `abstractgraph-ml` are
available from PyPI or local editable checkouts:

```bash
python -m pip install -e .
```

Inside the `abstractgraph-ecosystem` superproject, install sibling packages in
dependency order:

```bash
python -m pip install -e repos/abstractgraph --no-deps
python -m pip install -e repos/abstractgraph-ml --no-deps
python -m pip install -e repos/abstractgraph-generative --no-deps
```

## Dependencies

Sibling dependencies:

- `abstractgraph`
- `abstractgraph-ml`

Runtime dependencies declared in `pyproject.toml`:

- `networkx`
- `numpy`
- `matplotlib`

## Caveats

- Generative workflows build on the core graph representation and ML utilities;
  install and validate those sibling packages first when working locally.
- Text-oriented notebooks and backend-generator notebooks have moved to sibling
  repositories. This repository keeps the core generative workflows.
- Install with `--no-deps` only in a shared ecosystem environment where runtime
  dependencies are already managed.

## Local Validation

```bash
python -m pip install -e ../abstractgraph
python -m pip install -e ../abstractgraph-ml
python -m pip install -e .
python scripts/smoke_test.py
```
