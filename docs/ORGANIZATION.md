# abstractgraph-generative Organization

This document describes the maintained package layout, local setup, and
validation for `abstractgraph-generative`.

For the semantic role of this repository, see [overview.md](overview.md).

## Package Layout

- `conditional.py` implements conditional autoregressive graph generation.
- `conditional_batch.py` wraps conditional generation for datasets.
- `conditional_attributed.py` adds attributed/context-aware conditioning.
- `edge_generator.py` contains the `EdgeGenerator` workflow. Ranker fitting
  and persistence live in `_edge_ranker.py`; edge-neighbor, component-mixing,
  and regression-dataset helpers live in `_edge_utils.py` and are re-exported
  from the original module.
- `graph_generator.py` coordinates two-stage interpretation-graph and
  conditional base-graph generation.
- `interpolation_path.py` estimates paths through donor graph embeddings;
  `interpolation_generation.py` orchestrates interpolation-based generation.
- `rewrite.py` provides graph rewrite, cut-index, and virtual-rewrite
  operations. Shared anchor and component operations live in
  `_rewrite_components.py` and remain re-exported by `rewrite.py`.
- `repair.py` selects donor-based rewrites to repair a graph.
- `optimize.py` optimizes generator graph sets.
- `dataset_selection.py` selects graphs along edge-disjoint shortest paths in
  vector space.
- `generative_performance.py` contains sampling, scoring, and expected-gain
  performance evaluation utilities.
- `backends/` contains backend-specific support retained in this package.

## Documentation

- [README.md](README.md)
- [Overview](overview.md)
- [Notebook Guide](notebooks.md)
- [Graph Rewrite](graph-rewrite.md)
- [Conditional Autoregressive Generation](conditional-autoregressive-generation.md)
- [Edge Generator](edge-generator.md)
- [Graph Generator](graph-generator.md)

## Notebooks

- `notebooks/examples/` contains maintained generation workflows.
- `notebooks/archive/` and `notebooks/research/` retain historical and
  exploratory workflows, including interpolation, repair, optimization, and
  performance examples.
- Some text-oriented and backend-generator notebooks were extracted to the
  separate `abstractgraph-text` and `abstractgraph-generative-backends`
  projects. Those projects are not submodules of this ecosystem checkout.
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

Runtime dependencies declared in `pyproject.toml` include the sibling
packages `abstractgraph` and `abstractgraph-ml`, plus NetworkX, NumPy,
Matplotlib, SciPy, scikit-learn, joblib, toolz, and dill. These are required by
maintained runtime modules; there are no optional feature extras at present.

## Caveats

- Generative workflows build on the core graph representation and ML utilities;
  install and validate those sibling packages first when working locally.
- Text-oriented notebooks and backend-generator notebooks have moved to sibling
  repositories. This repository retains graph generation, rewrite, repair,
  interpolation, optimization, dataset-selection, and performance workflows.
- Install with `--no-deps` only in a shared ecosystem environment where runtime
  dependencies are already managed.

## Local Validation

```bash
python -m pip install -e ../abstractgraph
python -m pip install -e ../abstractgraph-ml
python -m pip install -e .
python scripts/smoke_test.py
```
