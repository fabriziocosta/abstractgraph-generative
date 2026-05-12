# Notebooks

The notebooks are examples and diagnostics for autoregressive, interpolation,
optimization, and edge-generation workflows built on top of `abstractgraph` and
`abstractgraph-ml`.

## Layout

- `notebooks/examples/`
  Supported generator workflows.
- `notebooks/research/`
  Exploratory generation notebooks.

## Highlighted Examples

- `notebooks/examples/example_edge_generator_from_pair.ipynb`
  Pair-conditioned edge generation from a stored retrieval corpus.
- `notebooks/examples/example_edge_generator_repair.ipynb`
  Retrieval-based graph repair by perturbing one stored graph and regrowing
  from surgically repaired infeasible starts.

## Bootstrap Behavior

Notebooks use `notebooks/_bootstrap.py` to:

- locate the repository root,
- prepend available sibling `src/` directories to `sys.path`,
- normalize the working directory to the repository root so relative paths are
  consistent across Jupyter launch locations.

## Extracted Notebooks

Text-oriented notebooks were extracted to the sibling `abstractgraph-text`
repository.

Backend-generator notebooks were extracted to the sibling
`abstractgraph-generative-backends` repository.
