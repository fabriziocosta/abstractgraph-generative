# Notebooks

The example folder keeps one notebook for each supported generation mode:
edge-based generation, conditional autoregressive generation, and their
hierarchical combination. Older, specialized, and overlapping examples are
preserved in `notebooks/archive/`.

## Layout

- `notebooks/examples/`
  The three current generator examples.
- `notebooks/research/`
  Exploratory generation notebooks.
- `notebooks/archive/`
  Older or specialized notebooks retained for reference.

## Highlighted Examples

- `notebooks/examples/edge-generator-zinc-molecule-generation.ipynb`
  Edge-by-edge generation from a partial molecule.
- `notebooks/examples/conditional-autoregressive-local-zinc-generation.ipynb`
  Conditional generation from a local neighborhood of stored ZINC molecules.
- `notebooks/examples/hierarchical-graph-generator-zinc-generation.ipynb`
  Hierarchical ZINC generation with one edge stage and two conditional stages.

## Bootstrap Behavior

Notebooks use `notebooks/_bootstrap.py` to:

- locate the repository root,
- prepend available sibling `src/` directories to `sys.path`,
- normalize the working directory to the repository root so relative paths are
  consistent across Jupyter launch locations.

## Extracted Notebooks

Some text-oriented notebooks were extracted to the separate
`abstractgraph-text` project, which is not included as a submodule in this
ecosystem checkout.

Backend-generator notebooks were extracted to the separate
`abstractgraph-generative-backends` project, which is also outside this
checkout.
