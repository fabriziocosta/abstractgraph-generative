# Generative notebooks

This folder contains notebooks centered on autoregressive, interpolation,
optimization, and story-graph generation workflows built on top of
`abstractgraph` and `abstractgraph-ml`.

Layout:
- `examples/` for supported generator workflows
- `research/` for exploratory generation notebooks

Bootstrap behavior:
- Notebooks use `notebooks/_bootstrap.py` to locate the repo root.
- They prepend available sibling `src/` directories to `sys.path`.
- They normalize the working directory to the repo root so relative paths are
  consistent across Jupyter launch locations.
