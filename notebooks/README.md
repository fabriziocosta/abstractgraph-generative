# Generative notebooks

This folder contains notebooks centered on autoregressive, interpolation, and
optimization workflows built on top of `abstractgraph` and
`abstractgraph-ml`.

Layout:
- `examples/` for supported generator workflows
- `research/` for exploratory generation notebooks

Highlighted examples:
- `examples/example_edge_generator_from_pair.ipynb`
  Pair-conditioned edge generation from a stored retrieval corpus.
- `examples/example_edge_generator_repair.ipynb`
  Retrieval-based graph repair by perturbing one stored graph and regrowing
  from surgically repaired infeasible starts.

Text-oriented notebooks were extracted to `/home/fabrizio/code/abstractgraph-text`.
Backend-generator notebooks were extracted to
`/home/fabrizio/code/abstractgraph-generative-backends`.

Bootstrap behavior:
- Notebooks use `notebooks/_bootstrap.py` to locate the repo root.
- They prepend available sibling `src/` directories to `sys.path`.
- They normalize the working directory to the repo root so relative paths are
  consistent across Jupyter launch locations.
