# Documentation

This directory is the documentation root for `abstractgraph-generative`.

## Guides

- [Autoregressive Generator and Rewrite](guides/autoregressive-generator-and-rewrite.md)
- [Conditional Autoregressive Generation](guides/conditional-autoregressive-generation.md)
- [Edge Generator](guides/edge-generator.md)

## Scope

This repo owns the graph-only generation layer built on top of
`abstractgraph` and `abstractgraph-ml`, including:

- autoregressive generation
- conditional autoregressive generation
- interpolation and path-construction workflows
- graph rewrite and repair utilities
- graph optimization helpers
- edge-based generation workflows


## Module Map

- `abstractgraph_generative.autoregressive`
  Autoregressive generator over mapped-subgraph rewrites.
- `abstractgraph_generative.rewrite`
  Rewrite primitives and boundary-compatible replacements.
- `abstractgraph_generative.conditional`
  Conditional autoregressive generator.
- `abstractgraph_generative.conditional_batch`
  Dataset-level conditional generation wrapper.
- `abstractgraph_generative.conditional_attributed`
  Context-aware conditional variant.
- `abstractgraph_generative.edge_generator`
  Edge-by-edge generator with feasibility, ranking, pair retrieval, and
  nearest-neighbor repair workflows.
- `abstractgraph_generative.interpolate`
  Interpolation estimator helpers.
- `abstractgraph_generative.interpolation`
  Interpolation generator.
- `abstractgraph_generative.optimize`
  Optimization helpers.
- `abstractgraph_generative.repair`
  Repair-style generation utilities.

## Common Workflows

- `generate(...)`
  Grow a graph to a requested edge count under feasibility constraints.
- `generate_from_pair(...)`
  Build a local fitting subset from a stored retrieval corpus and generate from
  a mixed pair start graph.
- `repair(...)`
  Fit on the nearest stored neighbors of one query graph, then regrow from a
  surgically repaired infeasible start back to the original edge count.

## Conventions

- Active narrative documentation lives in `docs/`.
- Long-form guides live in `docs/guides/`.
- Guide filenames use kebab-case.
- `README.md` files act as landing pages for directories.

## Notes

- Active code is migrating to `base_graph`, `interpretation_graph`, and
  `mapped_subgraph`.
- For repo-level dependency context, see [../ECOSYSTEM.md](../ECOSYSTEM.md).
