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

Text and story-graph workflows now live in the sibling repo
`/home/fabrizio/code/abstractgraph-text`.

Backend-specific generators such as DiGress, GRAN, and VGAE now live in the
sibling repo `/home/fabrizio/code/abstractgraph-generative-backends`.

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
  Edge-by-edge generator with feasibility and ranking.
- `abstractgraph_generative.interpolate`
  Interpolation estimator helpers.
- `abstractgraph_generative.interpolation`
  Interpolation generator.
- `abstractgraph_generative.optimize`
  Optimization helpers.
- `abstractgraph_generative.repair`
  Repair-style generation utilities.
- `abstractgraph_generative.legacy.conditional_v0_1`
  Preserved legacy conditional pipeline.

## Conventions

- Active narrative documentation lives in `docs/`.
- Long-form guides live in `docs/guides/`.
- Guide filenames use kebab-case.
- `README.md` files act as landing pages for directories.

## Notes

- Active code is migrating to `base_graph`, `interpretation_graph`, and
  `mapped_subgraph`.
- `abstractgraph_generative.legacy.*` intentionally preserves older
  terminology for backward compatibility.
- For repo-level dependency context, see [../ECOSYSTEM.md](../ECOSYSTEM.md).
