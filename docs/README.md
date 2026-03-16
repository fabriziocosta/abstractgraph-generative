# abstractgraph-generative docs

`abstractgraph-generative` contains generation and rewrite workflows built on
top of `abstractgraph` and `abstractgraph-ml`.

## Scope

This repo owns:
- autoregressive generators
- conditional autoregressive generators
- interpolation and path-construction workflows
- graph rewrite and repair utilities
- graph optimization helpers
- backend-specific generators such as VGAE, GRAN, and DiGress wrappers
- story-graph generation and text/graph round-tripping

## Module map

- `abstractgraph_generative.rewrite`
  rewrite primitives and boundary-compatible replacements
- `abstractgraph_generative.autoregressive`
  autoregressive generator
- `abstractgraph_generative.conditional`
  conditional autoregressive generator
- `abstractgraph_generative.conditional_batch`
  dataset-level conditional generation wrapper
- `abstractgraph_generative.interpolate`
  interpolation estimator
- `abstractgraph_generative.interpolation`
  interpolation generator
- `abstractgraph_generative.optimize`
  optimization helpers
- `abstractgraph_generative.repair`
  repair-style generation
- `abstractgraph_generative.story`
  story-graph extraction, validation, rendering, and packaging
- `abstractgraph_generative.legacy.conditional_v0_1`
  preserved legacy conditional pipeline

## Related docs

- [AUTOREGRESSIVE_GENERATOR_AND_REWRITE.md](AUTOREGRESSIVE_GENERATOR_AND_REWRITE.md)

Compatibility note:
- active code is migrating to `base_graph`, `interpretation_graph`, and
  `mapped_subgraph`
- `abstractgraph_generative.legacy.*` intentionally preserves the older
  terminology for backward compatibility

## Dependencies

- `abstractgraph`
- `abstractgraph-ml`

## Ecosystem

Sibling repositories:

- `abstractgraph`
  Path: `/home/fabrizio/work/abstractgraph`
- `abstractgraph-ml`
  Path: `/home/fabrizio/work/abstractgraph-ml`
- `abstractgraph-generative`
  Path: `/home/fabrizio/work/abstractgraph-generative`

See [../ECOSYSTEM.md](../ECOSYSTEM.md) for install order and dependency
direction.
