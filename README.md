<p align="center">
  <img src="docs/assets/AG_Logo.png" alt="AbstractGraph logo" width="220">
</p>

# abstractgraph-generative

`abstractgraph-generative` is the constructive layer of the AbstractGraph
ecosystem.

It uses graph semantics from `abstractgraph` and optional learned signals from
`abstractgraph-ml` to rewrite, generate, interpolate, optimize, and repair
graphs.

For package layout, local setup, validation commands, and documentation index,
see [docs/ORGANIZATION.md](docs/ORGANIZATION.md).

## Semantic Role

This repository closes the loop from analysis back to construction. Where
`abstractgraph` defines what graph transformations mean, and
`abstractgraph-ml` learns how to score or constrain graph structures,
`abstractgraph-generative` uses those signals to build new graph candidates or
repair existing ones.

It answers questions such as:

- How can a graph be rewritten while preserving meaningful structure?
- How can new graphs be generated conditionally?
- How can one graph be interpolated toward another?
- How can feasibility, ranking, and target guidance steer graph search?
- How can infeasible graphs be repaired through local edits?

## Main Generation Tracks

### Conditional Autoregressive Generation

Conditional autoregressive generation assembles reusable mapped subgraphs
against a target interpretation-graph scaffold. It treats generation as a
structured graph-building process guided by graph semantics rather than as an
unconstrained edge sampler.

See [Conditional Autoregressive Generation](docs/guides/conditional-autoregressive-generation.md).

### Edge-Based Generation and Repair

The edge-based generator starts from an input graph and grows it edge by edge
under feasibility constraints, graph ranking, optional target guidance, online
edge-risk penalties, pair retrieval, and local repair workflows.

One practical workflow is `repair(...)`: the generator retrieves similar stored
graphs, identifies feasibility violations, reduces the problematic start graph,
and regrows candidate graphs back toward the desired edge count.

See [Edge Generator](docs/guides/edge-generator.md).

### Rewriting and Interpolation

Rewrite and interpolation utilities provide constructive operations for moving
through graph space while retaining interpretable edit structure.

See [Autoregressive Generator and Rewrite](docs/guides/autoregressive-generator-and-rewrite.md).

## Extracted Workflows

Text and story-graph workflows were extracted into the sibling repo
`abstractgraph-text`.

Backend-specific generators were extracted into the sibling repo
`abstractgraph-generative-backends`.

## Ecosystem

See the [AbstractGraph ecosystem README](../../README.md) for how this
repository fits with the sibling repositories.
