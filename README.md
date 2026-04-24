# abstractgraph-generative

`abstractgraph-generative` is the graph generation and rewriting layer built on
top of `abstractgraph` and `abstractgraph-ml`.

It contains graph rewriting, autoregressive and conditional generation,
interpolation, optimization/repair workflows.

Two of the main active generation tracks in this repo are:

- conditional autoregressive generation
  This family generates a base graph by assembling reusable mapped subgraphs
  against a target interpretation-graph scaffold. It is implemented in
  `abstractgraph_generative.conditional`,
  `abstractgraph_generative.conditional_batch`, and the attributed extension in
  `abstractgraph_generative.conditional_attributed`.
  See [Conditional Autoregressive Generation](docs/guides/conditional-autoregressive-generation.md).

- edge-based generation
  This family starts from an input graph and grows it edge-by-edge under
  feasibility constraints, graph ranking, optional target guidance, optional
  online edge-risk penalties, and pair-retrieval and local repair workflows.
  It is implemented in `abstractgraph_generative.edge_generator`.
  See [Edge Generator](docs/guides/edge-generator.md).

One practical workflow in the edge-based generator is `repair(...)`:

- store a retrieval corpus with `generator.store(graphs, targets=...)`
- call `generator.repair(graph, n_neighbors=..., target=..., target_lambda=...)`
- the generator fits on the nearest stored neighbors of the query graph
- if the query graph is infeasible, it uses violating-edge sets from the final
  feasibility estimator to build surgically reduced start graphs
- it then regrows back to the original edge count

See [Edge Generator](docs/guides/edge-generator.md) for the full API and
examples.

Text and story-graph workflows were extracted into the sibling repo
`abstractgraph-text`.
Backend-specific generators were extracted into the sibling repo
`abstractgraph-generative-backends`.

## Ecosystem

This repo is one part of a three-repo stack:

- `abstractgraph`
  Path: `/home/fabrizio/work/abstractgraph`
- `abstractgraph-ml`
  Path: `/home/fabrizio/work/abstractgraph-ml`
- `abstractgraph-generative`
  Path: `/home/fabrizio/work/abstractgraph-generative`

See [ECOSYSTEM.md](ECOSYSTEM.md) for the dependency graph and install order.

## Package layout

- `src/abstractgraph_generative/rewrite.py`
- `src/abstractgraph_generative/autoregressive.py`
- `src/abstractgraph_generative/conditional.py`
- `src/abstractgraph_generative/conditional_batch.py`
- `src/abstractgraph_generative/interpolate.py`
- `src/abstractgraph_generative/interpolation.py`
- `src/abstractgraph_generative/optimize.py`
- `src/abstractgraph_generative/repair.py`
- `src/abstractgraph_generative/legacy/conditional_v0_1/`

## Documentation

- [docs/README.md](docs/README.md)
- [Autoregressive Generator and Rewrite](docs/guides/autoregressive-generator-and-rewrite.md)
- [Conditional Autoregressive Generation](docs/guides/conditional-autoregressive-generation.md)
- [Edge Generator](docs/guides/edge-generator.md)
- [ECOSYSTEM.md](ECOSYSTEM.md)

## Notebooks

- `notebooks/examples/` contains the remaining core generative workflows.
- Text-oriented notebooks now live in `/home/fabrizio/code/abstractgraph-text`.
- Backend-generator notebooks now live in
  `/home/fabrizio/code/abstractgraph-generative-backends`.
- `notebooks/research/` contains exploratory generation notebooks.
- Example and research notebooks now bootstrap imports and normalize the
  working directory automatically for the standard three-repo ecosystem layout.

## Dependencies

- `abstractgraph`
- `abstractgraph-ml`

## Local validation

```bash
python -m pip install -e ../abstractgraph --no-deps
python -m pip install -e ../abstractgraph-ml --no-deps
python -m pip install -e . --no-deps
python scripts/smoke_test.py
```

## Ecosystem

See the [AbstractGraph ecosystem README](../../README.md) for how this
repository fits with the sibling repositories.
