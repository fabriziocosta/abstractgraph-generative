# abstractgraph-generative

`abstractgraph-generative` is the graph generation and rewriting layer built on
top of `abstractgraph` and `abstractgraph-ml`.

It contains graph rewriting, autoregressive and conditional generation,
interpolation, optimization/repair workflows.

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
- [docs/AUTOREGRESSIVE_GENERATOR_AND_REWRITE.md](docs/AUTOREGRESSIVE_GENERATOR_AND_REWRITE.md)
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
