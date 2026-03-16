# AbstractGraph Ecosystem

The AbstractGraph stack is split across four sibling repositories:

- `abstractgraph`
  Path: `/home/fabrizio/work/abstractgraph`
  Role: core representation, operators, XML, hashing, vectorization, display,
  compatibility shims, and graph adapters

- `abstractgraph-graphicalizer`
  Path: `/home/fabrizio/work/abstractgraph-graphicalizer`
  Role: raw-data-to-NetworkX graphicalizers, including attention-driven
  base-graph induction and chemistry conversion/drawing

- `abstractgraph-ml`
  Path: `/home/fabrizio/work/abstractgraph-ml`
  Role: estimators, neural models, feasibility, importance, and top-k analysis

- `abstractgraph-generative`
  Path: `/home/fabrizio/work/abstractgraph-generative`
  Role: rewriting, autoregressive and conditional generation, interpolation,
  optimization/repair, and story-graph tooling

Dependency direction:

- `abstractgraph`
- `abstractgraph-graphicalizer` depends on no sibling repos
- `abstractgraph-ml` depends on `abstractgraph`
- `abstractgraph-generative` depends on `abstractgraph` and `abstractgraph-ml`

Editable install order:

```bash
python -m pip install -e /home/fabrizio/work/abstractgraph --no-deps
python -m pip install -e /home/fabrizio/work/abstractgraph-graphicalizer --no-deps
python -m pip install -e /home/fabrizio/work/abstractgraph-ml --no-deps
python -m pip install -e /home/fabrizio/work/abstractgraph-generative --no-deps
```
