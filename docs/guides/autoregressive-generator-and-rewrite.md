# Autoregressive Generator and Graph Rewrite

This document explains how the autoregressive generator in
`abstractgraph_generative.autoregressive` works and how it relies on the graph
rewrite utilities in `abstractgraph_generative.rewrite`. It focuses on the
end-to-end flow, data structures, and selection logic rather than every
parameter.

## Concepts: interpretation nodes and mapped subgraphs

`abstractgraph` uses two levels:
- Base graph: the original NetworkX graph.
- Interpretation graph: nodes represent subgraphs of the base graph. Each
  interpretation node stores a `mapped_subgraph` in its node attributes.

Graph rewriting swaps one mapped subgraph for another compatible mapped
subgraph and reconnects the boundary to preserve structure.

## Graph rewrite in `abstractgraph_generative.rewrite`

### Compatibility via cut signatures

For any mapped subgraph, the rewrite logic builds a boundary cut:
- Inner nodes: nodes belonging to the mapped subgraph.
- Cut edges: edges that cross from inner nodes to the outer graph.

Each cut edge is mapped to a per-edge key that captures local context:
- If `cut_radius is None`, the key is just the edge label hash (or empty if
  `cut_include_edge_label=False`).
- If `cut_radius >= 0`, the key combines hashes of the inner/outer
  radius-limited neighborhoods. The `cut_scope` flag can include inner only,
  outer only, or both.

The cut signature is the multiset of per-edge keys (stored as a sorted tuple).
Two mapped subgraphs are compatible if their signatures are identical. This ensures
boundary edges can be reconnected one-to-one by key.

### Rewrite algorithm (single step)

`rewrite(source, donors, ...)` performs one mapped-subgraph swap:
1. Build or reuse `AbstractGraph` for source and donors.
2. Index donors by cut signature.
3. Index source mapped subgraphs by cut signature.
4. Match source and donor entries by cut signature.
5. Pick a compatible source mapped subgraph and donor mapped subgraph.
6. Replace the source mapped subgraph with the donor mapped subgraph while preserving
   cut compatibility.
7. Repeat to produce `n_samples` rewrites.

### Context-aware selection

If `use_context_embedding=True` and `context_vectorizer` is provided, each cut
stores aggregated context embeddings and candidate source/donor pairs can be
scored by cosine similarity.

### Iterated rewrites

`iterated_rewrite` applies `rewrite` repeatedly for `n_iterations`, optionally
filtering each batch with a feasibility estimator.

## Autoregressive generator in `abstractgraph_generative.autoregressive`

The autoregressive generator grows graphs by inserting donor mapped subgraphs at
virtual cuts. It is autoregressive because each step proposes expansions based
on the current graph state and then selects the next graph from those
proposals.

### Fit phase

`AutoregressiveGraphGenerator.fit(generator_graphs)` builds:
1. pruning sequences from training graphs
2. a donor/cut index
3. an optional feasibility model
4. optional similarity embeddings for training graphs

### Candidate generation

At each step the generator:
1. samples a cut size
2. samples compatible cut nodes from the current graph
3. inserts donor mapped subgraphs compatible with the virtual cut

### Candidate selection

Candidates can be filtered and selected by:
- feasibility
- growth constraints
- similarity to training graphs
- context compatibility
- fallback heuristics

## Minimal usage sketch

```python
from abstractgraph.operators import node
from abstractgraph_ml.feasibility import FeasibilityEstimator
from abstractgraph_generative.autoregressive import AutoregressiveGraphGenerator

feasibility = FeasibilityEstimator(...)
gen = AutoregressiveGraphGenerator(
    feasibility_estimator=feasibility,
    nbits=8,
    decomposition_function=node(),
    use_similarity_selection=True,
)

gen.fit(training_graphs)
samples = gen.generate(n_samples=5)
```

## Practical notes

- Similarity selection uses training graphs as reference graphs.
- Cut signatures may optionally include edge labels.
- Candidate enumeration can grow quickly if replacement branching is not
  bounded.
- The legacy conditional pipeline remains available under
  `abstractgraph_generative.legacy.conditional_v0_1` when older workflows need
  to be preserved.
