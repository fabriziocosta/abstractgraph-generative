# Graph Rewrite

This guide covers the mapped-subgraph rewrite primitives in
`abstractgraph_generative.rewrite`, which support interpolation and repair
workflows.

## Concepts: interpretation nodes and mapped subgraphs

`abstractgraph` uses two levels:

- Base graph: the original NetworkX graph.
- Interpretation graph: nodes represent subgraphs of the base graph. Each
  interpretation node stores a `mapped_subgraph` in its node attributes.

Graph rewriting swaps one mapped subgraph for another compatible mapped
subgraph and reconnects the boundary to preserve structure.

## Compatibility via cut signatures

For any mapped subgraph, the rewrite logic builds a boundary cut:

- Inner nodes: nodes belonging to the mapped subgraph.
- Cut edges: edges that cross from inner nodes to the outer graph.

Each cut edge is mapped to a per-edge key that captures local context:

- If `cut_radius is None`, the key is just the edge label hash (or empty if
  `cut_include_edge_label=False`).
- If `cut_radius >= 0`, the key combines hashes of the inner and outer
  radius-limited neighborhoods. The `cut_scope` flag can include inner only,
  outer only, or both.

The cut signature is the multiset of per-edge keys (stored as a sorted tuple).
Two mapped subgraphs are compatible if their signatures are identical. This
ensures boundary edges can be reconnected one-to-one by key.

## Rewrite algorithm

`rewrite(source, donors, ...)` performs one mapped-subgraph swap:

1. Build or reuse `AbstractGraph` objects for the source and donors.
2. Index donor and source mapped subgraphs by cut signature.
3. Match source and donor entries with the same signature.
4. Choose a compatible pair and replace the source mapped subgraph while
   preserving cut compatibility.
5. Repeat to produce `n_samples` rewrites.

If `use_context_embedding=True` and a `context_vectorizer` is provided, each cut
stores aggregated context embeddings and candidate pairs can be scored by cosine
similarity. `iterated_rewrite` applies rewrites repeatedly for
`n_iterations`, optionally filtering each batch with a feasibility estimator.
