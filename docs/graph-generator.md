# GraphGenerator

`GraphGenerator` combines two existing generators into one two-stage
workflow:

1. generate a target interpretation graph with `EdgeGenerator`,
2. instantiate base graphs for that target with
   `ConditionalAutoregressiveGenerator`.

The implementation lives in `abstractgraph_generative.graph_generator`.

## Core Idea

`GraphGenerator` keeps two aligned corpora:

- `stored_graphs_`
  The original base graphs.
- `stored_interpretation_graphs_`
  The interpretation graphs produced from those base graphs, or supplied
  directly by the caller.

Sampling happens in interpretation-graph space first. For each sampled seed, the
generator retrieves nearby stored interpretation graphs, fits `EdgeGenerator` on
that local interpretation neighborhood, prunes edges from the seed
interpretation graph, and regrows a new interpretation graph.

The generated interpretation graph is then used as a query against the same
stored interpretation corpus. The nearest matches identify the corresponding
base graphs used to fit `ConditionalAutoregressiveGenerator`, which then
materializes concrete base graph instances for the generated target structure.

## Main API

```python
from abstractgraph_generative.graph_generator import GraphGenerator

generator = GraphGenerator(
    edge_generator=edge_generator,
    conditional_generator=conditional_generator,
    decomposition_function=decomposition_function,
    nbits=14,
    label_mode="histogram_values",
    interpretation_neighbor_vectorizer=None,
    seed=0,
)

generator.store(graphs, interpretation_graphs=interpretation_graphs)

samples = generator.sample(
    n_samples=4,
    n_interpretation_neighbors=30,
    n_conditional_neighbors=30,
    n_instances_per_sample=2,
    interpretation_edge_removal_size=0.5,
    random_state=0,
)
```

Constructor arguments:

- `edge_generator`
  Configured `EdgeGenerator` used on interpretation graphs.
- `conditional_generator`
  Configured `ConditionalAutoregressiveGenerator` used on base graphs.
- `decomposition_function`, `nbits`, `label_mode`
  The exact interpretation configuration used to compute or validate
  interpretation graphs.
- `interpretation_neighbor_vectorizer`
  Optional graph vectorizer for interpretation-neighbor retrieval. If omitted,
  `GraphGenerator` reuses `EdgeGenerator.store(...)` retrieval machinery.
- `seed`
  Default random seed for sampling.

## Store Phase

```python
generator.store(
    graphs,
    interpretation_graphs=None,
    targets=None,
)
```

`store(...)` requires at least two base graphs.

If `interpretation_graphs` is omitted, each interpretation graph is computed
with:

```python
graph_to_abstract_graph(
    graph,
    decomposition_function=decomposition_function,
    nbits=nbits,
    label_mode=label_mode,
).interpretation_graph
```

Stored attributes:

- `stored_graphs_`
- `stored_interpretation_graphs_`
- `stored_targets_`
- `stored_interpretation_hash_to_index_`
- `stored_interpretation_retrieval_vectors_`
- `stored_interpretation_distance_matrix_`

`targets` are optional and are forwarded to the local `EdgeGenerator.fit(...)`
calls when present.

## Sample Phase

For each requested seed, `sample(...)` does the following:

1. sample one stored interpretation graph index,
2. retrieve `n_interpretation_neighbors` nearby stored interpretation graphs,
3. fit `EdgeGenerator` on those interpretation graphs,
4. prune the seed interpretation graph with `remove_edges(...)`,
5. regrow one generated interpretation graph to the seed's original edge count,
6. retrieve `n_conditional_neighbors` stored interpretation graphs nearest to
   the generated interpretation graph,
7. fit `ConditionalAutoregressiveGenerator` on the aligned base graphs,
8. generate `n_instances_per_sample` base graphs for that generated
   interpretation graph.

The returned value is a flat `list[nx.Graph]`. With
`n_samples=4` and `n_instances_per_sample=2`, up to eight base graphs are
returned. Failed edge-stage seeds are skipped with a `RuntimeWarning`, so the
final count may be smaller.

`conditional_generate_kwargs` are forwarded to
`ConditionalAutoregressiveGenerator.generate(...)`, except `n_samples` and
`interpretation_graphs`, which are controlled by `GraphGenerator`.

`edge_generate_kwargs` are forwarded to `EdgeGenerator.generate(...)`, except
`return_path`, which is controlled internally.

## Bookkeeping

After `sample(...)`, these attributes describe the last run:

- `last_sampled_indices_`
- `last_interpretation_neighbor_indices_history_`
- `last_conditional_neighbor_indices_history_`
- `last_generated_interpretation_graphs_`
- `last_edge_generation_paths_`
- `last_conditional_training_graphs_history_`

These are useful for notebooks and diagnostics. For example:

```python
seed_graphs = [generator.stored_graphs_[i] for i in generator.last_sampled_indices_]
generated_targets = generator.last_generated_interpretation_graphs_
training_sets = generator.last_conditional_training_graphs_history_
```

## Interpretation Labels

The conditional stage preserves the target interpretation graph through the
existing `ConditionalAutoregressiveGenerator` postcondition. That means every
generated base graph must re-decompose to the generated interpretation graph
under the same `decomposition_function`, `nbits`, and `label_mode`.

For decomposition modes such as cycle/tree decomposition, this is the key
invariant: once the edge stage proposes a generated interpretation graph, the
conditional stage should instantiate a base graph whose re-decomposition matches
that proposal.

## ZINC Example

See:

- `notebooks/examples/example_graph_generator_zinc.ipynb`

The notebook uses a small ZINC slice, a cycle/tree interpretation
decomposition, an `EdgeGenerator` configured for interpretation graphs, and a
`ConditionalAutoregressiveGenerator` configured for base molecule
instantiation. It displays sampled source molecules, generated interpretation
graphs, generated molecules, and seed/target/final interpretation label counts.
