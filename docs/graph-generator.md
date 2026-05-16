# GraphGenerator

`GraphGenerator` combines one top-level `EdgeGenerator` with one or more
bottom-up `ConditionalAutoregressiveGenerator` stages.

The conditional generators are ordered from base graph to top interpretation
graph:

```text
base graph -> level 1 interpretation -> ... -> top interpretation
```

Sampling first generates a top-level target with `EdgeGenerator`, then walks
back down through the conditional generators in reverse order until final base
graphs are produced.

## Main API

```python
generator = GraphGenerator(
    edge_generator=edge_generator,
    conditional_generators=[
        base_to_interpretation_generator,
        interpretation_to_top_generator,
    ],
    seed=0,
    debug=False,
    require_new_interpretation_graph=True,
    max_same_interpretation_retries=3,
)

generator.store(graphs, targets=targets)

samples = generator.sample(
    n_samples=4,
    n_interpretation_neighbors=30,
    n_conditional_neighbors=[100, 30],
    n_instances_per_sample=2,
    interpretation_edge_removal_size=0.5,
    random_state=0,
    conditional_generate_kwargs=[
        {"max_backtracks": 2000},
        {"max_backtracks": 1000},
    ],
)
```

For backwards compatibility, a single stage can still be passed as:

```python
generator = GraphGenerator(
    edge_generator=edge_generator,
    conditional_generator=conditional_generator,
)
```

## Store Phase

`store(graphs, targets=None)` computes and stores all hierarchy levels. Explicit
`interpretation_graphs` are no longer accepted by the public store interface.
Each level is computed from the previous one with that stage's conditional
generator config:

```python
graph_to_abstract_graph(
    graph,
    decomposition_function=conditional_generators[i].decomposition_function,
    nbits=conditional_generators[i].nbits,
    label_mode=conditional_generators[i].label_mode,
).interpretation_graph
```

Stored attributes:

- `stored_level_graphs_[0]`: base graphs.
- `stored_level_graphs_[i + 1]`: interpretation graphs computed by stage `i`.
- `stored_graphs_`: compatibility alias for level `0`.
- `stored_interpretation_graphs_`: compatibility alias for the top level.
- `stored_targets_`: optional targets forwarded to top edge fitting.

The top retrieval index is owned by `edge_generator.store(top_level_graphs,
targets=targets)`. Intermediate conditional retrieval indexes use that stage's
`context_vectorizer` when present, otherwise a small graph descriptor fallback.

## Sample Phase

`sample(...)` treats `n_samples` as the number of successful top-level targets.
`n_instances_per_sample` controls the final number of base graphs per successful
target, not multiplicative fanout at every level.

Top-level controls remain scalar:

- `n_interpretation_neighbors`
- `interpretation_edge_removal_size`
- `edge_generate_kwargs`
- `max_seed_attempts`

Conditional controls accept either one value for every stage or a sequence
aligned with `conditional_generators`:

- `n_conditional_neighbors`
- `conditional_generate_kwargs`
- `deduplicate_conditional_neighbors`
- `exclude_seed_from_conditional_neighbors`
- `avoid_seed_components`

For each attempted seed, sampling:

1. selects a stored top-level seed graph,
2. optionally fits the edge generator on nearby top-level graphs and regrows a
   new top-level target,
3. for each conditional stage in reverse order, retrieves nearest stored
   upper-level graphs, fits on aligned lower-level graphs, generates lower-level
   candidates, and validates the generated interpretation postcondition,
4. returns final level-0 graphs as a flat `list[nx.Graph]`.

If `interpretation_edge_removal_size=0`, the edge stage is bypassed and the
sampled top-level seed is used directly as the first conditional target.

## Bookkeeping

After `sample(...)`, these attributes describe the last run:

- `last_sampled_indices_`: attempted top-level seed indices.
- `last_successful_sampled_indices_`: seeds that reached final base output.
- `last_interpretation_neighbor_indices_history_`: top edge-stage neighbors.
- `last_interpretation_neighbor_distances_history_`: top edge-stage distances.
- `last_edge_generation_paths_`: `[start_graph, generated_top_graph]` per
  successful seed.
- `last_level_seed_graphs_history_`: per-success seed graph at every stored
  level.
- `last_level_generated_graphs_history_`: per-success generated graphs at every
  level, where index `0` is final base output and the last index is the top
  target.
- `last_level_neighbor_indices_history_`: per-success, per-stage conditional
  neighbor indices.
- `last_level_training_graphs_history_`: per-success, per-stage conditional
  training graphs.

## Interpretation Ownership

Each conditional generator owns its own `decomposition_function`, `nbits`, and
`label_mode`. `GraphGenerator` validates that every stage provides these fields
and uses the stage's config for level construction and output validation.

The edge generator is singular and always operates on the top computed
interpretation level.

## ZINC Example

See:

- `notebooks/examples/example_graph_generator_zinc.ipynb`

The current notebook uses the backwards-compatible single conditional generator
form. A hierarchical notebook should pass `conditional_generators=[...]` in
bottom-up order and tune conditional parameters either as scalars or per-stage
lists.
