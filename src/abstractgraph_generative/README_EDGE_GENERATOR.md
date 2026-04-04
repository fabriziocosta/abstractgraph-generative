# EdgeGenerator

`EdgeGenerator` builds graphs by adding edges one step at a time, using:

- a feasibility model to reject invalid partial graphs
- a graph classifier to rank feasible candidates
- a beam search over partial constructions

The implementation lives in `abstractgraph_generative.edge_generator`.

## Main API

Key helpers:

- `edge_neighbors`
- `remove_edges`
- `make_edge_regression_dataset`
- `mix_connected_components`
- `EdgeGenerator`

Typical usage:

```python
from sklearn.ensemble import RandomForestClassifier

from abstractgraph.vectorize import AbstractGraphTransformer
from abstractgraph_ml.estimators import GraphEstimator
from abstractgraph_generative.edge_generator import EdgeGenerator

transformer = AbstractGraphTransformer(
    nbits=14,
    decomposition_function=df,
    return_dense=True,
    n_jobs=-1,
)

graph_estimator = GraphEstimator(
    transformer=transformer,
    estimator=RandomForestClassifier(
        random_state=0,
        n_estimators=300,
        n_jobs=-1,
        class_weight="balanced_subsample",
    ),
)

generator = EdgeGenerator(
    feasibility_estimator=feasibility_estimator,
    graph_estimator=graph_estimator,
    n_negative_per_positive=5,
    n_replicates=10,
    beam_size=2,
    max_restarts=3,
    fallback_base_steps=2,
    fallback_growth_factor=2.0,
    beam_growth_factor=1.5,
    max_beam_size=8,
    verbose=True,
    seed=0,
).fit(fit_graphs)

generated_path = generator.generate(
    start_graph,
    n_edges=target_n_edges,
    draw_graphs_fn=lambda graphs: display_graphs(graphs, n_graphs_per_line=1),
)
```

Component-mixing helper:

```python
from abstractgraph_generative.edge_generator import mix_connected_components

mixed_graph = mix_connected_components(graph_a, graph_b, seed=0)
```

`mix_connected_components` builds a new graph by taking the same number of
connected components from each source graph and choosing the component subsets
whose total node count is closest to the average node count of the two inputs.

## Generate Semantics

`generate()` is intentionally fault-tolerant.

For a single input graph:

- returns a generated path on success
- returns `[]` if no feasible construction can be found

For multiple input graphs:

- attempts each graph independently
- skips graphs that fail generation
- returns only successful paths
- returns `[]` if all requested graphs fail

With `verbose=True`, failed graphs emit a log line instead of aborting the full
batch.

## Search Strategy

At each depth, the generator:

1. expands the current beam by adding one missing edge
2. filters expanded graphs with the feasibility estimator
3. scores feasible survivors with the graph classifier
4. retains a beam made of:
   - the top-scoring candidates
   - a random sample from the remaining positive candidates

This keeps some exploration while still pushing toward high classifier score.

## Exponential Fallback

When the active beam reaches a dead end, the generator does not immediately
restart from the start graph.

Instead it uses an exponential fallback schedule.

For fallback index `i`:

```text
rollback_steps_i = ceil(fallback_base_steps * fallback_growth_factor**i)
beam_limit_i     = ceil(beam_size * beam_growth_factor**(i + 1))
```

The fallback depth is:

```text
fallback_depth = max(0, current_depth - rollback_steps_i)
```

If `fallback_depth == 0`, the fallback becomes a full restart from the start
graph.

### Example

With:

- `beam_size=2`
- `max_restarts=4`
- `fallback_base_steps=2`
- `fallback_growth_factor=2.0`
- `beam_growth_factor=1.5`

the fallback schedule is approximately:

1. rollback `2` steps, beam limit `3`
2. rollback `4` steps, beam limit `5`
3. rollback `8` steps, beam limit `7`
4. rollback `16` steps, beam limit `11`

If the current depth is smaller than the rollback distance, that stage becomes a
full restart.

`max_restarts` controls the number of fallback stages after the initial search
phase.

So the total number of search phases is:

```text
1 + max_restarts
```

This is why verbose logs report a `phase` index rather than only a restart
counter.

## Dead-End Banning

When a frontier at depth `d` has no feasible continuations, its states are
marked as dead ends for depth `d`.

After a rollback, if search regenerates candidates that would reproduce those
same dead-end states at the same depth, they are filtered out.

This prevents the search from repeatedly rebuilding exactly the same failed
suffix after a rollback.

## Tabu Dead-End Paths

The generator also keeps a path-level tabu set for exact dead-end trajectories.

When a frontier dies, each state in that frontier contributes:

- its state key at the current depth
- its full path signature from the start graph to that dead-end state

After rollback, a candidate is rejected if:

- it recreates a dead-end state at the same depth
- it recreates an exact dead-end path that already failed earlier

This path-level tabu is stricter than depth-local dead-end banning and helps
avoid cycling back into exactly the same failed branch history after rollback.

## Important Parameters

### Dataset / fitting

- `n_negative_per_positive`
  Number of negative neighbors sampled per positive training graph.
- `n_replicates`
  Number of edge-removal trajectories generated per seed graph.
- `fit_n_jobs`
  Parallelism for dataset generation.
- `fit_backend`
  Joblib backend for dataset generation.

### Search

- `beam_size`
  Base beam width used before any fallback.
- `max_restarts`
  Number of fallback stages allowed after the initial search phase.
- `fallback_base_steps`
  Initial rollback distance for the first fallback.
- `fallback_growth_factor`
  Exponential growth factor for rollback distance.
- `beam_growth_factor`
  Exponential growth factor for beam widening after each fallback.
- `max_beam_size`
  Optional cap on widened beam size.
- `allow_self_loops`
  Whether self-loop edge additions are allowed.

### Logging / control

- `verbose`
  Enables fit-time and search progress logging.
- `seed`
  Controls randomized parts of beam retention and dataset generation.

## Logging

With `verbose=True`, fit logs are printed in `Xm Y.Ys` format.

Search logs report:

- `phase`
- `depth`
- current frontier size
- number of generated candidates
- number of feasible candidates
- retained beam size
- cumulative tried candidates
- best classifier score
- remaining edges
- per-step time
- ETA
- active beam limit
- fallback count when a rollback happens

Fallback transitions are logged explicitly, including:

- fallback index
- rollback distance
- target depth
- widened beam limit

## Tradeoffs

Pros of the current approach:

- recovers from late dead ends without immediately discarding all progress
- widens search only when needed
- keeps compute focused on local repair before broader restart

Cons:

- wider fallback beams can still become expensive
- very aggressive classifier bias can still pull search back into the same basin
- the search is heuristic, so fallback schedules still need tuning per domain

## Tuning Guidance

If search dies late and often:

- increase `max_restarts`
- increase `beam_growth_factor`
- increase `max_beam_size`

If search is too slow:

- reduce `beam_size`
- reduce `max_restarts`
- reduce `beam_growth_factor`
- lower `n_negative_per_positive` or `n_replicates` during prototyping

If recovery is too conservative:

- increase `fallback_base_steps`
- increase `fallback_growth_factor`

If recovery is too aggressive:

- reduce `fallback_growth_factor`
- cap with a smaller `max_beam_size`
