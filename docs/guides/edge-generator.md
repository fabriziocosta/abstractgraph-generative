# EdgeGenerator

`EdgeGenerator` builds graphs by adding edges one step at a time, using:

- a partial-stage feasibility model to reject invalid intermediate graphs
- a final-graph feasibility model to validate completed graphs
- a graph classifier to rank feasible candidates
- an optional target model to bias generation toward a requested class or value
- an optional online edge-risk model to penalize edge decisions that tend to
  lead to infeasible descendants
- an optional decomposition-aware training trajectory to bias reconstruction toward complete subgraphs
- an optional stored retrieval corpus to select path-based fitting subsets from graph pairs
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDRegressor

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

target_estimator = GraphEstimator(
    transformer=transformer,
    estimator=RandomForestRegressor(
        random_state=0,
        n_estimators=300,
        n_jobs=-1,
    ),
)

edge_risk_estimator = GraphEstimator(
    transformer=transformer,
    estimator=SGDRegressor(
        loss="epsilon_insensitive",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        random_state=0,
    ),
)

# Example target: regress graph max degree.
fit_targets = [max(dict(graph.degree()).values(), default=0) for graph in fit_graphs]

generator = EdgeGenerator(
    feasibility_estimator=feasibility_estimator,
    graph_estimator=graph_estimator,
    target_estimator=target_estimator,
    edge_risk_estimator=edge_risk_estimator,
    target_estimator_mode="regression",
    decomposition_function=EDGE_DECOMPOSITION_FUNCTION,
    n_negative_per_positive=5,
    n_replicates=10,
    beam_size=2,
    max_restarts=3,
    fallback_base_steps=2,
    fallback_growth_factor=2.0,
    beam_growth_factor=1.5,
    max_beam_size=8,
    edge_risk_lambda=0.25,
    require_single_connected_component=True,
    verbose=True,
    seed=0,
).fit(fit_graphs, fit_targets)

generated_path = generator.generate(
    start_graph,
    n_edges=target_n_edges,
    target=desired_max_degree,
    target_lambda=0.5,
    return_path=True,
    draw_graphs_fn=lambda graphs: display_graphs(graphs, n_graphs_per_line=1),
)
```

Stored-corpus pair workflow:

```python
generator = EdgeGenerator(
    partial_feasibility_estimator=partial_feasibility_estimator,
    final_feasibility_estimator=final_feasibility_estimator,
    graph_estimator=graph_estimator,
    target_estimator=target_estimator,
    edge_risk_estimator=edge_risk_estimator,
    target_estimator_mode="regression",
    edge_risk_lambda=0.25,
    decomposition_function=EDGE_DECOMPOSITION_FUNCTION,
    verbose=True,
    seed=0,
)

generator.store(all_graphs, targets=all_targets)

generation_path = generator.generate_from_pair(
    graph_a,
    graph_b,
    size=0.5,
    n_paths=3,
    path_k=3,
    n_neighbors_per_path_graph=3,
    target=None,
    target_lambda=0.5,
    return_path=True,
    draw_graphs_fn=lambda graphs, **kwargs: display_graphs(graphs, **kwargs),
)
```

Stored-corpus repair workflow:

```python
repair_path = generator.repair(
    graph,
    n_neighbors=3,
    target=None,
    target_lambda=0.5,
    return_path=True,
    draw_graphs_fn=lambda graphs, **kwargs: display_graphs(
        graphs,
        n_graphs_per_line=7,
        **kwargs,
    ),
)
```

Use `repair(...)` when you already have a candidate graph and want to repair it
relative to a stored corpus, rather than mixing two endpoints.

Typical pattern:

1. call `generator.store(all_graphs, targets=all_targets)` once
2. pass the query graph to `generator.repair(...)`
3. choose `n_neighbors` to control how local the repair fit should be
4. optionally pass `target` and `target_lambda` to bias the repaired result

Behavior:

- the target edge count is always `graph.number_of_edges()`
- if the graph is already final-feasible, repair returns it unchanged
- before fitting the local repair models, `repair()` checks that the unique
  node-label set in the input graph is covered by the union of unique
  node labels present in the retrieved repair neighborhood; if any input label
  is missing from the neighborhood, repair fails fast and returns no repair
  instead of fitting incompatible feasibility models
- if the graph is final-infeasible, repair asks the final-feasibility estimator
  for violating edge sets, removes the most implicated edges according to the
  fallback rollback schedule, and regrows from those repaired starts
- with `return_path=True`, the returned path begins with the original graph,
  then the repaired start, then the regrowth sequence

Diagnostics:

- when the label-set compatibility check fails, the generator stores the
  details in `last_repair_label_set_mismatch_`
- in verbose mode, repair logs `missing_from_neighbors` and
  `extra_in_neighbors` so you can inspect neighborhood coverage

Component-mixing helper:

```python
from abstractgraph_generative.edge_generator import mix_connected_components

mixed_graph = mix_connected_components(graph_a, graph_b, seed=0)
```

`mix_connected_components` builds a new graph by taking the same number of
connected components from each source graph and choosing the component subsets
whose total node count is closest to the average node count of the two inputs.

## Feasibility Stages

`EdgeGenerator` supports two feasibility models:

- `partial_feasibility_estimator`: trained on partial reconstruction stages and
  used while edges are being added
- `final_feasibility_estimator`: trained on full graphs and used only when a
  candidate has reached the requested final edge count

This is useful when some constraints should apply only to completed graphs. For
example, an intermediate graph may temporarily contain aromatic bonds that do
not yet form a full aromatic cycle, while the final graph must satisfy that
constraint.

Backward compatibility:

- if you pass only `feasibility_estimator=...`, the generator uses it as the
  partial estimator and deep-copies it for the final estimator

Training behavior:

- the partial estimator is fit on the fragment set used during edge-regression
  training
- the final estimator is fit only on the full seed graphs

## Generate Semantics

`generate()` is intentionally fault-tolerant.

For a single input graph:

- with `return_path=True` returns a generated path on success
- with `return_path=False` returns only the final graph on success
- returns `[]` if no feasible construction can be found and `return_path=True`
- returns `None` if no feasible construction can be found and `return_path=False`

For multiple input graphs:

- attempts each graph independently
- skips graphs that fail generation
- returns only successful paths with `return_path=True`
- returns only successful final graphs with `return_path=False`
- returns `[]` if all requested graphs fail

With `verbose=True`, failed graphs emit a log line instead of aborting the full
batch.

`return_path=True` is still the default for compatibility.

When `require_single_connected_component=True` (the default), a graph is not
accepted as a terminal result unless it is both final-feasible and reduced to a
single connected component. If search reaches `n_edges` with a disconnected
final-feasible graph, it can continue for a bounded number of extra edges to
try to connect the remaining components.

## Stored Retrieval Corpus

`EdgeGenerator.store()` prepares a reusable corpus for multiple pair queries:

```python
generator.store(graphs, targets=None)
```

This:

- stores the graphs and optional per-graph targets
- deep-copies the graph-estimator transformer for retrieval use
- vectorizes the stored graphs once
- precomputes a dense pairwise distance matrix once

After that, `generate_from_pair()` can reuse the stored corpus:

```python
generator.generate_from_pair(
    graph_a,
    graph_b,
    size=0.5,
    n_paths=3,
    path_k=3,
    n_neighbors_per_path_graph=3,
    target=None,
    target_lambda=1.0,
    return_path=True,
)
```

`generate_from_pair()`:

1. resolves `graph_a` and `graph_b` to stored indices when their hashes match
2. otherwise transforms the query graphs on the fly and appends them to a temporary query corpus
3. builds an `MST + kNN` retrieval graph from the dense pairwise distance matrix, using `path_k` as the kNN degree
4. extracts `n_paths` edge-disjoint shortest paths by removing used path edges after each path
5. deduplicates the path graphs and augments that set with up to `n_neighbors_per_path_graph` nearest neighbors per selected graph
6. fits the generator on the resulting expanded set
7. removes edges from both endpoints
8. mixes connected components from the two reduced graphs
9. starts generation from that mixed graph

After one successful pair setup call, you can reuse the same fitted pair session
without recomputing paths or refitting:

```python
generator.generate_from_pair(
    None,
    None,
    target_lambda=1.0,
    return_path=True,
)
```

In that mode the generator:

- reuses the previously cached `graph_a`, `graph_b`, edge-removal fraction, and
  resolved target
- reuses the already fitted estimators from the prior pair call
- performs a fresh stochastic edge removal and component mix
- starts a new generation immediately

You can also override the cached edge-removal amount without rebuilding the
pair-retrieval context:

```python
generator.generate_from_pair(
    None,
    None,
    size_of_edge_removal=0.8,
    target_lambda=1.0,
    return_path=True,
)
```

If `size_of_edge_removal` is omitted in cached-session mode, the previously
cached value is reused.

If stored targets exist and no explicit `target=` is passed, `generate_from_pair()`
infers a pair target from the endpoint targets:

- regression mode: mean target
- classification mode: uniformly sample one of the two endpoint targets

`repair()` also reuses the stored corpus, but for a single graph query:

1. vectorize the query graph against the stored retrieval transformer
2. select the `n_neighbors` nearest stored graphs
3. fit the generator on that local neighborhood
4. keep the original graph edge count as the repair target
5. if the input graph is already final-feasible, return it directly
6. otherwise ask the final-feasibility estimator for violating edge sets
7. construct one or more surgically reduced start graphs by removing the most
   implicated edges according to the fallback rollback schedule
8. regrow from those repaired starts back to the original edge count

If the queried graph exactly matches a stored graph hash and stored targets are
available, `repair()` reuses that stored target when no explicit `target=` is
provided.

## Search Strategy

At each depth, the generator:

1. expands the current beam by adding one missing edge
2. filters expanded graphs with the feasibility estimator
3. scores feasible survivors with the graph classifier
4. optionally adds a target-model score from `target_estimator`
5. optionally subtracts an online edge-risk penalty from `edge_risk_estimator`
6. subtracts fallback repulsion when that mechanism is active
7. retains a beam made of:
   - the top-scoring candidates
   - a random sample from the remaining positive candidates

This keeps some exploration while still pushing toward high classifier score.

After the first fallback, the generator can also apply a similarity-based
repulsion term against previously failed states. In that case candidate ranking
uses:

For classification mode:

```text
target_score = P(target_class | graph)
selection_score = (
    classifier_score
    + target_lambda * target_score
    - edge_risk_lambda * risk_score
    - repulsion_lambda * repulsion
)
```

For regression mode:

```text
target_score = 1 / (1 + abs(predicted_target - requested_target))
selection_score = (
    classifier_score
    + target_lambda * target_score
    - edge_risk_lambda * risk_score
    - repulsion_lambda * repulsion
)
```

where:

- `classifier_score` is the downstream graph classifier probability for class 1
- `target_score` is either the requested class probability or a regression closeness score
- `target_lambda` is the user-controlled weight of the target objective
- `risk_score` is the predicted future infeasible-descendant ratio for the
  edge decision that produced the candidate
- `edge_risk_lambda` is the user-controlled weight of the risk penalty
- `repulsion` is the maximum cosine similarity to the failed-memory bank
- `repulsion_lambda` grows with the number of fallback stages already used

## Online Edge Risk Learning

When `edge_risk_estimator` is provided, the generator learns online from its
own search history.

For every closed parent-to-child transition produced by materializing one edge,
the estimator input is a disjoint graph made from:

- the parent graph before adding the edge
- the child graph after adding the edge

The target is computed when that transition closes, either because the search
solves, fails, or enters fallback and replaces the current beam:

```text
risk = infeasible_descendants / total_descendants
```

where `total_descendants` includes the child state itself plus all realized
search descendants observed below it in the current search policy. This means:

- an immediately infeasible leaf contributes one example with target `1.0`
- an immediately non-infeasible leaf contributes one example with target `0.0`
- an internal transition contributes one example with a fractional target that
  summarizes how much of its realized downstream subtree became infeasible

This model is updated online through an adapter with `partial_fit(...)`
semantics:

- if the wrapped estimator already supports `partial_fit`, that is used directly
- otherwise the adapter stores all past training examples in memory and refits
  the estimator on the full replay buffer each time it is updated

`GraphEstimator` also exposes `partial_fit(...)` natively: it uses the wrapped
estimator's own `partial_fit(...)` when available, otherwise it caches the
already-vectorized batches and refits on their concatenation. This means an
online SVM-style regressor such as
`GraphEstimator(..., estimator=SGDRegressor(loss="epsilon_insensitive", ...))`
can be trained incrementally for edge risk, while non-incremental estimators
still work through replayed refits on cached feature matrices.

## Target-Conditioned Fitting

`fit()` now accepts optional per-graph targets:

```python
generator.fit(graphs, targets)
```

You can also fit or overwrite the target model independently:

```python
generator.fit_target_estimator(graphs, targets)
```

When `targets` are provided:

- `graph_estimator` is still trained on the binary edge-regression dataset
- `target_estimator` is trained only on positive fragments
- `target_estimator_mode="classification"` uses class probabilities during generation
- `target_estimator_mode="regression"` uses distance to the requested numeric target
- every positive fragment derived from a seed graph inherits that seed graph's target

`fit()` and `fit_target_estimator()` use the same fragment-generation pipeline, so
the target model always sees the same positive fragments that are derived during
edge-regression dataset construction.

So if a seed graph has target value `t`, every graph encountered along the
edge-removal trajectory for that seed graph is also labeled with `t` for the
target model.

## Decomposition-Aware Training

`EdgeGenerator` can also accept an optional `decomposition_function`:

```python
generator = EdgeGenerator(
    ...,
    decomposition_function=EDGE_DECOMPOSITION_FUNCTION,
)
```

When provided, the training dataset no longer removes arbitrary edges uniformly.
Instead it:

- decomposes each training graph into mapped subgraphs using `abstractgraph`
- treats subgraphs with a `multi_owner` policy
- allows subgraphs to overlap, but once an edge is removed in one subgraph it is
  absent for all later subgraphs in that trajectory
- removes all remaining edges from one chosen subgraph before moving to another
  subgraph, randomizing order within each subgraph

This only changes the training trajectories. Inference and generation remain
edge-wise and the `generate()` API does not change. The user nudges the behavior
indirectly by choosing the decomposition used to build those training paths.

At generation time you can then request a class:

```python
generator.generate(start_graph, n_edges=target_n_edges, target=c, target_lambda=0.5)
```

If `target=None`, the target objective is disabled and scoring falls back to the
original classifier-plus-repulsion behavior.

## Surgical Backtracking

When the active beam reaches a dead end, the generator first tries to repair
the blocked frontier instead of immediately rewinding to an older beam.

It does this by looking at the infeasible one-edge expansions that were just
considered at the dead end:

1. keep the best-scoring infeasible candidates for each blocked beam state
2. ask the feasibility estimator for `violating_edge_sets(...)`
3. count how often each parent-edge appears inside those violating edge sets
4. break ties with candidate score and edge recency
5. remove exactly as many edges as the fallback policy says to remove

This makes the fallback logic answer two separate questions:

- how many edges to remove
- which edges to remove

The first still comes from the rollback schedule. The second now comes from
feasibility evidence whenever the estimator can expose structural violations.

### Rollback Schedule

For fallback index `i`:

```text
rollback_steps_i = ceil(fallback_base_steps * fallback_growth_factor**i)
beam_limit_i     = ceil(beam_size * beam_growth_factor**(i + 1))
```

At fallback stage `i`, the generator removes `rollback_steps_i` edges from each
repairable blocked state and resumes search with beam size `beam_limit_i`.

If no useful violating-edge evidence is available, it falls back to the older
rewind behavior and restores the beam at:

```text
fallback_depth = max(0, current_depth - rollback_steps_i)
```

If `fallback_depth == 0`, that rewind becomes a full restart from the start
graph.

### Why This Helps

The old fallback policy knew only how far to jump back. The new policy still
uses that distance, but applies it surgically:

- attractive but infeasible candidates are not discarded as pure noise
- violating motif edge sets identify which existing edges repeatedly block good
  continuations
- repeated evidence across several infeasible candidates increases the priority
  of removing that edge

This is especially effective with feasibility estimators such as
`FeasibilityEstimatorFeatureCannotExist`, because they can map forbidden motifs
back to concrete edge sets.

### Example

With:

- `beam_size=2`
- `max_restarts=4`
- `fallback_base_steps=2`
- `fallback_growth_factor=2.0`
- `beam_growth_factor=1.5`

the rollback schedule is approximately:

1. remove `2` edges, beam limit `3`
2. remove `4` edges, beam limit `5`
3. remove `8` edges, beam limit `7`
4. remove `16` edges, beam limit `11`

If the current depth is smaller than the rollback distance, that stage becomes a
full restart.

`max_restarts` controls the number of fallback stages after the initial search
phase, so the total number of search phases is:

```text
1 + max_restarts
```

Verbose logs therefore still report a `phase` index, but they can now also show
when a fallback stage was handled by surgical edge removal instead of plain beam
rewind.

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

## Similarity Repulsion

To avoid returning to the same basin after a rollback, the generator keeps a
memory of failed states and repels candidates that are too similar to them.

The implementation:

- uses `hash_graph(graph)` as the cache key for vector embeddings
- caches vectorizer outputs in an embedding cache
- stores embeddings only for failed-memory states
- computes cosine similarity between feasible candidates and the failed-memory bank
- uses the maximum cosine similarity as the repulsion value

Repulsion is activated only after the search has entered fallback mode.

For fallback index `i`:

```text
lambda_i = repulsion_weight * repulsion_growth_factor**i
```

So later fallback phases push the search further away from previously failed
regions.

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
- `target_estimator`
  Optional classifier or regressor trained on positive fragments when
  `fit(..., targets=...)` is used.
- `edge_risk_estimator`
  Optional online regressor trained from realized search descendants during
  generation.
- `target_estimator_mode`
  Either `"classification"` or `"regression"` for target scoring.
- `decomposition_function`
  Optional `abstractgraph` decomposition used to create subgraph-ordered
  edge-removal trajectories during training only.

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
- `edge_risk_lambda`
  Coefficient applied to the predicted edge-risk penalty in candidate ranking.
- `require_single_connected_component`
  When true, disconnected final-feasible graphs are not accepted at
  `n_edges`; search may spend a bounded number of extra edges trying to merge
  the remaining connected components.
- `use_similarity_repulsion`
  Whether to activate cosine-similarity repulsion after fallback begins.
- `repulsion_weight`
  Base coefficient for similarity repulsion.
- `repulsion_growth_factor`
  Geometric growth factor for repulsion strength across fallback stages.
- `max_repulsion_memory`
  Maximum number of failed-memory graph embeddings kept in the repulsion bank.
- `allow_self_loops`
  Whether self-loop edge additions are allowed.
- `target_lambda`
  Weight applied to the requested target-class probability during generation.

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
- best target score
- best repulsion-adjusted selection score
- best repulsion value
- active target lambda
- active repulsion lambda
- remaining edges
- per-step time
- ETA
- active beam limit
- fallback count when a rollback happens

If early stopping is enabled, it is evaluated only in the final search phase.
Earlier phases still have to exhaust their normal beam expansion and fallback
logic before a final-feasible shorter graph can terminate the run.

Fallback transitions are logged explicitly, including:

- fallback index
- rollback distance
- target depth
- widened beam limit
- current `edge_risk_training_set_size` when an edge-risk estimator is active

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
