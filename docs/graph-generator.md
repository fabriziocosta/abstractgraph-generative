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
generator retrieves a local stored-interpretation neighborhood, fits
`EdgeGenerator` on that neighborhood, prunes edges from the seed interpretation
graph, and regrows a new interpretation graph.

If `interpretation_edge_removal_size=0`, the edge stage is bypassed: the sampled
seed interpretation graph is used directly as the conditional target.

The edge-stage neighborhood uses the same label-coverage principle as
`EdgeGenerator.repair(...)`: candidates are scanned in nearest-neighbor order,
duplicate interpretation graphs are removed with `hash_graph(...)`, neighbors
that cover missing seed interpretation-node labels are selected first, and
remaining slots are filled with nearest unused candidates. The seed
interpretation graph is not used as evidence for its own feasibility. If no
stored interpretation-neighbor context covers the sampled seed's labels, that
seed is rejected and sampling continues with another seed.

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
    debug=False,
    require_new_interpretation_graph=True,
    max_same_interpretation_retries=3,
)

generator.store(graphs, interpretation_graphs=interpretation_graphs)

samples = generator.sample(
    n_samples=4,
    n_interpretation_neighbors=30,
    n_conditional_neighbors=30,
    n_instances_per_sample=2,
    interpretation_edge_removal_size=0.5,
    random_state=0,
    max_seed_attempts=None,
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
- `debug`
  If true, propagate debug/progress logging to the wrapped generators.
  `ConditionalAutoregressiveGenerator.debug` is set directly, and
  `EdgeGenerator.verbose` is enabled for edge-stage logs. The edge-stage
  neighbor log includes `neighbor_indices` and their retrieval
  `neighbor_distances` from the sampled seed interpretation graph.
- `require_new_interpretation_graph`
  If true, reject edge-stage outputs whose generated interpretation graph is
  identical to the sampled seed interpretation graph. This does not affect the
  explicit `interpretation_edge_removal_size=0` bypass, which intentionally
  uses the seed interpretation graph directly.
- `max_same_interpretation_retries`
  Number of additional edge-generation attempts to make for the same seed when
  `require_new_interpretation_graph=True` and the edge stage regenerates the
  seed interpretation graph. The default `3` allows four total attempts before
  the seed is skipped.

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

`sample(...)` treats `n_samples` as the number of successful interpretation
targets requested. Since seeds can be skipped, the generator may try more seed
indices than `n_samples`. By default, `max_seed_attempts=None` allows up to
`10 * n_samples` distinct seed attempts, capped by the stored corpus size. Set
`max_seed_attempts` explicitly to make this stricter or more exhaustive.

For each attempted seed, `sample(...)` does the following:

1. sample one stored interpretation graph index,
2. retrieve an interpretation-neighbor context using repair-style label
   coverage:
   - scan candidates in nearest-neighbor order,
   - deduplicate candidates by `abstractgraph.hashing.hash_graph(...)`,
   - select neighbors that cover missing seed interpretation-node labels,
   - fill any remaining slots up to `n_interpretation_neighbors`,
   - reject the seed if its labels cannot be covered by other stored
     interpretation graphs,
3. fit `EdgeGenerator` on those interpretation graphs,
4. prune the seed interpretation graph with `remove_edges(...)`,
5. regrow one generated interpretation graph to the seed's original edge count,
6. retrieve `n_conditional_neighbors` stored interpretation graphs nearest to
   the generated interpretation graph,
7. fit `ConditionalAutoregressiveGenerator` on the aligned base graphs,
8. generate `n_instances_per_sample` base graphs for that generated
   interpretation graph,
9. discard conditional outputs whose re-decomposed interpretation graph does not
   match the generated interpretation graph.

When `interpretation_edge_removal_size=0`, steps 2 through 5 are skipped and the
seed interpretation graph is used directly in step 6.
When `interpretation_edge_removal_size=1.0`, all edges are removed from the seed
interpretation graph before regrowth.

The returned value is a flat `list[nx.Graph]`. With
`n_samples=4` and `n_instances_per_sample=2`, up to eight base graphs are
returned. Failed edge-stage seeds, unsupported-label seeds, and unchanged
edge-stage outputs that exhaust their retry budget are skipped with a
`RuntimeWarning`; the generator then tries another distinct seed until it has
`n_samples` successful interpretation targets or exhausts `max_seed_attempts`.
The final count may still be smaller if too many attempts fail.

`conditional_generate_kwargs` are forwarded to
`ConditionalAutoregressiveGenerator.generate(...)`, except `n_samples` and
`interpretation_graphs`, which are controlled by `GraphGenerator`.

`edge_generate_kwargs` are forwarded to `EdgeGenerator.generate(...)`, except
`return_path`, which is controlled internally.

## Bookkeeping

After `sample(...)`, these attributes describe the last run:

- `last_sampled_indices_`
  Attempted seed indices, including skipped seeds.
- `last_successful_sampled_indices_`
  Seed indices that reached generated base-graph output. Use this for notebook
  displays aligned with `generated_graphs`.
- `last_interpretation_neighbor_indices_history_`
- `last_interpretation_neighbor_distances_history_`
- `last_conditional_neighbor_indices_history_`
- `last_seed_graphs_`
- `last_seed_interpretation_graphs_`
- `last_generated_interpretation_graphs_`
- `last_edge_generation_paths_`
- `last_conditional_training_graphs_history_`

The successful-stage histories only receive entries after their stage succeeds.
These are useful for notebooks and diagnostics. For example:

```python
attempted_seed_graphs = [
    generator.stored_graphs_[i] for i in generator.last_sampled_indices_
]
successful_seed_graphs = [
    generator.stored_graphs_[i] for i in generator.last_successful_sampled_indices_
]
generated_targets = generator.last_generated_interpretation_graphs_
training_sets = generator.last_conditional_training_graphs_history_
```

`last_seed_graphs_`, `last_seed_interpretation_graphs_`, and
`last_generated_interpretation_graphs_` are aligned by successful
interpretation target. Generated base graphs are returned flat; group them by
`n_instances_per_sample` to align them with each successful target.

## Interpretation Labels

The edge stage requires the local interpretation-neighbor context to cover the
sampled seed's interpretation-node labels. This is especially important with
label modes such as `"histogram"` or `"histogram_values"`, where labels can be
more specific than `"operator_hash"` labels. A seed with rare labels is skipped
unless those labels appear in other stored interpretation graphs. This avoids
fitting a feasibility estimator on a neighborhood that would necessarily mark
the seed's own structure infeasible.

This label-coverage requirement applies only when the edge stage runs. With
`interpretation_edge_removal_size=0`, no edge generator is fit, so the seed's
interpretation graph can be passed directly to the conditional stage.

The conditional stage preserves the target interpretation graph through the
existing `ConditionalAutoregressiveGenerator` postcondition. That means every
generated base graph must re-decompose to the generated interpretation graph
under the same `decomposition_function`, `nbits`, and `label_mode`.
`GraphGenerator` also applies this check defensively before recording histories
and returning generated base graphs.

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
