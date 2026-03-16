# Conditional Autoregressive Generation

This note explains the active conditional autoregressive generator implemented in:

- `abstractgraph_generative.conditional`
- `abstractgraph_generative.conditional_batch`

The goal of the model is to generate a new base graph while respecting a
target interpretation-graph structure learned through `AbstractGraph`
decomposition.

## Core Idea

The generator does not create a graph node-by-node from scratch. Instead, it:

1. decomposes training graphs into interpretation nodes and their mapped base
   subgraphs,
2. stores those mapped subgraphs as reusable components,
3. learns how components connect through shared boundary nodes,
4. generates a new graph by selecting components that match the target
   interpretation graph and merging them consistently.

The target interpretation graph acts as a structural scaffold. The generator
fills in that scaffold with concrete base-graph components taken from training
examples.

## Two Spaces

There are two graph spaces in the design:

- `base graph`
  The concrete graph we ultimately want to generate.

- `interpretation graph`
  The structural graph produced by the decomposition function. Each
  interpretation node represents a subgraph of the base graph through its
  `mapped_subgraph`.

An `AbstractGraph` is the pair of these two objects: the base graph and its
interpretation graph.

During training, each interpretation-node occurrence becomes one reusable
component. During generation, the solver chooses one component for each target
interpretation node.

## Training Phase

During `fit(graphs)` the generator processes each training graph as follows:

1. Convert the graph into an `AbstractGraph`.
2. Read its interpretation graph.
3. For each interpretation node:
   - extract the mapped base subgraph,
   - compute an interpretation-node signature,
   - record one port for each incident interpretation edge,
   - hash the boundary anchor types for each port.

The result is a library of `ComponentInstance` objects.

Each component stores:

- `comp_id`: unique component id,
- `img_type`: hash representing the local interpretation-node context,
- `deg`: interpretation-node degree,
- `subgraph`: the reusable base-graph fragment,
- `ports`: the exposed interfaces used to connect this component to neighbors.

## Retrieval Indexes

The fitted generator builds two indexes.

### Bucket Index

The coarse retrieval structure is a bucket keyed by:

- `(img_type, degree)`

This says: for a target interpretation node with a given abstract type and
degree, these are the candidate components that could fit.

### Inverted Anchor Index

The second index is more selective. It is keyed by:

- `(img_type, degree, anchor_type, multiplicity)`

This index supports pruning. If a partially built graph already requires a
boundary with specific anchor types, the solver intersects the relevant posting
lists before doing exact matching.

This reduces expensive search over clearly incompatible components.

## Generation Phase

Generation takes a target interpretation graph and tries to materialize a
consistent base graph.

At a high level, the solver repeats:

1. pick a target interpretation node to assign,
2. derive boundary requirements from already-assigned neighbors,
3. retrieve candidate components from the indexes,
4. test exact port matching,
5. materialize one candidate,
6. unify matched anchors,
7. continue or backtrack.

This is an autoregressive process because each decision is conditioned on the
partial graph built so far.

## Why "Conditional"

The process is conditional in two senses:

- it is conditioned on a target interpretation graph,
- each local choice is conditioned on already committed neighboring structure.

The generator is not sampling arbitrary components independently. It is solving
a constrained assembly problem.

## Attributed Variant

There is also a context-aware variant implemented in:

- `abstractgraph_generative.conditional_attributed`

This class inherits from the simplified
`ConditionalAutoregressiveGenerator` and keeps the same component
extraction, boundary matching, anchor unification, feasibility filtering, and
backtracking machinery. The difference is in how legal rewiring choices are
ranked before one branch is explored.

### Additional Inputs

The attributed variant adds:

- `context_vectorizer`
  A graph vectorizer with a `transform(graphs)` method that maps graphs to
  embedding vectors.

- `preimage_context_radius`
  The radius used to extract local base-graph context around anchor nodes.

- `num_context_rewirings`
  A cap on how many legal rewiring branches are materialized and scored before
  one is sampled.

There is intentionally no `image_context_radius` in this variant. The added
signal comes only from the base-graph-side attributed context.

### Training-Time Context Embeddings

During `fit(graphs)`, after the usual component library has been extracted, the
generator computes one extra embedding for each component:

1. collect the anchor nodes of that component in the training base graph,
2. extract the union of radius-`preimage_context_radius` neighborhoods centered
   on those anchors,
3. vectorize that union subgraph with `context_vectorizer`,
4. store the resulting embedding alongside the component id.

So every reusable mapped subgraph now carries not only its structural interface
(ports and anchor types), but also a summary of the base-graph context in
which it originally appeared.

### Generation-Time Context Scoring

At generation time, the solver still:

1. selects the next target interpretation node,
2. retrieves structurally legal candidate components,
3. checks exact anchor compatibility.

But after that, instead of immediately following the default candidate order, it
does the following:

1. materialize a bounded number of legal candidate rewiring branches,
2. for each branch, look at the anchors that are currently instantiated in the
   partial graph,
3. extract the union of radius-`preimage_context_radius` neighborhoods around
   those anchors in the current partially built graph,
4. vectorize that partial context,
5. compute cosine similarity against the stored training-time embedding of the
   candidate mapped subgraph,
6. convert the similarity into a non-negative sampling weight,
7. sample one branch with probability proportional to that weight.

If the current partial graph looks more similar to the original context in
which that mapped subgraph was observed during training, that rewiring is more
likely to be chosen.

### Why This Helps

The base generator enforces structural compatibility through interpretation-node type,
degree, and anchor multiset matching. That is necessary, but it can still leave
multiple legal components that are locally interchangeable from a pure boundary
point of view.

The attributed variant adds a softer preference:

- not just "can this component connect here?"
- but also "does this component appear in a base-graph context similar to the one
  we have built so far?"

This biases generation toward reusing associations in contexts that resemble
their original training environments, even when the current graph is still only
partially instantiated.

## Boundary Matching

The central problem is deciding whether a candidate component can connect to the
already materialized part of the graph.

Each assigned neighbor induces a boundary requirement:

- which global nodes are currently exposed,
- which anchor types those nodes have,
- which multiset of anchor types must be satisfied.

Each candidate component exposes ports. A port carries:

- a stable local slot id,
- a list of local anchor nodes,
- a list of aligned anchor types.

Matching works in two stages:

1. multiset containment:
   the port must contain at least the required anchor types,
2. concrete pairing:
   local anchor nodes are paired with the required global anchor nodes.

The solver also enforces injectivity: different requirements cannot reuse the
same port, and inconsistent pairings are rejected.

### What A Requirement Really Means

Suppose the target interpretation graph contains three interpretation nodes:

- `X`
- `Y`
- `Z`

Assume:

- `X` has already been assigned a component,
- `Y` has already been assigned a component,
- `Z` is the next interpretation node we want to materialize.

Also assume that in the target interpretation graph:

- `Z` is connected to `X`,
- `Z` is connected to `Y`.

Then `Z` must connect correctly to the partial graph already created by the
components of `X` and `Y`.

This produces two boundary requirements for `Z`:

- one requirement induced by neighbor `X`,
- one requirement induced by neighbor `Y`.

Each requirement says:

- which already-existing global nodes are exposed on that boundary,
- which anchor types those global nodes currently have,
- therefore which anchor-type multiset the new component for `Z` must satisfy.

### Concrete Example

Assume the partial working graph already contains these exposed boundary nodes:

- from the side of `X`: global nodes `(100, 101)`
- from the side of `Y`: global node `(205,)`

Assume their current anchor types are:

- node `100` has type `7`
- node `101` has type `11`
- node `205` has type `7`

Then the requirements for `Z` are:

- requirement from `X`
  - `global_nodes = (100, 101)`
  - `global_node_types = (7, 11)`
  - `required_types = Counter({7: 1, 11: 1})`

- requirement from `Y`
  - `global_nodes = (205,)`
  - `global_node_types = (7,)`
  - `required_types = Counter({7: 1})`

Now suppose one candidate component for `Z` has two ports:

- `port 0`
  - local anchor nodes `(0, 3)`
  - anchor types `(7, 11)`

- `port 1`
  - local anchor nodes `(2,)`
  - anchor types `(7,)`

This candidate is compatible:

- `port 0` can satisfy the requirement from `X`,
- `port 1` can satisfy the requirement from `Y`.

After matching, the concrete anchor pairs are:

- `(local 0 -> global 100)`
- `(local 3 -> global 101)`
- `(local 2 -> global 205)`

So when the component is materialized:

- the copied local node corresponding to local anchor `0` will be unified with
  global node `100`,
- the copied local node corresponding to local anchor `3` will be unified with
  global node `101`,
- the copied local node corresponding to local anchor `2` will be unified with
  global node `205`.

This is how the new component becomes attached to the graph built so far.

### Why Multisets Matter

The solver does not merely check whether a port contains the right anchor types
"in spirit". It checks multiplicities exactly.

For example:

- required types `Counter({7: 2, 11: 1})`

can be matched by a port with:

- anchor types `(7, 11, 7)`

but cannot be matched by:

- `(7, 11)`
- `(7, 7)`
- `(11, 11, 7)`

So anchor matching is multiset containment, not set overlap.

### Why Two Requirements Cannot Reuse The Same Port

Suppose `Z` has two already-assigned neighbors, so it has two distinct
requirements.

Even if one port contains enough anchor types to satisfy both requirements in
the abstract, the solver does not allow both requirements to map to the same
port.

Why:

- each port represents one interface corresponding to one incident image edge,
- using the same port twice would collapse two distinct image-graph adjacencies
  into one interface,
- that would break the intended decomposition semantics.

So the solver requires an injective requirement-to-port assignment:

- requirement `r0` must go to some port `p0`,
- requirement `r1` must go to some different port `p1`.

### Why Concrete Pairing Is Needed After Type Matching

Even after the type multiset matches, the generator still has to decide which
specific local anchor node is paired with which specific global anchor node.

This matters because:

- two anchors may share the same type,
- a boundary may contain repeated types,
- later unification must happen at the node level, not just at the type level.

For example, if:

- required global types are `(7, 7)`
- a port exposes local anchor types `(7, 7)`

then type matching alone only says "this could work".

The solver still has to build actual node pairs, such as:

- `(local 4 -> global 100)`
- `(local 5 -> global 205)`

Those pairings are then checked for consistency across all requirements. If one
global anchor would force two incompatible local anchors, the whole candidate is
rejected.

### Failure Example

Suppose the requirement from `X` is:

- `required_types = Counter({7: 1, 11: 1})`

and a candidate port has:

- anchor types `(7, 7)`

This candidate fails immediately because:

- it has no anchor of type `11`,
- so multiset containment fails.

Another failure case:

- requirement from `X` needs `Counter({7: 2})`
- requirement from `Y` needs `Counter({7: 1})`
- the candidate has only one remaining port with anchor types `(7, 7, 7)`

Even though that one port contains enough type-`7` anchors overall, it still
cannot satisfy both requirements at once, because the two requirements would be
forced to reuse the same port.

### Mental Picture

A useful way to think about boundary matching is:

- a requirement describes an already-open socket on the partial graph,
- a port describes a socket exposed by a candidate component,
- anchor types describe the shape of the pins in that socket,
- matching checks that the pin counts and pin types fit,
- pairing decides exactly which pin on the new component is welded to which pin
  on the existing graph.

Only after all of that succeeds can the component be safely materialized and
unified into the working graph.

## Materialization and Unification

Once a candidate is selected, the generator:

1. copies the component subgraph into the working graph,
2. obtains a local-to-global node map,
3. unifies matched anchor pairs,
4. rewrites cached references after contractions,
5. records new boundary bindings for unassigned neighbors.

Unification is crucial. It means that a node introduced by the new component and
a node that already exists in the working graph are merged into one
representative node when they correspond to the same boundary anchor.

If unification behaves unexpectedly or fails to contract as required, that
branch is rejected.

## Search Strategy

The generator uses a backtracking search.

Important heuristics:

- `frontier-first`
  Prefer unassigned interpretation nodes that touch already assigned neighbors.

- `fail-first`
  Prefer nodes with more assigned neighbors and stronger constraints.

- `rare bucket seeding`
  If nothing is assigned yet, prefer a node whose `(img_type, degree)` bucket is
  rare, because that reduces branching earlier.

- `forward checking`
  After a tentative commit, verify that every frontier node still has at least
  one candidate. If not, prune immediately.

This makes the generator a constrained combinatorial search procedure rather
than a purely neural decoder.

## Feasibility Filtering

An optional feasibility estimator can be attached during `fit()`.

Its role is not to construct the graph directly. Instead, it filters generated
graphs after construction and removes outputs that violate learned constraints.

This gives two layers of control:

- local structural consistency from anchor and port matching,
- global plausibility from the feasibility estimator.

## Limitations

The generator is structurally disciplined, but it is not complete. There are
important situations where valid-looking target interpretation graphs are still hard or
impossible to realize.

### Nearby Anchors Can Create Ordering Deadlocks

Anchor matching is based on local neighborhoods in the current partially
materialized graph.

This creates a practical limitation:

- if two anchors are close enough that each anchor's hash context depends on the
  other being already present,
- and the required context radius is controlled by `preimage_cut_radius`,
- then one anchor may fail to match because the other one has not yet been
  materialized.

In other words, when anchors lie within each other's relevant neighborhood, the
generator can face a circular dependency:

- anchor `a` needs anchor `b` to already exist to get the right current hash,
- anchor `b` needs anchor `a` for the same reason.

The backtracking search can sometimes work around this by choosing a different
component ordering, but not always. As `preimage_cut_radius` increases, this
kind of dependency becomes more common because anchor compatibility becomes more
context-sensitive.

This is why larger `preimage_cut_radius` values may improve local specificity
but also make generation more brittle.

### Local Compatibility Does Not Guarantee Global Validity

The generator makes decisions using mostly local information:

- interpretation-node type,
- interpretation-node degree,
- boundary anchor-type multisets,
- local port compatibility.

Those constraints are often sufficient to build a graph that is locally
consistent, but they may still be too weak to enforce the global properties you
actually care about.

Typical failure mode:

1. the search finds a sequence of components that match all local boundaries,
2. the assembled graph is syntactically coherent,
3. the final graph violates a global constraint,
4. the feasibility estimator removes it.

This means that if local constraints are weak, generation may spend a lot of
effort constructing graphs that are later discarded.

In practice, this often shows up as:

- many successful constructions but low post-filter retention,
- repeated rejection of outputs that "almost" fit,
- strong sensitivity to the quality of the decomposition function,
- strong sensitivity to the expressiveness of anchor hashes.

### Training Support Limits What Can Be Realized

The generator does not invent arbitrary new interfaces. It recombines component
interfaces seen in training.

So even if a target interpretation graph is valid in principle, it may be unrealizable
in practice when:

- the needed `(img_type, degree)` bucket is sparse,
- the required anchor-type multiplicities are rare,
- the training set never exposed enough compatible ports for a particular
  assembly pattern.

This is a support problem rather than a search bug: the model can only assemble
what its component library makes available.

### Search Heuristics Can Fail Before a Valid Construction Is Found

The solver uses bounded backtracking and forward checking. This is pragmatic but
incomplete.

As a result:

- a realizable target interpretation graph may still fail within the attempt budget,
- some targets are intrinsically harder because they create wider branching,
- some failures come from search complexity rather than impossible constraints.

So differences in success frequency across target interpretation graphs do not
necessarily imply hidden bias. They can arise from genuine differences in search
difficulty.

### Pool Frequency Still Matters

Even with permutation-invariant matching inside the solver, generation still
samples target interpretation graphs from the available interpretation-graph pool.

This means two distinct effects must be separated:

- `proposal frequency`: how often a target interpretation graph is attempted,
- `realization difficulty`: how often that target succeeds and survives
  feasibility filtering.

If one image-graph pattern appears more often in the pool, it may also appear
more often in final outputs even when the solver itself is unbiased with respect
to concrete node ids.

## Dataset-Level Wrapper

`ConditionalAutoregressiveGraphsGenerator` is a thin orchestration layer built
on top of the base generator.

It:

- splits training graphs by target class,
- fits one generator per class,
- generates graphs class-wise,
- predicts targets for generated graphs with a graph estimator.

This is useful when generation should preserve or target a label distribution.

## Practical Mental Model

A good mental model is:

- the interpretation graph is the blueprint,
- each component is a reusable building block,
- each port is a doorway on that block,
- each anchor is a typed attachment point on that doorway,
- generation is the process of selecting blocks and snapping matching doorways
  together while keeping the whole assembly consistent.

## Glossary

- `AbstractGraph`
  A pair of graphs: a base graph and an interpretation graph, where
  interpretation nodes carry mapped base subgraphs.

- `anchor`
  A boundary node used to connect one component to already materialized graph
  structure. Anchors are the concrete nodes that get unified across components.

- `anchor type`
  A hashed description of an anchor's local neighborhood. It is used to decide
  whether two anchors are compatible.

- `mapped subgraph`
  The base subgraph stored on an interpretation node. It is the concrete
  fragment represented by that interpretation node.

- `backtracking`
  Search procedure that undoes a tentative decision when it leads to
  inconsistency or dead ends.

- `boundary`
  The exposed interface between a component and the rest of the graph. In this
  implementation it is represented through anchors stored on ports and edge
  bindings.

- `boundary requirement`
  A constraint induced by an already assigned neighbor, describing which anchor
  types and global nodes a new component must connect to.

- `bucket`
  A coarse candidate set keyed by `(img_type, degree)`. Buckets group reusable
  components by target interpretation-node signature.

- `component`
  A reusable base-graph subgraph extracted from one training
  interpretation-node occurrence.

- `ComponentInstance`
  The concrete stored record for one reusable component, including its
  subgraph, interpretation signature, and ports.

- `degree`
  Here, usually the degree of an interpretation node in the target
  interpretation graph or in a training interpretation graph.

- `edge binding`
  Stored payload for an interpretation edge during generation. It records the
  currently exposed global boundary nodes and the required anchor-type multiset.

- `feasibility estimator`
  Optional model that filters generated graphs based on learned global
  constraints.

- `frontier`
  The set of unassigned target interpretation nodes adjacent to already
  assigned nodes. These nodes are the most constrained next choices.

- `interpretation graph`
  The structural graph produced by decomposition. Its nodes represent mapped
  base subgraphs.

- `img_type`
  Hash summarizing the local interpretation-node context. It is part of the
  coarse retrieval key.

- `injective port assignment`
  Matching rule requiring that distinct boundary requirements map to distinct
  ports on the candidate component.

- `inverted index`
  Secondary retrieval structure keyed by anchor-type information, used to prune
  bucket candidates before exact matching.

- `materialization`
  Copying a component subgraph into the working graph before anchor unification.

- `multiplicity`
  The number of times a given anchor type appears in a boundary multiset.

- `port`
  A component interface corresponding to one incident interpretation edge. A
  port exposes anchor nodes and anchor types that can be matched to neighbors.

- `Port`
  The dataclass storing one port's slot id, local anchor nodes, and anchor
  types.

- `base graph`
  The concrete graph being generated.

- `target interpretation graph`
  The interpretation graph given to the generator as the structure it should
  realize in base-graph space.

- `unification`
  Merging two graph nodes that represent the same anchor during assembly.

- `working graph`
  The partially constructed base graph maintained during one generation
  attempt.

## Public Entry Points

Use:

```python
from abstractgraph_generative.conditional import (
    ConditionalAutoregressiveGenerator,
)
from abstractgraph_generative.conditional_batch import (
    ConditionalAutoregressiveGraphsGenerator,
)
```

Legacy helper utilities from the older implementation remain in:

```python
from abstractgraph_generative.legacy.conditional_v0_1 import ...
```
