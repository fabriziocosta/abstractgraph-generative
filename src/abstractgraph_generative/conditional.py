"""Simplified conditional autoregressive generator.

`ConditionalAutoregressiveGenerator` learns reusable base-graph components
from training graphs, then assembles new graphs by conditioning on target
interpretation graphs and matching boundary anchors with backtracking.

At fit time, each interpretation-node occurrence is converted into a
`ComponentInstance` with:
- a local mapped base subgraph,
- one `Port` per incident interpretation edge,
- anchor hashes (`anchor_types`) aligned with local anchor nodes.

The fitted model builds two retrieval structures:
- a bucket index keyed by `(image_node_type, degree)`,
- an inverted index keyed by `(image_node_type, degree, anchor_type, multiplicity)`
  to prune candidates quickly from partial boundary constraints.

At generation time, the solver repeatedly:
1. selects a frontier interpretation node,
2. builds requirements from already-assigned neighbors,
3. retrieves and matches candidate components with injective port assignment,
4. commits one candidate by materializing and unifying anchors,
5. backtracks on any inconsistency.

Matching is multiset-based, so boundaries may carry multiple anchors (including
repeated types). State updates are fail-closed: malformed boundary payloads,
missing anchors, or non-contracting unifications reject the current branch.

Detailed execution model
------------------------
Training decomposition:
- Each training graph is converted to an `AbstractGraph` with a base graph
  and an interpretation graph.
- For every interpretation node, the mapped base subgraph becomes one
  component template (`ComponentInstance`).
- For each incident interpretation edge, the shared base nodes define a port
  boundary (`Port.anchor_local_nodes`) and aligned anchor hashes
  (`Port.anchor_types`).
- Components that violate the decomposition invariant (interpretation edge with no
  anchors) are rejected during fitting.

Retrieval and pruning:
- Candidate retrieval starts from a coarse signature key
  `(img_type, degree)`.
- Requirements induced by already-assigned neighbors are converted into anchor
  type multisets (`Counter`).
- The inverted index intersects candidates by rare required
  `(anchor_type, multiplicity)` keys before exact matching.
- Exact matching then enforces an injective requirement-to-port assignment.

Boundary semantics:
- A requirement stores `global_nodes` and aligned `global_node_types`.
- Matching first checks multiset containment
  (`Counter(port.anchor_types) >= Counter(required_types)`), then creates
  concrete local/global anchor pairs type-by-type.
- Pairing within each type bucket is deterministic to reduce search variance.
- Any length mismatch, duplicate boundary node, missing node, or inconsistent
  typed payload is treated as invalid and fails closed.

Commit and unification:
- `materialize_component` inserts a fresh copy of the chosen component into the
  working graph and returns a local-to-global node map.
- Anchor pairs are unified via `unify_anchors`.
- After each unification, cached references in `node_maps` and `edge_bindings`
  are rewritten through `_replace_global_id_everywhere` to prevent stale ids.
- If unification does not exhibit contraction behavior compatible with a single
  representative node, the branch is rejected.

Search strategy:
- Node selection is fail-first on the assigned frontier; when no frontier is
  available, seeding favors the rarest `(img_type, degree)` bucket.
- The solver branches over candidate components and, independently, over
  possible future neighbor-to-port assignments.
- `_frontier_has_candidate` performs forward-checking to prune dead branches
  early.
- Backtracking is bounded by `max_backtracks`; generation retries are bounded
  by attempt budgets in `generate()`.

Feasibility filtering:
- An optional feasibility estimator can be fitted during `fit()`.
- Generated graphs are filtered in batch (or per graph fallback) before being
  returned.
- Debug mode reports attempts, constructed candidates, filtered fraction, and
  final yield for diagnosis.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import copy
import os
from pprint import pformat
import random
import time
import warnings
from typing import Optional, Sequence

import networkx as nx

from abstractgraph.graphs import AbstractGraph, get_mapped_subgraph, graph_to_abstract_graph
from abstractgraph.hashing import hash_graph
from abstractgraph_generative.rewrite import (
    anchor_type_current,
    anchor_type_train,
    extract_ball,
    materialize_component,
    unify_anchors,
)


@dataclass(frozen=True)
class Port:
    """Interface definition for one incident image edge.

    Args:
        edge_slot: Stable local port index for the component.
        anchor_local_nodes: Local node ids in the component subgraph.
        anchor_types: Training-time anchor-type hashes aligned with local nodes.
        anchor_order_keys: Structural ordering keys aligned with local nodes.
    """

    edge_slot: int
    anchor_local_nodes: tuple
    anchor_types: tuple[int, ...]
    anchor_order_keys: tuple[tuple, ...]


@dataclass(frozen=True)
class ComponentInstance:
    """Swappable unit extracted from one interpretation-node occurrence.

    Args:
        comp_id: Unique component id.
        img_type: Interpretation-neighborhood hash bucket.
        deg: Interpretation-node degree.
        subgraph: Local mapped base-graph component.
        ports: Port definitions for all incident interpretation edges.
    """

    comp_id: int
    img_type: int
    deg: int
    subgraph: nx.Graph
    ports: tuple[Port, ...]


@dataclass
class _BoundaryRequirement:
    """Boundary constraints induced by already assigned neighbors.

    Args:
        neighbor: Assigned neighbor interpretation node.
        global_nodes: Current materialized boundary nodes.
        global_node_types: Anchor types aligned with ``global_nodes``.
        global_order_keys: Structural ordering keys aligned with ``global_nodes``.
        required_types: Required anchor-type multiset at generation time.
    """

    neighbor: object
    global_nodes: tuple
    global_node_types: tuple[int, ...]
    global_order_keys: tuple[tuple, ...]
    required_types: Counter


@dataclass
class _CandidateAssignment:
    """Concrete candidate with matched ports and anchor pairs.

    Args:
        component: Chosen component instance.
        req_to_port: Mapping from requirement index to port index.
        req_anchor_pairs: Mapping from requirement index to local/global anchor pairs.
    """

    component: ComponentInstance
    req_to_port: dict[int, int]
    req_anchor_pairs: dict[int, list[tuple[object, object]]]
    future_port_assignment: Optional[dict[object, int]] = None


@dataclass
class _GenerationState:
    """Mutable generation state for one target interpretation graph.

    Args:
        target_image: Target interpretation graph.
        target_signatures: Node signatures with interpretation hashes and degrees.
        graph: Current materialized base graph.
        assigned: Assigned flags per target interpretation node.
        comp_of: Selected component id per target interpretation node.
        node_maps: Local-to-global mapping per assigned interpretation node.
        edge_bindings: Boundary payload per interpretation edge. Each entry stores:
            - ``global_nodes``: boundary global node ids
            - ``required_types``: stable anchor-type multiset for matching
    """

    target_image: nx.Graph
    target_signatures: dict
    graph: nx.Graph
    assigned: dict
    comp_of: dict
    node_maps: dict
    edge_bindings: dict


_WORKER_GENERATOR: Optional["ConditionalAutoregressiveGenerator"] = None
_GRAPH_HASH_NBITS = 19


def _warn_deprecated_name(old_name: str, new_name: str) -> None:
    warnings.warn(
        f"`{old_name}` is deprecated and will be removed in a future release; use `{new_name}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _resolve_alias(
    *,
    canonical_name: str,
    canonical_value,
    deprecated_name: str,
    deprecated_value,
    default,
):
    if canonical_value is not None and deprecated_value is not None and canonical_value != deprecated_value:
        raise ValueError(
            f"Conflicting values provided for `{canonical_name}` and deprecated alias "
            f"`{deprecated_name}`."
        )
    if canonical_value is not None:
        return canonical_value
    if deprecated_value is not None:
        _warn_deprecated_name(deprecated_name, canonical_name)
        return deprecated_value
    return default


def _init_generate_worker(generator: "ConditionalAutoregressiveGenerator") -> None:
    """Initialize per-process generator snapshot for sample generation."""
    global _WORKER_GENERATOR
    _WORKER_GENERATOR = generator


def _generate_one_worker(
    target_interpretation_graph: nx.Graph,
    seed: int,
    max_backtracks: int,
) -> tuple[Optional[nx.Graph], dict]:
    """Worker entrypoint for parallel sample generation."""
    if _WORKER_GENERATOR is None:
        raise RuntimeError("Worker generator is not initialized.")
    rng = random.Random(int(seed))
    attempt_trace: dict = {}
    generated = _WORKER_GENERATOR._generate_one(
        target_interpretation_graph,
        rng,
        max_backtracks=int(max_backtracks),
        attempt_trace=attempt_trace,
    )
    return generated, attempt_trace


class ConditionalAutoregressiveGenerator:
    """Simplified conditional autoregressive generator with fit/generate API."""

    def __init__(
        self,
        *,
        decomposition_function,
        nbits: int,
        feasibility_estimator=None,
        base_cut_radius: Optional[int] = None,
        interpretation_cut_radius: Optional[int] = None,
        preimage_cut_radius: Optional[int] = None,
        image_cut_radius: Optional[int] = None,
        n_jobs: int = -1,
        debug: bool = False,
        debug_level: int = 1,
    ):
        """Initialize the simplified generator.

        Args:
            decomposition_function: Decomposition function for AbstractGraph conversion.
            nbits: Hash bit width for hashing base and interpretation neighborhoods.
            feasibility_estimator: Optional final-graph feasibility estimator.
            base_cut_radius: Canonical radius for anchor neighborhood hashes.
            interpretation_cut_radius: Canonical radius for interpretation-node neighborhood hashes.
            preimage_cut_radius: Deprecated alias for ``base_cut_radius``.
            image_cut_radius: Deprecated alias for ``interpretation_cut_radius``.
            n_jobs: Reserved for API compatibility.
            debug: If True, print generation diagnostics.
            debug_level: Debug verbosity level.
                - 0: disabled
                - 1: summaries
                - 2: detailed per-attempt/per-node traces

        Returns:
            None.
        """
        if decomposition_function is None:
            raise ValueError("decomposition_function is required.")
        if int(nbits) < 2:
            raise ValueError("nbits must be >= 2.")
        self.decomposition_function = decomposition_function
        self.nbits = int(nbits)
        self.feasibility_estimator = feasibility_estimator
        self._base_cut_radius = int(
            _resolve_alias(
                canonical_name="base_cut_radius",
                canonical_value=base_cut_radius,
                deprecated_name="preimage_cut_radius",
                deprecated_value=preimage_cut_radius,
                default=1,
            )
        )
        self._interpretation_cut_radius = int(
            _resolve_alias(
                canonical_name="interpretation_cut_radius",
                canonical_value=interpretation_cut_radius,
                deprecated_name="image_cut_radius",
                deprecated_value=image_cut_radius,
                default=1,
            )
        )
        self.n_jobs = int(n_jobs)
        requested_debug_level = max(0, int(debug_level))
        self.debug = bool(debug) or requested_debug_level > 0
        if self.debug:
            self.debug_level = max(1, requested_debug_level)
        else:
            self.debug_level = 0
        self.label_mode = "operator_hash"

        self._components: dict[int, ComponentInstance] = {}
        self._bucket: dict[tuple[int, int], list[int]] = {}
        self._inv: dict[tuple[int, int, int, int], set[int]] = {}
        self._inv_freq: dict[tuple[int, int, int, int], int] = {}
        self._interpretation_graph_pool: list[nx.Graph] = []
        self._is_fitted: bool = False
        self._zero_generation_streak: int = 0
        self._missing_anchor_sentinel = -1
        self._max_future_port_assignment_branches = 24
        self._fit_skipped_missing_anchor_components: int = 0

    @property
    def base_cut_radius(self) -> int:
        return self._base_cut_radius

    @base_cut_radius.setter
    def base_cut_radius(self, value: int) -> None:
        self._base_cut_radius = int(value)

    @property
    def interpretation_cut_radius(self) -> int:
        return self._interpretation_cut_radius

    @interpretation_cut_radius.setter
    def interpretation_cut_radius(self, value: int) -> None:
        self._interpretation_cut_radius = int(value)

    @property
    def preimage_cut_radius(self) -> int:
        _warn_deprecated_name("preimage_cut_radius", "base_cut_radius")
        return self._base_cut_radius

    @preimage_cut_radius.setter
    def preimage_cut_radius(self, value: int) -> None:
        _warn_deprecated_name("preimage_cut_radius", "base_cut_radius")
        self._base_cut_radius = int(value)

    @property
    def image_cut_radius(self) -> int:
        _warn_deprecated_name("image_cut_radius", "interpretation_cut_radius")
        return self._interpretation_cut_radius

    @image_cut_radius.setter
    def image_cut_radius(self, value: int) -> None:
        _warn_deprecated_name("image_cut_radius", "interpretation_cut_radius")
        self._interpretation_cut_radius = int(value)

    @property
    def interpretation_graph_pool(self) -> list[nx.Graph]:
        return self._interpretation_graph_pool

    @interpretation_graph_pool.setter
    def interpretation_graph_pool(self, value: list[nx.Graph]) -> None:
        self._interpretation_graph_pool = value

    @property
    def image_graph_pool(self) -> list[nx.Graph]:
        _warn_deprecated_name("image_graph_pool", "interpretation_graph_pool")
        return self._interpretation_graph_pool

    @image_graph_pool.setter
    def image_graph_pool(self, value: list[nx.Graph]) -> None:
        _warn_deprecated_name("image_graph_pool", "interpretation_graph_pool")
        self._interpretation_graph_pool = value

    def _dbg(self, level: int, event: str, **fields) -> None:
        """Emit one structured debug line when enabled at requested level.

        Args:
            level: Minimum debug level required for this line.
            event: Event tag.
            **fields: Structured key/value payload.

        Returns:
            None.
        """
        if not self.debug:
            return
        if int(self.debug_level) < int(level):
            return
        ordered_keys = sorted(fields.keys())
        if not ordered_keys:
            print(f"[DEBUG] event={event}")
            return

        # Use a block layout to keep long diagnostics readable.
        print(f"[DEBUG] event={event}")
        for key in ordered_keys:
            value = fields[key]
            if isinstance(value, (list, tuple, dict, set)):
                text = pformat(value, width=100, compact=True, sort_dicts=True)
            else:
                text = str(value)
            if "\n" in text or len(text) > 100:
                print(f"[DEBUG]   {key}=")
                for line in text.splitlines():
                    print(f"[DEBUG]     {line}")
            else:
                print(f"[DEBUG]   {key}={text}")

    @staticmethod
    def _format_min_mean_max(values: list[int]) -> str:
        """Format integer samples as min/mean/max text."""
        if not values:
            return "empty"
        return (
            f"min/mean/max=("
            f"{min(values)}/{(sum(values) / float(len(values))):.2f}/{max(values)})"
        )

    def _debug_print_fit_indexes(self) -> None:
        """Print fitted index statistics when debug mode is enabled."""
        if not self.debug:
            return
        components = list(self._components.values())
        bucket_sizes = [len(ids) for ids in self._bucket.values()]
        inv_member_sizes = [len(ids) for ids in self._inv.values()]
        degrees = [int(comp.deg) for comp in components]
        ports_per_component = [len(comp.ports) for comp in components]
        anchors_per_port = [
            len(port.anchor_types)
            for comp in components
            for port in comp.ports
        ]
        unique_anchor_types = len(
            {
                int(anchor_type)
                for comp in components
                for port in comp.ports
                for anchor_type in port.anchor_types
            }
        )
        inv_multiplicity_counter = Counter(int(key[3]) for key in self._inv.keys())
        anchors_per_preimage_subgraph = [
            sum(len(port.anchor_types) for port in comp.ports)
            for comp in components
        ]
        anchors_per_preimage_subgraph_hist = Counter(
            int(n_anchors) for n_anchors in anchors_per_preimage_subgraph
        )
        top_bucket_counts = sorted(
            ((len(ids), key) for key, ids in self._bucket.items()),
            reverse=True,
        )[:10]
        self._dbg(
            1,
            "fit_dictionaries",
            components=len(self._components),
            bucket_keys=len(self._bucket),
            inv_keys=len(self._inv),
            inv_freq_keys=len(self._inv_freq),
            image_pool=len(self._interpretation_graph_pool),
            skipped_missing_anchor_components=self._fit_skipped_missing_anchor_components,
        )
        self._dbg(
            1,
            "fit_distributions_index",
            bucket_size=self._format_min_mean_max(bucket_sizes),
            inv_members=self._format_min_mean_max(inv_member_sizes),
        )
        self._dbg(
            1,
            "fit_distributions_components",
            component_degree=self._format_min_mean_max(degrees),
            ports_per_component=self._format_min_mean_max(ports_per_component),
            anchors_per_port=self._format_min_mean_max(anchors_per_port),
        )
        self._dbg(
            1,
            "fit_anchors",
            unique_anchor_types=unique_anchor_types,
            inv_multiplicity_hist=dict(sorted(inv_multiplicity_counter.items())),
        )
        if anchors_per_preimage_subgraph_hist:
            self._dbg(
                1,
                "fit_anchor_hist",
                anchors_per_preimage_subgraph_hist=dict(sorted(anchors_per_preimage_subgraph_hist.items())),
            )
        if top_bucket_counts:
            self._dbg(
                1,
                "fit_top_buckets",
                top_buckets=[{"size": size, "key": key} for size, key in top_bucket_counts],
            )

    @staticmethod
    def _counter_contains(container: Counter, required: Counter) -> bool:
        """Return whether a multiset contains another multiset.

        Args:
            container: Candidate multiset.
            required: Required multiset.

        Returns:
            bool: True when all required counts are satisfied.
        """
        for key, count in required.items():
            if container.get(key, 0) < int(count):
                return False
        return True

    def _image_node_type(self, image_graph: nx.Graph, node) -> int:
        """Compute image-node hash under image cut radius.

        Args:
            image_graph: Target image graph.
            node: Image-node id.

        Returns:
            int: Image context hash bucket.
        """
        return hash_graph(
            extract_ball(image_graph, node, self.interpretation_cut_radius),
            nbits=_GRAPH_HASH_NBITS,
        )

    def _image_node_order_key(
        self,
        image_graph: nx.Graph,
        node,
        *,
        target_signatures: Optional[dict] = None,
    ) -> tuple:
        """Build a structural ordering key for an image node.

        Args:
            image_graph: Image graph containing ``node``.
            node: Image-node id.
            target_signatures: Optional cached ``(img_type, degree)`` signatures.

        Returns:
            tuple: Structural key independent of concrete node ids.
        """
        sig = (
            target_signatures.get(node)
            if target_signatures is not None
            else (
                self._image_node_type(image_graph, node),
                int(image_graph.degree(node)),
            )
        )
        refine_radius = max(int(self.interpretation_cut_radius) + 1, 1)
        refine_hash = hash_graph(
            extract_ball(image_graph, node, refine_radius),
            nbits=_GRAPH_HASH_NBITS,
        )
        return (sig[0], sig[1], int(refine_hash))

    def _preimage_node_order_key(self, graph: nx.Graph, node) -> tuple:
        """Build a structural ordering key for a preimage node.

        Args:
            graph: Preimage graph containing ``node``.
            node: Preimage node id.

        Returns:
            tuple: Structural key independent of concrete node ids.
        """
        near_hash = hash_graph(
            extract_ball(graph, node, max(int(self.base_cut_radius), 0)),
            nbits=_GRAPH_HASH_NBITS,
        )
        refine_hash = hash_graph(
            extract_ball(graph, node, max(int(self.base_cut_radius) + 1, 1)),
            nbits=_GRAPH_HASH_NBITS,
        )
        return (int(near_hash), int(refine_hash), int(graph.degree(node)))

    def _current_anchor_order_key(
        self,
        graph: nx.Graph,
        node,
        anchor_type: Optional[int] = None,
    ) -> tuple:
        """Build a structural ordering key for a materialized anchor node.

        Args:
            graph: Current working graph.
            node: Global node id.
            anchor_type: Optional precomputed anchor type.

        Returns:
            tuple: Structural key aligned with anchor matching.
        """
        anchor_type = (
            int(anchor_type_current(graph, node, self.base_cut_radius, nbits=self.nbits))
            if anchor_type is None
            else int(anchor_type)
        )
        return (anchor_type, *self._preimage_node_order_key(graph, node))

    def _build_component_instance(self, ag: AbstractGraph, image_node, comp_id: int) -> ComponentInstance:
        """Build one component instance from an interpretation-node occurrence.

        Args:
            ag: Source AbstractGraph.
            image_node: Image-node id.
            comp_id: Unique component id.

        Returns:
            ComponentInstance: Extracted component payload.
        """
        interpretation_graph = ag.interpretation_graph
        base_graph = ag.base_graph
        mapped_subgraph_u = get_mapped_subgraph(interpretation_graph.nodes[image_node])
        if not isinstance(mapped_subgraph_u, nx.Graph):
            mapped_subgraph_u = nx.Graph()

        global_nodes = sorted(list(mapped_subgraph_u.nodes()), key=lambda n: self._preimage_node_order_key(base_graph, n))
        global_to_local = {g: i for i, g in enumerate(global_nodes)}
        local_subgraph = nx.relabel_nodes(mapped_subgraph_u, global_to_local, copy=True)

        ports_with_keys: list[tuple[tuple, Port]] = []
        for neighbor in interpretation_graph.neighbors(image_node):
            mapped_subgraph_v = get_mapped_subgraph(interpretation_graph.nodes[neighbor])
            if not isinstance(mapped_subgraph_v, nx.Graph):
                mapped_subgraph_v = nx.Graph()
            shared_global = [n for n in global_nodes if n in mapped_subgraph_v]
            # Build local ids and types from the same traversal to guarantee
            # one-to-one alignment between anchor_local_nodes and anchor_types.
            aligned_pairs: list[tuple[object, int]] = []
            for n in shared_global:
                local_n = global_to_local[n]
                anchor_t = int(
                    anchor_type_train(
                        base_graph,
                        n,
                        self.base_cut_radius,
                        nbits=self.nbits,
                    )
                )
                aligned_pairs.append((local_n, anchor_t))
            aligned_pairs = sorted(
                aligned_pairs,
                key=lambda pair: (
                    int(pair[1]),
                    self._preimage_node_order_key(local_subgraph, pair[0]),
                ),
            )
            anchor_local_nodes = tuple(local_n for local_n, _ in aligned_pairs)
            anchor_types = tuple(anchor_t for _, anchor_t in aligned_pairs)
            anchor_order_keys = tuple(
                self._preimage_node_order_key(local_subgraph, local_n)
                for local_n in anchor_local_nodes
            )
            port_key = (
                self._image_node_order_key(interpretation_graph, neighbor),
                tuple(sorted(int(t) for t in anchor_types)),
                tuple(anchor_order_keys),
            )
            ports_with_keys.append(
                (
                    port_key,
                    Port(
                        edge_slot=-1,
                        anchor_local_nodes=anchor_local_nodes,
                        anchor_types=anchor_types,
                        anchor_order_keys=anchor_order_keys,
                    ),
                )
            )
        sorted_ports = sorted(ports_with_keys, key=lambda item: item[0])
        ports = tuple(
            Port(
                edge_slot=int(edge_slot),
                anchor_local_nodes=port.anchor_local_nodes,
                anchor_types=port.anchor_types,
                anchor_order_keys=port.anchor_order_keys,
            )
            for edge_slot, (_key, port) in enumerate(sorted_ports)
        )

        return ComponentInstance(
            comp_id=int(comp_id),
            img_type=self._image_node_type(interpretation_graph, image_node),
            deg=int(interpretation_graph.degree(image_node)),
            subgraph=local_subgraph,
            ports=ports,
        )

    def fit(self, graphs: Sequence[nx.Graph], **_) -> "ConditionalAutoregressiveGenerator":
        """Fit components and retrieval indexes from training graphs.

        Args:
            graphs: Training preimage graphs.
            **_: Ignored compatibility kwargs from previous versions.

        Returns:
            ConditionalAutoregressiveGenerator: Self.
        """
        if graphs is None:
            raise ValueError("graphs is required.")
        graph_list = list(graphs)
        if not graph_list:
            raise ValueError("graphs must be non-empty.")

        abstract_graphs = [
            graph_to_abstract_graph(
                g,
                decomposition_function=self.decomposition_function,
                nbits=self.nbits,
                label_mode=self.label_mode,
            )
            for g in graph_list
        ]

        components: dict[int, ComponentInstance] = {}
        bucket: dict[tuple[int, int], list[int]] = {}
        inv: dict[tuple[int, int, int, int], set[int]] = {}
        inv_freq: dict[tuple[int, int, int, int], int] = {}
        skipped_missing_anchor_components = 0

        comp_id = 0
        image_pool: list[nx.Graph] = []
        for ag in abstract_graphs:
            image_pool.append(ag.interpretation_graph.copy())
            for image_node in ag.interpretation_graph.nodes():
                comp = self._build_component_instance(ag, image_node, comp_id)
                if comp.deg > 0 and any(len(port.anchor_types) == 0 for port in comp.ports):
                    # Enforce decomposition invariant: image-edge interfaces should
                    # expose at least one anchor.
                    skipped_missing_anchor_components += 1
                    comp_id += 1
                    continue
                components[comp_id] = comp
                key = (comp.img_type, comp.deg)
                bucket.setdefault(key, []).append(comp_id)
                for port in comp.ports:
                    counts = Counter(port.anchor_types)
                    for anchor_type, count in counts.items():
                        for multiplicity in range(1, int(count) + 1):
                            inv_key = (comp.img_type, comp.deg, int(anchor_type), int(multiplicity))
                            members = inv.setdefault(inv_key, set())
                            size_before = len(members)
                            members.add(comp_id)
                            if len(members) > size_before:
                                inv_freq[inv_key] = inv_freq.get(inv_key, 0) + 1
                comp_id += 1

        if not components:
            raise ValueError("No components were extracted from training graphs.")

        self._components = components
        self._bucket = bucket
        self._inv = inv
        self._inv_freq = inv_freq
        self._interpretation_graph_pool = image_pool
        self._fit_skipped_missing_anchor_components = int(skipped_missing_anchor_components)
        self._is_fitted = True

        self._fit_feasibility_estimator(graph_list)
        self._debug_print_fit_indexes()

        return self

    def _compute_target_signatures(self, image_graph: nx.Graph) -> dict:
        """Precompute retrieval signatures for a target image graph.

        Args:
            image_graph: Target image graph.

        Returns:
            dict: Mapping image-node -> (img_type, degree).
        """
        out = {}
        for node in image_graph.nodes():
            out[node] = (
                self._image_node_type(image_graph, node),
                int(image_graph.degree(node)),
            )
        return out

    def _select_next_node(self, state: _GenerationState, rng: random.Random):
        """Pick next image node by fail-first frontier heuristic.

        Args:
            state: Current generation state.
            rng: Random number generator.

        Returns:
            object: Selected target image node.
        """
        unassigned = [u for u in state.target_image.nodes() if not state.assigned.get(u, False)]
        if not unassigned:
            return None
        frontier = [
            u for u in unassigned if any(state.assigned.get(v, False) for v in state.target_image.neighbors(u))
        ]
        if not frontier:
            return self._select_seed_node(state, rng, candidates=unassigned)
        pool = frontier

        def key_fn(node):
            assigned_neighbors = sum(1 for v in state.target_image.neighbors(node) if state.assigned.get(v, False))
            degree = int(state.target_image.degree(node))
            return (assigned_neighbors, degree)

        best = max(key_fn(node) for node in pool)
        ties = [node for node in pool if key_fn(node) == best]
        return rng.choice(ties)

    def _select_seed_node(
        self,
        state: _GenerationState,
        rng: random.Random,
        *,
        candidates: Optional[Sequence[object]] = None,
    ):
        """
        Select a seed node using rarest signature bucket, then degree.

        Args:
            state: Current generation state.
            rng: Random number generator.
            candidates: Optional candidate node subset.

        Returns:
            object: Selected seed node.
        """
        pool = list(candidates) if candidates is not None else list(state.target_image.nodes())
        if not pool:
            return None
        best_nodes = []
        best_key = None
        for node in pool:
            sig = state.target_signatures.get(node)
            if sig is None:
                continue
            img_type, degree = sig
            bucket_size = len(self._bucket.get((img_type, degree), []))
            # Treat unseen signatures as worst-case for seeding.
            if bucket_size <= 0:
                bucket_size = 10**12
            candidate_key = (
                int(bucket_size),
                -int(degree),
                self._image_node_order_key(
                    state.target_image,
                    node,
                    target_signatures=state.target_signatures,
                ),
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_nodes = [node]
            elif candidate_key == best_key:
                best_nodes.append(node)
        if best_nodes:
            return rng.choice(best_nodes)
        return rng.choice(pool)

    def _build_requirements(self, state: _GenerationState, node) -> list[_BoundaryRequirement]:
        """Build boundary requirements for one node from assigned neighbors.

        Args:
            state: Current generation state.
            node: Target image node.

        Returns:
            list[_BoundaryRequirement]: Boundary constraints.
        """
        reqs: list[_BoundaryRequirement] = []
        for neighbor in state.target_image.neighbors(node):
            if not state.assigned.get(neighbor, False):
                continue
            edge_key = frozenset((node, neighbor))
            binding = state.edge_bindings.get(edge_key, {})
            global_node_types: tuple[int, ...] = tuple()
            required_types = Counter()
            if isinstance(binding, dict):
                global_nodes = tuple(binding.get("global_nodes", ()))
                typed = tuple(binding.get("typed_global_nodes", ()))
                if typed:
                    typed_nodes = tuple(g for g, _t in typed)
                    typed_types = tuple(int(t) for _g, t in typed)
                    if len(typed_nodes) != len(typed_types):
                        global_node_types = tuple()
                        global_order_keys = tuple()
                        required_types = Counter({self._missing_anchor_sentinel: 1})
                        reqs.append(
                            _BoundaryRequirement(
                                neighbor=neighbor,
                                global_nodes=typed_nodes,
                                global_node_types=global_node_types,
                                global_order_keys=global_order_keys,
                                required_types=required_types,
                            )
                        )
                        continue
                    # Fail closed if stored typed nodes disagree with stored
                    # global nodes.
                    if "global_nodes" in binding and tuple(binding.get("global_nodes", ())) != typed_nodes:
                        global_node_types = tuple()
                        global_order_keys = tuple()
                        required_types = Counter({self._missing_anchor_sentinel: 1})
                        reqs.append(
                            _BoundaryRequirement(
                                neighbor=neighbor,
                                global_nodes=typed_nodes,
                                global_node_types=global_node_types,
                                global_order_keys=global_order_keys,
                                required_types=required_types,
                            )
                        )
                        continue
                    global_nodes = typed_nodes
                    global_node_types = typed_types
            else:
                # Backward-compat: older state may contain only a node tuple.
                global_nodes = tuple(binding)
            # Boundary interfaces are expected to reference distinct anchors.
            # If duplicates appear, treat the edge as inconsistent and fail closed.
            if len(set(global_nodes)) != len(global_nodes):
                global_node_types = tuple()
                global_order_keys = tuple()
                required_types = Counter({self._missing_anchor_sentinel: 1})
                reqs.append(
                    _BoundaryRequirement(
                        neighbor=neighbor,
                        global_nodes=global_nodes,
                        global_node_types=global_node_types,
                        global_order_keys=global_order_keys,
                        required_types=required_types,
                    )
                )
                continue
            if not global_nodes:
                # Missing boundary anchors means an inconsistent state for a
                # constrained edge; force no-match downstream.
                global_node_types = tuple()
                global_order_keys = tuple()
                required_types = Counter({self._missing_anchor_sentinel: 1})
            else:
                if any(g not in state.graph for g in global_nodes):
                    global_node_types = tuple()
                    global_order_keys = tuple()
                    required_types = Counter({self._missing_anchor_sentinel: 1})
                    reqs.append(
                        _BoundaryRequirement(
                            neighbor=neighbor,
                            global_nodes=global_nodes,
                            global_node_types=global_node_types,
                            global_order_keys=global_order_keys,
                            required_types=required_types,
                        )
                    )
                    continue
                if not global_node_types or len(global_node_types) != len(global_nodes):
                    try:
                        global_node_types = tuple(
                            int(
                                anchor_type_current(
                                    state.graph,
                                    g,
                                    self.base_cut_radius,
                                    nbits=self.nbits,
                                )
                            )
                            for g in global_nodes
                        )
                    except Exception:
                        global_node_types = tuple()
                        global_order_keys = tuple()
                        required_types = Counter({self._missing_anchor_sentinel: 1})
                        reqs.append(
                            _BoundaryRequirement(
                                neighbor=neighbor,
                                global_nodes=global_nodes,
                                global_node_types=global_node_types,
                                global_order_keys=global_order_keys,
                                required_types=required_types,
                            )
                        )
                        continue
                # Keep a single source of truth for containment and pairing.
                global_order_keys = tuple(
                    self._current_anchor_order_key(state.graph, g, anchor_type=t)
                    for g, t in zip(global_nodes, global_node_types)
                )
                required_types = Counter(global_node_types)
            reqs.append(
                _BoundaryRequirement(
                    neighbor=neighbor,
                    global_nodes=global_nodes,
                    global_node_types=global_node_types,
                    global_order_keys=global_order_keys,
                    required_types=required_types,
                )
            )
        return reqs

    def _match_port_to_requirement(
        self,
        port: Port,
        requirement: _BoundaryRequirement,
    ) -> Optional[list[tuple[object, object]]]:
        """Match one component port to one boundary requirement.

        Args:
            port: Candidate component port.
            requirement: Required boundary multiset.

        Returns:
            Optional[list[tuple[object, object]]]: Local/global anchor pairs when feasible.
        """
        if len(requirement.global_nodes) != len(requirement.global_node_types):
            return None
        port_counter = Counter(port.anchor_types)
        if not self._counter_contains(port_counter, requirement.required_types):
            return None

        by_type_local: dict[int, list[tuple[tuple, object]]] = {}
        for local, anchor_type, order_key in zip(
            port.anchor_local_nodes,
            port.anchor_types,
            port.anchor_order_keys,
        ):
            by_type_local.setdefault(int(anchor_type), []).append((order_key, local))

        # Build pairing by required types using the requirement's aligned
        # (global_node, global_node_type) payload.
        global_by_type: dict[int, list[tuple[tuple, object]]] = {}
        for global_node, node_type, order_key in zip(
            requirement.global_nodes,
            requirement.global_node_types,
            requirement.global_order_keys,
        ):
            global_by_type.setdefault(node_type, []).append((order_key, global_node))

        pairs: list[tuple[object, object]] = []
        for anchor_type, req_count in requirement.required_types.items():
            t = int(anchor_type)
            req_count = int(req_count)
            locals_for_type = sorted(
                by_type_local.get(t, []),
                key=lambda item: item[0],
            )
            globals_for_type = sorted(
                global_by_type.get(t, []),
                key=lambda item: item[0],
            )
            if len(locals_for_type) < req_count or len(globals_for_type) < req_count:
                return None
            for i in range(req_count):
                pairs.append((locals_for_type[i][1], globals_for_type[i][1]))
        return pairs

    def _exact_port_match(
        self,
        component: ComponentInstance,
        requirements: list[_BoundaryRequirement],
    ) -> Optional[tuple[dict[int, int], dict[int, list[tuple[object, object]]]]]:
        """Find injective requirement-to-port matching with multiset containment.

        Args:
            component: Candidate component.
            requirements: Boundary requirements.

        Returns:
            Optional[tuple[dict[int, int], dict[int, list[tuple[object, object]]]]]:
            Requirement-to-port mapping and concrete local/global anchor pairs.
        """
        if len(requirements) > len(component.ports):
            return None

        ordered_req_ids = sorted(
            range(len(requirements)),
            key=lambda i: (
                sum(int(v) for v in requirements[i].required_types.values()),
                len(requirements[i].required_types),
            ),
            reverse=True,
        )

        used_ports: set[int] = set()
        req_to_port: dict[int, int] = {}
        req_anchor_pairs: dict[int, list[tuple[object, object]]] = {}
        global_to_local_binding: dict[object, object] = {}
        local_to_global_binding: dict[object, object] = {}

        def dfs(pos: int) -> bool:
            if pos >= len(ordered_req_ids):
                return True
            req_id = ordered_req_ids[pos]
            req = requirements[req_id]
            for port_index, port in enumerate(component.ports):
                if port_index in used_ports:
                    continue
                if not self._counter_contains(Counter(port.anchor_types), req.required_types):
                    continue
                pairs = self._match_port_to_requirement(port, req)
                if pairs is None:
                    continue
                # Cross-requirement consistency for diamond/frontier cases:
                # the same global anchor cannot be matched to different local
                # anchors (and vice versa) across requirements.
                consistent = True
                for local_node, global_node in pairs:
                    prev_local = global_to_local_binding.get(global_node)
                    if prev_local is not None and prev_local != local_node:
                        consistent = False
                        break
                    prev_global = local_to_global_binding.get(local_node)
                    if prev_global is not None and prev_global != global_node:
                        consistent = False
                        break
                if not consistent:
                    continue
                used_ports.add(port_index)
                req_to_port[req_id] = port_index
                req_anchor_pairs[req_id] = pairs
                newly_bound_globals: list[object] = []
                newly_bound_locals: list[object] = []
                for local_node, global_node in pairs:
                    if global_node not in global_to_local_binding:
                        global_to_local_binding[global_node] = local_node
                        newly_bound_globals.append(global_node)
                    if local_node not in local_to_global_binding:
                        local_to_global_binding[local_node] = global_node
                        newly_bound_locals.append(local_node)
                if dfs(pos + 1):
                    return True
                for global_node in newly_bound_globals:
                    global_to_local_binding.pop(global_node, None)
                for local_node in newly_bound_locals:
                    local_to_global_binding.pop(local_node, None)
                used_ports.remove(port_index)
                req_to_port.pop(req_id, None)
                req_anchor_pairs.pop(req_id, None)
            return False

        if not dfs(0):
            return None
        return req_to_port, req_anchor_pairs

    def _retrieve_candidates(
        self,
        state: _GenerationState,
        node,
        rng: random.Random,
    ) -> tuple[list[_BoundaryRequirement], list[_CandidateAssignment]]:
        """Retrieve feasible candidate components for a target image node.

        Args:
            state: Current generation state.
            node: Target image node.
            rng: Random number generator.

        Returns:
            tuple[list[_BoundaryRequirement], list[_CandidateAssignment]]:
            Boundary requirements and matching candidates.
        """
        key = state.target_signatures[node]
        candidate_ids = set(self._bucket.get(key, []))
        if not candidate_ids:
            self._dbg(2, "retrieve_empty_bucket", node=node, signature=key)
            return [], []
        initial_candidates = len(candidate_ids)

        requirements = self._build_requirements(state, node)
        self._dbg(
            2,
            "retrieve_start",
            node=node,
            signature=key,
            bucket_candidates=initial_candidates,
            requirement_count=len(requirements),
        )

        for req_idx, req in enumerate(requirements):
            if not req.required_types:
                continue
            rarest_key = None
            rarest_freq = None
            for anchor_type, multiplicity in req.required_types.items():
                inv_key = (key[0], key[1], int(anchor_type), int(multiplicity))
                freq = self._inv_freq.get(inv_key, 0)
                if rarest_freq is None or freq < rarest_freq:
                    rarest_freq = freq
                    rarest_key = inv_key
            if rarest_key is None:
                continue
            before = len(candidate_ids)
            candidate_ids &= self._inv.get(rarest_key, set())
            after = len(candidate_ids)
            self._dbg(
                2,
                "retrieve_intersection",
                node=node,
                requirement_index=req_idx,
                rarest_key=rarest_key,
                rarest_freq=(0 if rarest_freq is None else int(rarest_freq)),
                candidates_before=before,
                candidates_after=after,
                required_types=dict(req.required_types),
            )
            if not candidate_ids:
                self._dbg(
                    2,
                    "retrieve_exhausted_after_intersection",
                    node=node,
                    requirement_index=req_idx,
                )
                return requirements, []

        ids = list(candidate_ids)
        rng.shuffle(ids)
        matches: list[_CandidateAssignment] = []
        rejected_exact = 0
        for comp_id in ids:
            component = self._components[comp_id]
            exact = self._exact_port_match(component, requirements)
            if exact is None:
                rejected_exact += 1
                continue
            req_to_port, req_anchor_pairs = exact
            matches.append(
                _CandidateAssignment(
                    component=component,
                    req_to_port=req_to_port,
                    req_anchor_pairs=req_anchor_pairs,
                )
            )
        self._dbg(
            2,
            "retrieve_done",
            node=node,
            signature=key,
            candidates_after_index=len(ids),
            rejected_exact=rejected_exact,
            matched=len(matches),
        )
        return requirements, matches

    @staticmethod
    def _replace_global_id_everywhere(state: _GenerationState, old_id, new_id) -> None:
        """
        Replace one global node id by another across state caches.

        Args:
            state: Mutable generation state.
            old_id: Removed global node id.
            new_id: Kept global node id.

        Returns:
            None.
        """
        if old_id == new_id:
            return
        for edge_key, binding in list(state.edge_bindings.items()):
            if not isinstance(binding, dict):
                nodes = tuple(new_id if x == old_id else x for x in tuple(binding))
                state.edge_bindings[edge_key] = {
                    "global_nodes": nodes,
                    "typed_global_nodes": tuple(),
                    "required_types": {},
                }
                continue
            nodes = tuple(new_id if x == old_id else x for x in tuple(binding.get("global_nodes", ())))
            typed = tuple(
                (new_id if g == old_id else g, int(t))
                for g, t in tuple(binding.get("typed_global_nodes", ()))
            )
            state.edge_bindings[edge_key] = {
                "global_nodes": nodes,
                "typed_global_nodes": typed,
                "required_types": dict(binding.get("required_types", {})),
            }
        for owner, mapping in list(state.node_maps.items()):
            if not isinstance(mapping, dict):
                continue
            state.node_maps[owner] = {
                local: (new_id if global_id == old_id else global_id)
                for local, global_id in mapping.items()
            }

    def _enumerate_future_port_assignments(
        self,
        state: _GenerationState,
        node,
        requirements: list[_BoundaryRequirement],
        candidate: _CandidateAssignment,
        rng: random.Random,
    ) -> list[dict[object, int]]:
        """
        Enumerate feasible port-to-neighbor bindings for unconstrained neighbors.

        Args:
            state: Current state.
            node: Node being expanded.
            requirements: Requirements already matched for assigned neighbors.
            candidate: Candidate component with matched requirement ports.
            rng: Random number generator.

        Returns:
            list[dict[object, int]]: Possible future neighbor->port assignments.
        """
        component = candidate.component
        assigned_neighbors = {req.neighbor for req in requirements}
        used_ports = set(candidate.req_to_port.values())
        remaining_ports = [i for i in range(len(component.ports)) if i not in used_ports]
        unassigned_neighbors = [
            nbr
            for nbr in sorted(
                list(state.target_image.neighbors(node)),
                key=lambda nbr: self._image_node_order_key(
                    state.target_image,
                    nbr,
                    target_signatures=state.target_signatures,
                ),
            )
            if nbr not in assigned_neighbors
        ]
        if len(remaining_ports) < len(unassigned_neighbors):
            return []
        if not unassigned_neighbors:
            return [{}]
        if len(remaining_ports) == len(unassigned_neighbors) == 1:
            return [{unassigned_neighbors[0]: remaining_ports[0]}]

        max_branches = max(1, int(self._max_future_port_assignment_branches))
        neighbors = tuple(unassigned_neighbors)
        k = len(neighbors)
        assignments: list[dict[object, int]] = []
        seen: set[tuple[int, ...]] = set()
        max_trials = max(10, max_branches * 10)
        for _ in range(max_trials):
            perm = tuple(rng.sample(remaining_ports, k))
            if perm in seen:
                continue
            seen.add(perm)
            assignments.append({nbr: port for nbr, port in zip(neighbors, perm)})
            if len(assignments) >= max_branches:
                break
        if assignments:
            return assignments
        # Deterministic fallback if random sampling failed to produce branches.
        return [{nbr: port for nbr, port in zip(neighbors, remaining_ports[:k])}]

    @staticmethod
    def _copy_state(state: _GenerationState) -> _GenerationState:
        """Create a branch copy for recursive backtracking.

        Args:
            state: Source state.

        Returns:
            _GenerationState: Deep branch copy.
        """
        return _GenerationState(
            target_image=state.target_image,
            target_signatures=state.target_signatures,
            graph=state.graph.copy(),
            assigned=dict(state.assigned),
            comp_of=dict(state.comp_of),
            node_maps=copy.deepcopy(state.node_maps),
            edge_bindings=copy.deepcopy(state.edge_bindings),
        )

    def _commit(
        self,
        state: _GenerationState,
        node,
        requirements: list[_BoundaryRequirement],
        candidate: _CandidateAssignment,
        future_port_assignment: Optional[dict[object, int]] = None,
    ) -> tuple[bool, str]:
        """Commit one candidate choice into generation state.

        Args:
            state: Mutable generation state.
            node: Target image node.
            requirements: Boundary requirements for this node.
            candidate: Candidate assignment to commit.

        Returns:
            tuple[bool, str]: Success flag and reason code.
        """
        component = candidate.component
        local_to_global = materialize_component(state.graph, component.subgraph)

        for req_idx, pairs in candidate.req_anchor_pairs.items():
            _ = req_idx
            for local_node, global_node in pairs:
                current_node = local_to_global.get(local_node)
                if current_node is None:
                    continue
                had_current = current_node in state.graph
                had_global = global_node in state.graph
                kept = unify_anchors(state.graph, current_node, global_node)
                # Safety: require contraction semantics. If both nodes survive
                # unchanged, this commit is inconsistent with anchor unification.
                if (
                    current_node != global_node
                    and had_current
                    and had_global
                    and current_node in state.graph
                    and global_node in state.graph
                ):
                    return False, "anchor_unification_non_contracting"
                for local_key, mapped_global in list(local_to_global.items()):
                    if mapped_global == current_node:
                        local_to_global[local_key] = kept
                removed = None
                if had_current and current_node not in state.graph:
                    removed = current_node
                elif had_global and global_node not in state.graph:
                    removed = global_node
                if removed is not None and removed != kept:
                    self._replace_global_id_everywhere(state, removed, kept)
                elif current_node != kept:
                    self._replace_global_id_everywhere(state, current_node, kept)
                elif global_node != kept:
                    self._replace_global_id_everywhere(state, global_node, kept)

        state.assigned[node] = True
        state.comp_of[node] = component.comp_id
        state.node_maps[node] = dict(local_to_global)

        assigned_neighbors = {req.neighbor for req in requirements}
        used_ports = set(candidate.req_to_port.values())
        remaining_ports = [i for i in range(len(component.ports)) if i not in used_ports]

        unassigned_neighbors = [
            nbr
            for nbr in sorted(
                list(state.target_image.neighbors(node)),
                key=lambda nbr: self._image_node_order_key(
                    state.target_image,
                    nbr,
                    target_signatures=state.target_signatures,
                ),
            )
            if nbr not in assigned_neighbors
        ]
        if len(remaining_ports) < len(unassigned_neighbors):
            return False, "not_enough_remaining_ports"

        for req in requirements:
            if len(req.global_nodes) != len(req.global_node_types):
                return False, "requirement_global_type_mismatch"
            edge_key = frozenset((node, req.neighbor))
            state.edge_bindings[edge_key] = {
                "global_nodes": tuple(req.global_nodes),
                "typed_global_nodes": tuple(zip(req.global_nodes, req.global_node_types)),
                "required_types": dict(req.required_types),
            }

        if future_port_assignment is not None:
            selected_pairs = [(nbr, int(future_port_assignment[nbr])) for nbr in unassigned_neighbors]
        else:
            selected_pairs = list(zip(unassigned_neighbors, remaining_ports))
        for neighbor, port_index in selected_pairs:
            port = component.ports[port_index]
            try:
                anchor_globals = [local_to_global[local_node] for local_node in port.anchor_local_nodes]
            except KeyError:
                return False, "missing_local_node_after_materialize"
            if any(global_node not in state.graph for global_node in anchor_globals):
                return False, "anchor_missing_in_state_graph"
            if len(anchor_globals) != len(port.anchor_types):
                return False, "anchor_type_length_mismatch"
            edge_key = frozenset((node, neighbor))
            state.edge_bindings[edge_key] = {
                "global_nodes": tuple(anchor_globals),
                "typed_global_nodes": tuple(
                    zip(anchor_globals, map(int, port.anchor_types))
                ),
                "required_types": dict(Counter(port.anchor_types)),
            }

        return True, "ok"

    def _frontier_has_candidate(self, state: _GenerationState) -> tuple[bool, Optional[dict]]:
        """Check whether each frontier node still has at least one candidate.

        Args:
            state: Current generation state.

        Returns:
            tuple[bool, Optional[dict]]:
            Whether frontier is satisfiable and optional failure payload.
        """
        rng = random.Random(0)
        for node in state.target_image.nodes():
            if state.assigned.get(node, False):
                continue
            if not any(state.assigned.get(nbr, False) for nbr in state.target_image.neighbors(node)):
                continue
            requirements, candidates = self._retrieve_candidates(state, node, rng)
            if not candidates:
                signature = state.target_signatures.get(node)
                info = {
                    "node": node,
                    "signature": signature,
                    "assigned_neighbor_count": sum(
                        1 for nbr in state.target_image.neighbors(node) if state.assigned.get(nbr, False)
                    ),
                    "requirement_count": len(requirements),
                    "requirement_anchor_sizes": [
                        int(sum(req.required_types.values())) for req in requirements
                    ],
                }
                self._dbg(2, "frontier_unsat", **info)
                return False, info
        return True, None

    def _is_feasible(self, graph: nx.Graph) -> bool:
        """Evaluate optional feasibility estimator on a full graph.

        Args:
            graph: Candidate generated graph.

        Returns:
            bool: Feasibility flag.
        """
        est = self.feasibility_estimator
        if est is None:
            return True
        if hasattr(est, "number_of_violations"):
            try:
                violations = est.number_of_violations([graph])
                if isinstance(violations, (list, tuple)) and violations:
                    return float(violations[0]) <= 0.0
            except Exception:
                pass
        if hasattr(est, "predict"):
            try:
                pred = est.predict([graph])
                if isinstance(pred, (list, tuple)):
                    return bool(pred[0])
                return bool(pred)
            except Exception:
                try:
                    pred = est.predict(graph)
                    return bool(pred)
                except Exception:
                    return True
        return True

    def _fit_feasibility_estimator(self, graphs: Sequence[nx.Graph]) -> None:
        """
        Fit the optional feasibility estimator on training generator graphs.

        Args:
            graphs: Generator graphs used for fitting this model.

        Returns:
            None.
        """
        est = self.feasibility_estimator
        if est is None or not hasattr(est, "fit"):
            return
        try:
            est.fit(graphs)
            return
        except TypeError:
            pass
        try:
            est.fit(graphs, None)
        except Exception as exc:
            warnings.warn(
                f"Failed to fit feasibility_estimator during fit(): {exc}",
                RuntimeWarning,
            )

    def _filter_feasible_graphs(self, graphs: Sequence[nx.Graph]) -> list[nx.Graph]:
        """
        Batch-filter graphs with the feasibility estimator.

        Args:
            graphs: Candidate graphs.

        Returns:
            list[nx.Graph]: Feasible subset.
        """
        if not graphs:
            return []
        est = self.feasibility_estimator
        if est is None:
            return list(graphs)
        if hasattr(est, "filter"):
            try:
                return list(est.filter(list(graphs)))
            except Exception:
                pass
        return [graph for graph in graphs if self._is_feasible(graph)]

    def _search(
        self,
        state: _GenerationState,
        rng: random.Random,
        counters: dict,
        max_backtracks: int,
    ) -> Optional[_GenerationState]:
        """Recursive backtracking search over image-node assignments.

        Args:
            state: Current state.
            rng: Random number generator.
            counters: Mutable recursion counters.
            max_backtracks: Maximum branching attempts.

        Returns:
            Optional[_GenerationState]: Solved state when successful.
        """
        if all(state.assigned.values()):
            return state
        if counters["branches"] >= max_backtracks:
            counters["backtrack_limit_hits"] = int(counters.get("backtrack_limit_hits", 0)) + 1
            return None

        node = self._select_next_node(state, rng)
        if node is None:
            return state

        requirements, candidates = self._retrieve_candidates(state, node, rng)
        if not candidates:
            counters["dead_no_candidates"] = int(counters.get("dead_no_candidates", 0)) + 1
            self._dbg(2, "search_dead_no_candidates", node=node, requirement_count=len(requirements))
            return None

        for candidate in candidates:
            counters["branches"] += 1
            if counters["branches"] > max_backtracks:
                counters["backtrack_limit_hits"] = int(counters.get("backtrack_limit_hits", 0)) + 1
                return None
            future_assignments = self._enumerate_future_port_assignments(state, node, requirements, candidate, rng)
            if not future_assignments:
                counters["dead_no_future_assignments"] = int(counters.get("dead_no_future_assignments", 0)) + 1
                self._dbg(
                    2,
                    "search_dead_no_future_assignments",
                    node=node,
                    component_id=candidate.component.comp_id,
                )
                continue
            for future_assignment in future_assignments:
                next_state = self._copy_state(state)
                commit_ok, commit_reason = self._commit(
                    next_state,
                    node,
                    requirements,
                    candidate,
                    future_port_assignment=future_assignment,
                )
                if not commit_ok:
                    counters["dead_commit"] = int(counters.get("dead_commit", 0)) + 1
                    commit_fail_reasons = counters.setdefault("commit_fail_reasons", Counter())
                    commit_fail_reasons[commit_reason] += 1
                    self._dbg(
                        2,
                        "search_dead_commit",
                        node=node,
                        component_id=candidate.component.comp_id,
                        reason=commit_reason,
                    )
                    continue
                frontier_ok, frontier_info = self._frontier_has_candidate(next_state)
                if not frontier_ok:
                    counters["dead_frontier_prune"] = int(counters.get("dead_frontier_prune", 0)) + 1
                    frontier_reasons = counters.setdefault("frontier_unsat_reasons", Counter())
                    if isinstance(frontier_info, dict):
                        reason_key = str(frontier_info.get("signature"))
                        frontier_reasons[reason_key] += 1
                    continue
                solved = self._search(next_state, rng, counters, max_backtracks)
                if solved is not None:
                    return solved
        return None

    def _generate_one(
        self,
        target_interpretation_graph: nx.Graph,
        rng: random.Random,
        *,
        max_backtracks: int,
        attempt_trace: Optional[dict] = None,
    ) -> Optional[nx.Graph]:
        """Generate one graph from a fixed target interpretation graph.

        Args:
            target_interpretation_graph: Target interpretation graph.
            rng: Random number generator.
            max_backtracks: Maximum backtracking attempts.
            attempt_trace: Optional mutable dictionary updated with attempt diagnostics.

        Returns:
            Optional[nx.Graph]: Generated graph when successful.
        """
        assigned = {node: False for node in target_interpretation_graph.nodes()}
        state = _GenerationState(
            target_image=target_interpretation_graph,
            target_signatures=self._compute_target_signatures(target_interpretation_graph),
            graph=nx.Graph(),
            assigned=assigned,
            comp_of={},
            node_maps={},
            edge_bindings={},
        )
        counters = {
            "branches": 0,
            "dead_no_candidates": 0,
            "dead_no_future_assignments": 0,
            "dead_commit": 0,
            "dead_frontier_prune": 0,
            "backtrack_limit_hits": 0,
            "commit_fail_reasons": Counter(),
            "frontier_unsat_reasons": Counter(),
        }
        solved = self._search(state, rng, counters, int(max_backtracks))
        if solved is None:
            if attempt_trace is not None:
                if int(counters.get("backtrack_limit_hits", 0)) > 0:
                    failure_stage = "backtrack_limit"
                elif int(counters.get("dead_no_candidates", 0)) > 0:
                    failure_stage = "no_candidates"
                elif int(counters.get("dead_frontier_prune", 0)) > 0:
                    failure_stage = "frontier_prune"
                elif int(counters.get("dead_commit", 0)) > 0:
                    failure_stage = "commit_failure"
                elif int(counters.get("dead_no_future_assignments", 0)) > 0:
                    failure_stage = "no_future_assignments"
                else:
                    failure_stage = "search_failed"
                attempt_trace.update(
                    {
                        "success": False,
                        "failure_stage": failure_stage,
                        "branches": int(counters.get("branches", 0)),
                        "dead_no_candidates": int(counters.get("dead_no_candidates", 0)),
                        "dead_no_future_assignments": int(counters.get("dead_no_future_assignments", 0)),
                        "dead_commit": int(counters.get("dead_commit", 0)),
                        "dead_frontier_prune": int(counters.get("dead_frontier_prune", 0)),
                        "backtrack_limit_hits": int(counters.get("backtrack_limit_hits", 0)),
                        "commit_fail_reasons": dict(counters.get("commit_fail_reasons", Counter())),
                    }
                )
            return None
        output = solved.graph.copy()
        output.graph["assigned_images"] = dict(solved.assigned)
        output.graph["comp_of"] = dict(solved.comp_of)
        if not self._is_feasible(output):
            if attempt_trace is not None:
                attempt_trace.update(
                    {
                        "success": False,
                        "failure_stage": "feasibility_filter",
                        "branches": int(counters.get("branches", 0)),
                    }
                )
            return None
        if attempt_trace is not None:
            attempt_trace.update(
                {
                    "success": True,
                    "failure_stage": "success",
                    "branches": int(counters.get("branches", 0)),
                }
            )
        return output

    def generate(
        self,
        n_samples: int = 1,
        *,
        interpretation_graphs: Optional[Sequence[nx.Graph]] = None,
        image_graphs: Optional[Sequence[nx.Graph]] = None,
        random_state: Optional[int] = None,
        max_backtracks: int = 5000,
        max_attempts_per_sample: int = 8,
        max_total_attempts: Optional[int] = None,
        progress_every_attempts: int = 100,
        progress_every_seconds: float = 10.0,
        parallel_queue_factor: int = 4,
    ) -> list[nx.Graph]:
        """Generate new base graphs by component assembly and backtracking.

        Args:
            n_samples: Number of graphs to generate.
            interpretation_graphs: Optional explicit pool of target interpretation graphs.
            image_graphs: Deprecated alias for ``interpretation_graphs``.
            random_state: Optional deterministic seed.
            max_backtracks: Maximum backtracking branches per sample.
            max_attempts_per_sample: Max retries with different targets per sample.
            max_total_attempts: Optional global attempt cap. If None, defaults
                to ``4 * n_samples * max_attempts_per_sample``.
            progress_every_attempts: Emit heartbeat every N attempts when
                debug is enabled. Set <= 0 to disable attempt-based heartbeat.
            progress_every_seconds: Emit heartbeat every T seconds when debug
                is enabled. Set <= 0 to disable time-based heartbeat.
            parallel_queue_factor: Number of in-flight futures per worker in
                parallel mode. Higher values improve CPU utilization when task
                durations are imbalanced.

        Returns:
            list[nx.Graph]: Generated graphs.
        """
        if not self._is_fitted:
            raise ValueError("Call fit(graphs) before generate().")
        n_samples = int(n_samples)
        if n_samples <= 0:
            return []

        resolved_interpretation_graphs = _resolve_alias(
            canonical_name="interpretation_graphs",
            canonical_value=interpretation_graphs,
            deprecated_name="image_graphs",
            deprecated_value=image_graphs,
            default=None,
        )
        pool = (
            list(resolved_interpretation_graphs)
            if resolved_interpretation_graphs is not None
            else list(self._interpretation_graph_pool)
        )
        if not pool:
            return []

        def _sequential_generate(
            rng_obj: random.Random,
            outputs_in: list[nx.Graph],
            attempts_in: int,
            constructed_in: int,
            filtered_out_in: int,
            *,
            capture_traces: bool = False,
        ) -> tuple[list[nx.Graph], int, int, int]:
            outputs_local = outputs_in
            attempts_local = attempts_in
            constructed_local = constructed_in
            filtered_local = filtered_out_in
            while len(outputs_local) < n_samples and attempts_local < max_total_attempts:
                attempts_local += 1
                target_local = rng_obj.choice(pool).copy()
                if capture_traces:
                    attempt_trace: dict = {}
                    generated_local = self._generate_one(
                        target_local,
                        rng_obj,
                        max_backtracks=int(max_backtracks),
                        attempt_trace=attempt_trace,
                    )
                    stage = str(attempt_trace.get("failure_stage", "unknown"))
                    attempt_outcomes[stage] += 1
                    aggregate_commit_fail_reasons.update(
                        Counter(attempt_trace.get("commit_fail_reasons", {}))
                    )
                    unseen_count = 0
                    for node in target_local.nodes():
                        sig = (
                            self._image_node_type(target_local, node),
                            int(target_local.degree(node)),
                        )
                        if sig not in self._bucket:
                            unseen_count += 1
                    nonlocal unseen_signature_nodes_total, unseen_signature_nodes_attempts
                    unseen_signature_nodes_total += int(unseen_count)
                    unseen_signature_nodes_attempts += 1
                    self._dbg(
                        2,
                        "generate_attempt",
                        attempt=attempts_local,
                        stage=stage,
                        success=bool(generated_local is not None),
                        branches=attempt_trace.get("branches", 0),
                        unseen_signature_nodes=unseen_count,
                        target_nodes=target_local.number_of_nodes(),
                        target_edges=target_local.number_of_edges(),
                    )
                else:
                    generated_local = self._generate_one(
                        target_local,
                        rng_obj,
                        max_backtracks=int(max_backtracks),
                    )
                if generated_local is None:
                    _maybe_log_progress(
                        attempts_now=attempts_local,
                        completed_now=attempts_local,
                        constructed_now=constructed_local,
                        filtered_now=filtered_local,
                        kept_now=len(outputs_local),
                        phase="sequential",
                    )
                    continue
                constructed_local += 1
                filtered_local_batch = self._filter_feasible_graphs([generated_local])
                if filtered_local_batch:
                    outputs_local.extend(filtered_local_batch)
                else:
                    filtered_local += 1
                    if capture_traces:
                        attempt_outcomes["post_filter_rejected"] += 1
                _maybe_log_progress(
                    attempts_now=attempts_local,
                    completed_now=attempts_local,
                    constructed_now=constructed_local,
                    filtered_now=filtered_local,
                    kept_now=len(outputs_local),
                    phase="sequential",
                )
            return outputs_local, attempts_local, constructed_local, filtered_local

        rng = random.Random(random_state)
        outputs: list[nx.Graph] = []
        per_sample_budget = max(1, int(max_attempts_per_sample))
        if max_total_attempts is None:
            max_total_attempts = int(max(1, 4 * n_samples * per_sample_budget))
        else:
            max_total_attempts = max(1, int(max_total_attempts))
        attempts = 0
        completed_futures = 0
        constructed_candidates = 0
        filtered_out = 0
        attempt_outcomes: Counter = Counter()
        aggregate_commit_fail_reasons: Counter = Counter()
        unseen_signature_nodes_total = 0
        unseen_signature_nodes_attempts = 0
        n_workers = int(self.n_jobs)
        if n_workers <= 0:
            n_workers = None
        progress_every_attempts = int(progress_every_attempts)
        progress_every_seconds = float(progress_every_seconds)
        parallel_queue_factor = max(1, int(parallel_queue_factor))
        progress_attempts_enabled = progress_every_attempts > 0
        progress_time_enabled = progress_every_seconds > 0.0
        start_time = time.time()
        last_progress_time = start_time
        last_progress_attempts = 0

        use_parallel = (n_workers is None or n_workers > 1) and n_samples > 1

        def _maybe_log_progress(
            *,
            attempts_now: int,
            completed_now: int,
            constructed_now: int,
            filtered_now: int,
            kept_now: int,
            phase: str = "runtime",
            force: bool = False,
        ) -> None:
            if not self.debug or int(self.debug_level) < 1:
                return
            nonlocal last_progress_time, last_progress_attempts
            now = time.time()
            due_attempts = progress_attempts_enabled and (
                attempts_now - last_progress_attempts >= progress_every_attempts
            )
            due_time = progress_time_enabled and (
                now - last_progress_time >= progress_every_seconds
            )
            if not force and not due_attempts and not due_time:
                return

            elapsed = max(1e-9, now - start_time)
            attempts_per_sec = float(attempts_now) / elapsed
            constructed_rate = float(constructed_now) / float(max(1, attempts_now))
            keep_rate = float(kept_now) / float(max(1, attempts_now))
            filtered_fraction = float(filtered_now) / float(max(1, constructed_now))
            remaining_samples = max(0, int(n_samples) - int(kept_now))
            if keep_rate > 0.0:
                eta_seconds = float(remaining_samples) / keep_rate / max(1e-9, attempts_per_sec)
                eta_seconds_text = f"{eta_seconds:.1f}"
            else:
                eta_seconds_text = "inf"
            self._dbg(
                1,
                "generate_progress",
                submitted_attempts=attempts_now,
                completed_futures=completed_now,
                phase=phase,
                kept=kept_now,
                remaining_samples=remaining_samples,
                constructed=constructed_now,
                filtered=filtered_now,
                attempts_per_sec=f"{attempts_per_sec:.2f}",
                constructed_rate=f"{constructed_rate:.3f}",
                keep_rate=f"{keep_rate:.3f}",
                filtered_fraction=f"{filtered_fraction:.3f}",
                eta_seconds=eta_seconds_text,
            )
            last_progress_time = now
            last_progress_attempts = int(attempts_now)
        if self.debug:
            pool_nodes = [g.number_of_nodes() for g in pool]
            pool_edges = [g.number_of_edges() for g in pool]
            self._dbg(
                1,
                "generate_indexes",
                components=len(self._components),
                bucket_keys=len(self._bucket),
                inv_keys=len(self._inv),
                inv_freq_keys=len(self._inv_freq),
                pool_graphs=len(pool),
            )
            self._dbg(
                1,
                "generate_pool_stats",
                nodes=self._format_min_mean_max(pool_nodes),
                edges=self._format_min_mean_max(pool_edges),
            )
            self._dbg(1, "generate_execution", parallel=use_parallel, n_jobs=self.n_jobs)
        if use_parallel:
            try:
                default_workers = max(1, int(os.cpu_count() or 1))
                effective_workers = default_workers if n_workers is None else int(n_workers)
                if self.debug:
                    self._dbg(
                        1,
                        "generate_workers",
                        default_cpu=default_workers,
                        effective_workers=effective_workers,
                    )
                worker_generator = copy.copy(self)
                # Generation uses fitted component indexes; decomposition and
                # feasibility are not needed inside workers and may be unpickleable.
                worker_generator.decomposition_function = None
                worker_generator.feasibility_estimator = None
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_init_generate_worker,
                    initargs=(worker_generator,),
                ) as executor:
                    while len(outputs) < n_samples and attempts < max_total_attempts:
                        remaining_attempts = int(max_total_attempts - attempts)
                        target_inflight = max(1, int(effective_workers) * parallel_queue_factor)
                        batch_size = max(1, min(remaining_attempts, target_inflight))
                        futures = []
                        for _ in range(batch_size):
                            target = rng.choice(pool).copy()
                            seed = rng.randrange(2**63)
                            futures.append(
                                executor.submit(
                                    _generate_one_worker,
                                    target,
                                    int(seed),
                                    int(max_backtracks),
                                )
                            )
                        attempts += len(futures)
                        generated_batch: list[nx.Graph] = []
                        for fut in as_completed(futures):
                            completed_futures += 1
                            generated, attempt_trace = fut.result()
                            if isinstance(attempt_trace, dict):
                                stage = str(attempt_trace.get("failure_stage", "unknown"))
                                attempt_outcomes[stage] += 1
                                aggregate_commit_fail_reasons.update(
                                    Counter(attempt_trace.get("commit_fail_reasons", {}))
                                )
                            if generated is None:
                                continue
                            constructed_candidates += 1
                            generated_batch.append(generated)
                        if generated_batch:
                            filtered = self._filter_feasible_graphs(generated_batch)
                            outputs.extend(filtered)
                            filtered_out += int(len(generated_batch) - len(filtered))
                        _maybe_log_progress(
                            attempts_now=attempts,
                            completed_now=completed_futures,
                            constructed_now=constructed_candidates,
                            filtered_now=filtered_out,
                            kept_now=len(outputs),
                            phase="parallel_post_filter",
                        )
            except Exception as exc:
                warnings.warn(
                    f"Parallel generation failed, falling back to sequential mode: {exc}",
                    RuntimeWarning,
                )
                outputs, attempts, constructed_candidates, filtered_out = _sequential_generate(
                    rng,
                    outputs,
                    attempts,
                    constructed_candidates,
                    filtered_out,
                    capture_traces=self.debug_level >= 2,
                )
        else:
            outputs, attempts, constructed_candidates, filtered_out = _sequential_generate(
                rng,
                outputs,
                attempts,
                constructed_candidates,
                filtered_out,
                capture_traces=self.debug_level >= 2,
            )

        outputs = outputs[:n_samples]
        _maybe_log_progress(
            attempts_now=attempts,
            completed_now=(completed_futures if use_parallel else attempts),
            constructed_now=constructed_candidates,
            filtered_now=filtered_out,
            kept_now=len(outputs),
            phase="final",
            force=True,
        )
        if self.debug:
            kept = len(outputs)
            if constructed_candidates > 0:
                filtered_fraction = float(filtered_out) / float(constructed_candidates)
            else:
                filtered_fraction = 0.0
            self._dbg(
                1,
                "generation_summary",
                attempts=attempts,
                constructed=constructed_candidates,
                kept=kept,
                filtered=filtered_out,
                filtered_fraction=f"{filtered_fraction:.3f}",
            )
            if attempt_outcomes:
                self._dbg(1, "generation_attempt_outcomes", outcomes=dict(attempt_outcomes))
            if aggregate_commit_fail_reasons:
                self._dbg(
                    1,
                    "generation_commit_fail_reasons",
                    reasons=dict(aggregate_commit_fail_reasons),
                )
            if unseen_signature_nodes_attempts > 0:
                unseen_avg = float(unseen_signature_nodes_total) / float(unseen_signature_nodes_attempts)
                self._dbg(
                    1,
                    "generation_unseen_signatures",
                    attempts=unseen_signature_nodes_attempts,
                    avg_unseen_nodes=f"{unseen_avg:.2f}",
                    total_unseen_nodes=unseen_signature_nodes_total,
                )
        if len(outputs) < n_samples:
            warnings.warn(
                f"generate requested n_samples={n_samples} but produced {len(outputs)} "
                f"after {attempts} attempts (budget={max_total_attempts}).",
                RuntimeWarning,
            )
        if outputs:
            self._zero_generation_streak = 0
        else:
            self._zero_generation_streak += 1
            if self.base_cut_radius > 0 and self._zero_generation_streak >= 2:
                warnings.warn(
                    "ConditionalAutoregressiveGenerator.generate returned 0 samples "
                    f"for {self._zero_generation_streak} consecutive calls with "
                    f"base_cut_radius={self.base_cut_radius}. This may be "
                    "over-constraining anchor matching. Consider base_cut_radius=0 "
                    "or larger generator sets / attempt budgets.",
                    RuntimeWarning,
                )
        return outputs

from abstractgraph_generative.conditional_batch import (  # noqa: E402,F401
    ConditionalAutoregressiveGraphsGenerator,
)

__all__ = [
    "ConditionalAutoregressiveGenerator",
    "ConditionalAutoregressiveGraphsGenerator",
    "ComponentInstance",
    "Port",
]
