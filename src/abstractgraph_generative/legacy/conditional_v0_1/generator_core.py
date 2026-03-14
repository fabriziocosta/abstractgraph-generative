"""Core conditional autoregressive generator implementation."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Callable, Optional, Sequence
import itertools
import math
import pickle
import random
import threading
import time
import warnings
try:
    import cloudpickle
except Exception:  # pragma: no cover - optional dependency via joblib stack
    try:
        from joblib.externals import cloudpickle  # type: ignore[no-redef]
    except Exception:
        cloudpickle = None

from joblib import Parallel, delayed

import networkx as nx
import matplotlib.pyplot as plt

from abstractgraph.display import display as _display_abstract_graph
from abstractgraph.hashing import hash_graph
from abstractgraph.graphs import AbstractGraph, graph_to_abstract_graph
from abstractgraph_generative.legacy.conditional_v0_1.types import CutArchiveEntry
from abstractgraph_generative.legacy.conditional_v0_1.utils import _available_cpu_count
from abstractgraph_generative.legacy.conditional_v0_1.index_building import (
    _assoc_map_from_image_associations,
    _cut_hash_with_image,
    _default_image_label,
    _default_preimage_label,
    build_image_conditioned_cut_index_from_pruning,
    cached_context_embedding_sum_for_nodes,
    cached_image_node_hash_map,
    cached_preimage_node_hash_map,
    group_graphs_by_image_hash,
    select_image_group,
)
from abstractgraph_generative.legacy.conditional_v0_1.materialization import _merge_with_anchors
from abstractgraph_generative.rewrite import (
    _cosine_similarity,
)


def display_conditioned_graphs(
    preimage_graphs: Sequence[nx.Graph],
    image_graph,
    *,
    decomposition_function: Optional[Callable[[AbstractGraph], AbstractGraph]] = None,
    nbits: Optional[int] = None,
    label_mode: str = "graph_hash",
    per_graph_image_graphs: bool = False,
    titles: Optional[Sequence[str]] = None,
    n_graphs_per_line: int = 3,
    size: tuple[int, int] = (6, 4),
    show: bool = True,
) -> plt.Figure:
    """
    Display preimage graphs against a fixed image graph in a grid.

    Args:
        preimage_graphs: Sequence of preimage graphs to display.
        image_graph: Fixed image graph or per-step list of image graphs.
        decomposition_function: Optional decomposition used to rebuild per-graph image graphs.
        nbits: Hash bit width used when rebuilding per-graph image graphs.
        label_mode: Label mode passed to graph_to_abstract_graph.
        per_graph_image_graphs: If True, compute image graphs per preimage graph.
        titles: Optional per-graph titles.
        n_graphs_per_line: Number of columns in the grid.
        size: Size of each subplot (width, height).
        show: If True, call plt.show().

    Returns:
        matplotlib.figure.Figure: Matplotlib figure containing the grid.
    """
    graphs = list(preimage_graphs)
    image_steps = None
    if per_graph_image_graphs:
        if decomposition_function is None or nbits is None:
            raise ValueError("per_graph_image_graphs requires decomposition_function and nbits.")
        image_steps = [
            graph_to_abstract_graph(
                g,
                decomposition_function=decomposition_function,
                nbits=nbits,
                label_mode=label_mode,
            ).image_graph
            for g in graphs
        ]
    elif isinstance(image_graph, (list, tuple)):
        image_steps = list(image_graph)
    n = len(graphs)
    if n == 0:
        fig, ax = plt.subplots(figsize=(size[0], size[1]))
        ax.axis("off")
        if show:
            plt.show()
        return fig

    cols = max(1, int(n_graphs_per_line))
    rows = (n + cols - 1) // cols
    figsize = (size[0] * cols, size[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for i in range(rows * cols):
        ax = axes_list[i]
        if i < n:
            ag = AbstractGraph()
            ag.preimage_graph = graphs[i].copy()
            image_step = (
                image_steps[i].copy()
                if image_steps is not None and i < len(image_steps)
                else image_graph.copy()
            )
            conditioned_image_graph = ag.preimage_graph.graph.get("conditioned_image_graph")
            if isinstance(conditioned_image_graph, nx.Graph):
                image_step = conditioned_image_graph.copy()
            pre_nodes = set(ag.preimage_graph.nodes())
            assigned = ag.preimage_graph.graph.get("assigned_images")
            image_associations = ag.preimage_graph.graph.get("image_associations")
            removed_images = image_step.graph.get("removed_images")
            for node, data in image_step.nodes(data=True):
                assoc = data.get("association")
                if image_associations is not None:
                    assoc_nodes = image_associations.get(node, set())
                    data["association"] = ag.preimage_graph.subgraph(assoc_nodes).copy()
                    continue
                if isinstance(assoc, nx.Graph):
                    if removed_images is not None and node in removed_images:
                        data["association"] = nx.Graph()
                    elif assigned is not None and node not in assigned:
                        data["association"] = nx.Graph()
                    else:
                        data["association"] = assoc.subgraph(pre_nodes).copy()
            ag.image_graph = image_step
            _display_abstract_graph(ag, ax=ax)
            if titles is not None and i < len(titles):
                ax.set_title(str(titles[i]))
        else:
            ax.axis("off")

    if show:
        plt.show()
    return fig


# =================================================================================================
# Pruning (index building)
# =================================================================================================

class ConditionalAutoregressiveGenerator:
    """Autoregressive generator that grows preimage graphs under fixed image context."""

    def __init__(
        self,
        *,
        decomposition_function=None,
        nbits: Optional[int] = None,
        n_pruning_iterations: int = 5,
        feasibility_estimator=None,

        preimage_cut_radius: Optional[int] = None,
        image_cut_radius: Optional[int] = None,
        cut_radius: Optional[int] = 0,
        constraint_level: Optional[int] = None,
        # Backward-compat alias; prefer preimage_cut_radius.
        pre_image_cut_radius: Optional[int] = None,
        preimage_context_radius: Optional[int] = None,
        image_context_radius: Optional[int] = None,
        context_vectorizer=None,
        use_context_embedding: bool = False,
        apply_feasibility_during_construction: bool = False,
        max_num_anchors: int = 7,
        max_num_anchor_sets_retry: int = 128,
        label_mode: str = "operator_hash",
        feasibility_fit_source: str = "generator",
        max_anchor_matches: int = 1000,
        preimage_label_fn: Optional[Callable[[Optional[nx.Graph], object], str]] = None,
        image_label_fn: Optional[Callable[[nx.Graph, object], str]] = None,
        n_jobs: int = -1,
        parallel_backend: Optional[str] = None,
        min_attempts_per_job: int = 4,
        verbose_parallel_stats: bool = False,
        dead_end_reset_after_zero_batches: int = 4,
        max_dfs_nodes: int = 5000,
        max_dfs_seconds: float = 3.0,
        max_candidates_per_bfs_node: int = 24,
        max_entry_subset_retries: int = 10,
        candidate_budget_multiplier: int = 20,
        entry_pool_budget_multiplier: int = 12,
        enforce_image_edge_constraints: bool = True,
        adaptive_entry_retry: bool = True,
        adaptive_entry_retry_multiplier: int = 4,
        adaptive_retry_disable_context: bool = True,
        context_exploration_fraction: float = 0.35,
        fallback_disable_context_on_failure: bool = True,
        use_transition_fallback: bool = True,
        use_transition_primary: bool = False,
        transition_primary_uniform_root: bool = True,
        transition_primary_round_robin_root: bool = False,
        transition_fallback_entry_budget: int = 256,
        enable_pairwise_rejection_learning: bool = False,
        pairwise_rejection_discount_alpha: float = 0.6,
        pairwise_rejection_beta: float = 0.35,
        pairwise_rejection_edge_weight: float = 1.0,
        pairwise_rejection_anchor_weight: float = 0.6,
        pairwise_rejection_feasibility_weight: float = 0.8,
        usage_penalty_beta: float = 1.0,
        dead_state_skip_threshold: float = 6.0,
        dead_state_skip_escape_prob: float = 0.03,
        dead_state_decay_per_attempt: float = 0.98,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the conditional autoregressive generator.

        Args:
            decomposition_function: Decomposition used to build image graphs.
                A more expressive decomposition increases representational power
                but may reduce donor-rule reuse.
            nbits: Hash bit width for image labels.
                Larger values reduce hash collisions but increase memory footprint.
            n_pruning_iterations: Number of pruning iterations per generator graph.
                Higher values improve archive coverage and diversity at the cost
                of slower fitting/index construction.
            feasibility_estimator: Optional feasibility estimator for filtering donors.
                When provided, candidate quality tends to improve but generation
                time increases due to additional predict calls.
            label_mode: Label mode passed to graph_to_abstract_graph.
                Controls what semantics are preserved in role/cut hashing.
            preimage_cut_radius: Radius for preimage anchor hashes.
                Larger radii make anchor matching stricter and more local-context
                aware, often reducing admissible donor matches.
            image_cut_radius: Radius for image-context hashes.
                Larger radii make role bucketing stricter, reducing donor pool size.
            cut_radius: Fallback radius when preimage_cut_radius is None.
                Kept for backward compatibility; prefer preimage_cut_radius.
            constraint_level: Optional override for cut radii. When provided as an
                integer, it overwrites the pre/image cut radii as follows:
                - 0 -> preimage_cut_radius=0, image_cut_radius=0
                - 1 -> preimage_cut_radius=1, image_cut_radius=0
                - 2 -> preimage_cut_radius=1, image_cut_radius=1
                - 3 -> preimage_cut_radius=2, image_cut_radius=1
                - 4 -> preimage_cut_radius=2, image_cut_radius=2
                - 5 -> preimage_cut_radius=3, image_cut_radius=2
                When set, it also sets context radii to
                preimage_context_radius=preimage_cut_radius+1 and
                image_context_radius=image_cut_radius+1.
            preimage_context_radius: Context radius for preimage embeddings.
                If None in downstream embedding helpers, full-graph context is used.
                Larger values increase compute and smooth local distinctions.
            image_context_radius: Context radius for image embeddings.
                If None in downstream embedding helpers, full-image context is used.
                Larger values provide broader semantic context but reduce locality.
            context_vectorizer: Vectorizer used for context embeddings.
                Must support transform on graph collections or single graphs.
            use_context_embedding: If True, use context embeddings for scoring.
                Usually improves contextual alignment but increases runtime.
            apply_feasibility_during_construction: If False, skip feasibility
                filtering during graph construction and only apply feasibility
                checks externally on final samples.
            max_num_anchors: Maximum subset size for anchor sampling.
                Larger values improve exactness of subset-based compatibility tests
                but increase combinatorial cost.
            max_num_anchor_sets_retry: Cap on number of subsets per size.
                Controls approximation level in subset hashing when combinations
                are too many to enumerate.
            feasibility_fit_source: Which graphs to fit feasibility on.
                "generator" fits on the original generator graphs.
                "pruned" fits on pruning-sequence graphs.
            max_anchor_matches: Cap on anchor match candidates.
                Higher values broaden search but can raise per-step latency.
            n_jobs: Parallelism for generation; joblib semantics (-1 = all cores).
                More workers improve throughput until memory/serialization limits.
            parallel_backend: Optional joblib backend override (e.g. "multiprocessing").
                Choose backend based on pickling overhead and platform behavior.
            min_attempts_per_job: Minimum attempts per worker batch to amortize overhead.
                Higher values reduce scheduler overhead, lower values improve responsiveness.
            verbose_parallel_stats: If True, print per-batch timing and acceptance stats.
            dead_end_reset_after_zero_batches: Number of consecutive parallel
                batches with zero accepted samples that triggers a lightweight
                dead-end reset (clears last-association cache).
                Set to 0 to disable the reset behavior.
            max_dfs_nodes: Max DFS node expansions per sample attempt.
                Limits search breadth/depth and prevents runaway recursion.
            max_dfs_seconds: Max DFS wall-clock seconds per sample attempt.
                Hard timeout per sample attempt; lower values improve responsiveness
                but can reduce acceptance rate.
            max_candidates_per_bfs_node: Max candidate rewires tried per BFS node.
                Primary branching-factor control during DFS backtracking.
            max_entry_subset_retries: Number of random entry-subset retries per
                BFS node before giving up and backtracking.
                Higher values improve robustness in sparse/strict settings at
                additional runtime cost.
            candidate_budget_multiplier: Multiplier used to build the internal
                candidate-evaluation budget:
                candidate_budget = max_candidates_per_bfs_node * multiplier.
                Larger values improve candidate recall but increase per-node cost.
            entry_pool_budget_multiplier: Multiplier used to size the random
                presampled entry pool before context scoring:
                entry_pool_budget = max(candidate_budget * multiplier, target).
                Larger values reduce sampling variance but increase runtime.
            context_exploration_fraction: Fraction of the early candidate list
                reserved for uniform-random exploration when context scoring is
                enabled (range [0, 1]).
                Higher values reduce context bias and improve robustness when
                similarity-guided choices lead to dead ends.
            fallback_disable_context_on_failure: If True, when one full DFS
                attempt fails under context scoring, retry once with context
                scoring disabled before declaring failure for that sample attempt.
                This improves robustness in hard regimes at some extra runtime.
            use_transition_fallback: If True, enable a restrictive fallback
                candidate source built from observed association-adjacency
                transitions in training image graphs. This is only consulted
                after normal candidate search fails for a BFS node.
            use_transition_primary: If True, query the observed-transition
                graph first for each BFS node, then fall back to the normal
                cut-index candidate search when transition-guided candidates
                are unavailable.
            transition_primary_uniform_root: If True, transition-primary picks
                a root association uniformly among observed root associations
                for the role hash, instead of always favoring the highest-count
                root transition.
            transition_primary_round_robin_root: If True, transition-primary
                root associations are ordered in round-robin per role hash so
                repeated attempts cycle through all observed root starts.
            transition_fallback_entry_budget: Max number of cut entries loaded
                from the transition fallback index per BFS node.
            enable_pairwise_rejection_learning: If True, learn pairwise
                transition difficulty between association hashes from rejected
                candidates and use it to downweight future transitions.
            pairwise_rejection_discount_alpha: Distance discount in (0, 1]
                applied as alpha^d over image-graph hop distance.
            pairwise_rejection_beta: Strength of pairwise rejection downweight.
            pairwise_rejection_edge_weight: Pairwise penalty increment for
                edge-constraint rejections.
            pairwise_rejection_anchor_weight: Pairwise penalty increment for
                anchor-mapping/filter rejections.
            pairwise_rejection_feasibility_weight: Pairwise penalty increment
                for feasibility rejections.
            usage_penalty_beta: Penalty strength for reusing the same
                (role_hash, assoc_hash) within a generation attempt.
                Higher values encourage broader donor usage.
            random_seed: Optional default RNG seed used across fit/prepare/generate
                when call-level seeds/RNG objects are not provided.
            preimage_label_fn: Optional label function for preimage nodes.
            image_label_fn: Optional label function for image nodes.

        Returns:
            None.
        """
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.n_pruning_iterations = int(n_pruning_iterations)
        self.feasibility_estimator = feasibility_estimator
        self.label_mode = label_mode
        self.constraint_level = constraint_level
        self.feasibility_fit_source = str(feasibility_fit_source or "generator").lower()

        # Resolve effective cut radii, optionally overridden by constraint_level.
        if self.constraint_level is not None:
            try:
                m = int(self.constraint_level)
            except Exception:
                m = None
            if m is not None:
                preimage_cut_radius = max(0, (m + 1) // 2)
                image_cut_radius = max(0, m // 2)
                preimage_context_radius = (preimage_cut_radius if preimage_cut_radius is not None else 0) + 1
                image_context_radius = image_cut_radius + 1
        # Fallback behavior (unchanged) when constraint_level is None or invalid.
        # Prefer the new name; fall back to legacy alias if provided.
        if preimage_cut_radius is None:
            preimage_cut_radius = pre_image_cut_radius
        if preimage_cut_radius is None:
            preimage_cut_radius = cut_radius
        if image_cut_radius is None:
            image_cut_radius = 0
        self.preimage_cut_radius = preimage_cut_radius
        self.image_cut_radius = image_cut_radius
        self.cut_radius = self.preimage_cut_radius
        self.preimage_context_radius = 1 if preimage_context_radius is None else preimage_context_radius
        self.image_context_radius = 1 if image_context_radius is None else image_context_radius
        self.context_vectorizer = context_vectorizer
        self.use_context_embedding = bool(use_context_embedding)
        self.apply_feasibility_during_construction = bool(apply_feasibility_during_construction)
        self.max_num_anchors = int(max_num_anchors)
        self.max_num_anchor_sets_retry = int(max_num_anchor_sets_retry)
        self.max_anchor_matches = int(max_anchor_matches)
        self.n_jobs = 1 if n_jobs is None else int(n_jobs)
        self.parallel_backend = parallel_backend
        self.min_attempts_per_job = max(1, int(min_attempts_per_job))
        self.verbose_parallel_stats = bool(verbose_parallel_stats)
        self.dead_end_reset_after_zero_batches = max(
            0, int(dead_end_reset_after_zero_batches)
        )
        self.max_dfs_nodes = max(1, int(max_dfs_nodes))
        self.max_dfs_seconds = max(0.05, float(max_dfs_seconds))
        self.max_candidates_per_bfs_node = max(1, int(max_candidates_per_bfs_node))
        self.max_entry_subset_retries = max(1, int(max_entry_subset_retries))
        self.candidate_budget_multiplier = max(1, int(candidate_budget_multiplier))
        self.entry_pool_budget_multiplier = max(1, int(entry_pool_budget_multiplier))
        self.enforce_image_edge_constraints = bool(enforce_image_edge_constraints)
        self.adaptive_entry_retry = bool(adaptive_entry_retry)
        self.adaptive_entry_retry_multiplier = max(2, int(adaptive_entry_retry_multiplier))
        self.adaptive_retry_disable_context = bool(adaptive_retry_disable_context)
        self.context_exploration_fraction = min(
            1.0, max(0.0, float(context_exploration_fraction))
        )
        self.fallback_disable_context_on_failure = bool(fallback_disable_context_on_failure)
        self.use_transition_fallback = bool(use_transition_fallback)
        self.use_transition_primary = bool(use_transition_primary)
        self.transition_primary_uniform_root = bool(transition_primary_uniform_root)
        self.transition_primary_round_robin_root = bool(transition_primary_round_robin_root)
        self.transition_fallback_entry_budget = max(1, int(transition_fallback_entry_budget))
        self.enable_pairwise_rejection_learning = bool(enable_pairwise_rejection_learning)
        self.pairwise_rejection_discount_alpha = min(
            1.0, max(0.0, float(pairwise_rejection_discount_alpha))
        )
        self.pairwise_rejection_beta = max(0.0, float(pairwise_rejection_beta))
        self.pairwise_rejection_edge_weight = max(0.0, float(pairwise_rejection_edge_weight))
        self.pairwise_rejection_anchor_weight = max(0.0, float(pairwise_rejection_anchor_weight))
        self.pairwise_rejection_feasibility_weight = max(0.0, float(pairwise_rejection_feasibility_weight))
        self.usage_penalty_beta = max(0.0, float(usage_penalty_beta))
        self.dead_state_skip_threshold = max(0.0, float(dead_state_skip_threshold))
        self.dead_state_skip_escape_prob = min(
            1.0, max(0.0, float(dead_state_skip_escape_prob))
        )
        self.dead_state_decay_per_attempt = min(
            1.0, max(0.0, float(dead_state_decay_per_attempt))
        )
        self.random_seed = None if random_seed is None else int(random_seed)
        # Failure-aware sampling state for backtracking/search.
        self.failure_penalty_edge_weight = 3
        self.failure_penalty_anchor_weight = 1
        self.failure_penalty_feasibility_weight = 2
        self.failure_penalty_beta = 0.35
        self.state_penalty_beta = 0.20
        self.state_backprop_alpha = 1.0
        self.state_backprop_lambda = 0.45
        self._reject_counts_by_state_assoc: dict = defaultdict(float)
        self._reject_state_scores: dict = defaultdict(float)
        self._reject_pair_scores: dict = defaultdict(float)
        self._preimage_hash_accept_stats: dict = defaultdict(
            lambda: {"success": 0, "fail": 0}
        )
        # Generation policy controls:
        # - BFS + backtracking search is bounded by node/time budgets.
        # - Branching is bounded per BFS node.
        # - Candidate ordering can penalize donor reuse within a sample.
        self.preimage_label_fn = preimage_label_fn
        self.image_label_fn = image_label_fn
        self.cut_index: dict = defaultdict(list)
        self.generator_graphs: list[nx.Graph] = []
        # Optional training image graphs for generate() when image_graph=None.
        self.image_graphs: list[nx.Graph] = []
        self.fixed_image_graph: Optional[nx.Graph] = None
        self.last_prepared_image_graph: Optional[nx.Graph] = None
        # Track last association hash per image node to avoid immediate repeats.
        self._last_assoc_hash_by_image: dict = {}
        # Fallback transition index:
        # (prev_assoc_hash|None, next_role_hash) -> Counter(next_assoc_hash -> count).
        self._transition_fallback_counts: dict = {}
        # Pointer map to concrete cut rules used by rewiring.
        self._cut_entries_by_role_assoc: dict = defaultdict(list)
        # Per-role cursor used by round-robin transition-primary root ordering.
        self._root_rr_cursor_by_role: dict = {}
        self._runtime_verbose = False
        self._warned_message_types: set[str] = set()
        self._runtime_target_n_samples: Optional[int] = None
        self._runtime_completed_n_samples: int = 0
        self._runtime_sample_backtracks: int = 0
        self._runtime_dead_state_skips: int = 0
        self._generation_stats_lock = threading.Lock()
        self._generation_stats: Optional[dict] = None
        self.last_generation_stats: dict = {}

    def __getstate__(self) -> dict:
        """
        Return picklable state for deepcopy/joblib serialization.

        Returns:
            dict: Instance state without transient thread locks.
        """
        state = self.__dict__.copy()
        state.pop("_generation_stats_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restore state and recreate transient synchronization primitives.

        Args:
            state: Pickled instance state.

        Returns:
            None.
        """
        self.__dict__.update(state)
        self._generation_stats_lock = threading.Lock()

    def _resolve_seed(self, seed: Optional[int]) -> Optional[int]:
        """
        Resolve an effective seed, falling back to the generator default.

        Args:
            seed: Optional call-level seed.

        Returns:
            Optional[int]: Effective seed or None.
        """
        if seed is None:
            return self.random_seed
        return int(seed)

    def _warn_verbose(
        self,
        message: str,
        *,
        exc: Optional[Exception] = None,
        verbose: Optional[bool] = None,
        message_type: Optional[str] = None,
    ) -> None:
        """
        Emit a runtime warning when verbose mode is enabled.

        Warnings are rate-limited to one emission per message type per
        generator instance.

        Args:
            message: Warning message context.
            exc: Optional exception to append.
            verbose: Optional explicit verbose override.
            message_type: Optional category key for rate limiting.

        Returns:
            None.
        """
        is_verbose = self._runtime_verbose if verbose is None else bool(verbose)
        if not is_verbose:
            return
        category = message if message_type is None else str(message_type)
        if category in self._warned_message_types:
            return
        self._warned_message_types.add(category)
        if exc is not None:
            message = f"{message}: {type(exc).__name__}: {exc}"
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    @staticmethod
    def _print_cut_size_summary(cut_index: dict) -> None:
        """
        Print the number of associations grouped by cut size.

        Args:
            cut_index: Mapping role_hash -> list[CutArchiveEntry].

        Returns:
            None.
        """
        size_counts = Counter()
        for entries in (cut_index or {}).values():
            for entry in entries:
                if getattr(entry, "anchor_nodes", None):
                    size = len(entry.anchor_nodes)
                elif getattr(entry, "anchor_pairs", None):
                    size = len(entry.anchor_pairs)
                else:
                    size = 0
                size_counts[int(size)] += 1
        if not size_counts:
            print("Cut size summary: no associations.")
            return
        parts = [f"{size}:{size_counts[size]}" for size in sorted(size_counts)]
        print("Cut size summary (size:count): " + ", ".join(parts))

    def _build_cut_entry_lookup(self) -> None:
        """
        Build a lookup from (role_hash, assoc_hash) to concrete cut entries.

        Returns:
            None.
        """
        lookup: dict = defaultdict(list)
        for role_hash, entries in (self.cut_index or {}).items():
            for entry in entries:
                lookup[(role_hash, int(entry.assoc_hash))].append(entry)
        self._cut_entries_by_role_assoc = lookup

    def _reset_generation_stats(self) -> None:
        """Reset per-generate aggregated counters."""
        self._generation_stats = {
            "candidate_calls": 0,
            "transition_primary_calls": 0,
            "transition_primary_success": 0,
            "transition_fallback_calls": 0,
            "transition_fallback_success": 0,
            "candidate_accept_total": 0,
            "anchor_filter_reject": 0,
            "anchor_map_reject": 0,
            "edge_constraint_reject": 0,
            "feasibility_reject": 0,
        }

    def _accumulate_generation_stats(self, stats: Counter) -> None:
        """Merge one candidate-evaluation stats block into run-level counters."""
        if self._generation_stats is None:
            return
        with self._generation_stats_lock:
            self._generation_stats["candidate_calls"] += 1
            self._generation_stats["transition_primary_calls"] += int(stats.get("transition_primary", 0))
            self._generation_stats["transition_primary_success"] += int(stats.get("transition_primary_success", 0))
            self._generation_stats["transition_fallback_calls"] += int(stats.get("transition_fallback", 0))
            self._generation_stats["transition_fallback_success"] += int(stats.get("transition_fallback_success", 0))
            self._generation_stats["candidate_accept_total"] += int(stats.get("accepted", 0))
            self._generation_stats["anchor_filter_reject"] += int(stats.get("anchor_filter_reject", 0))
            self._generation_stats["anchor_map_reject"] += int(stats.get("anchor_map_reject", 0))
            self._generation_stats["edge_constraint_reject"] += int(stats.get("edge_constraint_reject", 0))
            self._generation_stats["feasibility_reject"] += int(stats.get("feasibility_reject", 0))

    @staticmethod
    def _print_generation_summary(stats: dict) -> None:
        """Print a compact per-run transition and rejection summary."""
        if not stats:
            return
        tp_calls = int(stats.get("transition_primary_calls", 0))
        tp_succ = int(stats.get("transition_primary_success", 0))
        tf_calls = int(stats.get("transition_fallback_calls", 0))
        tf_succ = int(stats.get("transition_fallback_success", 0))
        parts = [
            f"candidate_calls={int(stats.get('candidate_calls', 0))}",
            f"accept_total={int(stats.get('candidate_accept_total', 0))}",
            f"tp={tp_succ}/{tp_calls}",
            f"tf={tf_succ}/{tf_calls}",
            f"rej(edge={int(stats.get('edge_constraint_reject', 0))},"
            f"anchor_filter={int(stats.get('anchor_filter_reject', 0))},"
            f"anchor_map={int(stats.get('anchor_map_reject', 0))},"
            f"feas={int(stats.get('feasibility_reject', 0))})",
        ]
        print("[generation-summary] " + " ".join(parts))

    def _build_transition_fallback_index(self, *, image_graphs: Sequence[nx.Graph]) -> None:
        """
        Build a restrictive association-transition fallback index.

        Nodes are association hashes and directed edges are labeled by the
        destination image-role hash. A virtual root (None) emits transitions
        only to valid start nodes (max-degree image nodes).

        Args:
            image_graphs: Conditioned image graphs used as evidence.

        Returns:
            None.
        """
        counts: dict = defaultdict(Counter)
        image_label_fn = self.image_label_fn or _default_image_label
        graphs = [g for g in (image_graphs or []) if isinstance(g, nx.Graph)]
        for image_graph in graphs:
            if image_graph.number_of_nodes() == 0:
                continue
            role_by_img = cached_image_node_hash_map(
                image_graph,
                radius=self.image_cut_radius,
                image_label_fn=image_label_fn,
            )
            assoc_by_img: dict = {}
            for img_node, data in image_graph.nodes(data=True):
                assoc = data.get("association")
                if not isinstance(assoc, nx.Graph):
                    continue
                assoc_by_img[img_node] = int(hash_graph(assoc))
            if not assoc_by_img:
                continue

            degrees = dict(image_graph.degree())
            max_deg = max(degrees.values()) if degrees else 0
            start_nodes = [n for n, d in degrees.items() if d == max_deg and n in assoc_by_img]
            for start_node in start_nodes:
                role_hash = role_by_img.get(start_node)
                assoc_hash = assoc_by_img[start_node]
                counts[(None, role_hash)][assoc_hash] += 1

            for u, v in image_graph.edges():
                if u not in assoc_by_img or v not in assoc_by_img:
                    continue
                u_assoc = assoc_by_img[u]
                v_assoc = assoc_by_img[v]
                u_role = role_by_img.get(u)
                v_role = role_by_img.get(v)
                counts[(u_assoc, v_role)][v_assoc] += 1
                counts[(v_assoc, u_role)][u_assoc] += 1
        self._transition_fallback_counts = counts

    def _rebuild_fallback_indexes(self, *, image_graphs: Sequence[nx.Graph]) -> None:
        """
        Rebuild all fallback indexes derived from cut-index and image graphs.

        Args:
            image_graphs: Conditioned image graphs used for transition fallback.

        Returns:
            None.
        """
        self._build_cut_entry_lookup()
        self._build_transition_fallback_index(image_graphs=image_graphs)
        self._root_rr_cursor_by_role = {}

    def _transition_fallback_entries(
        self,
        *,
        image_graph: nx.Graph,
        img_node,
        assigned_images: set,
        image_assoc_hashes: Optional[dict],
        image_hash_map: dict,
        rng: random.Random,
        for_primary: bool = False,
    ) -> list[CutArchiveEntry]:
        """
        Retrieve fallback entries suggested by observed association transitions.

        Args:
            image_graph: Conditioned image graph.
            img_node: Current image node to materialize.
            assigned_images: Assigned image nodes in current DFS state.
            image_assoc_hashes: Mapping assigned image node -> chosen assoc hash.
            image_hash_map: Cached role hash per image node.
            rng: Random generator for tie-breaking.
            for_primary: If True, apply primary-specific root sampling policy.

        Returns:
            list[CutArchiveEntry]: Ordered candidate entries from fallback index.
        """
        if not bool(getattr(self, "use_transition_fallback", True)):
            return []
        role_hash = image_hash_map.get(img_node)
        if not self._cut_entries_by_role_assoc:
            return []
        if not self._transition_fallback_counts:
            return []

        score_by_assoc = Counter()
        has_neighbor_evidence = False
        assigned_neighbors = [
            n for n in image_graph.neighbors(img_node) if n in assigned_images
        ]
        if image_assoc_hashes:
            for neigh in assigned_neighbors:
                prev_assoc = image_assoc_hashes.get(neigh)
                if prev_assoc is None:
                    continue
                has_neighbor_evidence = True
                for assoc_hash, cnt in self._transition_fallback_counts.get(
                    (int(prev_assoc), role_hash), {}
                ).items():
                    score_by_assoc[int(assoc_hash)] += int(cnt)

        if not score_by_assoc and not has_neighbor_evidence:
            for assoc_hash, cnt in self._transition_fallback_counts.get((None, role_hash), {}).items():
                score_by_assoc[int(assoc_hash)] += int(cnt)
        if not score_by_assoc:
            return []

        is_root = not has_neighbor_evidence
        use_uniform_root = bool(
            for_primary
            and is_root
            and getattr(self, "transition_primary_uniform_root", True)
        )
        use_rr_root = bool(
            for_primary
            and is_root
            and getattr(self, "transition_primary_round_robin_root", False)
        )
        if use_rr_root:
            assoc_keys = sorted(int(k) for k in score_by_assoc.keys())
            if assoc_keys:
                cursor = int(self._root_rr_cursor_by_role.get(role_hash, 0)) % len(assoc_keys)
                rotated = assoc_keys[cursor:] + assoc_keys[:cursor]
                self._root_rr_cursor_by_role[role_hash] = (cursor + 1) % len(assoc_keys)
                ranked = [(int(k), int(score_by_assoc.get(k, 0))) for k in rotated]
            else:
                ranked = []
        elif use_uniform_root:
            assoc_keys = list(score_by_assoc.keys())
            rng.shuffle(assoc_keys)
            ranked = [(int(k), int(score_by_assoc.get(k, 0))) for k in assoc_keys]
        else:
            ranked = list(score_by_assoc.items())
            rng.shuffle(ranked)
            ranked.sort(key=lambda item: item[1], reverse=True)
        budget = max(1, int(getattr(self, "transition_fallback_entry_budget", 256)))
        out: list[CutArchiveEntry] = []
        for assoc_hash, _score in ranked:
            key = (role_hash, int(assoc_hash))
            entries = list(self._cut_entries_by_role_assoc.get(key, []))
            if not entries:
                continue
            rng.shuffle(entries)
            out.extend(entries)
            if len(out) >= budget:
                break
        if len(out) > budget:
            out = out[:budget]
        return out

    def fit(
        self,
        generator_graphs: Sequence[nx.Graph],
        *,
        n_pruning_iterations: Optional[int] = None,
        constraint_level: Optional[int] = None,
        use_context_embedding: Optional[bool] = None,
        seed: Optional[int] = None,
        label_mode: Optional[str] = None,
        verbose: bool = False,
    ) -> "ConditionalAutoregressiveGenerator":
        """
        Build and cache the cut index from generator graphs.

        Args:
            generator_graphs: Source graphs used to build the cut index.
            n_pruning_iterations: Override number of pruning iterations per graph.
            constraint_level: Optional override for cut radii.
            use_context_embedding: Optional override for context embeddings.
            seed: RNG seed for pruning order.
            label_mode: Optional label mode override for image graphs.
            verbose: If True, emit warning context for recoverable failures.

        Returns:
            ConditionalAutoregressiveGenerator: Self.
        """
        if self.decomposition_function is None:
            raise ValueError("decomposition_function must be provided at initialization.")
        if self.nbits is None:
            raise ValueError("nbits must be provided at initialization.")
        graphs = list(generator_graphs or [])
        if not graphs:
            raise ValueError("generator_graphs must be non-empty.")

        if label_mode is not None:
            self.label_mode = label_mode
        if constraint_level is not None:
            self.constraint_level = constraint_level
            try:
                m = int(self.constraint_level)
            except Exception:
                m = None
            if m is not None:
                self.preimage_cut_radius = max(0, (m + 1) // 2)
                self.image_cut_radius = max(0, m // 2)
                self.cut_radius = self.preimage_cut_radius
                self.preimage_context_radius = (self.preimage_cut_radius or 0) + 1
                self.image_context_radius = self.image_cut_radius + 1
        if use_context_embedding is not None:
            self.use_context_embedding = bool(use_context_embedding)

        n_iters = self.n_pruning_iterations if n_pruning_iterations is None else int(n_pruning_iterations)

        seed = self._resolve_seed(seed)
        cut_index, donors, fixed_image_graph, _assoc_map = build_image_conditioned_cut_index_from_pruning(
            graphs,
            self.decomposition_function,
            int(self.nbits),
            n_pruning_iterations=n_iters,
            preimage_cut_radius=self.preimage_cut_radius,
            image_cut_radius=self.image_cut_radius,
            preimage_context_radius=self.preimage_context_radius,
            image_context_radius=self.image_context_radius,
            context_vectorizer=self.context_vectorizer,
            use_context_embedding=self.use_context_embedding,
            max_num_anchors=self.max_num_anchors,
            max_num_anchor_sets_retry=self.max_num_anchor_sets_retry,
            seed=seed,
            label_mode=self.label_mode,
        )

        self.generator_graphs = graphs
        self.cut_index = cut_index
        self.fixed_image_graph = fixed_image_graph
        self._last_assoc_hash_by_image = {}
        self._reject_counts_by_state_assoc = defaultdict(float)
        self._reject_state_scores = defaultdict(float)
        self._reject_pair_scores = defaultdict(float)
        if verbose:
            self._print_cut_size_summary(cut_index)

        # Cache per-generator image graphs for sampling when image_graph=None.
        self.image_graphs = [
            graph_to_abstract_graph(
                g,
                decomposition_function=self.decomposition_function,
                nbits=int(self.nbits),
                label_mode=self.label_mode,
            ).image_graph.copy()
            for g in graphs
        ]
        self._rebuild_fallback_indexes(image_graphs=self.image_graphs)

        if self.feasibility_estimator is not None:
            fit_source = self.feasibility_fit_source
            if fit_source == "pruned":
                fit_graphs = donors
            else:
                fit_graphs = graphs
            try:
                self.feasibility_estimator.fit(fit_graphs)
            except Exception as exc:
                self._warn_verbose(
                    "feasibility_estimator.fit failed during fit(); continuing without fitted feasibility model",
                    exc=exc,
                    verbose=verbose,
                )

        return self


    def prepare_from_graphs(
        self,
        graphs: Sequence[nx.Graph],
        *,
        generator_size: int = 1,
        n_pruning_iterations: Optional[int] = None,
        constraint_level: Optional[int] = None,
        use_context_embedding: Optional[bool] = None,
        mode: str = "same_image",
        label_mode: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> tuple[list[nx.Graph], Optional[nx.Graph]]:
        """
        Prepare cut index and image graphs for generation from a graph pool.

        Args:
            graphs: Candidate preimage graphs.
            generator_size: Number of generator graphs to use.
            n_pruning_iterations: Override number of pruning iterations per graph.
            constraint_level: Optional override for cut radii.
            use_context_embedding: Optional override for context embeddings.
            mode: 'same_image' or 'nn' selection strategy.
            label_mode: Optional label mode override.
            seed: RNG seed for selection and pruning.
            verbose: If True, print selection diagnostics.

        Returns:
            tuple[list[nx.Graph], Optional[nx.Graph]]: (generator_graphs, image_graph).
        """
        if label_mode is not None:
            self.label_mode = label_mode
        if constraint_level is not None:
            self.constraint_level = constraint_level
            try:
                m = int(self.constraint_level)
            except Exception as exc:
                m = None
                self._warn_verbose(
                    "Invalid constraint_level value in prepare_from_graphs(); keeping current cut/context radii",
                    exc=exc,
                    verbose=verbose,
                )
            if m is not None:
                self.preimage_cut_radius = max(0, (m + 1) // 2)
                self.image_cut_radius = max(0, m // 2)
                self.cut_radius = self.preimage_cut_radius
                self.preimage_context_radius = (self.preimage_cut_radius or 0) + 1
                self.image_context_radius = self.image_cut_radius + 1
        if use_context_embedding is not None:
            self.use_context_embedding = bool(use_context_embedding)

        if mode not in ("same_image", "nn"):
            raise ValueError("mode must be 'same_image' or 'nn'")
        seed = self._resolve_seed(seed)
        rng = random.Random(seed)

        if mode == "nn":
            vec = self.context_vectorizer
            if vec is None:
                raise ValueError("generator.context_vectorizer is required for mode='nn'")
            import numpy as _np
            feats = _np.asarray(vec.fit_transform(list(graphs)), dtype=float)
            if feats.ndim != 2 or feats.shape[0] != len(list(graphs)):
                raise ValueError('vectorizer returned invalid shape')
            seed_idx = rng.randrange(len(graphs))
            ref = feats[seed_idx]
            norms = _np.linalg.norm(feats, axis=1)
            rn = _np.linalg.norm(ref)
            if rn == 0.0:
                rn = 1.0
            norms[norms == 0.0] = 1.0
            sims = (feats @ ref) / (norms * rn)
            order = _np.argsort(-sims)
            k = max(1, min(int(generator_size), len(graphs)))
            sel_idx = order[:k].tolist()
            generator_graphs = [graphs[i] for i in sel_idx]
        else:
            buckets = group_graphs_by_image_hash(
                graphs,
                decomposition_function=self.decomposition_function,
                nbits=self.nbits,
                label_mode=self.label_mode,
            )
            generator_graphs, used_fallback = select_image_group(
                buckets,
                generator_size,
                rng=rng,
            )
            if used_fallback and verbose:
                print('No image-graph group meets generator_size; using largest available group.')
            if not generator_graphs:
                return [], None

        if verbose:
            print(f'Generator graph group size: {len(generator_graphs)}')

        image_graphs = None
        image_graph = None
        if mode == "same_image":
            ag = graph_to_abstract_graph(
                generator_graphs[0],
                decomposition_function=self.decomposition_function,
                nbits=self.nbits,
                label_mode=self.label_mode,
            )
            image_graph = ag.image_graph.copy()
            image_graphs = [image_graph]
        elif mode == "nn":
            image_graphs = [
                graph_to_abstract_graph(
                    g,
                    decomposition_function=self.decomposition_function,
                    nbits=self.nbits,
                    label_mode=self.label_mode,
                ).image_graph.copy()
                for g in generator_graphs
            ]
            image_graph = rng.choice(image_graphs) if image_graphs else None

        n_iters = self.n_pruning_iterations if n_pruning_iterations is None else int(n_pruning_iterations)
        cut_index, donors, _image_graph, _assoc_map = build_image_conditioned_cut_index_from_pruning(
            generator_graphs,
            self.decomposition_function,
            int(self.nbits),
            n_pruning_iterations=n_iters,
            preimage_cut_radius=self.preimage_cut_radius,
            image_cut_radius=self.image_cut_radius,
            preimage_context_radius=self.preimage_context_radius,
            image_context_radius=self.image_context_radius,
            context_vectorizer=self.context_vectorizer,
            use_context_embedding=self.use_context_embedding,
            max_num_anchors=self.max_num_anchors,
            max_num_anchor_sets_retry=self.max_num_anchor_sets_retry,
            seed=seed,
            label_mode=self.label_mode,
        )

        self.generator_graphs = list(generator_graphs)
        self.cut_index = cut_index
        self.fixed_image_graph = image_graph if mode == "same_image" else None
        self.last_prepared_image_graph = image_graph.copy() if isinstance(image_graph, nx.Graph) else None
        self.image_graphs = list(image_graphs) if image_graphs is not None else ([] if image_graph is None else [image_graph])
        self._rebuild_fallback_indexes(image_graphs=self.image_graphs)
        self._last_assoc_hash_by_image = {}
        self._reject_counts_by_state_assoc = defaultdict(float)
        self._reject_state_scores = defaultdict(float)
        self._reject_pair_scores = defaultdict(float)
        if verbose:
            self._print_cut_size_summary(cut_index)

        if self.feasibility_estimator is not None:
            fit_source = self.feasibility_fit_source
            fit_graphs = donors if fit_source == "pruned" else generator_graphs
            try:
                self.feasibility_estimator.fit(fit_graphs)
            except Exception as exc:
                self._warn_verbose(
                    "feasibility_estimator.fit failed during prepare_from_graphs(); continuing",
                    exc=exc,
                    verbose=verbose,
                )

        return generator_graphs, image_graph

    def generate_from_graphs(
        self,
        graphs: Sequence[nx.Graph],
        *,
        n_samples: int = 1,
        generator_size: int = 1,
        n_pruning_iterations: Optional[int] = None,
        constraint_level: Optional[int] = None,
        use_context_embedding: Optional[bool] = None,
        mode: str = "same_image",
        label_mode: Optional[str] = None,
        max_attempts_multiplier: int = 30,
        return_history: bool = False,
        verbose: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Prepare from graphs and generate samples in one call.

        Args:
            graphs: Candidate preimage graphs.
            n_samples: Number of samples to generate.
            generator_size: Number of generator graphs to use.
            n_pruning_iterations: Override pruning iterations.
            constraint_level: Optional override for cut radii.
            use_context_embedding: Optional override for context embeddings.
            mode: 'same_image' or 'nn' selection strategy.
            label_mode: Optional label mode override.
            max_attempts_multiplier: Attempts per sample multiplier.
            return_history: If True, return histories.
            verbose: If True, print progress.
            seed: RNG seed for selection and pruning.

        Returns:
            list[nx.Graph] | tuple[list[nx.Graph], list[list[nx.Graph]]]: Samples (and histories).
        """
        generator_graphs, image_graph = self.prepare_from_graphs(
            graphs,
            generator_size=generator_size,
            n_pruning_iterations=n_pruning_iterations,
            constraint_level=constraint_level,
            use_context_embedding=use_context_embedding,
            mode=mode,
            label_mode=label_mode,
            seed=seed,
            verbose=verbose,
        )
        if not generator_graphs:
            return [] if not return_history else ([], [])
        seed = self._resolve_seed(seed)
        max_attempts = max(1, int(n_samples) * max(1, int(max_attempts_multiplier)))
        image_graph_for_generation = image_graph if mode == "same_image" else None
        return self.generate(
            n_samples=n_samples,
            image_graph=image_graph_for_generation,
            rng=random.Random(seed),
            return_history=return_history,
            max_attempts=max_attempts,
            verbose=verbose,
        )

    def _pick_training_image_graph(
        self,
        *,
        image_graph: Optional[nx.Graph],
        rng: random.Random,
    ) -> nx.Graph:
        """
        Choose the image graph to condition on.

        Args:
            image_graph: Optional image graph provided by the caller.
            rng: Random generator used for tie-breaking.

        Returns:
            nx.Graph: Image graph used for generation.
        """
        if image_graph is not None:
            return image_graph.copy()
        candidates = list(getattr(self, "image_graphs", []))
        if candidates:
            return rng.choice(candidates).copy()
        default_graph = getattr(self, "fixed_image_graph", None)
        if isinstance(default_graph, nx.Graph):
            return default_graph.copy()
        raise ValueError(
            "No image_graph provided and no training image graphs stored. "
            "Set generator.image_graphs or generator.fixed_image_graph."
        )

    @staticmethod
    def _bfs_image_order(image_graph: nx.Graph, start_node, rng: random.Random) -> list:
        """
        Produce a breadth-first traversal order over the image graph.
        Neighbor visitation order is randomized using the provided RNG.

        Args:
            image_graph: Image graph to traverse.
            start_node: Node id to use as BFS root.
            rng: Random generator used to shuffle neighbor visitation order.

        Returns:
            list: BFS order of image nodes.
        """
        if start_node is None or start_node not in image_graph:
            return list(image_graph.nodes())
        visited = set([start_node])
        order = [start_node]
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            neighbors = [n for n in image_graph.neighbors(node) if n not in visited]
            rng.shuffle(neighbors)
            for neigh in neighbors:
                if neigh in visited:
                    continue
                visited.add(neigh)
                order.append(neigh)
                queue.append(neigh)
        return order

    @staticmethod
    def _highest_degree_node(image_graph: nx.Graph, rng: random.Random):
        """
        Pick the image node with highest degree (random tie-break).

        Args:
            image_graph: Image graph to inspect.
            rng: Random generator for tie-breaking.

        Returns:
            Optional[node]: Selected node id or None if graph is empty.
        """
        if image_graph.number_of_nodes() == 0:
            return None
        degrees = dict(image_graph.degree())
        max_deg = max(degrees.values())
        candidates = [n for n, d in degrees.items() if d == max_deg]
        return rng.choice(candidates)

    def _is_feasible(self, graph: nx.Graph) -> bool:
        """
        Check whether a candidate graph is feasible under the estimator.

        Args:
            graph: Candidate preimage graph.

        Returns:
            bool: True if feasible or if no estimator is configured.
        """
        if not getattr(self, "apply_feasibility_during_construction", True):
            return True
        if self.feasibility_estimator is None:
            return True
        try:
            preds = self.feasibility_estimator.predict([graph])
            return bool(preds[0])
        except Exception as exc:
            self._warn_verbose(
                "feasibility_estimator.predict failed during construction-time feasibility check; rejecting candidate",
                exc=exc,
            )
            return False

    def _build_anchor_context(
        self,
        *,
        current: nx.Graph,
        image_graph: nx.Graph,
        img_node,
        assigned_images: set,
        image_associations: dict,
    ) -> Optional[dict]:
        """
        Build the anchor context needed for anchored insertion.

        Args:
            current: Current preimage graph.
            image_graph: Fixed image graph (context).
            img_node: Image node id being inserted.
            assigned_images: Set of already assigned image nodes.
            image_associations: Mapping image_node -> set of preimage nodes.

        Returns:
            Optional[dict]: Anchor context dictionary or None if anchors are missing.
        """
        assigned_neighbors = [
            n for n in image_graph.neighbors(img_node) if n in assigned_images
        ]
        if not assigned_neighbors:
            return None
        current_assoc_map = (
            _assoc_map_from_image_associations(image_associations)
            if image_associations
            else {}
        )
        anchor_pool = set()
        for neigh in assigned_neighbors:
            anchor_pool |= set(image_associations.get(neigh, set()))
        if not anchor_pool:
            return None
        preimage_label_fn = self.preimage_label_fn or _default_preimage_label
        image_label_fn = self.image_label_fn or _default_image_label
        pre_cut_radius = self.preimage_cut_radius
        pre_hash_map = cached_preimage_node_hash_map(
            current,
            radius=pre_cut_radius,
            preimage_label_fn=preimage_label_fn,
        )
        outer_hash_by_node = {
            node: pre_hash_map.get(node, 0)
            for node in anchor_pool
        }
        outer_hash_to_nodes: dict = defaultdict(list)
        for node, h in outer_hash_by_node.items():
            outer_hash_to_nodes[h].append(node)
        image_hash_map = cached_image_node_hash_map(
            image_graph,
            radius=self.image_cut_radius,
            image_label_fn=image_label_fn,
        )
        image_cut_hash = image_hash_map.get(img_node)
        return {
            "assigned_neighbor_set": set(assigned_neighbors),
            "current_assoc_map": current_assoc_map,
            "anchor_pool": anchor_pool,
            "outer_hash_by_node": outer_hash_by_node,
            "outer_hash_to_nodes": outer_hash_to_nodes,
            "image_cut_hash": image_cut_hash,
        }

    def _respects_image_edge_constraints(
        self,
        *,
        image_graph: nx.Graph,
        img_node,
        assigned_images: set,
        image_associations: dict,
        id_map: dict,
    ) -> bool:
        """
        Check that a candidate does not induce forbidden image-graph edges.

        A forbidden edge occurs when the candidate association shares at least one
        preimage node with an already assigned image node that is not adjacent to
        `img_node` in the conditioned image graph.

        Args:
            image_graph: Conditioned image graph.
            img_node: Image node currently being materialized.
            assigned_images: Already assigned image nodes.
            image_associations: Mapping image_node -> associated preimage nodes.
            id_map: Mapping donor assoc node -> current preimage node ids.

        Returns:
            bool: True if no forbidden overlaps are present.
        """
        if not bool(getattr(self, "enforce_image_edge_constraints", True)):
            return True
        candidate_nodes = set(id_map.values())
        if not candidate_nodes:
            return True
        for other_img in assigned_images:
            other_nodes = set(image_associations.get(other_img, set()))
            actual_shared = len(candidate_nodes & other_nodes)

            if image_graph.has_edge(img_node, other_img):
                edge_data = image_graph.get_edge_data(img_node, other_img) or {}
                expected_shared = edge_data.get("shared_preimage_nodes")
                # If edge overlap metadata exists, enforce exact overlap equality.
                if expected_shared is not None and actual_shared != int(expected_shared):
                    return False
                continue

            if actual_shared > 0:
                return False
        return True

    def _entry_matches_assigned_neighbor_overlap_signature(
        self,
        *,
        entry: CutArchiveEntry,
        image_graph: nx.Graph,
        image_hash_map: dict,
        img_node,
        assigned_images: set,
    ) -> bool:
        """
        Check whether an entry can satisfy assigned-neighbor overlap requirements.

        The check only applies to image edges that carry `shared_preimage_nodes`.
        """
        entry_signature = getattr(entry, "neighbor_overlap_signature", ())
        if not entry_signature:
            return True

        required = Counter(
            self._required_neighbor_overlap_signature(
                image_graph=image_graph,
                image_hash_map=image_hash_map,
                img_node=img_node,
                assigned_images=assigned_images,
            )
        )

        if not required:
            return True

        available = Counter(entry_signature)
        return all(available.get(key, 0) >= count for key, count in required.items())

    def _required_neighbor_overlap_signature(
        self,
        *,
        image_graph: nx.Graph,
        image_hash_map: dict,
        img_node,
        assigned_images: set,
    ) -> tuple:
        """Return required (neighbor_role_hash, shared_count) pairs for assigned neighbors."""
        required = []
        for neigh in assigned_images:
            if not image_graph.has_edge(img_node, neigh):
                continue
            edge_data = image_graph.get_edge_data(img_node, neigh) or {}
            shared = edge_data.get("shared_preimage_nodes")
            if shared is None:
                continue
            try:
                shared = int(shared)
            except Exception:
                continue
            neigh_role_hash = image_hash_map.get(neigh)
            required.append((neigh_role_hash, shared))
        required.sort()
        return tuple(required)

    def _pool_cut_hashes_for_size(
        self,
        size: int,
        *,
        anchor_pool: set,
        outer_hash_by_node: dict,
        image_cut_hash: Optional[int],
        rng: random.Random,
        cache: dict,
    ) -> tuple[set[int], bool]:
        """
        Compute or retrieve cut hashes for all/sampled anchor subsets of a size.

        Args:
            size: Anchor subset size to evaluate.
            anchor_pool: Candidate preimage anchor nodes.
            outer_hash_by_node: Mapping preimage node -> outer hash.
            image_cut_hash: Image-context hash to include in cut hash.
            rng: Random generator for sampling.
            cache: Cache dictionary keyed by size.

        Returns:
            tuple[set[int], bool]: (hashes, complete) where complete=True means
                all subsets were enumerated.
        """
        if size in cache:
            return cache[size]
        if size <= 0 or size > len(anchor_pool):
            result = (set(), True)
            cache[size] = result
            return result
        nodes = list(anchor_pool)
        max_subsets = max(1, int(self.max_num_anchor_sets_retry))
        try:
            total = math.comb(len(nodes), size)
        except Exception:
            total = max_subsets + 1
        hashes: set[int] = set()
        complete = total <= max_subsets
        if complete:
            for subset in itertools.combinations(nodes, size):
                hashes.add(
                    _cut_hash_with_image(
                        [outer_hash_by_node[n] for n in subset],
                        image_cut_hash,
                    )
                )
        else:
            attempts = 0
            while len(hashes) < max_subsets and attempts < max_subsets * 5:
                subset = rng.sample(nodes, size)
                hashes.add(
                    _cut_hash_with_image(
                        [outer_hash_by_node[n] for n in subset],
                        image_cut_hash,
                    )
                )
                attempts += 1
        cache[size] = (hashes, complete)
        return hashes, complete

    def _context_scoring_enabled(self) -> bool:
        """
        Return whether context-embedding scoring is enabled.

        Returns:
            bool: True when enabled and a vectorizer is available.
        """
        return bool(self.use_context_embedding and self.context_vectorizer is not None)

    def _image_node_context_embedding(self, image_graph: nx.Graph, img_node):
        """
        Compute the context embedding for the current image node.

        Args:
            image_graph: Conditioned image graph.
            img_node: Current image node id.

        Returns:
            Optional[object]: Context embedding, or None if unavailable.
        """
        if not self._context_scoring_enabled():
            return None
        if image_graph is None or img_node not in image_graph:
            return None
        return cached_context_embedding_sum_for_nodes(
            image_graph,
            nodes=[img_node],
            radius=self.image_context_radius,
            context_vectorizer=self.context_vectorizer,
        )

    def _preimage_anchor_context_embedding(self, preimage_graph: nx.Graph, anchor_nodes):
        """
        Compute the summed context embedding for mapped anchor nodes.

        Args:
            preimage_graph: Current preimage graph.
            anchor_nodes: Anchor nodes in the current preimage graph.

        Returns:
            Optional[object]: Summed anchor context embedding, or None.
        """
        if not self._context_scoring_enabled():
            return None
        nodes = [n for n in anchor_nodes if n in preimage_graph]
        if not nodes:
            return None
        return cached_context_embedding_sum_for_nodes(
            preimage_graph,
            nodes=nodes,
            radius=self.preimage_context_radius,
            context_vectorizer=self.context_vectorizer,
        )

    @staticmethod
    def _similarity_to_weight(similarity: Optional[float]) -> Optional[float]:
        """
        Convert cosine similarity in [-1, 1] to a non-negative sampling weight.

        Args:
            similarity: Cosine similarity.

        Returns:
            Optional[float]: Weight in [0, 1], or None if unavailable.
        """
        if similarity is None:
            return None
        return max(0.0, (float(similarity) + 1.0) * 0.5)

    def _context_weight_for_entry(
        self,
        *,
        entry: CutArchiveEntry,
        current_image_ctx=None,
        current_preimage_ctx=None,
    ) -> float:
        """
        Compute context-driven sampling weight for one cut entry.

        Args:
            entry: Candidate cut entry.
            current_image_ctx: Current image-node context embedding.
            current_preimage_ctx: Current mapped-anchor context embedding.

        Returns:
            float: Multiplicative context weight.
        """
        if not self._context_scoring_enabled():
            return 1.0
        scores: list[float] = []
        image_weight = self._similarity_to_weight(
            _cosine_similarity(current_image_ctx, entry.image_ctx)
        )
        if image_weight is not None:
            scores.append(float(image_weight))
        preimage_weight = self._similarity_to_weight(
            _cosine_similarity(current_preimage_ctx, entry.preimage_ctx)
        )
        if preimage_weight is not None:
            scores.append(float(preimage_weight))
        if not scores:
            return 1.0
        return float(sum(scores) / len(scores))

    def _usage_weight(self, *, role_hash, assoc_hash: int, usage_counts: dict) -> float:
        """
        Compute usage-penalty weight for a candidate entry.

        Args:
            role_hash: Role hash key for the current image node.
            assoc_hash: Association hash for the candidate entry.
            usage_counts: Path-local usage counter.

        Returns:
            float: Multiplicative weight.
        """
        beta = float(getattr(self, "usage_penalty_beta", 0.0))
        if beta <= 0.0:
            return 1.0
        used = int(usage_counts.get((role_hash, assoc_hash), 0))
        return 1.0 / ((1.0 + float(used)) ** beta)

    def _register_reject(
        self,
        *,
        state_key: tuple,
        assoc_hash: int,
        reason: str,
        image_graph: Optional[nx.Graph] = None,
        img_node=None,
        image_assoc_hashes: Optional[dict] = None,
    ) -> None:
        """Accumulate failure counts used to downweight repeated dead candidates."""
        self._mark_preimage_hash_reject(int(assoc_hash))
        # Propagate a discounted reject signal to already-assigned neighboring
        # preimage hashes (distance-1 parents around the current image node).
        if (
            image_graph is not None
            and image_assoc_hashes
            and (img_node in image_graph)
        ):
            parent_decay = float(getattr(self, "pairwise_rejection_discount_alpha", 0.6))
            if parent_decay > 0.0:
                for neigh in image_graph.neighbors(img_node):
                    parent_hash = image_assoc_hashes.get(neigh)
                    if parent_hash is None:
                        continue
                    self._mark_preimage_hash_reject(int(parent_hash), weight=parent_decay)
        if not state_key:
            return
        if reason == "edge":
            weight = int(self.failure_penalty_edge_weight)
        elif reason == "feasibility":
            weight = int(self.failure_penalty_feasibility_weight)
        else:
            weight = int(self.failure_penalty_anchor_weight)
        key = (state_key, int(assoc_hash))
        self._reject_counts_by_state_assoc[key] = float(
            self._reject_counts_by_state_assoc.get(key, 0.0) + float(weight)
        )
        self._register_pairwise_reject(
            assoc_hash=int(assoc_hash),
            reason=reason,
            image_graph=image_graph,
            img_node=img_node,
            image_assoc_hashes=image_assoc_hashes,
        )

    def _register_pairwise_reject(
        self,
        *,
        assoc_hash: int,
        reason: str,
        image_graph: Optional[nx.Graph],
        img_node,
        image_assoc_hashes: Optional[dict],
    ) -> None:
        """Accumulate discounted pairwise rejection penalties by image-hop distance."""
        if not bool(getattr(self, "enable_pairwise_rejection_learning", False)):
            return
        if image_graph is None or img_node not in image_graph:
            return
        if not image_assoc_hashes:
            return
        if reason == "edge":
            base = float(getattr(self, "pairwise_rejection_edge_weight", 0.0))
        elif reason == "feasibility":
            base = float(getattr(self, "pairwise_rejection_feasibility_weight", 0.0))
        else:
            base = float(getattr(self, "pairwise_rejection_anchor_weight", 0.0))
        if base <= 0.0:
            return
        alpha = float(getattr(self, "pairwise_rejection_discount_alpha", 0.6))
        if alpha <= 0.0:
            return

        lengths = nx.single_source_shortest_path_length(image_graph, img_node)
        target_hash = int(assoc_hash)
        for other_img, prev_hash in image_assoc_hashes.items():
            if other_img == img_node:
                continue
            if prev_hash is None:
                continue
            dist = lengths.get(other_img)
            if dist is None or dist <= 0:
                continue
            delta = base * (alpha ** float(dist))
            key = (int(prev_hash), target_hash)
            self._reject_pair_scores[key] = float(
            self._reject_pair_scores.get(key, 0.0) + float(delta)
            )

    def _mark_preimage_hash_accept(self, assoc_hash: int, *, weight: float = 1.0) -> None:
        """Record one successful acceptance for a candidate assoc hash."""
        if weight <= 0.0:
            return
        key = int(assoc_hash)
        bucket = self._preimage_hash_accept_stats[key]
        bucket["success"] = float(bucket.get("success", 0.0)) + float(weight)

    def _mark_preimage_hash_reject(self, assoc_hash: int, *, weight: float = 1.0) -> None:
        """Record one rejection for a candidate assoc hash."""
        if weight <= 0.0:
            return
        key = int(assoc_hash)
        bucket = self._preimage_hash_accept_stats[key]
        bucket["fail"] = float(bucket.get("fail", 0.0)) + float(weight)

    def _preimage_hash_accept_prob(self, assoc_hash) -> Optional[float]:
        """
        Return empirical acceptance probability for a candidate assoc hash.

        Unseen hashes default to 1.0. If a hash has only failures, probability is 0.
        """
        if assoc_hash is None:
            return None
        try:
            key = int(assoc_hash)
        except Exception:
            return None
        bucket = self._preimage_hash_accept_stats.get(key)
        if not bucket:
            return 1.0
        succ = max(0.0, float(bucket.get("success", 0.0)))
        fail = max(0.0, float(bucket.get("fail", 0.0)))
        total = succ + fail
        if total <= 0:
            return 1.0
        return float(succ / float(total))

    def _parent_hashes_with_probs(
        self,
        *,
        image_graph: Optional[nx.Graph],
        img_node,
        assigned_images: set,
        image_assoc_hashes: Optional[dict],
    ) -> list[tuple[int, float]]:
        """Return assigned neighbor assoc hashes with their current accept probs."""
        if image_graph is None or image_assoc_hashes is None or img_node not in image_graph:
            return []
        out: list[tuple[int, float]] = []
        for neigh in image_graph.neighbors(img_node):
            if neigh not in assigned_images:
                continue
            assoc_hash = image_assoc_hashes.get(neigh)
            if assoc_hash is None:
                continue
            h = int(assoc_hash)
            p = self._preimage_hash_accept_prob(h)
            out.append((h, 1.0 if p is None else float(p)))
        out.sort(key=lambda t: t[0])
        return out

    def _pairwise_reject_weight(
        self,
        *,
        assoc_hash: int,
        image_graph: Optional[nx.Graph],
        img_node,
        image_assoc_hashes: Optional[dict],
    ) -> float:
        """Return multiplicative downweight from learned pairwise reject scores."""
        if not bool(getattr(self, "enable_pairwise_rejection_learning", False)):
            return 1.0
        beta = float(getattr(self, "pairwise_rejection_beta", 0.0))
        if beta <= 0.0:
            return 1.0
        if image_graph is None or img_node not in image_graph:
            return 1.0
        if not image_assoc_hashes:
            return 1.0
        alpha = float(getattr(self, "pairwise_rejection_discount_alpha", 0.6))
        if alpha <= 0.0:
            return 1.0

        lengths = nx.single_source_shortest_path_length(image_graph, img_node)
        score = 0.0
        target_hash = int(assoc_hash)
        for other_img, prev_hash in image_assoc_hashes.items():
            if other_img == img_node or prev_hash is None:
                continue
            dist = lengths.get(other_img)
            if dist is None or dist <= 0:
                continue
            pair_penalty = float(self._reject_pair_scores.get((int(prev_hash), target_hash), 0.0))
            if pair_penalty <= 0.0:
                continue
            score += pair_penalty * (alpha ** float(dist))
        if score <= 0.0:
            return 1.0
        return 1.0 / ((1.0 + score) ** beta)

    def _failure_weight_for_entry(
        self,
        *,
        state_key: tuple,
        assoc_hash: int,
    ) -> float:
        """Return multiplicative weight that decays with repeated failures."""
        beta = float(getattr(self, "failure_penalty_beta", 0.0))
        if beta <= 0.0 or not state_key:
            return 1.0
        key = (state_key, int(assoc_hash))
        penalty = float(self._reject_counts_by_state_assoc.get(key, 0.0))
        if penalty <= 0.0:
            return 1.0
        return 1.0 / ((1.0 + penalty) ** beta)

    def _state_weight(self, *, state_key: tuple) -> float:
        """Return multiplicative weight from accumulated state-level failures."""
        beta = float(getattr(self, "state_penalty_beta", 0.0))
        if beta <= 0.0 or not state_key:
            return 1.0
        penalty = float(self._reject_state_scores.get(state_key, 0.0))
        if penalty <= 0.0:
            return 1.0
        return 1.0 / ((1.0 + penalty) ** beta)

    def _backpropagate_state_failure(self, *, path_states: list[tuple]) -> None:
        """
        Apply exponentially decayed penalties to all states on a failing DFS path.
        """
        if not path_states:
            return
        alpha = float(getattr(self, "state_backprop_alpha", 0.0))
        lam = float(getattr(self, "state_backprop_lambda", 0.0))
        if alpha <= 0.0:
            return
        for dist, state_key in enumerate(reversed(path_states)):
            if not state_key:
                continue
            penalty = alpha * math.exp(-lam * float(dist))
            self._reject_state_scores[state_key] = float(
                self._reject_state_scores.get(state_key, 0.0) + penalty
            )

    def _decay_dead_state_memory(self) -> None:
        """Apply per-attempt decay to state-failure memory to allow recovery."""
        decay = float(getattr(self, "dead_state_decay_per_attempt", 1.0))
        if decay >= 1.0:
            return
        if not self._reject_state_scores:
            return
        for key in list(self._reject_state_scores.keys()):
            val = float(self._reject_state_scores.get(key, 0.0)) * decay
            if val <= 1e-9:
                self._reject_state_scores.pop(key, None)
            else:
                self._reject_state_scores[key] = val

    def _should_skip_dead_state(self, *, state_key: tuple, rng: random.Random) -> bool:
        """Return True when a state is heavily penalized and should be skipped."""
        if not state_key:
            return False
        threshold = float(getattr(self, "dead_state_skip_threshold", 0.0))
        if threshold <= 0.0:
            return False
        score = float(self._reject_state_scores.get(state_key, 0.0))
        if score < threshold:
            return False
        escape = float(getattr(self, "dead_state_skip_escape_prob", 0.0))
        if escape > 0.0 and rng.random() < escape:
            return False
        return True

    @staticmethod
    def _weighted_order_without_replacement(
        items: Sequence,
        weights: Sequence[float],
        rng: random.Random,
    ) -> list:
        """
        Return a weighted random order without replacement.

        Args:
            items: Items to order.
            weights: Non-negative weights aligned with items.
            rng: Random generator.

        Returns:
            list: Items in weighted random order.
        """
        if not items:
            return []
        if len(items) != len(weights):
            out = list(items)
            rng.shuffle(out)
            return out
        cleaned = [max(0.0, float(w)) for w in weights]
        if not any(w > 0.0 for w in cleaned):
            out = list(items)
            rng.shuffle(out)
            return out
        keyed = []
        for item, weight in zip(items, cleaned):
            if weight <= 0.0:
                key = float("inf")
            else:
                u = max(1e-12, rng.random())
                key = -math.log(u) / weight
            keyed.append((key, item))
        keyed.sort(key=lambda t: t[0])
        return [item for _, item in keyed]

    def _order_candidate_records(
        self,
        *,
        candidate_records: list[tuple[tuple[nx.Graph, dict, int, Optional[int]], float]],
        target_candidates: int,
        rng: random.Random,
    ) -> list[tuple[nx.Graph, dict, int, Optional[int]]]:
        """
        Order candidate records with context bias plus optional random exploration.

        Args:
            candidate_records: List of (candidate_tuple, weight).
            target_candidates: Number of leading candidates to prioritize.
            rng: Random generator.

        Returns:
            list[tuple[nx.Graph, dict, int, Optional[int]]]: Ordered candidates.
        """
        if not candidate_records:
            return []
        records = [rec for rec, _w in candidate_records]
        weights = [w for _rec, w in candidate_records]
        weighted_idx = self._weighted_order_without_replacement(
            list(range(len(records))),
            weights,
            rng,
        )
        if not self._context_scoring_enabled() or target_candidates <= 1:
            return [records[i] for i in weighted_idx]
        frac = float(getattr(self, "context_exploration_fraction", 0.0))
        if frac <= 0.0:
            return [records[i] for i in weighted_idx]
        k = max(1, int(target_candidates))
        k_explore = min(k - 1, int(round(k * frac)))
        k_model = max(1, k - k_explore)
        prefix = weighted_idx[:k_model]
        prefix_set = set(prefix)
        remaining = [i for i in range(len(records)) if i not in prefix_set]
        rng.shuffle(remaining)
        explore = remaining[:k_explore]
        ordered_prefix = prefix + explore
        ordered_prefix_set = set(ordered_prefix)
        ordered_idx = ordered_prefix + [i for i in weighted_idx if i not in ordered_prefix_set]
        return [records[i] for i in ordered_idx]

    def _entry_accept_prob_weight(self, assoc_hash: int) -> float:
        """
        Return non-negative sampling weight from learned accept probability.

        Args:
            assoc_hash: Candidate association hash.

        Returns:
            float: Weight used for entry subset sampling.
        """
        p = self._preimage_hash_accept_prob(int(assoc_hash))
        if p is None:
            return 1.0
        return max(0.0, float(p))

    def _sample_entries_by_accept_prob(
        self,
        *,
        entries: Sequence[CutArchiveEntry],
        subset_size: int,
        rng: random.Random,
    ) -> list[CutArchiveEntry]:
        """
        Sample/reorder entries using learned accept-probability weights.

        This avoids uniform pre-sampling followed by probability-based rejection.
        """
        if not entries:
            return []
        size = max(1, int(subset_size))
        items = list(entries)
        weights = [self._entry_accept_prob_weight(int(e.assoc_hash)) for e in items]
        if any(w > 0.0 for w in weights):
            order = self._weighted_order_without_replacement(
                list(range(len(items))),
                weights,
                rng,
            )
        else:
            order = list(range(len(items)))
            rng.shuffle(order)
        ordered = [items[i] for i in order]
        if len(ordered) > size:
            return ordered[:size]
        return ordered

    def _materialize_anchorless(
        self,
        entries: Sequence[CutArchiveEntry],
        *,
        current: nx.Graph,
        rng: random.Random,
        img_node=None,
    ) -> Optional[tuple[nx.Graph, dict]]:
        """
        Materialize an anchorless association, filtering with feasibility when set.

        Args:
            entries: Candidate CutArchiveEntry list.
            current: Current preimage graph.
            rng: Random generator for selection.

        Returns:
            Optional[tuple[nx.Graph, dict]]: (new_graph, id_map) if successful.
        """
        entries = list(entries)
        if not entries:
            return None
        last_hash = self._last_assoc_hash_by_image.get(img_node) if img_node is not None else None
        if last_hash is not None:
            filtered = [e for e in entries if e.assoc_hash != last_hash]
            if filtered:
                entries = filtered
        anchorless = [e for e in entries if not e.anchor_pairs]
        if anchorless:
            pool = anchorless
        else:
            current_nodes = set(current.nodes())
            disjoint = [e for e in entries if set(e.assoc.nodes()).isdisjoint(current_nodes)]
            pool = disjoint if disjoint else entries
        rng.shuffle(pool)
        for entry in pool:
            candidate, id_map = _merge_with_anchors(current, entry.assoc, {})
            if self._is_feasible(candidate):
                if img_node is not None:
                    self._last_assoc_hash_by_image[img_node] = entry.assoc_hash
                return candidate, id_map
        return None

    def _materialize_with_anchors(
        self,
        entries: Sequence[CutArchiveEntry],
        *,
        current: nx.Graph,
        anchor_ctx: dict,
        rng: random.Random,
        img_node=None,
    ) -> Optional[tuple[nx.Graph, dict]]:
        """
        Materialize an association using anchor matching, filtering with feasibility when set.

        Args:
            entries: Candidate CutArchiveEntry list.
            current: Current preimage graph.
            anchor_ctx: Anchor context dictionary from _build_anchor_context.
            rng: Random generator for sampling.

        Returns:
            Optional[tuple[nx.Graph, dict]]: (new_graph, id_map) if successful.
        """
        outer_hash_to_nodes = anchor_ctx["outer_hash_to_nodes"]
        assigned_neighbor_set = anchor_ctx["assigned_neighbor_set"]
        current_assoc_map = anchor_ctx["current_assoc_map"]
        anchor_pool = anchor_ctx["anchor_pool"]
        outer_hash_by_node = anchor_ctx["outer_hash_by_node"]
        image_cut_hash = anchor_ctx["image_cut_hash"]
        pool_cut_hashes: dict[int, tuple[set[int], bool]] = {}

        entries = list(entries)
        last_hash = self._last_assoc_hash_by_image.get(img_node) if img_node is not None else None
        if last_hash is not None:
            filtered = [e for e in entries if e.assoc_hash != last_hash]
            if filtered:
                entries = filtered
        rng.shuffle(entries)
        for entry in entries:
            if not self._entry_passes_anchor_filters(
                entry,
                outer_hash_to_nodes=outer_hash_to_nodes,
                image_cut_hash=image_cut_hash,
                anchor_pool=anchor_pool,
                outer_hash_by_node=outer_hash_by_node,
                rng=rng,
                pool_cache=pool_cut_hashes,
            ):
                continue
            mapping = self._sample_anchor_mapping(
                entry,
                outer_hash_to_nodes=outer_hash_to_nodes,
                assigned_neighbor_set=assigned_neighbor_set,
                current_assoc_map=current_assoc_map,
                rng=rng,
            )
            if mapping is None:
                continue
            candidate, id_map = _merge_with_anchors(current, entry.assoc, mapping)
            if self._is_feasible(candidate):
                if img_node is not None:
                    self._last_assoc_hash_by_image[img_node] = entry.assoc_hash
                return candidate, id_map
        return None

    def _entry_passes_anchor_filters(
        self,
        entry: CutArchiveEntry,
        *,
        outer_hash_to_nodes: dict,
        image_cut_hash: Optional[int],
        anchor_pool: set,
        outer_hash_by_node: dict,
        rng: random.Random,
        pool_cache: dict,
        enforce_cut_hash: bool = True,
        enforce_outer_count: bool = True,
        stats: Optional[Counter] = None,
    ) -> bool:
        """
        Check whether an entry is compatible with the available anchor pool.

        Args:
            entry: Candidate CutArchiveEntry.
            outer_hash_to_nodes: Mapping outer hash -> preimage nodes.
            image_cut_hash: Image-context hash.
            anchor_pool: Candidate preimage anchor nodes.
            outer_hash_by_node: Mapping preimage node -> outer hash.
            rng: Random generator for subset sampling.
            pool_cache: Cache of cut hashes by size.

        Returns:
            bool: True if the entry passes hash/size filters.
        """
        anchor_pairs = list(entry.anchor_pairs)
        if not anchor_pairs:
            return False
        entry_outer_hashes = [outer_h for outer_h, _inner_h, _inner_id in anchor_pairs]
        outer_counts = Counter(entry_outer_hashes)
        if enforce_outer_count:
            if any(len(outer_hash_to_nodes.get(h, [])) < c for h, c in outer_counts.items()):
                if stats is not None:
                    stats["anchor_outer_count_reject"] += 1
                    stats["anchor_filter_reject"] += 1
                return False
        else:
            if not any(len(outer_hash_to_nodes.get(h, [])) > 0 for h in outer_counts):
                if stats is not None:
                    stats["anchor_outer_empty_reject"] += 1
                    stats["anchor_filter_reject"] += 1
                return False
        anchor_size = len(entry_outer_hashes)
        if enforce_cut_hash and anchor_size <= self.max_num_anchors:
            hashes, complete = self._pool_cut_hashes_for_size(
                anchor_size,
                anchor_pool=anchor_pool,
                outer_hash_by_node=outer_hash_by_node,
                image_cut_hash=image_cut_hash,
                rng=rng,
                cache=pool_cache,
            )
            entry_cut_hash = _cut_hash_with_image(entry_outer_hashes, image_cut_hash)
            if complete and entry_cut_hash not in hashes:
                if stats is not None:
                    stats["anchor_cut_hash_reject"] += 1
                    stats["anchor_filter_reject"] += 1
                return False
        return True

    def _sample_anchor_mapping(
        self,
        entry: CutArchiveEntry,
        *,
        outer_hash_to_nodes: dict,
        assigned_neighbor_set: set,
        current_assoc_map: dict,
        rng: random.Random,
        max_attempts_override: Optional[int] = None,
        allow_partial: bool = False,
        require_full_neighbor_coverage: bool = True,
    ) -> Optional[dict]:
        """
        Sample a mapping from entry anchor nodes to existing preimage nodes.

        Args:
            entry: Candidate CutArchiveEntry.
            outer_hash_to_nodes: Mapping outer hash -> preimage nodes.
            assigned_neighbor_set: Assigned image neighbors for coverage checks.
            current_assoc_map: Mapping preimage node -> assigned image nodes.
            rng: Random generator for sampling.

        Returns:
            Optional[dict]: Mapping from entry anchor node -> preimage node.
        """
        anchor_pairs = list(entry.anchor_pairs)
        if not anchor_pairs:
            return None
        outer_to_inner_ids: dict = defaultdict(list)
        for outer_h, _inner_h, inner_id in anchor_pairs:
            outer_to_inner_ids[outer_h].append(inner_id)
        max_attempts = max(1, int(max_attempts_override if max_attempts_override is not None else self.max_anchor_matches))
        for _ in range(max_attempts):
            attempt = {}
            used = set()
            failed = False
            for outer_h, inner_ids in outer_to_inner_ids.items():
                outer_candidates = list(outer_hash_to_nodes.get(outer_h, []))
                rng.shuffle(outer_candidates)
                if allow_partial:
                    available = [n for n in outer_candidates if n not in used]
                    if not available:
                        continue
                    rng.shuffle(inner_ids)
                    n_to_map = min(len(inner_ids), len(available))
                    selected_inner_ids = inner_ids[:n_to_map]
                else:
                    selected_inner_ids = inner_ids
                for inner_id in selected_inner_ids:
                    choices = [n for n in outer_candidates if n not in used]
                    if not choices:
                        failed = True
                        break
                    pick = rng.choice(choices)
                    used.add(pick)
                    attempt[inner_id] = pick
                if failed:
                    break
            if failed:
                continue
            if not attempt:
                continue
            covered_neighbors = set()
            for mapped_node in attempt.values():
                covered_neighbors |= (
                    current_assoc_map.get(mapped_node, set()) & assigned_neighbor_set
                )
            if assigned_neighbor_set:
                if require_full_neighbor_coverage:
                    if not assigned_neighbor_set.issubset(covered_neighbors):
                        continue
                else:
                    if not covered_neighbors:
                        continue
            return attempt
        return None

    def _materialize_image_node(
        self,
        *,
        current: nx.Graph,
        image_graph: nx.Graph,
        img_node,
        rng: random.Random,
        assigned_images: set,
        image_associations: dict,
        image_assoc_hashes: Optional[dict] = None,
    ) -> Optional[tuple[nx.Graph, dict]]:
        """
        Materialize the preimage association for a single image node.

        This tries to anchor on already-assigned neighbor associations when possible.
        If no neighbor anchors exist, it inserts an anchorless association.

        Args:
            current: Current preimage graph.
            image_graph: Fixed image graph (context).
            img_node: Image node id to materialize.
            rng: Random generator for selection.
            assigned_images: Set of already assigned image nodes.
            image_associations: Mapping image_node -> set of preimage nodes.
            image_assoc_hashes: Optional mapping image_node -> selected assoc hash.

        Returns:
            Optional[tuple[nx.Graph, dict]]: (new_graph, id_map) if successful.
        """
        candidates = self._candidate_materializations_for_image_node(
            current=current,
            image_graph=image_graph,
            img_node=img_node,
            rng=rng,
            assigned_images=assigned_images,
            image_associations=image_associations,
            image_assoc_hashes=image_assoc_hashes,
            usage_counts={},
        )
        if not candidates:
            return None
        candidate_graph, id_map, assoc_hash, _role_hash = candidates[0]
        if img_node is not None:
            self._last_assoc_hash_by_image[img_node] = int(assoc_hash)
        return candidate_graph, id_map

    def _candidate_materializations_for_image_node(
        self,
        *,
        current: nx.Graph,
        image_graph: nx.Graph,
        img_node,
        rng: random.Random,
        assigned_images: set,
        image_associations: dict,
        image_assoc_hashes: Optional[dict],
        usage_counts: dict,
        entries_override: Optional[Sequence[CutArchiveEntry]] = None,
        use_transition_fallback: bool = True,
        use_transition_primary: bool = True,
        emit_stats: bool = True,
    ) -> list[tuple[nx.Graph, dict, int, Optional[int]]]:
        """
        Enumerate feasible materializations for a specific BFS-selected image node.

        Args:
            current: Current preimage graph.
            image_graph: Fixed image graph (context).
            img_node: Image node id to materialize.
            rng: Random generator for candidate ordering/sampling.
            assigned_images: Set of already assigned image nodes.
            image_associations: Mapping image_node -> set of preimage nodes.
            image_assoc_hashes: Mapping image_node -> selected assoc hash.
            usage_counts: Path-local usage count for (role_hash, assoc_hash).
            entries_override: Optional candidate-entry pool override.
            use_transition_fallback: Whether to trigger transition fallback when
                the current candidate source yields no feasible options.
            use_transition_primary: Whether to query transition fallback before
                the normal cut-index path.
            emit_stats: If False, suppress per-call debug stat printing.

        Returns:
            list[tuple[nx.Graph, dict, int, Optional[int]]]:
                Feasible (new_graph, id_map, assoc_hash, role_hash) candidates.
        """
        stats = Counter()

        def _print_stats(selected_preimage_hash=None) -> None:
            if (not self._runtime_verbose) or (not emit_stats):
                return
            try:
                p_accept = self._preimage_hash_accept_prob(selected_preimage_hash)
                parent_pairs = self._parent_hashes_with_probs(
                    image_graph=image_graph,
                    img_node=img_node,
                    assigned_images=assigned_images,
                    image_assoc_hashes=image_assoc_hashes,
                )
                if not parent_pairs:
                    parent_field = "-"
                else:
                    parent_field = ",".join(
                        f"{h}:{p:.4f}" for h, p in parent_pairs[:8]
                    )
                    if len(parent_pairs) > 8:
                        parent_field += f",...(+{len(parent_pairs) - 8})"
                fields = [
                    (
                        "done",
                        (
                            f"{int(getattr(self, '_runtime_completed_n_samples', 0))}/"
                            f"{int(getattr(self, '_runtime_target_n_samples', 0))}"
                            if getattr(self, "_runtime_target_n_samples", None) is not None
                            else "-"
                        ),
                    ),
                    (
                        "assigned",
                        (
                            f"{len(assigned_images)}/{int(image_graph.number_of_nodes())}"
                            if isinstance(image_graph, nx.Graph)
                            else "-"
                        ),
                    ),
                    ("backtracks", int(getattr(self, "_runtime_sample_backtracks", 0))),
                    ("dead_state_skips", int(getattr(self, "_runtime_dead_state_skips", 0))),
                    ("img_node", img_node),
                    (
                        "preimage_hash",
                        "-" if selected_preimage_hash is None else int(selected_preimage_hash),
                    ),
                    ("accept_prob", "-" if p_accept is None else f"{p_accept:.4f}"),
                    ("parent_preimages", parent_field),
                    ("entries", int(stats.get("entries", 0))),
                    ("zero_prob_skip", int(stats.get("zero_prob_skip", 0))),
                    ("anchor_filter_reject", int(stats.get("anchor_filter_reject", 0))),
                    ("anchor_outer_count_reject", int(stats.get("anchor_outer_count_reject", 0))),
                    ("anchor_outer_empty_reject", int(stats.get("anchor_outer_empty_reject", 0))),
                    ("anchor_cut_hash_reject", int(stats.get("anchor_cut_hash_reject", 0))),
                    ("anchor_map_reject", int(stats.get("anchor_map_reject", 0))),
                    ("edge_constraint_reject", int(stats.get("edge_constraint_reject", 0))),
                    ("feasibility_reject", int(stats.get("feasibility_reject", 0))),
                    ("relaxed_anchor_phase", int(stats.get("relaxed_anchor_phase", 0))),
                    ("partial_anchor_phase", int(stats.get("partial_anchor_phase", 0))),
                    ("transition_primary", int(stats.get("transition_primary", 0))),
                    ("transition_fallback", int(stats.get("transition_fallback", 0))),
                    ("accepted", int(stats.get("accepted", 0))),
                ]
                tokens = []
                for key, value in fields:
                    if isinstance(value, int):
                        tokens.append(f"{key}={value:>7d}")
                    else:
                        tokens.append(f"{key}={str(value):>7}")
                print(
                    "[candidate-stats] " + " ".join(tokens)
                )
            except Exception:
                pass

        def _finalize(
            candidates: list[tuple[nx.Graph, dict, int, Optional[int]]]
        ) -> list[tuple[nx.Graph, dict, int, Optional[int]]]:
            selected_hash = int(candidates[0][2]) if candidates else None
            # Count success only for the committed choice at the outer call.
            if emit_stats and selected_hash is not None:
                self._mark_preimage_hash_accept(selected_hash)
            _print_stats(selected_hash)
            self._accumulate_generation_stats(stats)
            return candidates

        image_hash_map = cached_image_node_hash_map(
            image_graph,
            radius=self.image_cut_radius,
            image_label_fn=_default_image_label,
        )
        role_hash = image_hash_map.get(img_node)
        # Policy: donor lookup is permutation-invariant and keyed only by
        # rooted image-context hash (no image node id in the cut-index key).
        if entries_override is not None:
            entries = list(entries_override)
        else:
            entries = list(self.cut_index.get(role_hash, []))
        stats["entries"] += len(entries)
        if use_transition_primary and entries_override is None and bool(getattr(self, "use_transition_primary", False)):
            primary_entries = self._transition_fallback_entries(
                image_graph=image_graph,
                img_node=img_node,
                assigned_images=assigned_images,
                image_assoc_hashes=image_assoc_hashes,
                image_hash_map=image_hash_map,
                rng=rng,
                for_primary=True,
            )
            if primary_entries:
                stats["transition_primary"] += 1
                primary_candidates = self._candidate_materializations_for_image_node(
                    current=current,
                    image_graph=image_graph,
                    img_node=img_node,
                    rng=rng,
                    assigned_images=assigned_images,
                    image_associations=image_associations,
                    image_assoc_hashes=image_assoc_hashes,
                    usage_counts=usage_counts,
                    entries_override=primary_entries,
                    use_transition_fallback=False,
                    use_transition_primary=False,
                    emit_stats=False,
                )
                if primary_candidates:
                    stats["transition_primary_success"] += 1
                    stats["accepted"] += len(primary_candidates)
                    return _finalize(primary_candidates)
        if not entries:
            return _finalize([])

        assigned_neighbors = [
            n for n in image_graph.neighbors(img_node) if n in assigned_images
        ]
        required_overlap_signature = self._required_neighbor_overlap_signature(
            image_graph=image_graph,
            image_hash_map=image_hash_map,
            img_node=img_node,
            assigned_images=assigned_images,
        )
        state_key = (role_hash, required_overlap_signature)
        anchor_ctx = self._build_anchor_context(
            current=current,
            image_graph=image_graph,
            img_node=img_node,
            assigned_images=assigned_images,
            image_associations=image_associations,
        )
        if anchor_ctx is None and assigned_neighbors:
            return _finalize([])

        max_candidates = max(1, int(self.max_anchor_matches))
        # Bound per-node candidate work; DFS later keeps only a small prefix.
        candidate_budget = max(1, int(self.max_candidates_per_bfs_node)) * max(
            1, int(getattr(self, "candidate_budget_multiplier", 8))
        )
        target_candidates = min(max_candidates, candidate_budget)
        # Randomly pre-sample a larger pool of rule entries before context scoring.
        entry_pool_budget = max(
            target_candidates,
            candidate_budget * max(1, int(getattr(self, "entry_pool_budget_multiplier", 4))),
        )
        current_image_ctx = self._image_node_context_embedding(image_graph, img_node)
        candidate_records: list[tuple[tuple[nx.Graph, dict, int, Optional[int]], float]] = []
        base_retries = max(1, int(self.max_entry_subset_retries))
        retry_phases = [
            {
                "retries": base_retries,
                "candidate_budget": candidate_budget,
                "entry_pool_budget": entry_pool_budget,
                "use_context": True,
                "max_anchor_attempts": None,
            }
        ]
        if bool(getattr(self, "adaptive_entry_retry", True)):
            retry_mult = max(2, int(getattr(self, "adaptive_entry_retry_multiplier", 4)))
            retry_phases.append(
                {
                    "retries": max(base_retries, base_retries * retry_mult),
                    "candidate_budget": max(candidate_budget, candidate_budget * retry_mult),
                    "entry_pool_budget": max(entry_pool_budget, entry_pool_budget * retry_mult),
                    "use_context": not bool(getattr(self, "adaptive_retry_disable_context", True)),
                    "max_anchor_attempts": max(1, int(self.max_anchor_matches)) * retry_mult,
                }
            )
            retry_phases.append(
                {
                    "retries": max(base_retries, base_retries * retry_mult),
                    "candidate_budget": max(candidate_budget, candidate_budget * retry_mult),
                    "entry_pool_budget": max(entry_pool_budget, entry_pool_budget * retry_mult),
                    "use_context": False,
                    "max_anchor_attempts": max(1, int(self.max_anchor_matches)) * retry_mult,
                    "relaxed_anchor_filter": True,
                    "relaxed_outer_count": True,
                    "allow_partial_anchor_mapping": True,
                    "require_full_neighbor_coverage": False,
                }
            )

        if anchor_ctx is None:
            anchorless = [e for e in entries if not e.anchor_pairs]
            if anchorless:
                pool = anchorless
            else:
                current_nodes = set(current.nodes())
                disjoint = [e for e in entries if set(e.assoc.nodes()).isdisjoint(current_nodes)]
                pool = disjoint if disjoint else entries
            pool = list(pool)
            for phase_idx, phase in enumerate(retry_phases):
                if phase_idx > 0:
                    stats["adaptive_retry"] += 1
                candidate_records = []
                phase_retries = max(1, int(phase["retries"]))
                phase_candidate_budget = max(1, int(phase["candidate_budget"]))
                phase_entry_pool_budget = max(1, int(phase["entry_pool_budget"]))
                phase_use_context = bool(phase["use_context"])
                phase_image_ctx = current_image_ctx if phase_use_context else None
                for _retry in range(phase_retries):
                    candidate_records = []
                    sampled_pool = self._sample_entries_by_accept_prob(
                        entries=pool,
                        subset_size=phase_entry_pool_budget,
                        rng=rng,
                    )
                    for entry in sampled_pool:
                        assoc_hash = int(entry.assoc_hash)
                        if not self._entry_matches_assigned_neighbor_overlap_signature(
                            entry=entry,
                            image_graph=image_graph,
                            image_hash_map=image_hash_map,
                            img_node=img_node,
                            assigned_images=assigned_images,
                        ):
                            stats["edge_constraint_reject"] += 1
                            self._register_reject(
                                state_key=state_key,
                                assoc_hash=assoc_hash,
                                reason="edge",
                                image_graph=image_graph,
                                img_node=img_node,
                                image_assoc_hashes=image_assoc_hashes,
                            )
                            continue
                        candidate, id_map = _merge_with_anchors(current, entry.assoc, {})
                        if not self._respects_image_edge_constraints(
                            image_graph=image_graph,
                            img_node=img_node,
                            assigned_images=assigned_images,
                            image_associations=image_associations,
                            id_map=id_map,
                        ):
                            stats["edge_constraint_reject"] += 1
                            self._register_reject(
                                state_key=state_key,
                                assoc_hash=assoc_hash,
                                reason="edge",
                                image_graph=image_graph,
                                img_node=img_node,
                                image_assoc_hashes=image_assoc_hashes,
                            )
                            continue
                        if not self._is_feasible(candidate):
                            stats["feasibility_reject"] += 1
                            self._register_reject(
                                state_key=state_key,
                                assoc_hash=assoc_hash,
                                reason="feasibility",
                                image_graph=image_graph,
                                img_node=img_node,
                                image_assoc_hashes=image_assoc_hashes,
                            )
                            continue
                        context_weight = self._context_weight_for_entry(
                            entry=entry,
                            current_image_ctx=phase_image_ctx,
                            current_preimage_ctx=None,
                        )
                        usage_weight = self._usage_weight(
                            role_hash=role_hash,
                            assoc_hash=assoc_hash,
                            usage_counts=usage_counts,
                        )
                        failure_weight = self._failure_weight_for_entry(
                            state_key=state_key,
                            assoc_hash=assoc_hash,
                        )
                        pairwise_weight = self._pairwise_reject_weight(
                            assoc_hash=assoc_hash,
                            image_graph=image_graph,
                            img_node=img_node,
                            image_assoc_hashes=image_assoc_hashes,
                        )
                        state_weight = self._state_weight(state_key=state_key)
                        weight = max(
                            0.0,
                            float(context_weight)
                            * float(usage_weight)
                            * float(failure_weight)
                            * float(pairwise_weight)
                            * float(state_weight),
                        )
                        candidate_records.append(
                            ((candidate, id_map, assoc_hash, role_hash), weight)
                        )
                        stats["accepted"] += 1
                        if len(candidate_records) >= phase_candidate_budget:
                            break
                    if candidate_records:
                        ordered = self._order_candidate_records(
                            candidate_records=candidate_records,
                            target_candidates=target_candidates,
                            rng=rng,
                        )
                        return _finalize(ordered[:target_candidates])
            if use_transition_fallback and entries_override is None:
                fallback_entries = self._transition_fallback_entries(
                    image_graph=image_graph,
                    img_node=img_node,
                    assigned_images=assigned_images,
                    image_assoc_hashes=image_assoc_hashes,
                    image_hash_map=image_hash_map,
                    rng=rng,
                    for_primary=False,
                )
                if fallback_entries:
                    stats["transition_fallback"] += 1
                    fallback_candidates = self._candidate_materializations_for_image_node(
                        current=current,
                        image_graph=image_graph,
                        img_node=img_node,
                        rng=rng,
                        assigned_images=assigned_images,
                        image_associations=image_associations,
                        image_assoc_hashes=image_assoc_hashes,
                        usage_counts=usage_counts,
                        entries_override=fallback_entries,
                        use_transition_fallback=False,
                        emit_stats=False,
                    )
                    if fallback_candidates:
                        stats["transition_fallback_success"] += 1
                        stats["accepted"] += len(fallback_candidates)
                        return _finalize(fallback_candidates)
            return _finalize([])

        outer_hash_to_nodes = anchor_ctx["outer_hash_to_nodes"]
        assigned_neighbor_set = anchor_ctx["assigned_neighbor_set"]
        current_assoc_map = anchor_ctx["current_assoc_map"]
        anchor_pool = anchor_ctx["anchor_pool"]
        outer_hash_by_node = anchor_ctx["outer_hash_by_node"]
        image_cut_hash = anchor_ctx["image_cut_hash"]
        pool_cut_hashes: dict[int, tuple[set[int], bool]] = {}

        entries = list(entries)
        for phase_idx, phase in enumerate(retry_phases):
            if phase_idx > 0:
                stats["adaptive_retry"] += 1
            phase_retries = max(1, int(phase["retries"]))
            phase_candidate_budget = max(1, int(phase["candidate_budget"]))
            phase_entry_pool_budget = max(1, int(phase["entry_pool_budget"]))
            phase_use_context = bool(phase["use_context"])
            phase_image_ctx = current_image_ctx if phase_use_context else None
            phase_anchor_attempts = phase.get("max_anchor_attempts")
            phase_relaxed_anchor_filter = bool(phase.get("relaxed_anchor_filter", False))
            phase_relaxed_outer_count = bool(phase.get("relaxed_outer_count", False))
            phase_allow_partial_anchor_mapping = bool(phase.get("allow_partial_anchor_mapping", False))
            phase_require_full_neighbor_coverage = bool(phase.get("require_full_neighbor_coverage", True))
            if phase_relaxed_anchor_filter:
                stats["relaxed_anchor_phase"] += 1
            if phase_allow_partial_anchor_mapping:
                stats["partial_anchor_phase"] += 1
            for _retry in range(phase_retries):
                candidate_records = []
                pool_cut_hashes = {}
                sampled_entries = self._sample_entries_by_accept_prob(
                    entries=entries,
                    subset_size=phase_entry_pool_budget,
                    rng=rng,
                )
                pre_ctx_cache: dict[tuple, Optional[object]] = {}
                for entry in sampled_entries:
                    assoc_hash = int(entry.assoc_hash)
                    if not self._entry_matches_assigned_neighbor_overlap_signature(
                        entry=entry,
                        image_graph=image_graph,
                        image_hash_map=image_hash_map,
                        img_node=img_node,
                        assigned_images=assigned_images,
                    ):
                        stats["edge_constraint_reject"] += 1
                        self._register_reject(
                            state_key=state_key,
                            assoc_hash=assoc_hash,
                            reason="edge",
                            image_graph=image_graph,
                            img_node=img_node,
                            image_assoc_hashes=image_assoc_hashes,
                        )
                        continue
                    if not self._entry_passes_anchor_filters(
                        entry,
                        outer_hash_to_nodes=outer_hash_to_nodes,
                        image_cut_hash=image_cut_hash,
                        anchor_pool=anchor_pool,
                        outer_hash_by_node=outer_hash_by_node,
                        rng=rng,
                        pool_cache=pool_cut_hashes,
                        enforce_cut_hash=not phase_relaxed_anchor_filter,
                        enforce_outer_count=not phase_relaxed_outer_count,
                        stats=stats,
                    ):
                        continue
                    mapping = self._sample_anchor_mapping(
                        entry,
                        outer_hash_to_nodes=outer_hash_to_nodes,
                        assigned_neighbor_set=assigned_neighbor_set,
                        current_assoc_map=current_assoc_map,
                        rng=rng,
                        max_attempts_override=phase_anchor_attempts,
                        allow_partial=phase_allow_partial_anchor_mapping,
                        require_full_neighbor_coverage=phase_require_full_neighbor_coverage,
                    )
                    if mapping is None:
                        stats["anchor_map_reject"] += 1
                        self._register_reject(
                            state_key=state_key,
                            assoc_hash=assoc_hash,
                            reason="anchor",
                            image_graph=image_graph,
                            img_node=img_node,
                            image_assoc_hashes=image_assoc_hashes,
                        )
                        continue
                    candidate, id_map = _merge_with_anchors(current, entry.assoc, mapping)
                    if not self._respects_image_edge_constraints(
                        image_graph=image_graph,
                        img_node=img_node,
                        assigned_images=assigned_images,
                        image_associations=image_associations,
                        id_map=id_map,
                    ):
                        stats["edge_constraint_reject"] += 1
                        self._register_reject(
                            state_key=state_key,
                            assoc_hash=assoc_hash,
                            reason="edge",
                            image_graph=image_graph,
                            img_node=img_node,
                            image_assoc_hashes=image_assoc_hashes,
                        )
                        continue
                    if not self._is_feasible(candidate):
                        stats["feasibility_reject"] += 1
                        self._register_reject(
                            state_key=state_key,
                            assoc_hash=assoc_hash,
                            reason="feasibility",
                            image_graph=image_graph,
                            img_node=img_node,
                            image_assoc_hashes=image_assoc_hashes,
                        )
                        continue
                    mapped_nodes_key = tuple(sorted(mapping.values()))
                    if mapped_nodes_key not in pre_ctx_cache:
                        pre_ctx_cache[mapped_nodes_key] = self._preimage_anchor_context_embedding(
                            current,
                            list(mapped_nodes_key),
                        )
                    current_preimage_ctx = pre_ctx_cache[mapped_nodes_key] if phase_use_context else None
                    context_weight = self._context_weight_for_entry(
                        entry=entry,
                        current_image_ctx=phase_image_ctx,
                        current_preimage_ctx=current_preimage_ctx,
                    )
                    usage_weight = self._usage_weight(
                        role_hash=role_hash,
                        assoc_hash=assoc_hash,
                        usage_counts=usage_counts,
                    )
                    failure_weight = self._failure_weight_for_entry(
                        state_key=state_key,
                        assoc_hash=assoc_hash,
                    )
                    pairwise_weight = self._pairwise_reject_weight(
                        assoc_hash=assoc_hash,
                        image_graph=image_graph,
                        img_node=img_node,
                        image_assoc_hashes=image_assoc_hashes,
                    )
                    state_weight = self._state_weight(state_key=state_key)
                    weight = max(
                        0.0,
                        float(context_weight)
                        * float(usage_weight)
                        * float(failure_weight)
                        * float(pairwise_weight)
                        * float(state_weight),
                    )
                    candidate_records.append(
                        ((candidate, id_map, assoc_hash, role_hash), weight)
                    )
                    stats["accepted"] += 1
                    if len(candidate_records) >= phase_candidate_budget:
                        break
                if candidate_records:
                    ordered = self._order_candidate_records(
                        candidate_records=candidate_records,
                        target_candidates=target_candidates,
                        rng=rng,
                    )
                    return _finalize(ordered[:target_candidates])
        if use_transition_fallback and entries_override is None:
            fallback_entries = self._transition_fallback_entries(
                image_graph=image_graph,
                img_node=img_node,
                assigned_images=assigned_images,
                image_assoc_hashes=image_assoc_hashes,
                image_hash_map=image_hash_map,
                rng=rng,
                for_primary=False,
            )
            if fallback_entries:
                stats["transition_fallback"] += 1
                fallback_candidates = self._candidate_materializations_for_image_node(
                    current=current,
                    image_graph=image_graph,
                    img_node=img_node,
                    rng=rng,
                    assigned_images=assigned_images,
                    image_associations=image_associations,
                    image_assoc_hashes=image_assoc_hashes,
                    usage_counts=usage_counts,
                    entries_override=fallback_entries,
                    use_transition_fallback=False,
                    emit_stats=False,
                )
                if fallback_candidates:
                    stats["transition_fallback_success"] += 1
                    stats["accepted"] += len(fallback_candidates)
                    return _finalize(fallback_candidates)
        return _finalize([])

    def _generate_once(
        self,
        *,
        image_graph: Optional[nx.Graph],
        rng: Optional[random.Random] = None,
        rng_seed: Optional[int] = None,
        return_history: bool = False,
    ):
        """
        Attempt to generate a single sample.

        Args:
            image_graph: Optional fixed image graph.
            rng: Optional Random instance.
            rng_seed: Optional seed for a new Random instance.
            return_history: If True, also return intermediate graphs.

        Returns:
            tuple: (graph or None, history or None)
        """
        if rng is None:
            rng = random.Random(rng_seed)
        img = self._pick_training_image_graph(image_graph=image_graph, rng=rng)
        current = nx.Graph()
        # Disable graph-attached conditional caches on the evolving preimage
        # graph to avoid large attribute copies during DFS branching.
        current.graph["__ag_disable_conditional_cache__"] = True
        # Policy: each history state carries the exact conditioned image graph
        # used for this sample, so visualization is step-consistent.
        current.graph["conditioned_image_graph"] = img.copy()
        current.graph["assigned_images"] = set()
        current.graph["image_associations"] = {}
        current.graph["image_assoc_hashes"] = {}
        history0: list[nx.Graph] = [current.copy()]
        t0 = time.perf_counter()
        dfs_nodes = 0
        backtracks = 0
        self._runtime_sample_backtracks = 0
        self._runtime_dead_state_skips = 0
        self._decay_dead_state_memory()

        start_node = self._highest_degree_node(img, rng)
        order = self._bfs_image_order(img, start_node, rng)
        image_hash_map = cached_image_node_hash_map(
            img,
            radius=self.image_cut_radius,
            image_label_fn=_default_image_label,
        )
        # Policy: materialization order is fixed by highest-degree-rooted BFS.
        if not order:
            return current, history0 if return_history else None

        def _dfs(
            idx: int,
            current_graph: nx.Graph,
            assigned_images: set,
            image_associations: dict,
            image_assoc_hashes: dict,
            history: list[nx.Graph],
            usage_counts: dict,
            tried_by_state: dict,
            path_states: list[tuple],
        ) -> Optional[tuple[nx.Graph, list[nx.Graph]]]:
            # Policy: terminate quickly by bounding DFS search effort.
            nonlocal dfs_nodes, backtracks
            dfs_nodes += 1
            if dfs_nodes > self.max_dfs_nodes:
                return None
            if (time.perf_counter() - t0) > self.max_dfs_seconds:
                return None
            if idx >= len(order):
                return current_graph, history
            current_img_node = order[idx]
            required_sig = self._required_neighbor_overlap_signature(
                image_graph=img,
                image_hash_map=image_hash_map,
                img_node=current_img_node,
                assigned_images=assigned_images,
            )
            state_key = (image_hash_map.get(current_img_node), required_sig)
            if self._should_skip_dead_state(state_key=state_key, rng=rng):
                self._runtime_dead_state_skips = int(
                    getattr(self, "_runtime_dead_state_skips", 0) + 1
                )
                return None
            candidates = self._candidate_materializations_for_image_node(
                current=current_graph,
                image_graph=img,
                img_node=current_img_node,
                rng=rng,
                assigned_images=assigned_images,
                image_associations=image_associations,
                image_assoc_hashes=image_assoc_hashes,
                usage_counts=usage_counts,
            )
            if not candidates:
                self._backpropagate_state_failure(path_states=path_states)
                return None
            if len(candidates) > self.max_candidates_per_bfs_node:
                # Policy: cap branching factor at each BFS step.
                candidates = candidates[: self.max_candidates_per_bfs_node]
            tabu = tried_by_state.setdefault(state_key, set())
            next_path_states = path_states + [state_key]
            for candidate_graph, id_map, assoc_hash, role_hash in candidates:
                if assoc_hash in tabu:
                    continue
                next_assigned = set(assigned_images)
                next_assigned.add(current_img_node)
                next_assoc = dict(image_associations)
                next_assoc[current_img_node] = set(id_map.values())
                next_assoc_hashes = dict(image_assoc_hashes)
                next_assoc_hashes[current_img_node] = int(assoc_hash)
                candidate_graph.graph["assigned_images"] = set(next_assigned)
                candidate_graph.graph["image_associations"] = dict(next_assoc)
                candidate_graph.graph["image_assoc_hashes"] = dict(next_assoc_hashes)
                candidate_graph.graph["conditioned_image_graph"] = img.copy()
                next_history = history + [candidate_graph.copy()]
                usage_key = (role_hash, assoc_hash)
                # Policy: usage counts are path-local; increment on descent,
                # decrement on backtrack so sibling branches are unaffected.
                usage_counts[usage_key] = int(usage_counts.get(usage_key, 0)) + 1
                solved = _dfs(
                    idx + 1,
                    candidate_graph,
                    next_assigned,
                    next_assoc,
                    next_assoc_hashes,
                    next_history,
                    usage_counts,
                    tried_by_state,
                    next_path_states,
                )
                if solved is not None:
                    return solved
                # Backtracking policy: avoid retrying the same association for
                # the same local role/overlap state within this DFS attempt.
                backtracks += 1
                self._runtime_sample_backtracks = int(backtracks)
                tabu.add(assoc_hash)
                self._backpropagate_state_failure(path_states=next_path_states)
                usage_counts[usage_key] = max(0, int(usage_counts.get(usage_key, 0)) - 1)
                if usage_counts[usage_key] == 0:
                    usage_counts.pop(usage_key, None)
            return None

        solved = _dfs(0, current, set(), {}, {}, history0, {}, {}, [])
        if (
            solved is None
            and self._context_scoring_enabled()
            and bool(getattr(self, "fallback_disable_context_on_failure", True))
        ):
            prev_context_flag = bool(self.use_context_embedding)
            try:
                self.use_context_embedding = False
                solved = _dfs(0, current, set(), {}, {}, history0, {}, {}, [])
            finally:
                self.use_context_embedding = prev_context_flag
        if solved is None:
            return None, history0 if return_history else None
        solved_graph, solved_history = solved
        return solved_graph, solved_history if return_history else None

    def _resolve_n_jobs_eff(self, n_samples: int) -> int:
        """
        Resolve the effective parallelism for generation.

        Args:
            n_samples: Number of samples requested.

        Returns:
            int: Effective number of workers to use.
        """
        # Normalize n_jobs to a concrete worker count.
        n_jobs_eff = int(self.n_jobs) if self.n_jobs is not None else 1
        if n_jobs_eff < 0:
            # joblib convention: -1 means "all cores".
            n_jobs_eff = _available_cpu_count()
        if n_samples <= 1:
            # Avoid parallel overhead for single-sample requests.
            return 1
        return max(1, int(n_jobs_eff))

    def _resolve_parallel_backend(self, *, verbose: bool) -> str:
        """
        Resolve a joblib backend robustly for this generator instance.

        Process backends require picklable generator state. Notebook-local
        callables (for example feasibility predicates defined in ``__main__``)
        may fail stdlib pickle checks even though ``loky`` can serialize them
        through cloudpickle. Validate ``loky`` with cloudpickle when available,
        and only fall back to ``threading`` when serialization is expected to
        fail at worker startup.

        Args:
            verbose: Whether to emit warning details.

        Returns:
            str: Effective joblib backend name.
        """
        backend = str(self.parallel_backend or "multiprocessing")
        process_backends = {"multiprocessing", "loky"}
        if backend not in process_backends:
            return backend
        if threading.current_thread() is not threading.main_thread():
            self._warn_verbose(
                "process backend requested from a non-main thread; using threading backend to avoid nested loky constraints",
                verbose=verbose,
            )
            return "threading"
        if backend == "loky":
            try:
                if cloudpickle is not None:
                    cloudpickle.dumps(self)
                else:
                    pickle.dumps(self)
                return backend
            except Exception as exc:
                self._warn_verbose(
                    "generator state is not process-picklable; falling back to threading backend",
                    exc=exc,
                    verbose=verbose,
                )
                return "threading"
        # Multiprocessing relies on stdlib pickle; if that fails but cloudpickle
        # works, prefer loky to keep process-based parallelism.
        try:
            pickle.dumps(self)
            return backend
        except Exception as exc:
            if cloudpickle is not None:
                try:
                    cloudpickle.dumps(self)
                    self._warn_verbose(
                        "multiprocessing pickle failed; switching to loky backend",
                        exc=exc,
                        verbose=verbose,
                    )
                    return "loky"
                except Exception:
                    pass
            self._warn_verbose(
                "generator state is not process-picklable; falling back to threading backend",
                exc=exc,
                verbose=verbose,
            )
            return "threading"

    def _prepare_parallel_batch(
        self,
        *,
        batch: int,
        image_graph: Optional[nx.Graph],
        rng: random.Random,
        round_robin: bool = False,
        rr_index: int = 0,
    ) -> tuple[list[Optional[nx.Graph]], list[int], int]:
        """
        Prepare image graphs and seeds for a parallel batch.

        Args:
            batch: Number of parallel tasks to run.
            image_graph: Optional fixed image graph.
            rng: Random generator for sampling.

        Returns:
            tuple[list[Optional[nx.Graph]], list[int], int]:
                (image_graphs, seeds, next_rr_index)
        """
        if image_graph is None:
            candidates = list(getattr(self, "image_graphs", []))
            if candidates:
                if round_robin:
                    # Deterministic coverage when n_samples > n_generators.
                    start = rr_index % len(candidates)
                    batch_image_graphs = [
                        candidates[(start + i) % len(candidates)] for i in range(batch)
                    ]
                    rr_index = (start + batch) % len(candidates)
                else:
                    # Randomized diversity when coverage is not required.
                    if batch <= len(candidates):
                        batch_image_graphs = rng.sample(candidates, batch)
                    else:
                        batch_image_graphs = [rng.choice(candidates) for _ in range(batch)]
            else:
                # Fall back to None if no training image graphs are available.
                batch_image_graphs = [None] * batch
        else:
            # Caller explicitly fixed the image graph.
            batch_image_graphs = [image_graph] * batch
        seeds = [rng.randint(0, 10**9) for _ in range(batch)]
        return batch_image_graphs, seeds, rr_index

    def _generate_sequential(
        self,
        *,
        n_samples: int,
        max_attempts: int,
        image_graph: Optional[nx.Graph],
        rng: random.Random,
        record_history: bool,
        verbose: bool,
        round_robin: bool = False,
    ) -> tuple[list[nx.Graph], list[list[nx.Graph]]]:
        """
        Generate samples sequentially (single-worker path).

        Args:
            n_samples: Target number of samples.
            max_attempts: Maximum attempts to reach target.
            image_graph: Optional fixed image graph.
            rng: Random generator for sampling.
            record_history: Whether to store histories.
            verbose: Whether to print progress.

        Returns:
            tuple[list[nx.Graph], list[list[nx.Graph]]]: (graphs, histories)
        """
        graphs: list[nx.Graph] = []
        histories: list[list[nx.Graph]] = []
        attempts = 0
        rr_index = 0
        zero_accept_streak = 0
        while len(graphs) < n_samples and attempts < max_attempts:
            self._runtime_completed_n_samples = int(len(graphs))
            attempts += 1
            chosen_image_graph = image_graph
            if image_graph is None and round_robin:
                # Cycle through generators to guarantee at-least-once usage.
                candidates = list(getattr(self, "image_graphs", []))
                if candidates:
                    chosen_image_graph = candidates[rr_index % len(candidates)]
                    rr_index = (rr_index + 1) % len(candidates)
            current, history = self._generate_once(
                image_graph=chosen_image_graph,
                rng=rng,
                return_history=record_history,
            )
            if current is None:
                continue
            if self.feasibility_estimator is not None:
                # Final feasibility filter on fully instantiated samples.
                try:
                    keep = bool(self.feasibility_estimator.predict([current])[0])
                except Exception as exc:
                    self._warn_verbose(
                        "feasibility_estimator.predict failed during final sequential filtering; rejecting candidate",
                        exc=exc,
                        verbose=verbose,
                    )
                    keep = False
                if not keep:
                    continue
            graphs.append(current)
            if record_history:
                histories.append(history)
            if verbose:
                try:
                    print(f"Generated {len(graphs)}/{n_samples}")
                except Exception:
                    pass
        return graphs, histories

    def _generate_parallel(
        self,
        *,
        n_samples: int,
        max_attempts: int,
        image_graph: Optional[nx.Graph],
        rng: random.Random,
        record_history: bool,
        verbose: bool,
        n_jobs_eff: int,
        round_robin: bool = False,
    ) -> tuple[list[nx.Graph], list[list[nx.Graph]]]:
        """
        Generate samples in parallel batches.

        Args:
            n_samples: Target number of samples.
            max_attempts: Maximum attempts to reach target.
            image_graph: Optional fixed image graph.
            rng: Random generator for sampling.
            record_history: Whether to store histories.
            verbose: Whether to print progress.
            n_jobs_eff: Effective number of workers.

        Returns:
            tuple[list[nx.Graph], list[list[nx.Graph]]]: (graphs, histories)
        """
        graphs: list[nx.Graph] = []
        histories: list[list[nx.Graph]] = []
        attempts = 0
        rr_index = 0
        zero_accept_streak = 0

        def _generate_batch_local(
            *,
            n_attempts: int,
            image_graph: Optional[nx.Graph],
            image_graphs_pool: Optional[Sequence[nx.Graph]],
            rng_seed: int,
            record_history: bool,
        ) -> list[tuple[Optional[nx.Graph], Optional[list[nx.Graph]]]]:
            rng_local = random.Random(rng_seed)
            out = []
            for _ in range(max(0, int(n_attempts))):
                chosen_image = image_graph
                if chosen_image is None and image_graphs_pool:
                    chosen_image = rng_local.choice(image_graphs_pool)
                out.append(
                    self._generate_once(
                        image_graph=chosen_image,
                        rng=rng_local,
                        return_history=record_history,
                    )
                )
            return out

        while len(graphs) < n_samples and attempts < max_attempts:
            self._runtime_completed_n_samples = int(len(graphs))
            remaining = max_attempts - attempts
            need = n_samples - len(graphs)
            # Aim to keep all workers busy by running full batches when possible.
            target_attempts = min(remaining, max(need, n_jobs_eff))
            batch_jobs = n_jobs_eff if remaining >= n_jobs_eff else max(1, remaining)
            # Ensure enough work per worker to amortize joblib overhead.
            min_attempts_per_job = max(1, int(getattr(self, "min_attempts_per_job", 4)))
            attempts_per_job = max(
                min_attempts_per_job,
                int(math.ceil(target_attempts / batch_jobs)),
            )
            # Clamp to avoid exceeding remaining attempts in the final batch.
            if attempts_per_job * batch_jobs > remaining:
                attempts_per_job = max(1, remaining // batch_jobs)
            batch_start = time.perf_counter() if self.verbose_parallel_stats else None
            batch_image_graphs, seeds, rr_index = self._prepare_parallel_batch(
                batch=batch_jobs,
                image_graph=image_graph,
                rng=rng,
                round_robin=round_robin,
                rr_index=rr_index,
            )
            image_graphs_pool = None
            if image_graph is None:
                candidates = list(getattr(self, "image_graphs", []))
                if candidates:
                    # Preserve explicit per-worker assignment in round-robin mode.
                    # When round_robin is disabled, allow each worker attempt to
                    # sample from the full pool for extra diversity.
                    if round_robin:
                        image_graphs_pool = None
                    else:
                        image_graphs_pool = candidates
                        batch_image_graphs = [None] * batch_jobs
            backend = self._resolve_parallel_backend(verbose=verbose)
            prefer = "threads" if backend == "threading" else "processes"
            results = Parallel(n_jobs=batch_jobs, prefer=prefer, backend=backend)(
                delayed(_generate_batch_local)(
                    n_attempts=attempts_per_job,
                    image_graph=batch_image_graphs[i],
                    image_graphs_pool=image_graphs_pool,
                    rng_seed=seed,
                    record_history=record_history,
                )
                for i, seed in enumerate(seeds)
            )
            attempts += attempts_per_job * batch_jobs
            batch_before = len(graphs)
            # Flatten batch results and apply feasibility filtering in one pass.
            batch_candidates = []
            batch_histories = []
            for batch_result in results:
                for current, history in batch_result:
                    if current is None:
                        continue
                    batch_candidates.append(current)
                    if record_history:
                        batch_histories.append(history)
            if batch_candidates:
                if self.feasibility_estimator is not None:
                    try:
                        keep_mask = list(self.feasibility_estimator.predict(batch_candidates))
                    except Exception as exc:
                        self._warn_verbose(
                            "feasibility_estimator.predict failed during final parallel filtering; rejecting batch",
                            exc=exc,
                            verbose=verbose,
                        )
                        keep_mask = [False] * len(batch_candidates)
                else:
                    keep_mask = [True] * len(batch_candidates)
                for idx, keep in enumerate(keep_mask):
                    if not keep:
                        continue
                    graphs.append(batch_candidates[idx])
                    if record_history:
                        histories.append(batch_histories[idx])
                    if verbose:
                        try:
                            print(f"Generated {len(graphs)}/{n_samples}")
                        except Exception:
                            pass
                    if len(graphs) >= n_samples:
                        break
            accepted = len(graphs) - batch_before
            attempted = attempts_per_job * batch_jobs
            construction_success = len(batch_candidates)
            construction_fail = max(0, attempted - construction_success)
            feasibility_reject = max(0, construction_success - accepted)
            if accepted == 0 and attempted > 0:
                zero_accept_streak += 1
            else:
                zero_accept_streak = 0
            if (
                self.dead_end_reset_after_zero_batches > 0
                and zero_accept_streak >= self.dead_end_reset_after_zero_batches
            ):
                # Break repeated dead-end loops by allowing associations to be retried.
                self._last_assoc_hash_by_image = {}
                zero_accept_streak = 0
                if verbose and self.verbose_parallel_stats:
                    print(
                        "[parallel-reset] "
                        "reason=zero-accept-streak "
                        f"cleared_last_assoc_hashes=1"
                    )
            if verbose and self.verbose_parallel_stats and batch_start is not None:
                batch_time = max(1e-9, time.perf_counter() - batch_start)
                rate = accepted / attempted if attempted else 0.0
                print(
                    f"[parallel] jobs={batch_jobs} attempts/job={attempts_per_job} "
                    f"attempts={attempted} accepted={accepted} "
                    f"acc_rate={rate:.3f} batch_time={batch_time:.2f}s "
                    f"construction_fail={construction_fail} "
                    f"feasibility_reject={feasibility_reject} "
                    f"zero_accept_streak={zero_accept_streak}"
                )
        return graphs, histories

    def generate(
        self,
        n_samples: int = 1,
        *,
        image_graph: Optional[nx.Graph] = None,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        return_history: bool = False,
        max_attempts: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Generate graphs by materializing image-node associations in BFS order.

        Workflow:
        1) Select the image graph (provided or from training set).
        2) Start from the highest-degree image node and materialize its association.
        3) Traverse the image graph in breadth-first order, inserting associations
           with anchor matching when neighbors are already assigned.
        4) If a feasibility estimator is configured, reject candidates that violate it.
        5) Apply a final feasibility filter on completed samples.

        Args:
            n_samples: Number of graphs to generate.
            image_graph: Optional fixed image graph. If None, one is chosen
                from `self.image_graphs` (or `self.fixed_image_graph`).
            rng: Optional Random instance for reproducibility.
            seed: Optional seed used when `rng` is not provided. If None,
                defaults to `self.random_seed`.
            return_history: If True, return (graphs, histories), where each history
                is a list of intermediate graphs after each insertion.
            max_attempts: Optional cap on total attempts to obtain n_samples.
                Defaults to 5 * n_samples.
            verbose: If True, print a running counter each time a
                new sample is successfully generated.

        Returns:
            list[nx.Graph] | tuple[list[nx.Graph], list[list[nx.Graph]]]:
                Generated graphs (and optional histories). Graphs that do not
                fully instantiate the image graph are discarded.
        """
        prev_runtime_verbose = self._runtime_verbose
        self._runtime_verbose = bool(verbose)
        self._runtime_target_n_samples = None
        self._runtime_completed_n_samples = 0
        self._reset_generation_stats()
        if rng is None:
            seed = self._resolve_seed(seed)
            rng = random.Random(seed)
        try:
            n_samples = max(1, int(n_samples))
            self._runtime_target_n_samples = int(n_samples)
            if max_attempts is None:
                max_attempts = max(1, n_samples * 5)
            max_attempts = max(1, int(max_attempts))
            n_jobs_eff = self._resolve_n_jobs_eff(n_samples)
            candidates = list(getattr(self, "image_graphs", []))
            # If we need more samples than generators, use round-robin to ensure coverage.
            round_robin = image_graph is None and bool(candidates) and n_samples > len(candidates)
            if n_jobs_eff == 1 or n_samples <= 1:
                graphs, histories = self._generate_sequential(
                    n_samples=n_samples,
                    max_attempts=max_attempts,
                    image_graph=image_graph,
                    rng=rng,
                    record_history=bool(return_history),
                    verbose=verbose,
                    round_robin=round_robin,
                )
            else:
                graphs, histories = self._generate_parallel(
                    n_samples=n_samples,
                    max_attempts=max_attempts,
                    image_graph=image_graph,
                    rng=rng,
                    record_history=bool(return_history),
                    verbose=verbose,
                    n_jobs_eff=n_jobs_eff,
                    round_robin=round_robin,
                )

            self.last_generation_stats = dict(self._generation_stats or {})
            if verbose:
                self._print_generation_summary(self.last_generation_stats)
            if return_history:
                return graphs, histories
            return graphs
        finally:
            if not self.last_generation_stats:
                self.last_generation_stats = dict(self._generation_stats or {})
            self._generation_stats = None
            self._runtime_target_n_samples = None
            self._runtime_completed_n_samples = 0
            self._runtime_sample_backtracks = 0
            self._runtime_verbose = prev_runtime_verbose
