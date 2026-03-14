"""Dataset-level orchestration for conditional autoregressive generation."""

from __future__ import annotations

import copy
from typing import Sequence
import time

import networkx as nx

from abstractgraph_generative.legacy.conditional_v0_1.generator_core import (
    ConditionalAutoregressiveGenerator,
)
from abstractgraph_generative.legacy.conditional_v0_1.utils import _fmt_elapsed


class ConditionalAutoregressiveGraphsGenerator:
    """
    Generate graph datasets with class-partitioned conditional autoregression.

    This orchestrator wraps a fitted/fittable
    ``ConditionalAutoregressiveGenerator`` and a graph estimator.
    Generation is run independently per target class and target labels for
    generated samples are always produced via ``graph_estimator.predict``.
    """

    def __init__(
        self,
        generator: ConditionalAutoregressiveGenerator,
        graph_estimator,
        *,
        min_cluster_size: int = 20,
        size_factor: float = 1.0,
        constraint_level: int = 1,
        use_context_embedding: bool = False,
        max_attempts_multiplier: int = 30,
        verbose: bool = True,
    ):
        """
        Initialize the dataset-level conditional autoregressive generator.

        Args:
            generator: Conditional autoregressive generator used for sampling.
                Must be initialized with decomposition/hash settings and a
                context vectorizer.
            graph_estimator: Estimator used to fit on input data and predict
                targets for generated samples.
                Also defines the target labels assigned to generated graphs.
            min_cluster_size: Minimum target size for generation groups within
                each class.
                Smaller values increase the number of work items (more local
                conditioning, more overhead); larger values smooth groups and
                reduce per-group specialization.
            size_factor: Reserved for backward compatibility.
                Kept as a constructor argument but not used by ``fit``/``generate``.
            constraint_level: Constraint-level setting forwarded to ``generator``.
                Higher values generally enforce stricter structural/contextual
                matching and can reduce acceptance while improving fidelity.
            use_context_embedding: Context-embedding flag forwarded to ``generator``.
                When True, generation uses context-similarity-biased sampling;
                this can improve contextual alignment but is computationally heavier.
            max_attempts_multiplier: Attempt multiplier used to compute
                ``max_attempts`` per cluster during ``generate``.
                Higher values raise the chance of filling requested counts in
                hard regions at the cost of longer runtime.
            verbose: If True, print progress.
                Helpful for long runs; disable for cleaner batch logs.

        Returns:
            None.
        """
        self.generator = generator
        self.graph_estimator = graph_estimator
        self.min_cluster_size = int(min_cluster_size)
        self.size_factor = float(size_factor)
        self.constraint_level = int(constraint_level)
        self.use_context_embedding = bool(use_context_embedding)
        self.max_attempts_multiplier = int(max_attempts_multiplier)
        self.verbose = bool(verbose)

        if self.min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be > 0")
        if self.max_attempts_multiplier <= 0:
            raise ValueError("max_attempts_multiplier must be > 0")

        self._cluster_specs: list[dict] = []
        self._cluster_sizes: list[int] = []
        self._is_fitted: bool = False

    def _validate_inputs(self, graphs: Sequence[nx.Graph], targets: Sequence) -> None:
        """
        Validate required inputs and generator prerequisites.

        Args:
            graphs: Input graphs.
            targets: Input targets.

        Returns:
            None.
        """
        if self.generator is None:
            raise ValueError("generator is required.")
        if self.graph_estimator is None:
            raise ValueError("graph_estimator is required.")
        if len(graphs) != len(targets):
            raise ValueError("targets must have the same length as graphs.")
        if getattr(self.generator, "context_vectorizer", None) is None:
            raise ValueError("generator.context_vectorizer is required.")
        if getattr(self.generator, "decomposition_function", None) is None:
            raise ValueError("generator.decomposition_function must be initialized.")
        if getattr(self.generator, "nbits", None) is None:
            raise ValueError("generator.nbits must be initialized.")

    def _require_fitted(self) -> None:
        """
        Validate that cluster models have been built with fit().

        Returns:
            None.
        """
        if not self._is_fitted:
            raise ValueError("Call fit(graphs, targets) before generate().")

    @staticmethod
    def _partition_by_target(targets: Sequence) -> dict:
        """
        Partition example indices by class label while preserving order.

        Args:
            targets: Sequence of target labels.

        Returns:
            dict: Mapping ``target -> list[index]``.
        """
        partitions: dict = {}
        for idx, target in enumerate(targets):
            partitions.setdefault(target, []).append(idx)
        return partitions

    def _build_groups(self, graphs: Sequence[nx.Graph]) -> list[list[int]]:
        """
        Build embedding-based groups for a class-specific graph subset.

        Groups are built with Ward clustering, then split into chunks of
        ``min_cluster_size`` and finally merged when undersized.

        Args:
            graphs: Graphs belonging to a single target class.

        Returns:
            list[list[int]]: Local indices grouped for generation.
        """
        import math as _math
        import numpy as _np

        try:
            from scipy.cluster.hierarchy import linkage, fcluster
        except Exception as exc:
            raise ImportError("scipy is required for Ward hierarchical clustering.") from exc

        n = len(graphs)
        if n <= 0:
            return []
        if n == 1:
            return [[0]]

        feats = _np.asarray(self.generator.context_vectorizer.fit_transform(graphs), dtype=float)
        if feats.ndim != 2 or feats.shape[0] != n:
            raise ValueError("context_vectorizer returned invalid feature shape.")

        n_clusters = max(1, int(_math.ceil(float(n) / float(self.min_cluster_size))))
        z = linkage(feats, method="ward")
        labels = _np.asarray(fcluster(z, t=n_clusters, criterion="maxclust"), dtype=int)

        clustered: dict[int, list[int]] = {}
        for idx, lbl in enumerate(labels.tolist()):
            clustered.setdefault(int(lbl), []).append(idx)

        groups: list[list[int]] = []
        for lbl in sorted(clustered.keys()):
            idxs = clustered[lbl]
            for start in range(0, len(idxs), self.min_cluster_size):
                groups.append(idxs[start:start + self.min_cluster_size])

        if len(groups) > 1 and n >= self.min_cluster_size:
            while True:
                small = [gi for gi, g in enumerate(groups) if len(g) < self.min_cluster_size]
                if not small or len(groups) <= 1:
                    break
                gi = small[0]
                center_i = feats[_np.asarray(groups[gi], dtype=int)].mean(axis=0)
                best_j = None
                best_d = None
                for gj, g in enumerate(groups):
                    if gj == gi:
                        continue
                    center_j = feats[_np.asarray(g, dtype=int)].mean(axis=0)
                    dist = float(_np.linalg.norm(center_i - center_j))
                    if best_d is None or dist < best_d:
                        best_d = dist
                        best_j = gj
                if best_j is None:
                    break
                groups[best_j] = groups[best_j] + groups[gi]
                del groups[gi]

        return groups

    @staticmethod
    def _estimate_eta_seconds(recent_elapsed: Sequence[float], remaining: int) -> float:
        """
        Estimate ETA from recent per-cluster elapsed times.

        Args:
            recent_elapsed: Recent elapsed durations in seconds.
            remaining: Number of remaining clusters.

        Returns:
            float: Estimated remaining time in seconds.
        """
        if remaining <= 0:
            return 0.0
        if not recent_elapsed:
            return 0.0
        avg_recent = sum(float(x) for x in recent_elapsed) / float(len(recent_elapsed))
        return max(0.0, avg_recent * float(remaining))

    @staticmethod
    def _allocate_proportional_counts(total: int, weights: Sequence[int]) -> list[int]:
        """
        Allocate integer sample counts proportionally to cluster sizes.

        Args:
            total: Total number of samples to allocate.
            weights: Positive cluster-size weights.

        Returns:
            list[int]: Per-cluster integer counts summing to ``total``.
        """
        import math as _math

        if total <= 0:
            return [0 for _ in weights]
        if not weights:
            return []
        weight_sum = sum(max(0, int(w)) for w in weights)
        if weight_sum <= 0:
            counts = [0 for _ in weights]
            counts[0] = int(total)
            return counts

        raw = [(float(total) * float(max(0, int(w))) / float(weight_sum)) for w in weights]
        base = [int(_math.floor(x)) for x in raw]
        remainder = int(total) - sum(base)
        order = sorted(range(len(raw)), key=lambda i: (raw[i] - float(base[i])), reverse=True)
        for i in order[:remainder]:
            base[i] += 1
        return base

    @staticmethod
    def _log_progress_header() -> None:
        """
        Print the progress-table header.

        Returns:
            None.
        """
        print(
            f"{'cluster':>9} | {'size':>4} | {'generated':>11} | "
            f"{'cumulative':>20} | {'elapsed':>10} | {'eta':>10}"
        )

    @staticmethod
    def _log_progress_row(
        *,
        cluster_index: int,
        n_clusters: int,
        cluster_size: int,
        generated: int,
        requested: int,
        cumulative: int,
        total_requested: int,
        elapsed_seconds: float,
        eta_seconds: float,
    ) -> None:
        """
        Print one progress-table row.

        Args:
            cluster_index: 1-based cluster index.
            n_clusters: Total number of clusters.
            cluster_size: Number of input graphs in the current cluster.
            generated: Number of generated samples in this cluster.
            requested: Number of requested samples in this cluster.
            cumulative: Total generated samples so far.
            total_requested: Total requested samples across all clusters.
            elapsed_seconds: Current cluster elapsed time.
            eta_seconds: Estimated remaining time.

        Returns:
            None.
        """
        frac_pct = (100.0 * float(cumulative) / float(total_requested)) if total_requested > 0 else 0.0
        cluster_str = f"{cluster_index:>2}/{n_clusters:<2}"
        generated_str = f"{generated:>3}/{requested:<3}"
        cumulative_str = f"{cumulative:>4}/{total_requested:<4} ({frac_pct:5.1f}%)"
        elapsed_str = _fmt_elapsed(elapsed_seconds)
        eta_str = _fmt_elapsed(eta_seconds)
        print(
            f"{cluster_str:>9} | {cluster_size:>4} | {generated_str:>11} | "
            f"{cumulative_str:>20} | {elapsed_str:>10} | {eta_str:>10}"
        )

    def fit(self, graphs: Sequence[nx.Graph], targets: Sequence, *, verbose: bool | None = None) -> "ConditionalAutoregressiveGraphsGenerator":
        """
        Fit class/cluster-specific generator models and the target estimator.

        Args:
            graphs: Input graphs.
            targets: Input target labels.
            verbose: Optional fit-level verbosity override.

        Returns:
            ConditionalAutoregressiveGraphsGenerator: Self.
        """
        graphs = [] if graphs is None else list(graphs)
        targets = [] if targets is None else list(targets)
        if not graphs:
            raise ValueError("graphs must be non-empty.")
        self._validate_inputs(graphs, targets)

        is_verbose = self.verbose if verbose is None else bool(verbose)
        partitions = self._partition_by_target(targets)
        cluster_specs: list[dict] = []

        vec = getattr(self.generator, "context_vectorizer", None)
        vec_had_n_jobs = hasattr(vec, "n_jobs")
        vec_original_n_jobs = getattr(vec, "n_jobs", None) if vec_had_n_jobs else None
        if vec_had_n_jobs:
            vec.n_jobs = 1
        try:
            for class_value in partitions.keys():
                class_indices = partitions[class_value]
                class_graphs = [graphs[i] for i in class_indices]
                groups = self._build_groups(class_graphs)
                for local_idxs in groups:
                    group_graphs = [class_graphs[j] for j in local_idxs]
                    if not group_graphs:
                        continue
                    cluster_generator = copy.deepcopy(self.generator)
                    cluster_generator.fit(
                        group_graphs,
                        constraint_level=self.constraint_level,
                        use_context_embedding=self.use_context_embedding,
                        label_mode=getattr(self.generator, "label_mode", None),
                        verbose=False,
                    )
                    cluster_specs.append(
                        {
                            "class_value": class_value,
                            "cluster_graphs": group_graphs,
                            "cluster_size": len(group_graphs),
                            "generator": cluster_generator,
                        }
                    )
        finally:
            if vec_had_n_jobs:
                vec.n_jobs = vec_original_n_jobs

        self._cluster_specs = cluster_specs
        self._cluster_sizes = [int(spec["cluster_size"]) for spec in cluster_specs]
        self.graph_estimator.fit(graphs, targets)
        self._is_fitted = True

        if is_verbose:
            print(f"fit clusters={len(self._cluster_specs)}")
            for idx, spec in enumerate(self._cluster_specs, start=1):
                class_value = spec["class_value"]
                if hasattr(class_value, "item"):
                    try:
                        class_value = class_value.item()
                    except Exception:
                        pass
                if isinstance(class_value, int):
                    class_repr = str(class_value)
                else:
                    class_repr = repr(class_value)
                print(
                    f"cluster={idx:>3} class={class_repr} "
                    f"size={int(spec['cluster_size'])}"
                )
        return self

    def generate(self, n_samples: int | None = None) -> tuple[list[nx.Graph], list]:
        """
        Generate samples from fitted cluster-specific generators.

        Behavior:
        - Requires ``fit(...)`` to be called first.
        - If ``n_samples is None``, each cluster generates exactly its
          training size.
        - If ``n_samples`` is provided, samples are allocated to clusters
          proportionally to training cluster sizes.
        - Generated labels are always predicted with ``graph_estimator.predict``.

        Args:
            n_samples: Optional total number of samples to generate.

        Returns:
            tuple[list[nx.Graph], list]: Generated graphs and predicted targets.
        """
        self._require_fitted()
        if not self._cluster_specs:
            return [], []

        if n_samples is None:
            requests = [int(size) for size in self._cluster_sizes]
        else:
            n_samples = int(n_samples)
            if n_samples < 0:
                raise ValueError("n_samples must be >= 0.")
            requests = self._allocate_proportional_counts(n_samples, self._cluster_sizes)

        out: list[nx.Graph] = []
        n_groups = len(self._cluster_specs)
        total_requested = sum(requests)
        # Use a short trailing window for ETA stability without over-smoothing.
        recent_elapsed: list[float] = []
        if self.verbose:
            self._log_progress_header()
        for i, (spec, requested) in enumerate(zip(self._cluster_specs, requests), start=1):
            requested = int(requested)
            t0 = time.perf_counter()
            if requested > 0:
                max_attempts = max(1, requested * self.max_attempts_multiplier)
                samples = spec["generator"].generate(
                    n_samples=requested,
                    max_attempts=max_attempts,
                    return_history=False,
                    verbose=False,
                )
            else:
                samples = []
            elapsed = time.perf_counter() - t0
            out.extend(samples)
            recent_elapsed.append(float(elapsed))
            if len(recent_elapsed) > 2:
                recent_elapsed = recent_elapsed[-2:]
            if self.verbose:
                cum = len(out)
                remaining = max(0, n_groups - i)
                eta_seconds = self._estimate_eta_seconds(recent_elapsed, remaining)
                self._log_progress_row(
                    cluster_index=i,
                    n_clusters=n_groups,
                    cluster_size=int(spec["cluster_size"]),
                    generated=len(samples),
                    requested=requested,
                    cumulative=cum,
                    total_requested=total_requested,
                    elapsed_seconds=elapsed,
                    eta_seconds=eta_seconds,
                )

        if self.verbose:
            print(f"total generated={len(out)} across {n_groups} clusters")

        # Targets for generated graphs are always estimator predictions.
        predicted = [] if not out else list(self.graph_estimator.predict(out))
        return out, predicted
