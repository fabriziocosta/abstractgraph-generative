"""Dataset-level wrapper around the simplified conditional generator."""

from __future__ import annotations

import copy
from typing import Sequence

import networkx as nx


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds into a compact elapsed-time string.

    Args:
        seconds: Duration in seconds.

    Returns:
        str: Formatted elapsed string.
    """
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class ConditionalAutoregressiveGraphsGenerator:
    """Generate datasets with one conditional generator per target class."""

    def __init__(self, generator, graph_estimator, *, verbose: bool = True):
        """Initialize the dataset-level wrapper.

        Args:
            generator: Base conditional generator template.
            graph_estimator: Estimator used to predict targets for generated graphs.
            verbose: Whether to print progress.

        Returns:
            None.
        """
        self.generator = generator
        self.graph_estimator = graph_estimator
        self.verbose = bool(verbose)
        self._class_generators: dict = {}
        self._class_sizes: dict = {}
        self._is_fitted = False

    @staticmethod
    def _partition_by_target(targets: Sequence) -> dict:
        """Partition indices by target label.

        Args:
            targets: Target sequence.

        Returns:
            dict: Mapping target -> list of indices.
        """
        out = {}
        for idx, target in enumerate(targets):
            out.setdefault(target, []).append(idx)
        return out

    def fit(self, graphs: Sequence[nx.Graph], targets: Sequence) -> "ConditionalAutoregressiveGraphsGenerator":
        """Fit one generator per class and the graph estimator.

        Args:
            graphs: Training graphs.
            targets: Training targets.

        Returns:
            ConditionalAutoregressiveGraphsGenerator: Self.
        """
        if graphs is None or targets is None:
            raise ValueError("graphs and targets are required.")
        graphs = list(graphs)
        targets = list(targets)
        if not graphs:
            raise ValueError("graphs must be non-empty.")
        if len(graphs) != len(targets):
            raise ValueError("targets must have the same length as graphs.")

        partitions = self._partition_by_target(targets)
        class_generators = {}
        class_sizes = {}
        for cls, idxs in partitions.items():
            subset = [graphs[i] for i in idxs]
            if not subset:
                continue
            cls_gen = copy.deepcopy(self.generator)
            cls_gen.fit(subset)
            class_generators[cls] = cls_gen
            class_sizes[cls] = len(subset)
            if self.verbose:
                print(f"fit class={cls!r} size={len(subset)}")

        if not class_generators:
            raise ValueError("No class-specific generators were fitted.")

        self._class_generators = class_generators
        self._class_sizes = class_sizes
        if hasattr(self.graph_estimator, "fit"):
            self.graph_estimator.fit(graphs, targets)
        self._is_fitted = True
        return self

    def generate(
        self,
        n_samples: int | None = None,
        debug_level: int | None = None,
        **generator_generate_kwargs,
    ) -> tuple[list[nx.Graph], list]:
        """Generate dataset samples and infer targets via graph estimator.

        Args:
            n_samples: Total number of requested samples. If None, uses class sizes.
            debug_level: Optional debug verbosity level propagated to each
                class-specific generator before generation.
            **generator_generate_kwargs: Extra keyword arguments forwarded to
                each class generator ``generate`` call (for example
                ``max_attempts_per_sample`` or ``max_total_attempts``).

        Returns:
            tuple[list[nx.Graph], list]: Generated graphs and predicted targets.
        """
        if not self._is_fitted:
            raise ValueError("Call fit(graphs, targets) before generate().")
        if not self._class_generators:
            return [], []

        classes = list(self._class_generators.keys())
        if n_samples is None:
            requests = {cls: int(self._class_sizes.get(cls, 0)) for cls in classes}
        else:
            n_samples = int(n_samples)
            if n_samples < 0:
                raise ValueError("n_samples must be >= 0.")
            total_weight = sum(int(self._class_sizes.get(cls, 0)) for cls in classes)
            if total_weight <= 0:
                requests = {classes[0]: n_samples}
                for cls in classes[1:]:
                    requests[cls] = 0
            else:
                raw = {
                    cls: float(n_samples) * float(int(self._class_sizes.get(cls, 0))) / float(total_weight)
                    for cls in classes
                }
                base = {cls: int(raw[cls]) for cls in classes}
                remainder = int(n_samples) - sum(base.values())
                order = sorted(classes, key=lambda c: raw[c] - float(base[c]), reverse=True)
                for cls in order[:remainder]:
                    base[cls] += 1
                requests = base

        all_graphs: list[nx.Graph] = []
        for cls in classes:
            req = int(requests.get(cls, 0))
            if req <= 0:
                continue
            cls_generator = self._class_generators[cls]
            if debug_level is not None and hasattr(cls_generator, "debug_level"):
                cls_generator.debug_level = max(0, int(debug_level))
                if hasattr(cls_generator, "debug"):
                    cls_generator.debug = bool(cls_generator.debug_level > 0)
            generated = self._class_generators[cls].generate(
                n_samples=req,
                **generator_generate_kwargs,
            )
            all_graphs.extend(generated)
            if self.verbose:
                print(f"generate class={cls!r} requested={req} produced={len(generated)}")

        if not all_graphs:
            return [], []

        if hasattr(self.graph_estimator, "predict"):
            predicted = self.graph_estimator.predict(all_graphs)
            if not isinstance(predicted, list):
                try:
                    predicted = list(predicted)
                except Exception:
                    predicted = [predicted]
        else:
            predicted = [None for _ in all_graphs]
        return all_graphs, predicted


__all__ = [
    "ConditionalAutoregressiveGraphsGenerator",
    "_fmt_elapsed",
]
