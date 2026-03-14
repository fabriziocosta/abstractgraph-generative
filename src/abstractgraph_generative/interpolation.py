"""
Generation via interpolation with model-driven selection.

Overview
- This module orchestrates graph generation by repeatedly:
  1) Selecting disjoint pairs from a pool of donor graphs.
  2) Interpolating a path between each pair using `InterpolationEstimator`.
  3) Scoring the interpolated graphs with a downstream estimator.
  4) Keeping graphs whose predicted probabilities fall within a user‑defined
     acceptance interval, optionally restricted to a single target class.
  5) De‑duplicating accepted graphs by hash and adding them to the pool.

Key classes
- `InterpolationGenerator`: Parallel, feature‑rich generator. Handles process‑level
  parallelism, periodic demo streaming, progress logging, and per‑round refits of
  the interpolation estimator on the expanding pool.
- `SerialInterpolationGenerator`: Minimal, single‑process variant for quick runs
  or constrained environments.

Scoring and acceptance
- Supervised: when training labels are provided, a `RandomForestClassifier` is
  trained and `predict_proba` is used for selection.
- Unsupervised: when no labels are provided, `IsolationForestProba` is used; it
  exposes two classes `[0=outlier, 1=inlier]` and returns calibrated probabilities
  for compatibility with the same selection code.
- Acceptance interval: graphs are selected if any (or a chosen) class probability
  falls within an inclusive interval `[low, high]` (default `(0.5, 1.0)`).
- Class restriction: `accept_only_class` can be set (e.g., `1` to keep only
  inliers with `IsolationForestProba`). When `None`, all classes may contribute to
  acceptance.

Rounds and pool growth
- Each round creates new pairs from the current pool, interpolates, selects, and
  appends accepted graphs to the pool. The interpolation estimator is re‑fit at the
  start of each round on the latest pool to keep interpolation neighborhoods fresh.
- The downstream scoring model is not re‑fit across rounds by default; this keeps
  the scoring calibration stable relative to the initial training data distribution.

Parallel execution
- Uses `ProcessPoolExecutor` with a spawn context. The interpolation estimator is
  serialized (with `dill`) and shipped to workers to avoid re‑construction costs.
- Worker payload carries the serialized estimator, decomposition operator (XML or
  pickled), and optional feasibility estimator.
- To avoid CPU oversubscription, thread counts for the main process scoring
  pipeline are reduced while workers are active.

Deduplication and hashing
- A hash‑based index (see `hash_graph`) prevents adding duplicates to the pool
  and allows efficient round‑local deduplication.

Reproducibility
- Pair selection uses a numpy RNG (configurable or seeded). Interpolation
  randomness is controlled by the estimator’s RNG; workers receive deterministic
  seeds derived from `(round_idx, chunk_id, chunk_len)`.

Visualization
- Optional periodic demos show one example interpolation path (source, destination,
  and a subset of midpoint graphs) with overlaid probabilities for each class.
"""

import os
import random
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Any, Optional, Protocol, Sequence, Tuple, Union

import dill
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from abstractgraph.hashing import GraphHashDeduper, hash_graph
from abstractgraph.vectorize import AbstractGraphTransformer
from abstractgraph_generative.interpolate import InterpolationEstimator
from abstractgraph_ml.estimators import GraphEstimator, IsolationForestProba


def _interpolate_pairs_chunk_worker(
    pool_graphs: Sequence[nx.Graph],
    pair_chunk: Sequence[Tuple[int, int]],
    *,
    interpolator_config: dict,
    best_of: int,
    n_iterations: int,
    rng_seed: Optional[int] = None,
    demo_limit: int = 8,
):
    """Worker: interpolate a chunk of pairs and return graphs and one demo.

    Args:
        pool_graphs: Donor/pool graphs for the round.
        pair_chunk: Indices of pairs to process.
        interpolator_config: Dict with keys {nbits, k, n_samples, n_iterations,
            degree_penalty, degree_penalty_mode, cut_radius, cut_scope,
            decomposition_xml, decomposition_bytes, feasibility_bytes, interpolator_bytes}.
        best_of: Attempts per interpolation when choosing best path.
        n_iterations: Midpoint refinement iterations per interpolation.
        rng_seed: Optional seed to initialize worker RNG.
        demo_limit: Max graphs to include in the returned demo path.

    Returns:
        dict with keys:
          - 'graphs': list of interpolated graphs across pairs in the chunk.
          - 'demo': tuple (source_graph, dest_graph, path_graphs) or None.
    """
    rng = random.Random(rng_seed) if rng_seed is not None else None
    interpolator_bytes = interpolator_config.get("interpolator_bytes")
    if interpolator_bytes is not None:
        estimator = dill.loads(interpolator_bytes)
        estimator.rng = rng
        if hasattr(estimator, "graph_transformer") and hasattr(estimator.graph_transformer, "n_jobs"):
            estimator.graph_transformer.n_jobs = 1
        if hasattr(estimator, "feasibility_estimator") and hasattr(estimator.feasibility_estimator, "n_jobs"):
            estimator.feasibility_estimator.n_jobs = 1
    else:
        # Build a local interpolation estimator mirroring the main one, but with
        # transformer parallelism disabled to avoid oversubscription.
        nbits = interpolator_config.get("nbits")
        # Rebuild decomposition function from XML if provided to avoid pickling closures
        decomposition_function = None
        decomposition_xml = interpolator_config.get("decomposition_xml")
        decomposition_bytes = interpolator_config.get("decomposition_bytes")
        if decomposition_xml is not None:
            try:
                from abstractgraph.xml import operator_from_xml_string
                decomposition_function = operator_from_xml_string(decomposition_xml)
            except Exception:
                pass
        if decomposition_function is None and decomposition_bytes is not None:
            try:
                decomposition_function = dill.loads(decomposition_bytes)
            except Exception:
                decomposition_function = None
        transformer = AbstractGraphTransformer(
            nbits=nbits,
            decomposition_function=decomposition_function,
            return_dense=True,
            n_jobs=1,
        )
        feasibility_estimator = None
        feasibility_bytes = interpolator_config.get("feasibility_bytes")
        if feasibility_bytes is not None:
            try:
                feasibility_estimator = dill.loads(feasibility_bytes)
            except Exception:
                feasibility_estimator = None
        estimator = InterpolationEstimator(
            graph_transformer=transformer,
            rng=rng,
            n_samples=interpolator_config.get("n_samples", 1),
            n_iterations=interpolator_config.get("n_iterations", 1),
            k=interpolator_config.get("k", 5),
            degree_penalty=interpolator_config.get("degree_penalty", "auto"),
            degree_penalty_mode=interpolator_config.get("degree_penalty_mode", "multiplicative"),
            cut_radius=interpolator_config.get("cut_radius", None),
            cut_scope=interpolator_config.get("cut_scope", "both"),
            feasibility_estimator=feasibility_estimator,
        )
        estimator.fit(pool_graphs)

    out_graphs: list[nx.Graph] = []
    demo = None
    for idx, (i, j) in enumerate(pair_chunk):
        try:
            graphs = estimator.iterated_interpolate_idxs(
                i, j, n_iterations=n_iterations, best_of=best_of
            )
        except Exception:
            graphs = []
        out_graphs.extend(graphs)
        if demo is None and graphs:
            # Include endpoints for nicer visualization
            src = pool_graphs[i]
            dst = pool_graphs[j]
            path = list(graphs)
            if demo_limit and len(path) > demo_limit:
                mid = max(1, demo_limit // 2)
                path = path[:mid] + path[-(demo_limit - mid):]
            demo = (src, dst, path)

    try:
        out_hashes = [hash_graph(g) for g in out_graphs]
    except Exception:
        out_hashes = None
    return {"graphs": out_graphs, "hashes": out_hashes, "demo": demo}


class _ChunkProcessingContextProtocol(Protocol):
    fut_to_chunk: dict
    start_times: dict
    chunks: list[list[Tuple[int, int]]]
    pool_graphs: list[nx.Graph]
    prob_for_acceptance_interval: tuple[float, float]
    round_elite_graphs: list[nx.Graph]
    round_elite_targets: list[Any]
    elite_targets: list[Any]
    pairs_len: int
    round_idx: int
    n_rounds: int
    round_step_t0: float
    interp_elapsed: float
    verbose: bool
    draw_func: object
    stream_plots_every: int
    completed: int
    pairs_seen_round: int
    total_interpolated_round: int
    n_chunks: int
    accept_only_class: Optional[int]


def _fmt_elapsed(seconds):
    """Format elapsed seconds as s, m, or h with two decimals.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Human-readable duration string.
    """
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{seconds/60:.2f}m"
    return f"{seconds/3600:.2f}h"


def make_pairs(size, n_pairs, rng):
    """Create up to n_pairs disjoint index pairs using random permutations.

    Args:
        size: Number of items to pair.
        n_pairs: Maximum number of pairs to return.
        rng: Random generator with permutation method.

    Returns:
        List of index pairs.
    """
    pairs = []
    n_pairs_per_perm = size // 2
    while len(pairs) < n_pairs and n_pairs_per_perm > 0:
        perm = rng.permutation(size)
        for i_perm in range(0, len(perm) - 1, 2):
            pairs.append((int(perm[i_perm]), int(perm[i_perm + 1])))
            if len(pairs) >= n_pairs:
                break
    pairs = pairs[:n_pairs]
    return pairs


def _class_index_from_estimator(graph_estimator, target_id) -> int:
    """Return the probability column index for the target label."""
    estimator = getattr(graph_estimator, "estimator_", None)
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        raise ValueError("Estimator classes are unavailable; fit the estimator before scoring.")
    matches = np.where(classes == target_id)[0]
    if matches.size == 0:
        raise ValueError(f"Target id {target_id} not found in estimator classes.")
    return int(matches[0])


def _compute_query_to_donor_distances(
    graph: nx.Graph,
    *,
    transformer: AbstractGraphTransformer,
    donor_vectors: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute distances from a query graph to donor embeddings.

    Args:
        graph: Query graph to embed.
        transformer: Transformer used to embed graphs.
        donor_vectors: Precomputed embeddings for donor graphs.
        metric: Distance metric to use for `cdist`.

    Returns:
        1D array of distances from the query to each donor.
    """
    from scipy.spatial.distance import cdist

    q_vec = transformer.transform([graph])[0]
    dists = cdist(donor_vectors, q_vec[None, :], metric=metric).ravel()
    return dists


def _k_nearest_indices_from_distances(
    dists: np.ndarray,
    k: int,
    *,
    exclude_zero: bool = True,
) -> list[int]:
    """Return indices of the k nearest entries in a distance vector.

    Args:
        dists: 1D array of distances.
        k: Desired number of nearest neighbors.
        exclude_zero: If True, skip entries with distance exactly 0.0.

    Returns:
        List of up to k indices sorted by increasing distance.
    """
    order = np.argsort(dists)
    out: list[int] = []
    k_eff = max(1, min(int(k), int(dists.shape[0])))
    for idx in order:
        if exclude_zero and float(dists[idx]) == 0.0:
            continue
        out.append(int(idx))
        if len(out) >= k_eff:
            break
    if not out:
        # Fallback to the closest entry even if zero
        out = [int(order[0])] if order.size else []
    return out


def _class_probability_weights(
    graphs: Sequence[nx.Graph],
    graph_estimator: GraphEstimator,
    target_class: int,
) -> np.ndarray:
    """Return nonnegative weights equal to P(class=target_class) for each graph.

    Args:
        graphs: List of graphs to score.
        graph_estimator: Estimator providing ``predict_proba``.
        target_class: Desired class label or column index.

    Returns:
        1D array of nonnegative weights aligned with ``graphs``.
    """
    if not graphs:
        return np.asarray([], dtype=float)
    probs = np.asarray(graph_estimator.predict_proba(graphs, log=False), dtype=float)
    if probs.ndim != 2 or probs.size == 0:
        return np.zeros(len(graphs), dtype=float)
    try:
        col = _class_index_from_estimator(graph_estimator, target_class)
    except Exception:
        # Fallback: treat as numeric column index when classes are unavailable
        idx = int(target_class)
        col = idx if 0 <= idx < probs.shape[1] else 0
    weights = probs[:, int(col)].astype(float)
    # Sanitize weights: clip negatives, replace non-finite with 0
    weights = np.clip(weights, 0.0, None)
    weights[~np.isfinite(weights)] = 0.0
    return weights


def _sample_index_with_weights(weights: np.ndarray, rng: np.random.Generator) -> Optional[int]:
    """Sample an index given weights; uniform fallback when degenerate.

    Args:
        weights: Nonnegative weights.
        rng: Numpy RNG used to sample.

    Returns:
        Selected index, or None when ``weights`` is empty.
    """
    if weights.size == 0:
        return None
    if not np.isfinite(weights).any() or np.all(weights <= 0):
        return int(rng.integers(0, len(weights)))
    w = np.clip(weights, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return int(rng.integers(0, len(weights)))
    p = (w / s).ravel()
    return int(rng.choice(len(weights), p=p))


def _select_graph_for_class(
    graphs: Sequence[nx.Graph],
    graph_estimator: GraphEstimator,
    target_class: int,
    rng: np.random.Generator,
) -> Optional[nx.Graph]:
    """Choose a graph from ``graphs`` proportional to P(class=target_class).

    Args:
        graphs: Candidate graphs to choose from.
        graph_estimator: Estimator providing ``predict_proba``.
        target_class: Desired class label or column index.
        rng: Numpy RNG for sampling.

    Returns:
        Selected graph or None when the input list is empty.
    """
    if not graphs:
        return None
    weights = _class_probability_weights(graphs, graph_estimator, target_class)
    idx = _sample_index_with_weights(weights, rng)
    if idx is None:
        return None
    return graphs[int(idx)]


def plot_sample(
    draw_func,
    graphs,
    graph_estimator,
    n_show: int = 7,
    prob_for_acceptance_interval: Optional[Tuple[float, float]] = None,
):
    """Draw top and bottom scoring graphs and show all class probabilities.

    Prints probabilities for all classes in the title of each subplot.
    If an acceptance interval is provided, highlights entries within it by
    wrapping them in square brackets.

    Args:
        draw_func: Callable used to render graphs.
        graphs: Graphs to score and display.
        graph_estimator: Estimator providing predict_proba.
        n_show: Number of top and bottom graphs to display.
        prob_for_acceptance_interval: Inclusive (low, high) interval; entries within
            [low, high] are wrapped in square brackets in the titles.

    Returns:
        None.
    """
    probs = graph_estimator.predict_proba(graphs, log=False)
    probs = np.asarray(probs, dtype=float)
    graphs_probs = np.max(probs, axis=1) if probs.size else np.array([])
    order = np.argsort(graphs_probs)
    top_idx = order[::-1][:n_show]
    bottom_idx = order[:n_show][::-1]
    top_graphs = [graphs[i] for i in top_idx]
    bottom_graphs = [graphs[i] for i in bottom_idx]
    estimator = getattr(graph_estimator, "estimator_", None)
    classes = getattr(estimator, "classes_", None)

    # Fallback to numeric class indices if class labels are unavailable.
    if classes is None and probs.size:
        classes = np.arange(probs.shape[1])

    low_high = None
    if prob_for_acceptance_interval is not None:
        try:
            low, high = prob_for_acceptance_interval
            low_high = (float(low), float(high))
        except Exception:
            low_high = None

    def _title_for_index(idx: int) -> str:
        if probs.size == 0 or classes is None:
            return ""
        row = probs[idx]
        parts = []
        for cls, p in zip(classes, row):
            text = f"{p:.2f}"
            if low_high is not None and (low_high[0] <= p <= low_high[1]):
                text = f"*{text}*"
            parts.append(text)
        return "  ".join(parts)

    top_titles = [_title_for_index(i) for i in top_idx]
    bottom_titles = [_title_for_index(i) for i in bottom_idx]

    # If fewer graphs than n_show, plot a single row only
    if len(graphs) < n_show:
        single_idx = order[::-1]  # all graphs, best first
        single_graphs = [graphs[i] for i in single_idx]
        single_titles = [_title_for_index(i) for i in single_idx]
        draw_func(single_graphs, titles=single_titles, n_graphs_per_line=len(single_graphs))
        return

    draw_func(top_graphs, titles=top_titles, n_graphs_per_line=n_show)
    print("...")
    draw_func(bottom_graphs, titles=bottom_titles, n_graphs_per_line=n_show)


class InterpolationGenerator:
    """Generate new graphs via interpolation and model-based filtering in parallel.

    Args:
        interpolation_estimator: Configured interpolation estimator instance.
        rng: Optional numpy RNG for reproducible pairing.
        seed: Optional seed to initialize a numpy RNG.

    Returns:
        None.
    """

    def __init__(
        self,
        interpolation_estimator,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        """Initialize with an interpolation estimator and empty training state.

        Args:
            interpolation_estimator: Configured interpolation estimator instance.
            rng: Optional numpy RNG for reproducible pairing.
            seed: Optional seed to initialize a numpy RNG.

        Returns:
            None.
        """
        self.interpolation_estimator = interpolation_estimator
        self.graph_estimator = None
        self.graphs = []
        self.targets = None
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        # Hash index of current pool for fast dedup across rounds (set of hash keys)
        self._pool_hash_index: Optional[set] = None

    def fit(self, train_graphs, train_targets=None):
        """Fit the estimator and store training data.

        Works with or without targets:
        - Supervised: if ``train_targets`` is provided, fit a classifier
          (RandomForest by default) and use class probabilities.
        - Unsupervised: if ``train_targets`` is ``None``, fit a generative
          density model (GaussianMixture with 3 components by default) and use
          per-component responsibilities as "probabilities" for selection.

        Args:
            train_graphs: Training graphs.
            train_targets: Optional training labels.

        Returns:
            Self for chaining.
        """
        if self.interpolation_estimator.feasibility_estimator is not None:
            self.interpolation_estimator.feasibility_estimator.fit(train_graphs)
        self.interpolation_estimator.fit(train_graphs)
        graph_transformer = self.interpolation_estimator.graph_transformer
        if graph_transformer is None:
            raise ValueError("interpolation_estimator.graph_transformer is required")

        # Choose supervised vs generative estimator
        if train_targets is None:
            # Unsupervised: IsolationForest with ECDF calibration for probabilities
            try:
                seed = int(self.rng.integers(0, 2**32 - 1))
            except Exception:
                seed = None
            estimator = IsolationForestProba(random_state=seed)
            self.targets = None
            y_fit = np.zeros(len(train_graphs), dtype=int)
        else:
            estimator = RandomForestClassifier(n_estimators=300, n_jobs=-1)
            self.targets = np.asarray(train_targets)
            y_fit = train_targets

        self.graph_estimator = GraphEstimator(
            estimator=estimator,
            transformer=graph_transformer,
        )
        self.graph_estimator.fit(train_graphs, y_fit)
        self.graphs = list(train_graphs)
        # Reset pool hash index; rebuilt lazily during generate
        self._pool_hash_index = None
        return self

    def improve(self, graph: nx.Graph, k: int = 5, target_class: int = 1) -> Optional[nx.Graph]:
        """Improve a graph by interpolating towards a random nearby donor and sampling by class probability.

        The method:
        1) Finds the k nearest neighbors of ``graph`` among ``self.graphs`` using the
           interpolation estimator's embedding space.
        2) Samples uniformly one neighbor from these k.
        3) Computes the interpolation graphs between ``graph`` and the sampled neighbor.
        4) Scores all interpolated graphs with the downstream estimator and returns
           one sampled with probability proportional to ``P(class=target_class)``.

        Args:
            graph: The query graph to improve.
            k: Number of nearest neighbors to consider for uniform sampling.
            target_class: Class label whose probability is used as sampling weight.

        Returns:
            One interpolated graph sampled proportionally to its target class probability,
            or None if interpolation produced no candidates.
        """
        if self.graph_estimator is None or self.interpolation_estimator is None:
            raise ValueError("fit must be called before improve.")
        if not self.graphs:
            return None

        # Ensure the interpolation estimator is fit to current donors
        self.interpolation_estimator.fit(self.graphs)

        # Embed donors and the query graph
        transformer = self.interpolation_estimator.graph_transformer
        if transformer is None:
            raise ValueError("interpolation_estimator.graph_transformer is required")
        donor_vectors = self.interpolation_estimator.donor_vectors
        if donor_vectors is None:
            donor_vectors = transformer.transform(self.graphs)

        # Compute distances to all donors and select k nearest (excluding exact self-match when obvious)
        dists = _compute_query_to_donor_distances(
            graph,
            transformer=transformer,
            donor_vectors=donor_vectors,
            metric=getattr(self.interpolation_estimator, "distance_metric", "euclidean"),
        )
        neighbor_indices = _k_nearest_indices_from_distances(dists, k, exclude_zero=True)

        # Uniformly sample one neighbor among the k nearest
        neighbor_idx = int(self.rng.choice(neighbor_indices))
        neighbor_graph = self.graphs[neighbor_idx]

        # Interpolate between the query graph and the sampled neighbor
        interpolated = self.interpolation_estimator.iterated_interpolate(
            graph, neighbor_graph
        )
        if not interpolated:
            return None

        # Score and choose proportional to P(class=target_class)
        selected = _select_graph_for_class(
            interpolated, self.graph_estimator, target_class, self.rng
        )
        if selected is None:
            return interpolated[int(self.rng.integers(0, len(interpolated)))]
        return selected

    def _resolve_rng(self, rng: Optional[Union[np.random.Generator, int]]):
        """Return a numpy RNG from an existing generator or seed."""
        if rng is None:
            return self.rng
        if isinstance(rng, (int, np.integer)):
            return np.random.default_rng(int(rng))
        return rng

    @staticmethod
    def _log(verbose: bool, message):
        """Print a message or message factory when verbose."""
        if not verbose:
            return
        if callable(message):
            message = message()
        print(message)

    def _maybe_demo_plot(
        self,
        *,
        source_orig: nx.Graph,
        destination_orig: nx.Graph,
        interpolated_graphs: Sequence[nx.Graph],
        classes,
        draw_func,
        verbose: bool,
        demo_periodic_interval: int,
        pair_id: int,
        prob_for_acceptance_interval: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Optionally draw a demo path with all class probabilities.

        Titles list p(class)=prob for all classes; entries within the acceptance
        interval are wrapped in square brackets, e.g., [p(A)=0.72].
        """
        if not demo_periodic_interval:
            return
        if pair_id % demo_periodic_interval != 0:
            return
        if not verbose and draw_func is None:
            return
        demo_graphs = [source_orig] + list(interpolated_graphs) + [destination_orig]
        demo_probs = self.graph_estimator.predict_proba(demo_graphs, log=False)
        demo_probs = np.asarray(demo_probs, dtype=float)
        demo_classes = np.asarray(classes) if classes is not None else (
            np.arange(demo_probs.shape[1]) if demo_probs.size else np.array([])
        )

        low_high = None
        if prob_for_acceptance_interval is not None:
            try:
                lo, hi = prob_for_acceptance_interval
                low_high = (float(lo), float(hi))
            except Exception:
                low_high = None

        titles: list[str] = []
        if demo_probs.size and demo_classes.size:
            for row in demo_probs:
                parts: list[str] = []
                for cls, p in zip(demo_classes, row):
                    text = f"{p:.2f}"
                    if low_high is not None and (low_high[0] <= p <= low_high[1]):
                        text = f"[{text}]"
                    parts.append(text)
                titles.append("  ".join(parts))
        else:
            titles = [""] * len(demo_graphs)
        if verbose and draw_func is not None:
            draw_func(demo_graphs, titles=titles, n_graphs_per_line=7)

    def _get_classes(self):
        """Return class labels for the downstream estimator if available."""
        est = getattr(self.graph_estimator, "estimator_", None)
        if est is None:
            return None
        return getattr(est, "classes_", None)

    # ---------------- Round helpers: modularize generate -----------------
    def _resolve_n_jobs_eff(self, n_jobs, n_pairs: int) -> int:
        """Resolve effective worker processes for the round.

        Returns an integer in [1, n_pairs].
        """
        if isinstance(n_jobs, str) and n_jobs == "auto":
            return max(1, min(os.cpu_count() or 1, int(n_pairs)))
        try:
            val = int(n_jobs)
        except Exception:
            val = 1
        return max(1, min(val, int(n_pairs)))

    def _fit_interpolator_for_round(
        self,
        pool_graphs: Sequence[nx.Graph],
        *,
        interp_transformer,
        interp_tr_prev_jobs,
        interp_fe,
        interp_fe_prev_jobs,
        verbose: bool,
    ) -> float:
        """Fit interpolation estimator on current pool and log elapsed."""
        t0 = time.time()
        if interp_transformer is not None and interp_tr_prev_jobs is not None:
            interp_transformer.n_jobs = interp_tr_prev_jobs
        if interp_fe is not None and interp_fe_prev_jobs is not None:
            interp_fe.n_jobs = interp_fe_prev_jobs
        self.interpolation_estimator.fit(pool_graphs)
        elapsed = time.time() - t0
        self._log(
            verbose,
            lambda: (
                f"Interpolation estimator trained on {len(pool_graphs)} graphs. "
                f"[{_fmt_elapsed(elapsed)}]"
            ),
        )
        return float(elapsed)

    def _prepare_scoring_jobs(self, *, n_jobs, est_prev_jobs: Optional[int], tr_prev_jobs: Optional[int]):
        """Adjust scoring thread counts when using process-level parallelism."""
        parallel_requested = (isinstance(n_jobs, str) and n_jobs == "auto") or (
            isinstance(n_jobs, (int, np.integer)) and int(n_jobs) > 1
        )
        if not parallel_requested or self.graph_estimator is None:
            return est_prev_jobs, tr_prev_jobs
        est = getattr(self.graph_estimator, "estimator_", None)
        tr = getattr(self.graph_estimator, "transformer_", None)
        if hasattr(est, "n_jobs"):
            est_prev_jobs = est.n_jobs
        if hasattr(tr, "n_jobs"):
            tr_prev_jobs = tr.n_jobs
        # Heuristic: keep a small pool for scoring to leave CPUs for workers
        cpu = os.cpu_count() or 8
        score_jobs = max(1, min(8, cpu // 4))
        if hasattr(est, "n_jobs"):
            est.n_jobs = score_jobs
        if hasattr(tr, "n_jobs"):
            tr.n_jobs = score_jobs
        return est_prev_jobs, tr_prev_jobs

    def _chunk_pairs(self, pairs: Sequence[Tuple[int, int]], n_jobs_local: int, chunk_size: Union[int, str]):
        """Return a list of pair chunks according to the requested chunking policy."""
        if isinstance(chunk_size, str) and chunk_size == "auto":
            target_chunks = max(1, min(n_jobs_local, len(pairs)))
            n_total = len(pairs)
            num_chunks = 1
            for k in range(target_chunks, 0, -1):
                if n_total % k == 0:
                    num_chunks = k
                    break
            eff_size = max(1, n_total // num_chunks)
            chunks = [pairs[i:i + eff_size] for i in range(0, n_total, eff_size)]
        else:
            try:
                eff_size = int(chunk_size)
            except Exception:
                eff_size = 4
            if eff_size <= 0:
                eff_size = 4
            chunks = [pairs[i:i + eff_size] for i in range(0, len(pairs), eff_size)]
        return chunks

    def _log_parallel_setup(self, *, verbose: bool, chunks: list[list[Tuple[int, int]]], n_pairs: int) -> None:
        """Log parallel chunk layout for the round."""
        if not verbose:
            return
        chunk_sizes = [len(chunk) for chunk in chunks]
        min_chunk = min(chunk_sizes) if chunk_sizes else 0
        max_chunk = max(chunk_sizes) if chunk_sizes else 0
        chunk_range = f"{min_chunk}" if min_chunk == max_chunk else f"{min_chunk}:{max_chunk}"
        self._log(verbose, lambda: (f"Parallel run: {len(chunks)} chunks for {n_pairs} pairs; pairs per chunk: {chunk_range}"))

    def _log_parallel_progress(
        self,
        *,
        verbose: bool,
        pairs_seen_round: int,
        n_pairs: int,
        n_rounds: int,
        round_idx: int,
        completed: int,
        n_chunks: int,
        raw_chunk_count: int,
        kept_in_chunk: int,
        total_interpolated_round: int,
        round_elite_targets: list[Any],
        elite_targets: list[Any],
        round_step_t0: float,
        interp_elapsed: float,
        elapsed_secs: float,
        chunk_len: int,
    ) -> None:
        """Log throughput and ETA for parallel execution."""
        if not verbose:
            return
        elapsed_round = max(1e-6, time.time() - round_step_t0)
        throughput = pairs_seen_round / elapsed_round
        if throughput <= 0:
            throughput = chunk_len / max(1e-6, elapsed_secs)
        remaining_pairs = max(0, n_pairs - pairs_seen_round)
        remaining_rounds = max(0, n_rounds - (round_idx + 1))
        eta_round_secs = remaining_pairs / max(throughput, 1e-9)
        per_round_overhead = float(interp_elapsed) if interp_elapsed is not None else 0.0
        eta_total_secs = eta_round_secs + remaining_rounds * (n_pairs / max(throughput, 1e-9) + per_round_overhead)
        eta_round = _fmt_elapsed(eta_round_secs)
        eta_total = _fmt_elapsed(eta_total_secs)
        eta_clock = time.strftime("%H:%M", time.localtime(time.time() + eta_total_secs))
        pairs_seen_total = (round_idx * n_pairs) + pairs_seen_round
        total_pairs = n_pairs * n_rounds
        total_generated_so_far = len(elite_targets) + len(round_elite_targets)
        avg_generated_per_pair = total_generated_so_far / max(1, pairs_seen_total)
        estimated_total_generated = int(round(avg_generated_per_pair * total_pairs))
        self._log(
            verbose,
            lambda: (
                f"[pairs: {pairs_seen_round:2d}/{n_pairs}  round: {round_idx+1:2d}/{n_rounds}]  "
                f"gen: {raw_chunk_count:2d}  accepted: {kept_in_chunk:>3}  "
                f"round tot: {total_interpolated_round:>3}  "
                f"#gen curr round: {len(round_elite_targets):>3}  "
                f"#gen: {total_generated_so_far:>4}  "
                f"#to be gen: {estimated_total_generated:>4}  "
                f"chunk: {completed:>3}/{n_chunks:<3}  done in: {_fmt_elapsed(elapsed_secs):>7} for {chunk_len} pairs  "
                f"ETA round: {eta_round:>6}  ETA: {eta_total:>6} (at {eta_clock})"
            ),
        )

    def _process_completed_chunk(
        self,
        *,
        fut,
        ctx: _ChunkProcessingContextProtocol,
    ) -> None:
        """Process one completed chunk future and update round counters."""
        chunk_id = ctx.fut_to_chunk[fut]
        res = fut.result()
        ctx.completed += 1
        elapsed_secs = time.time() - ctx.start_times[chunk_id]
        chunk_len = len(ctx.chunks[chunk_id])
        ctx.pairs_seen_round += chunk_len
        graphs_chunk = res.get("graphs", [])
        hashes_chunk = res.get("hashes", None)
        raw_chunk_count = len(graphs_chunk)
        ctx.total_interpolated_round += raw_chunk_count
        selected_graphs_chunk, selected_targets_chunk, selected_hashes_chunk = self._score_and_select_chunk(
            graphs_chunk=graphs_chunk,
            hashes_chunk=hashes_chunk,
            prob_for_acceptance_interval=ctx.prob_for_acceptance_interval,
            accept_only_class=ctx.accept_only_class,
        )
        kept_in_chunk = self._dedup_selected_chunk(
            selected_graphs=selected_graphs_chunk,
            selected_targets=selected_targets_chunk,
            selected_hashes=selected_hashes_chunk,
            pool_graphs=ctx.pool_graphs,
            round_elite_graphs=ctx.round_elite_graphs,
            round_elite_targets=ctx.round_elite_targets,
        )
        self._maybe_stream_demo_chunk(
            res=res,
            completed=ctx.completed,
            stream_plots_every=ctx.stream_plots_every,
            draw_func=ctx.draw_func,
            verbose=ctx.verbose,
            prob_for_acceptance_interval=ctx.prob_for_acceptance_interval,
        )

        self._log_parallel_progress(
            verbose=ctx.verbose,
            pairs_seen_round=ctx.pairs_seen_round,
            n_pairs=ctx.pairs_len,
            n_rounds=ctx.n_rounds,
            round_idx=ctx.round_idx,
            completed=ctx.completed,
            n_chunks=ctx.n_chunks,
            raw_chunk_count=raw_chunk_count,
            kept_in_chunk=kept_in_chunk,
            total_interpolated_round=ctx.total_interpolated_round,
            round_elite_targets=ctx.round_elite_targets,
            elite_targets=ctx.elite_targets,
            round_step_t0=ctx.round_step_t0,
            interp_elapsed=ctx.interp_elapsed,
            elapsed_secs=elapsed_secs,
            chunk_len=chunk_len,
        )
        return None

    def _score_and_select_chunk(
        self,
        *,
        graphs_chunk: Sequence[nx.Graph],
        hashes_chunk: Optional[Sequence[int]],
        prob_for_acceptance_interval: tuple[float, float],
        accept_only_class: Optional[int],
    ) -> tuple[list[nx.Graph], list[Any], list[int]]:
        """Score chunk graphs and select those within the acceptance interval."""
        if not graphs_chunk:
            return [], [], []
        probs = self.graph_estimator.predict_proba(graphs_chunk, log=False)
        probs = np.asarray(probs, dtype=float)
        classes = self._get_classes()
        low, high = prob_for_acceptance_interval
        selected_graphs: list[nx.Graph] = []
        selected_targets: list[Any] = []
        selected_hashes: list[int] = []
        # Determine which probability columns are eligible
        col_indices_allowed: Optional[np.ndarray] = None
        if accept_only_class is not None:
            if classes is not None:
                matches = np.where(classes == accept_only_class)[0]
                if matches.size == 0:
                    raise ValueError(f"accept_only_class={accept_only_class} not found in estimator classes.")
                col_indices_allowed = matches
            else:
                # Fallback: treat accept_only_class as column index if in range
                idx = int(accept_only_class)
                if 0 <= idx < probs.shape[1]:
                    col_indices_allowed = np.array([idx])
                else:
                    raise ValueError("accept_only_class provided but classes are unavailable.")
        for ii, (graph, row) in enumerate(zip(graphs_chunk, probs)):
            mask = (row >= low) & (row <= high)
            hit_indices = np.where(mask)[0]
            if col_indices_allowed is not None:
                hit_indices = np.intersect1d(hit_indices, col_indices_allowed, assume_unique=False)
            for idx_c in hit_indices:
                selected_graphs.append(graph)
                selected_targets.append(classes[idx_c] if classes is not None else int(idx_c))
                h = None
                if hashes_chunk is not None and ii < len(hashes_chunk):
                    h = hashes_chunk[ii]
                else:
                    try:
                        h = hash_graph(graph)
                    except Exception:
                        h = None
                selected_hashes.append(-1 if h is None else h)
        return selected_graphs, selected_targets, selected_hashes

    def _dedup_selected_chunk(
        self,
        *,
        selected_graphs: Sequence[nx.Graph],
        selected_targets: Sequence[Any],
        selected_hashes: Sequence[int],
        pool_graphs: Sequence[nx.Graph],
        round_elite_graphs: list[nx.Graph],
        round_elite_targets: list[Any],
    ) -> int:
        """Deduplicate selected graphs against the pool hash index."""
        if not selected_graphs:
            return 0
        if self._pool_hash_index is None:
            pool_index = GraphHashDeduper().build_index(pool_graphs)
            self._pool_hash_index = set(pool_index.keys())
        new_hashes: list[int] = []
        for h, g, t in zip(selected_hashes, selected_graphs, selected_targets):
            if h not in self._pool_hash_index:
                round_elite_graphs.append(g)
                round_elite_targets.append(t)
                new_hashes.append(h)
        if new_hashes:
            self._pool_hash_index.update(new_hashes)
        return len(new_hashes)

    def _maybe_stream_demo_chunk(
        self,
        *,
        res: dict,
        completed: int,
        stream_plots_every: int,
        draw_func,
        verbose: bool,
        prob_for_acceptance_interval: tuple[float, float],
    ) -> None:
        """Optionally stream a demo plot for a completed chunk."""
        if not stream_plots_every or draw_func is None or res.get("demo") is None:
            return
        if completed % int(stream_plots_every) != 0:
            return
        src, dst, path = res["demo"]
        classes = self._get_classes()
        self._maybe_demo_plot(
            source_orig=src,
            destination_orig=dst,
            interpolated_graphs=path,
            classes=classes,
            draw_func=draw_func,
            verbose=verbose,
            demo_periodic_interval=1,
            pair_id=0,
            prob_for_acceptance_interval=prob_for_acceptance_interval,
        )

    def _run_round_parallel(
        self,
        *,
        pairs: Sequence[Tuple[int, int]],
        pool_graphs: list[nx.Graph],
        round_idx: int,
        n_rounds: int,
        n_iterations: int,
        best_of: int,
        prob_for_acceptance_interval: tuple[float, float],
        draw_func,
        verbose: bool,
        demo_periodic_interval: int,
        round_step_t0: float,
        interp_elapsed: float,
        elite_targets: list[Any],
        stream_plots_every: int,
        n_jobs_eff: int,
        chunk_size: Union[int, str],
        accept_only_class: Optional[int],
    ) -> tuple[list[nx.Graph], list[Any]]:
        """Process one round in parallel across pair chunks (n_jobs_eff > 1)."""
        # Build interpolator config dict; ensure it is picklable.
        ie = self.interpolation_estimator
        decomposition_function = getattr(ie, "decomposition_function", None)
        decomposition_xml = None
        try:
            from abstractgraph.xml import operator_to_xml_string
            if decomposition_function is not None:
                decomposition_xml = operator_to_xml_string(decomposition_function)
        except Exception:
            decomposition_xml = None
        decomposition_bytes = None
        if decomposition_xml is None and decomposition_function is not None:
            try:
                decomposition_bytes = dill.dumps(decomposition_function)
            except Exception:
                decomposition_bytes = None
        feasibility_estimator = getattr(ie, "feasibility_estimator", None)
        feasibility_bytes = None
        if feasibility_estimator is not None:
            try:
                feasibility_bytes = dill.dumps(feasibility_estimator)
            except Exception:
                feasibility_bytes = None
        interpolator_bytes = None
        try:
            interpolator_bytes = dill.dumps(ie)
        except Exception:
            interpolator_bytes = None
        interpolator_config = dict(
            nbits=getattr(getattr(ie, "graph_transformer", None), "nbits", 19),
            k=getattr(ie, "k", 5),
            n_samples=getattr(ie, "n_samples", 1),
            n_iterations=getattr(ie, "n_iterations", 1),
            degree_penalty=getattr(ie, "degree_penalty", "auto"),
            degree_penalty_mode=getattr(ie, "degree_penalty_mode", "multiplicative"),
            cut_radius=getattr(ie, "cut_radius", None),
            cut_scope=getattr(ie, "cut_scope", "both"),
            decomposition_xml=decomposition_xml,
            decomposition_bytes=decomposition_bytes,
            feasibility_bytes=feasibility_bytes,
            interpolator_bytes=interpolator_bytes,
        )
        # Validate payload
        payload_ok = True
        try:
            dill.dumps((pool_graphs[:1], [(0, 0)], interpolator_config))
        except Exception:
            payload_ok = False
        if decomposition_xml is None and decomposition_bytes is None:
            payload_ok = False
        if feasibility_estimator is not None and feasibility_bytes is None:
            payload_ok = False
        if interpolator_bytes is None:
            payload_ok = False
        if not payload_ok:
            raise ValueError("Parallel payload not picklable; unable to run in parallel mode.")
        n_jobs_local = n_jobs_eff

        @dataclass
        class _ChunkProcessingContext:
            fut_to_chunk: dict
            start_times: dict
            chunks: list[list[Tuple[int, int]]]
            pool_graphs: list[nx.Graph]
            prob_for_acceptance_interval: tuple[float, float]
            round_elite_graphs: list[nx.Graph]
            round_elite_targets: list[Any]
            elite_targets: list[Any]
            pairs_len: int
            round_idx: int
            n_rounds: int
            round_step_t0: float
            interp_elapsed: float
            verbose: bool
            draw_func: object
            stream_plots_every: int
            completed: int
            pairs_seen_round: int
            total_interpolated_round: int
            n_chunks: int
            accept_only_class: Optional[int]

        chunks = self._chunk_pairs(pairs, n_jobs_local, chunk_size)
        self._log_parallel_setup(verbose=verbose, chunks=chunks, n_pairs=len(pairs))

        round_elite_graphs: list[nx.Graph] = []
        round_elite_targets: list[Any] = []
        with ProcessPoolExecutor(max_workers=n_jobs_local, mp_context=mp.get_context("spawn")) as ex:
            futures = []
            start_times = {}
            for chunk_id, chunk in enumerate(chunks):
                seed = int(hash((round_idx, chunk_id, len(chunk))) % (2**31 - 1))
                fut = ex.submit(
                    _interpolate_pairs_chunk_worker,
                    pool_graphs,
                    chunk,
                    interpolator_config=interpolator_config,
                    best_of=best_of,
                    n_iterations=n_iterations,
                    rng_seed=seed,
                )
                futures.append((chunk_id, fut))
                start_times[chunk_id] = time.time()

            completed = 0
            total_interpolated_round = 0
            fut_to_chunk = {f: cid for cid, f in futures}
            pairs_seen_round = 0
            ctx = _ChunkProcessingContext(
                fut_to_chunk=fut_to_chunk,
                start_times=start_times,
                chunks=chunks,
                pool_graphs=pool_graphs,
                prob_for_acceptance_interval=prob_for_acceptance_interval,
                round_elite_graphs=round_elite_graphs,
                round_elite_targets=round_elite_targets,
                elite_targets=elite_targets,
                pairs_len=len(pairs),
                round_idx=round_idx,
                n_rounds=n_rounds,
                round_step_t0=round_step_t0,
                interp_elapsed=interp_elapsed,
                verbose=verbose,
                draw_func=draw_func,
                stream_plots_every=stream_plots_every,
                completed=completed,
                pairs_seen_round=pairs_seen_round,
                total_interpolated_round=total_interpolated_round,
                n_chunks=len(chunks),
                accept_only_class=accept_only_class,
            )
            for fut in as_completed(fut_to_chunk):
                self._process_completed_chunk(
                    fut=fut,
                    ctx=ctx,
                )
                completed = ctx.completed
                pairs_seen_round = ctx.pairs_seen_round
                total_interpolated_round = ctx.total_interpolated_round

        return round_elite_graphs, round_elite_targets

    def generate(
        self,
        *,
        n_rounds=2,
        n_pairs=64,
        n_iterations=3,
        best_of=3,
        prob_for_acceptance_interval: tuple[float, float] = (0.5, 1.0),
        accept_only_class: Optional[int] = None,
        draw_func=None,
        verbose=False,
        demo_periodic_interval=0,
        rng: Optional[Union[np.random.Generator, int]] = None,
        n_jobs: Union[int, str] = "auto",
        chunk_size: Union[int, str] = "auto",
        stream_plots_every: int = 0,
    ):
        """Generate new graphs by interpolating pairs and filtering by probability.

        Args:
            n_rounds: Number of generation rounds.
            n_pairs: Number of pairs per round.
            n_iterations: Iterations per interpolation.
            best_of: Number of attempts per interpolation.
            prob_for_acceptance_interval: Inclusive probability interval for keeping graphs per class.
            accept_only_class: If not None, keep only predictions for this class label
                (e.g., 1 to keep only inliers when using IsolationForestProba). When None
                consider all classes within the acceptance interval.
            draw_func: Optional drawing function for progress visualization.
            verbose: Whether to print progress messages.
            demo_periodic_interval: Interval for periodic demo visualizations.
            rng: Optional numpy RNG or seed used for pairing.
            n_jobs: Process-level parallelism across pair chunks. If 'auto', uses
                available CPUs (capped by n_pairs).
            chunk_size: Pairs per worker task. If 'auto', chooses equal-sized chunks
                so that num_chunks * chunk_size == n_pairs with num_chunks as large
                as possible but <= n_jobs.
            stream_plots_every: If >0, plot one demo after every N completed chunks.

        Returns:
            Tuple of (generated_graphs, generated_targets) where generated_targets
            mirrors the generated graphs' assigned class labels. Graphs are unique
            after hash-based deduplication within each round.
        """
        if self.graph_estimator is None or self.interpolation_estimator is None:
            raise ValueError("fit must be called before generate.")
        # Unsupervised mode allowed (self.targets may be None)
        if (
            prob_for_acceptance_interval is None
            or len(prob_for_acceptance_interval) != 2
        ):
            raise ValueError("prob_for_acceptance_interval must be a (low, high) tuple.")
        low_prob, high_prob = prob_for_acceptance_interval
        if low_prob > high_prob:
            raise ValueError("prob_for_acceptance_interval lower bound must be <= upper bound.")
        t0 = time.time()
        pool_graphs = list(self.graphs)
        elite_graphs: list[nx.Graph] = []
        elite_targets: list[Any] = []
        rng = self._resolve_rng(rng)

        # Preserve parallel settings for interpolation estimator fit.
        interp_transformer = getattr(self.interpolation_estimator, "graph_transformer", None)
        interp_tr_prev_jobs = getattr(interp_transformer, "n_jobs", None) if interp_transformer is not None else None
        interp_fe = getattr(self.interpolation_estimator, "feasibility_estimator", None)
        interp_fe_prev_jobs = getattr(interp_fe, "n_jobs", None) if interp_fe is not None else None

        # Avoid CPU oversubscription when using process-level parallelism by
        # limiting the scoring (transform + predict) thread count in main.
        est_prev_jobs = None
        tr_prev_jobs = None
        est_prev_jobs, tr_prev_jobs = self._prepare_scoring_jobs(
            n_jobs=n_jobs,
            est_prev_jobs=est_prev_jobs,
            tr_prev_jobs=tr_prev_jobs,
        )

        for round_idx in range(n_rounds):
            pairs = make_pairs(len(pool_graphs), n_pairs, rng)
            # Fit interpolation estimator for current pool
            interp_elapsed = self._fit_interpolator_for_round(
                pool_graphs,
                interp_transformer=interp_transformer,
                interp_tr_prev_jobs=interp_tr_prev_jobs,
                interp_fe=interp_fe,
                interp_fe_prev_jobs=interp_fe_prev_jobs,
                verbose=verbose,
            )
            round_step_t0 = time.time()
            n_jobs_eff = self._resolve_n_jobs_eff(n_jobs, len(pairs))
            round_elite_graphs, round_elite_targets = self._run_round_parallel(
                pairs=pairs,
                pool_graphs=pool_graphs,
                round_idx=round_idx,
                n_rounds=n_rounds,
                n_iterations=n_iterations,
                best_of=best_of,
                prob_for_acceptance_interval=prob_for_acceptance_interval,
                draw_func=draw_func,
                verbose=verbose,
                demo_periodic_interval=demo_periodic_interval,
                round_step_t0=round_step_t0,
                interp_elapsed=interp_elapsed,
                elite_targets=elite_targets,
                stream_plots_every=stream_plots_every,
                n_jobs_eff=int(n_jobs_eff),
                chunk_size=chunk_size,
                accept_only_class=accept_only_class,
            )

            # Accumulate and log per-round results
            elite_graphs.extend(round_elite_graphs)
            elite_targets.extend(round_elite_targets)
            pool_graphs.extend(round_elite_graphs)
            self._log(
                verbose,
                lambda: (
                    f"Round {round_idx+1}/{n_rounds}: added {len(round_elite_graphs)} instances "
                    f"(per class: "
                    + ", ".join(
                        f"{str(c)}:{round_elite_targets.count(c)}" for c in set(round_elite_targets)
                    )
                    + ")    "
                    f"total pool: {len(pool_graphs)}. [{_fmt_elapsed(time.time() - round_step_t0)}]"
                ),
            )
            if verbose and draw_func is not None and round_elite_graphs:
                # Plot a separate sample for each class accepted in this round
                classes_in_round = list(set(round_elite_targets))
                for cls in classes_in_round:
                    cls_graphs = [g for g, t in zip(round_elite_graphs, round_elite_targets) if t == cls]
                    if not cls_graphs:
                        continue
                    print(f"Class {cls}: showing samples ({len(cls_graphs)} accepted this round)")
                    plot_sample(
                        draw_func,
                        cls_graphs,
                        self.graph_estimator,
                        n_show=7,
                        prob_for_acceptance_interval=prob_for_acceptance_interval,
                    )

        self._log(
            verbose,
            lambda: f"Generated size: {len(elite_targets)}. [{_fmt_elapsed(time.time() - t0)}]",
        )
        if draw_func is not None and elite_graphs:
            # Plot a separate sample for each class in the final generated set
            classes_overall = list(set(elite_targets))
            for cls in classes_overall:
                cls_graphs = [g for g, t in zip(elite_graphs, elite_targets) if t == cls]
                if not cls_graphs:
                    continue
                print(f"Class {cls}: final samples ({len(cls_graphs)} generated)")
                plot_sample(
                    draw_func,
                    cls_graphs,
                    self.graph_estimator,
                    n_show=7,
                    prob_for_acceptance_interval=prob_for_acceptance_interval,
                )

        # Restore scoring jobs
        if est_prev_jobs is not None and self.graph_estimator is not None:
            est = getattr(self.graph_estimator, "estimator_", None)
            if hasattr(est, "n_jobs"):
                est.n_jobs = est_prev_jobs
        if tr_prev_jobs is not None and self.graph_estimator is not None:
            tr = getattr(self.graph_estimator, "transformer_", None)
            if hasattr(tr, "n_jobs"):
                tr.n_jobs = tr_prev_jobs

        return elite_graphs, elite_targets


class SerialInterpolationGenerator:
    """Serial-only interpolation generator with minimal configuration."""

    def __init__(
        self,
        interpolation_estimator,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        """Initialize with an interpolation estimator and empty training state.

        Args:
            interpolation_estimator: Configured interpolation estimator instance.
            rng: Optional numpy RNG for reproducible pairing.
            seed: Optional seed to initialize a numpy RNG.

        Returns:
            None.
        """
        self.interpolation_estimator = interpolation_estimator
        self.graph_estimator = None
        self.graphs = []
        self.targets = None
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        self._pool_hash_index: Optional[set] = None

    def fit(self, train_graphs, train_targets=None):
        """Fit estimator; supports supervised and unsupervised modes.

        - If ``train_targets`` provided → supervised classifier (RandomForest).
        - If ``None`` → generative density (GaussianMixture with 3 components).
        """
        if self.interpolation_estimator.feasibility_estimator is not None:
            self.interpolation_estimator.feasibility_estimator.fit(train_graphs)
        self.interpolation_estimator.fit(train_graphs)
        graph_transformer = self.interpolation_estimator.graph_transformer
        if graph_transformer is None:
            raise ValueError("interpolation_estimator.graph_transformer is required")

        if train_targets is None:
            try:
                seed = int(self.rng.integers(0, 2**32 - 1))
            except Exception:
                seed = None
            estimator = IsolationForestProba(random_state=seed)
            y_fit = np.zeros(len(train_graphs), dtype=int)
            self.targets = None
        else:
            estimator = RandomForestClassifier(n_estimators=300, n_jobs=-1)
            y_fit = train_targets
            self.targets = np.asarray(train_targets)

        self.graph_estimator = GraphEstimator(
            estimator=estimator,
            transformer=graph_transformer,
        )
        self.graph_estimator.fit(train_graphs, y_fit)
        self.graphs = list(train_graphs)
        self._pool_hash_index = None
        return self

    def _get_classes(self):
        est = getattr(self.graph_estimator, "estimator_", None)
        if est is None:
            return None
        return getattr(est, "classes_", None)

    def _resolve_rng(self, rng: Optional[Union[np.random.Generator, int]]):
        """Return a numpy RNG from an existing generator or seed."""
        if rng is None:
            return self.rng
        if isinstance(rng, (int, np.integer)):
            return np.random.default_rng(int(rng))
        return rng

    def generate(
        self,
        *,
        n_rounds=2,
        n_pairs=64,
        n_iterations=3,
        best_of=3,
        prob_for_acceptance_interval: tuple[float, float] = (0.5, 1.0),
        accept_only_class: Optional[int] = None,
        rng: Optional[Union[np.random.Generator, int]] = None,
    ):
        """Generate new graphs by interpolating pairs and filtering by probability.

        Args:
            n_rounds: Number of generation rounds.
            n_pairs: Number of pairs per round.
            n_iterations: Iterations per interpolation.
            best_of: Number of attempts per interpolation.
            prob_for_acceptance_interval: Inclusive probability interval for keeping graphs per class.
            accept_only_class: If not None, keep only predictions for this class label
                (e.g., 1 to keep only inliers when using IsolationForestProba). When None
                consider all classes within the acceptance interval.
            rng: Optional numpy RNG or seed used for pairing.

        Returns:
            Tuple of (generated_graphs, generated_targets) where generated_targets
            mirrors the generated graphs' assigned class labels. Graphs are unique
            after hash-based deduplication within each round.
        """
        if self.graph_estimator is None or self.interpolation_estimator is None:
            raise ValueError("fit must be called before generate.")
        # Unsupervised mode allowed (self.targets may be None)
        if (
            prob_for_acceptance_interval is None
            or len(prob_for_acceptance_interval) != 2
        ):
            raise ValueError("prob_for_acceptance_interval must be a (low, high) tuple.")
        low_prob, high_prob = prob_for_acceptance_interval
        if low_prob > high_prob:
            raise ValueError("prob_for_acceptance_interval lower bound must be <= upper bound.")

        pool_graphs = list(self.graphs)
        elite_graphs: list[nx.Graph] = []
        elite_targets: list[Any] = []
        rng = self._resolve_rng(rng)

        for _round_idx in range(n_rounds):
            pairs = make_pairs(len(pool_graphs), n_pairs, rng)
            self.interpolation_estimator.fit(pool_graphs)
            round_elite_graphs: list[nx.Graph] = []
            round_elite_targets: list[Any] = []
            for i, j in pairs:
                interpolated_graphs = self.interpolation_estimator.iterated_interpolate_idxs(
                    i, j, n_iterations=n_iterations, best_of=best_of
                )
                if not interpolated_graphs:
                    continue
                probs = self.graph_estimator.predict_proba(interpolated_graphs, log=False)
                probs = np.asarray(probs, dtype=float)
                classes = self._get_classes()
                selected_graphs: list[nx.Graph] = []
                selected_targets: list[Any] = []
                # Resolve acceptable columns for selection
                col_indices_allowed: Optional[np.ndarray] = None
                if accept_only_class is not None:
                    if classes is not None:
                        matches = np.where(classes == accept_only_class)[0]
                        if matches.size == 0:
                            raise ValueError(f"accept_only_class={accept_only_class} not found in estimator classes.")
                        col_indices_allowed = matches
                    else:
                        idx = int(accept_only_class)
                        if 0 <= idx < probs.shape[1]:
                            col_indices_allowed = np.array([idx])
                        else:
                            raise ValueError("accept_only_class provided but classes are unavailable.")
                for graph, row in zip(interpolated_graphs, probs):
                    mask = (row >= low_prob) & (row <= high_prob)
                    hit_indices = np.where(mask)[0]
                    if col_indices_allowed is not None:
                        hit_indices = np.intersect1d(hit_indices, col_indices_allowed, assume_unique=False)
                    for idx in hit_indices:
                        selected_graphs.append(graph)
                        selected_targets.append(classes[idx] if classes is not None else int(idx))
                if selected_graphs:
                    if self._pool_hash_index is None:
                        pool_index = GraphHashDeduper().build_index(pool_graphs)
                        self._pool_hash_index = set(pool_index.keys())
                    hashes = [hash_graph(g) for g in selected_graphs]
                    for h, g, t in zip(hashes, selected_graphs, selected_targets):
                        if h not in self._pool_hash_index:
                            round_elite_graphs.append(g)
                            round_elite_targets.append(t)
                            self._pool_hash_index.add(h)

            elite_graphs.extend(round_elite_graphs)
            elite_targets.extend(round_elite_targets)
            pool_graphs.extend(round_elite_graphs)

        return elite_graphs, elite_targets
