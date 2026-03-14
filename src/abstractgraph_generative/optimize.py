"""Graph optimization with combinatorial Thompson Sampling over generator sets.

This module implements a practical optimizer for target-conditioned graph
generation when the generator quality depends on the *set* used during `fit`.
The core setting is:

- A generator (e.g., conditional autoregressive) that recombines parts seen in
  the graphs provided at fit time.
- A scalar scoring objective in `[0, 1]` provided by `score_function`.
- A generator set that must be both high quality and structurally diverse so it
  can support useful part swapping/composition during generation.

High-level idea
---------------
We optimize a generator set `S` iteratively. At each iteration:

1. Fit generator on current generator set and generate candidate graphs.
2. Build a candidate pool from current generator-set members + novel graphs.
3. Select the next generator set with combinatorial Thompson Sampling:
   - sample uncertainty-aware utility per candidate from Beta posteriors,
   - combine with current known score term,
   - build the generator set sequentially with a redundancy penalty.
4. Evaluate selected set by rollout reward `R(S)`:
   - refit generator on selected generator set,
   - generate a probe batch,
   - compute `R(S)` as a chosen quantile of probe-generation scores.
5. Update TS posteriors with *uniform credit assignment*:
   - normalize reward `r = clip(R(S), 0, 1)`,
   - for each generator-set member hash: `alpha += r`, `beta += (1-r)`.
6. Optionally run bounded local swap refinement:
   - attempt replacing weak generator-set members with strong non-selected candidates,
   - accept swaps only when `R(S)` improves by a minimum threshold.

Why this design
---------------
- Set-level reward aligns optimization with the true downstream objective:
  \"does this generator set enable good generation?\" rather than only \"are members
  individually similar to target?\"
- Thompson Sampling provides adaptive exploration without fixed random schedules.
- Combinatorial sequential construction + redundancy penalty reduces generator-set
  collapse to near-duplicates.
- Local swaps provide recovery from early greedy mistakes at low extra cost.

Posterior model note (Beta with fractional updates)
---------------------------------------------------
The Thompson Sampling posterior for each graph hash is modeled as a Beta
distribution `Beta(alpha, beta)`, which is conjugate to Bernoulli rewards.
In this optimizer, the observed set reward is a continuous value in `[0, 1]`
rather than an exact binary outcome. We therefore use a *fractional* (soft)
update rule:

- `alpha += r`
- `beta += (1 - r)`

with `r = clip(R(S), 0, 1)`.

This can be interpreted as adding a fractional pseudo-success and
pseudo-failure each iteration. It is a pragmatic approximation of Bernoulli TS
that preserves simple sampling and uncertainty behavior while supporting
continuous reward signals.

Step-by-step iteration example
------------------------------
Suppose:
- current generator set size is 3,
- selected set in this iteration becomes `S = {g1, g2, g3}`,
- posterior params before update are:
  - `g1: (alpha=2.0, beta=1.0)`
  - `g2: (alpha=1.5, beta=1.5)`
  - `g3: (alpha=1.0, beta=2.0)`.

One optimization iteration proceeds as:

1. Fit generator on current set and generate a main candidate batch.
2. Build candidate pool (`fit_graphs + novel_generated`) and score candidates.
3. Sample TS utilities from each candidate Beta posterior.
4. Construct new set sequentially with redundancy penalty.
5. Refit generator on selected set `S` and generate probe batch.
6. Compute set reward:
   - if probe scores are `[0.70, 0.80, 0.90, 0.60]`,
   - then `R(S) = P75 = 0.825`,
   - normalized reward is `r = clip(0.825, 0, 1) = 0.825`.
7. Apply uniform fractional Beta update to all selected members:
   - `g1: alpha=2.0+0.75=2.75, beta=1.0+0.25=1.25`
   - `g2: alpha=1.5+0.75=2.25, beta=1.5+0.25=1.75`
   - `g3: alpha=1.0+0.75=1.75, beta=2.0+0.25=2.25`.

Interpretation: every selected member receives the same set-quality signal for
that iteration; members in strong sets become more likely to be sampled again,
unless diversity penalties make them redundant in a given context.

Redundancy penalty and score mixing
-----------------------------------
During sequential generator-set construction, each candidate `i` is ranked by:

`utility_i = sampled_ts_i + ts_similarity_weight * score_i - ts_diversity_penalty * max_redundancy_i`

where:
- `sampled_ts_i` is the Thompson sample from the candidate's Beta posterior,
- `score_i` is the known score from `score_function` in `[0, 1]`,
- `max_redundancy_i` is the maximum similarity between candidate `i` and the
  already-selected set members (computed via `similarity_function`).

Parameters controlling the mix:
- `ts_similarity_weight`: scales the deterministic score contribution.
  Higher values push toward greedy high-score selection.
- `ts_diversity_penalty`: scales the redundancy subtraction.
  Higher values push toward more diverse/non-overlapping members.
- `similarity_function`: defines what redundancy means numerically.
  Its calibration directly affects the effective penalty magnitude.

Tuning intuition:
- If the set collapses to near-duplicates, increase `ts_diversity_penalty`.
- If diversity is too strong and quality drops, decrease
  `ts_diversity_penalty` and/or increase `ts_similarity_weight`.

Main public API
---------------
- `GraphOptimizer(generator, score_function, similarity_function=None)`:
  receives generation and scoring components.
- `fit(graphs)`:
  caches optimization pool.
- `optimize(...)`:
  runs the iterative loop and returns `GraphOptimizationResult`.
- `plot_score_progress(...)`:
  plots best/mean generator-set score across search budget.

Notes
-----
- Deduplication and novelty filtering are hash-based (`hash_graph`).
- Progress history stores both member-level metrics (best/mean generator-set score)
  and set-level signal (`set_reward`) used for posterior updates.
"""

from __future__ import annotations

import os
import pickle
import random
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from abstractgraph.hashing import GraphHashDeduper, hash_graph


def _fmt_elapsed(seconds: float) -> str:
    """
    Format elapsed seconds into a compact human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        str: Formatted elapsed time.
    """
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass
class GraphOptimizationResult:
    """Container for optimization outputs.

    Args:
        generator_set_graphs: Final generator-set graphs.
        generator_set_scores: Scores for final generator-set graphs.
        best_generator_set_graphs: Best-so-far generator-set graphs.
        best_generator_set_scores: Scores for best-so-far generator-set graphs.
        best_generated_from_best_set: Generated graphs produced when evaluating
            the best generator set.
        history: Per-iteration optimization history.
        all_generated_novel: All novel generated graphs across iterations.
    """

    generator_set_graphs: list[nx.Graph]
    generator_set_scores: np.ndarray
    best_generator_set_graphs: list[nx.Graph]
    best_generator_set_scores: np.ndarray
    best_generated_from_best_set: list[nx.Graph]
    history: list[dict[str, Any]]
    all_generated_novel: list[nx.Graph]


@dataclass
class _OptimizationRunState:
    """Mutable state for a single optimize() run."""

    generator_set_graphs: list[nx.Graph]
    generator_set_scores: np.ndarray
    best_generator_set_graphs: list[nx.Graph]
    best_generator_set_scores: np.ndarray
    best_generated_from_best_set: list[nx.Graph]
    alternatives_tried: int
    history: list[dict[str, Any]]
    printed_table_header: bool
    best_so_far: float
    no_improve_counter: int
    all_generated_novel: list[nx.Graph]
    optimize_t0: float


@dataclass
class _IterationStepResult:
    """Outputs from one optimization iteration step."""

    stop_requested: bool
    stop_message: str | None
    generated_count: int
    novel_count: int
    avg_ts_sample: float
    set_reward: float
    probe_generated: int
    accepted_swaps: int
    best_iter: float
    mean_iter: float
    iteration_elapsed_seconds: float
    total_elapsed_seconds: float


class GraphOptimizer:
    """Optimize a generator set with combinatorial Thompson Sampling.

    This optimizer targets scenarios where generation quality depends on the
    *set* of graphs used to fit the generator. It repeatedly:
    1) generates candidate graphs from the current generator set,
2) builds the next generator set via combinatorial TS with redundancy
       control,
    3) evaluates set reward from probe generations,
    4) updates per-graph TS posteriors with uniform credit.

    The optimization objective is defined by `score_function`, which must
    return a scalar in `[0, 1]` where higher is better. Scores are memoized by
    graph hash, so each unique graph is scored once and reused afterward.

    Args:
        generator: Graph generator exposing `fit(...)` and `generate(...)`.
        score_function: Callable that can receive either one graph or a list
            of graphs and return score(s) in `[0, 1]`. Batched scoring is
            preferred for efficiency.
        similarity_function: Optional callable receiving two graph lists and
            returning an `(n_a, n_b)` similarity matrix in `[0, 1]`.
            Used only to compute pairwise redundancy penalties during set
            construction. If omitted, redundancy penalties are effectively
            disabled.
            Note: all fitted graphs are reconsidered each iteration, so the
            candidate pool is global (`fit_graphs + novel_generated`).
    """

    def __init__(
        self,
        generator: Any,
        score_function: Any,
        similarity_function: Any | None = None,
    ) -> None:
        self.generator = generator
        self.score_function = score_function
        self.similarity_function = similarity_function
        self.graphs: list[nx.Graph] | None = None
        self.sorted_indices: list[int] | None = None
        self.scores: np.ndarray | None = None
        self._score_cache: dict[Any, float] = {}
        self._pairwise_similarity_cache: dict[tuple[Any, Any], float] = {}
        self._similarity_seen_hashes: set[Any] = set()
        self._graph_by_hash: dict[Any, nx.Graph] = {}
        self.last_result: GraphOptimizationResult | None = None

    def fit(self, graphs: list[nx.Graph]) -> "GraphOptimizer":
        """Cache optimization context for later runs.

        Args:
            graphs: Graph collection used as source pool.

        Returns:
            Self with cached scores and sorted candidate indices.
        """

        self.graphs = list(graphs)
        self._score_cache = {}
        self._pairwise_similarity_cache = {}
        self._similarity_seen_hashes = set()
        self._graph_by_hash = {}
        self._register_graphs(self.graphs)
        self.scores = self.score_graphs(self.graphs)
        self.sorted_indices = np.argsort(-self.scores).tolist()
        return self

    def score_graphs(self, candidate_graphs: list[nx.Graph]) -> np.ndarray:
        """Score candidate graphs with the user-provided score function.

        Args:
            candidate_graphs: Graphs to score.

        Returns:
            Scores in `[0, 1]` as a 1D array.
        """

        if not candidate_graphs:
            return np.asarray([])

        out = np.empty(len(candidate_graphs), dtype=float)
        missing_idx: list[int] = []
        missing_graphs: list[nx.Graph] = []
        missing_hashes: list[Any] = []

        for i, graph in enumerate(candidate_graphs):
            graph_hash = hash_graph(graph)
            if graph_hash not in self._graph_by_hash:
                self._graph_by_hash[graph_hash] = graph
            cached = self._score_cache.get(graph_hash)
            if cached is None:
                missing_idx.append(i)
                missing_graphs.append(graph)
                missing_hashes.append(graph_hash)
            else:
                out[i] = cached

        if missing_graphs:
            batch_scores: np.ndarray | None = None
            try:
                maybe_scores = self.score_function(missing_graphs)
                maybe_arr = np.asarray(maybe_scores, dtype=float).ravel()
                if len(maybe_arr) == len(missing_graphs):
                    batch_scores = maybe_arr
            except Exception:
                batch_scores = None

            if batch_scores is None:
                batch_scores = np.asarray(
                    [float(self.score_function(graph)) for graph in missing_graphs],
                    dtype=float,
                )

            batch_scores = np.clip(batch_scores, 0.0, 1.0)
            for i, graph_hash, score in zip(missing_idx, missing_hashes, batch_scores):
                score_f = float(score)
                self._score_cache[graph_hash] = score_f
                out[i] = score_f

        return out

    @staticmethod
    def _pair_key(hash_a: Any, hash_b: Any) -> tuple[Any, Any]:
        """Return a canonical cache key for an unordered hash pair."""

        return (hash_a, hash_b) if str(hash_a) <= str(hash_b) else (hash_b, hash_a)

    def _register_graphs(self, graphs: list[nx.Graph]) -> list[Any]:
        """Register graphs in hash->graph cache and return aligned hashes.

        Args:
            graphs: Graphs to register.

        Returns:
            Hashes aligned with input order.
        """

        hashes: list[Any] = []
        for graph in graphs:
            graph_hash = hash_graph(graph)
            hashes.append(graph_hash)
            if graph_hash not in self._graph_by_hash:
                self._graph_by_hash[graph_hash] = graph
        return hashes

    def _pairwise_similarity_matrix(self, graphs: list[nx.Graph], graph_hashes: list[Any]) -> np.ndarray:
        """Build pairwise similarity matrix using cached pairs and novel updates.

        Pair similarities are memoized by hash-pair key. At each call, only
        missing pairs involving newly seen hashes are computed.
        """

        n = len(graphs)
        if n == 0 or self.similarity_function is None:
            return np.zeros((n, n), dtype=float)

        self._register_graphs(graphs)
        unique_hashes = list(dict.fromkeys(graph_hashes))
        novel_hashes = [h for h in unique_hashes if h not in self._similarity_seen_hashes]

        if novel_hashes:
            row_graphs = [self._graph_by_hash[h] for h in novel_hashes]
            col_graphs = [self._graph_by_hash[h] for h in unique_hashes]
            sim_block = np.asarray(self.similarity_function(row_graphs, col_graphs), dtype=float)
            if sim_block.shape != (len(row_graphs), len(col_graphs)):
                raise ValueError(
                    "similarity_function returned invalid shape "
                    f"{sim_block.shape}, expected {(len(row_graphs), len(col_graphs))}."
                )
            sim_block = np.clip(sim_block, 0.0, 1.0)
            for r, hash_a in enumerate(novel_hashes):
                for c, hash_b in enumerate(unique_hashes):
                    self._pairwise_similarity_cache[self._pair_key(hash_a, hash_b)] = float(sim_block[r, c])
            self._similarity_seen_hashes.update(novel_hashes)

        sim = np.eye(n, dtype=float)

        for i in range(n):
            hash_i = graph_hashes[i]
            for j in range(i + 1, n):
                hash_j = graph_hashes[j]
                key = self._pair_key(hash_i, hash_j)
                cached = self._pairwise_similarity_cache.get(key)
                if cached is None:
                    fallback = np.asarray(
                        self.similarity_function([self._graph_by_hash[hash_i]], [self._graph_by_hash[hash_j]]),
                        dtype=float,
                    )
                    cached = float(np.clip(fallback.ravel()[0], 0.0, 1.0))
                    self._pairwise_similarity_cache[key] = cached
                sim[i, j] = cached
                sim[j, i] = cached
        return sim

    def plot_score_progress(
        self,
        history: list[dict[str, Any]] | None = None,
        figsize: tuple[float, float] = (8, 5),
        show: bool = True,
    ) -> None:
        """Plot best/mean generator-set score versus alternatives tried.

        Args:
            history: Optional history list. Defaults to last run history.
            figsize: Figure size passed to Matplotlib.
            show: Whether to call `plt.show()`.

        Returns:
            None.
        """

        local_history = history
        if local_history is None:
            if self.last_result is None:
                raise RuntimeError("No optimization history available to plot.")
            local_history = self.last_result.history

        x_vals = [h["alternatives_tried"] for h in local_history]
        y_best = [h["best_score"] for h in local_history]
        y_mean = [h["mean_generator_set_score"] for h in local_history]

        plt.figure(figsize=figsize)
        plt.plot(x_vals, y_best, marker="o", label="Best generator-set score")
        plt.plot(x_vals, y_mean, marker="s", label="Mean generator-set score")
        plt.xlabel("Number of alternatives tried")
        plt.ylabel("Score [0, 1]")
        plt.title("Generator-set score progression during conditional generation")
        plt.grid(alpha=0.3)
        plt.legend()
        if show:
            plt.show()

    @staticmethod
    def _validate_optimize_params(
        generator_set_size: int,
        ts_posterior_forget: float,
        set_reward_probe_samples: int,
        set_reward_attempts_multiplier: int,
        set_reward_quantile: float,
        local_search_max_swaps: int,
        local_search_candidates: int,
        local_search_bottom_m: int,
    ) -> None:
        """Validate optimize() numeric parameters.

        Args:
            generator_set_size: Target generator-set size.
            ts_posterior_forget: Posterior decay factor in [0, 1].
            set_reward_probe_samples: Probe generation sample count.
            set_reward_attempts_multiplier: Probe attempts multiplier.
            set_reward_quantile: Quantile in `(0, 1]` used for set reward.
            local_search_max_swaps: Max accepted local swaps.
            local_search_candidates: Candidate swap-in count.
            local_search_bottom_m: Weakest members eligible for swap-out.

        Returns:
            None.
        """

        if generator_set_size <= 0:
            raise ValueError("generator_set_size must be > 0.")
        if not (0.0 <= ts_posterior_forget <= 1.0):
            raise ValueError("ts_posterior_forget must be in [0, 1].")
        if set_reward_probe_samples <= 0:
            raise ValueError("set_reward_probe_samples must be > 0.")
        if set_reward_attempts_multiplier <= 0:
            raise ValueError("set_reward_attempts_multiplier must be > 0.")
        if not (0.0 < set_reward_quantile <= 1.0):
            raise ValueError("set_reward_quantile must be in (0, 1].")
        if local_search_max_swaps < 0:
            raise ValueError("local_search_max_swaps must be >= 0.")
        if local_search_candidates < 0:
            raise ValueError("local_search_candidates must be >= 0.")
        if local_search_bottom_m <= 0:
            raise ValueError("local_search_bottom_m must be > 0.")

    def _configure_generation_runtime(
        self,
        alternatives_per_iteration: int,
        generation_n_jobs: int,
        generation_backend: str,
        verbose: bool,
    ) -> tuple[int, str]:
        """Apply generator parallel settings and return effective runtime info.

        Args:
            alternatives_per_iteration: Requested generation batch size.
            generation_n_jobs: Requested worker count.
            generation_backend: Requested backend.
            verbose: Verbosity flag used by backend resolver.

        Returns:
            Tuple of `(effective_workers, effective_backend)`.
        """

        if hasattr(self.generator, "n_jobs"):
            self.generator.n_jobs = generation_n_jobs
        if hasattr(self.generator, "parallel_backend"):
            self.generator.parallel_backend = generation_backend

        workers = generation_n_jobs
        if hasattr(self.generator, "_resolve_n_jobs_eff"):
            workers = self.generator._resolve_n_jobs_eff(alternatives_per_iteration)

        backend = generation_backend
        if hasattr(self.generator, "_resolve_parallel_backend"):
            backend = self.generator._resolve_parallel_backend(verbose=verbose)
        return workers, backend

    @staticmethod
    def _build_resume_signature(
        *,
        generator_set_size: int,
        alternatives_per_iteration: int,
        constraint_level: int,
        use_context_embedding: bool,
        attempts_multiplier: int,
        seed: int,
        ts_similarity_weight: float,
        ts_diversity_penalty: float,
        ts_prior_alpha: float,
        ts_prior_beta: float,
        ts_posterior_forget: float,
        set_reward_probe_samples: int,
        set_reward_attempts_multiplier: int,
        set_reward_quantile: float,
        local_search_max_swaps: int,
        local_search_candidates: int,
        local_search_bottom_m: int,
        local_search_min_improvement: float,
    ) -> dict[str, Any]:
        """Build parameter signature used to validate checkpoint compatibility.

        Args:
            generator_set_size: Target generator-set size.
            alternatives_per_iteration: Main generation sample count.
            constraint_level: Generator fit constraint level.
            use_context_embedding: Generator fit context-embedding flag.
            attempts_multiplier: Main generation attempts multiplier.
            seed: Base random seed.
            ts_similarity_weight: Deterministic score weight in TS utility.
            ts_diversity_penalty: Redundancy penalty in TS utility.
            ts_prior_alpha: Beta prior alpha.
            ts_prior_beta: Beta prior beta.
            ts_posterior_forget: Posterior decay factor in `[0, 1]`.
            set_reward_probe_samples: Probe generation sample count.
            set_reward_attempts_multiplier: Probe generation attempts multiplier.
            set_reward_quantile: Probe-score quantile used as set reward.
            local_search_max_swaps: Max accepted swaps in local refinement.
            local_search_candidates: Candidate swap-in count for local refinement.
            local_search_bottom_m: Weakest members considered for swap-out.
            local_search_min_improvement: Minimum reward gain for accepted swap.

        Returns:
            Dictionary containing resume-critical optimization parameters.
        """

        return {
            "generator_set_size": generator_set_size,
            "alternatives_per_iteration": alternatives_per_iteration,
            "constraint_level": constraint_level,
            "use_context_embedding": use_context_embedding,
            "attempts_multiplier": attempts_multiplier,
            "seed": seed,
            "ts_similarity_weight": ts_similarity_weight,
            "ts_diversity_penalty": ts_diversity_penalty,
            "ts_prior_alpha": ts_prior_alpha,
            "ts_prior_beta": ts_prior_beta,
            "ts_posterior_forget": ts_posterior_forget,
            "set_reward_probe_samples": set_reward_probe_samples,
            "set_reward_attempts_multiplier": set_reward_attempts_multiplier,
            "set_reward_quantile": set_reward_quantile,
            "local_search_max_swaps": local_search_max_swaps,
            "local_search_candidates": local_search_candidates,
            "local_search_bottom_m": local_search_bottom_m,
            "local_search_min_improvement": local_search_min_improvement,
        }

    @staticmethod
    def _save_checkpoint(checkpoint_path: str, payload: dict[str, Any]) -> None:
        """Persist checkpoint payload atomically to disk.

        Args:
            checkpoint_path: Destination path for checkpoint pickle.
            payload: Serializable state payload.

        Returns:
            None.
        """

        directory = os.path.dirname(os.path.abspath(checkpoint_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".graph_optimize_ckpt_",
            suffix=".tmp",
            dir=directory if directory else None,
        )
        try:
            with os.fdopen(fd, "wb") as file_obj:
                pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, checkpoint_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
        """Load checkpoint payload from disk.

        Args:
            checkpoint_path: Checkpoint pickle path.

        Returns:
            Loaded payload dictionary.
        """

        with open(checkpoint_path, "rb") as file_obj:
            payload = pickle.load(file_obj)
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint payload is invalid: expected dict.")
        return payload

    @staticmethod
    def _validate_resume_payload(
        payload: dict[str, Any],
        expected_signature: dict[str, Any],
        source_graph_hashes: list[Any],
    ) -> None:
        """Validate checkpoint compatibility with current optimization config.

        Args:
            payload: Loaded checkpoint payload.
            expected_signature: Signature built from current optimize args.
            source_graph_hashes: Hashes of current fitted source graphs.

        Returns:
            None.
        """

        version = int(payload.get("checkpoint_version", -1))
        if version != 1:
            raise ValueError(f"Unsupported checkpoint version: {version}.")

        saved_signature = payload.get("resume_signature")
        if saved_signature != expected_signature:
            raise ValueError(
                "Checkpoint parameters are incompatible with current optimize() call."
            )

        saved_source_hashes = payload.get("source_graph_hashes")
        if saved_source_hashes != source_graph_hashes:
            raise ValueError("Checkpoint source graph set does not match current fit() graphs.")

    @staticmethod
    def _filter_unseen_by_history(
        candidate_graphs: list[nx.Graph],
        generated_history_hashes: set[Any],
    ) -> list[nx.Graph]:
        """Keep only graphs not seen in generation history.

        Args:
            candidate_graphs: Candidate graphs.
            generated_history_hashes: Hashes seen in previous iterations.

        Returns:
            Filtered list with unseen graphs only.
        """

        fresh: list[nx.Graph] = []
        for graph in candidate_graphs:
            if hash_graph(graph) not in generated_history_hashes:
                fresh.append(graph)
        return fresh

    @staticmethod
    def _update_generated_history_hashes(
        candidate_graphs: list[nx.Graph],
        generated_history_hashes: set[Any],
    ) -> None:
        """Add graph hashes to generated-history set.

        Args:
            candidate_graphs: Graphs to register as seen.
            generated_history_hashes: Hash set updated in place.

        Returns:
            None.
        """

        for graph in candidate_graphs:
            generated_history_hashes.add(hash_graph(graph))

    @staticmethod
    def _get_posterior_params(
        graph_hashes: list[Any],
        ts_posterior: dict[Any, tuple[float, float]],
        ts_prior_alpha: float,
        ts_prior_beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fetch aligned Beta posterior parameters for graph hashes.

        Args:
            graph_hashes: Hashes for which to retrieve posteriors.
            ts_posterior: Hash -> (alpha, beta) map.
            ts_prior_alpha: Prior alpha fallback.
            ts_prior_beta: Prior beta fallback.

        Returns:
            Tuple `(alpha, beta)` aligned with `graph_hashes`.
        """

        alpha = np.empty(len(graph_hashes), dtype=float)
        beta = np.empty(len(graph_hashes), dtype=float)
        for i, graph_hash in enumerate(graph_hashes):
            a, b = ts_posterior.get(graph_hash, (float(ts_prior_alpha), float(ts_prior_beta)))
            alpha[i] = a
            beta[i] = b
        return alpha, beta

    @staticmethod
    def _update_posteriors(
        selected_hashes: list[Any],
        reward_norm: float,
        ts_posterior: dict[Any, tuple[float, float]],
        ts_prior_alpha: float,
        ts_prior_beta: float,
        ts_posterior_forget: float,
    ) -> None:
        """Apply uniform fractional reward updates to selected posteriors.

        Args:
            selected_hashes: Hashes of selected generator-set members.
            reward_norm: Reward in [0, 1] assigned uniformly.
            ts_posterior: Hash -> (alpha, beta) map updated in place.
            ts_prior_alpha: Prior alpha.
            ts_prior_beta: Prior beta.
            ts_posterior_forget: Forget factor in [0, 1].

        Returns:
            None.
        """

        for graph_hash in selected_hashes:
            a, b = ts_posterior.get(graph_hash, (float(ts_prior_alpha), float(ts_prior_beta)))
            if ts_posterior_forget < 1.0:
                a = float(ts_prior_alpha) + (a - float(ts_prior_alpha)) * float(ts_posterior_forget)
                b = float(ts_prior_beta) + (b - float(ts_prior_beta)) * float(ts_posterior_forget)
            a += float(reward_norm)
            b += float(1.0 - reward_norm)
            ts_posterior[graph_hash] = (a, b)

    def _evaluate_set_reward(
        self,
        generator_set: list[nx.Graph],
        iteration: int,
        *,
        use_context_embedding: bool,
        constraint_level: int,
        seed: int,
        generation_verbose: bool,
        set_reward_probe_samples: int,
        set_reward_attempts_multiplier: int,
        set_reward_quantile: float = 0.75,
    ) -> tuple[float, float, int, list[nx.Graph]]:
        """Estimate set-level reward by probe generation from a fitted set.

        Args:
            generator_set: Generator-set used for probe fit.
            iteration: Iteration index used for deterministic seeding.
            use_context_embedding: Generator fit flag.
            constraint_level: Generator fit constraint level.
            seed: Base random seed.
            generation_verbose: Generator verbosity.
            set_reward_probe_samples: Probe sample count.
            set_reward_attempts_multiplier: Probe attempts multiplier.
            set_reward_quantile: Quantile in `(0, 1]` used for set reward.

        Returns:
            Tuple `(reward, reward_norm, probe_count, probe_graphs)`.
        """

        self.generator.fit(
            generator_set,
            use_context_embedding=use_context_embedding,
            constraint_level=constraint_level,
            seed=seed + 100_000 + iteration,
            verbose=generation_verbose,
        )
        probe_generated = self.generator.generate(
            n_samples=set_reward_probe_samples,
            max_attempts=set_reward_probe_samples * set_reward_attempts_multiplier,
            verbose=generation_verbose,
        )
        if probe_generated:
            scores = np.asarray(self.score_graphs(probe_generated), dtype=float)
            reward = float(np.quantile(scores, set_reward_quantile)) if scores.size else 0.0
        else:
            reward = 0.0
        reward_norm = float(np.clip(reward, 0.0, 1.0))
        return reward, reward_norm, len(probe_generated), list(probe_generated)

    def _select_generator_set_with_ts(
        self,
        candidate_pool: list[nx.Graph],
        candidate_scores: np.ndarray,
        *,
        dedupe: Any,
        generator_set_size: int,
        ts_similarity_weight: float,
        ts_diversity_penalty: float,
        ts_posterior: dict[Any, tuple[float, float]],
        ts_prior_alpha: float,
        ts_prior_beta: float,
    ) -> tuple[list[nx.Graph], np.ndarray, float, list[Any], list[int]]:
        """Build next generator set using combinatorial Thompson Sampling.

        Args:
            candidate_pool: Candidate graphs.
            candidate_scores: Candidate scores aligned to `candidate_pool`.
            dedupe: Dedupe callable for graph lists.
            generator_set_size: Desired size of selected set.
            ts_similarity_weight: Deterministic score weight in TS utility.
            ts_diversity_penalty: Redundancy penalty weight.
            ts_posterior: Hash -> (alpha, beta) map.
            ts_prior_alpha: Prior alpha.
            ts_prior_beta: Prior beta.

        Returns:
            Tuple `(selected_graphs, selected_scores, ts_mean, selected_hashes, ranked_non_selected_idx)`.
        """

        n_candidates = len(candidate_pool)
        if n_candidates == 0:
            return [], np.asarray([]), 0.0, [], []

        candidate_hashes = [hash_graph(graph) for graph in candidate_pool]
        alpha, beta = self._get_posterior_params(
            candidate_hashes,
            ts_posterior,
            ts_prior_alpha,
            ts_prior_beta,
        )
        ts_sampled = np.random.beta(alpha, beta)
        score_term = np.clip(np.asarray(candidate_scores, dtype=float), 0.0, 1.0)
        base_utility = ts_sampled + float(ts_similarity_weight) * score_term
        pairwise_sim = self._pairwise_similarity_matrix(candidate_pool, candidate_hashes)

        selected_idx: list[int] = []
        available = set(range(n_candidates))
        target_k = min(generator_set_size, n_candidates)
        while len(selected_idx) < target_k and available:
            best_idx = -1
            best_utility = -np.inf
            for idx in available:
                if selected_idx:
                    max_redundancy = float(np.max(pairwise_sim[idx, selected_idx]))
                else:
                    max_redundancy = 0.0
                utility = float(base_utility[idx]) - float(ts_diversity_penalty) * max_redundancy
                if utility > best_utility:
                    best_utility = utility
                    best_idx = idx
            selected_idx.append(best_idx)
            available.remove(best_idx)

        selected_graphs = [candidate_pool[i] for i in selected_idx]
        selected_graphs = dedupe(selected_graphs)[:generator_set_size]
        selected_scores = self.score_graphs(selected_graphs)
        selected_hashes = [hash_graph(graph) for graph in selected_graphs]
        selected_set = set(selected_idx)
        ranked_non_selected_idx = [i for i in np.argsort(-base_utility).tolist() if i not in selected_set]
        return (
            selected_graphs,
            selected_scores,
            float(np.mean(ts_sampled)),
            selected_hashes,
            ranked_non_selected_idx,
        )

    def _refine_generator_set_with_local_swaps(
        self,
        candidate_pool: list[nx.Graph],
        selected_graphs: list[nx.Graph],
        ranked_non_selected_idx: list[int],
        iteration: int,
        *,
        dedupe: Any,
        local_search_max_swaps: int,
        local_search_candidates: int,
        local_search_bottom_m: int,
        local_search_min_improvement: float,
        use_context_embedding: bool,
        constraint_level: int,
        seed: int,
        generation_verbose: bool,
        set_reward_probe_samples: int,
        set_reward_attempts_multiplier: int,
        set_reward_quantile: float,
    ) -> tuple[list[nx.Graph], list[Any], float, float, int, list[nx.Graph], int]:
        """Apply bounded local swap refinement on selected generator set.

        Args:
            candidate_pool: Candidate graphs.
            selected_graphs: Initially selected generator set.
            ranked_non_selected_idx: Candidate indices ranked by utility.
            iteration: Current iteration index.
            dedupe: Dedupe callable for graph lists.
            local_search_max_swaps: Max accepted swap count.
            local_search_candidates: Number of candidate swap-ins to consider.
            local_search_bottom_m: Number of weakest members eligible for removal.
            local_search_min_improvement: Minimum set reward improvement.
            use_context_embedding: Generator fit flag.
            constraint_level: Generator fit constraint level.
            seed: Base random seed.
            generation_verbose: Generator verbosity.
            set_reward_probe_samples: Probe sample count.
            set_reward_attempts_multiplier: Probe attempts multiplier.
            set_reward_quantile: Quantile in `(0, 1]` used for set reward.

        Returns:
            Tuple `(graphs, hashes, reward, reward_norm, probe_count, probe_graphs, accepted_swaps)`.
        """

        current_graphs = list(selected_graphs)
        current_hashes = [hash_graph(graph) for graph in current_graphs]
        (
            current_reward,
            current_reward_norm,
            probe_generated,
            probe_generated_graphs,
        ) = self._evaluate_set_reward(
            current_graphs,
            iteration,
            use_context_embedding=use_context_embedding,
            constraint_level=constraint_level,
            seed=seed,
            generation_verbose=generation_verbose,
            set_reward_probe_samples=set_reward_probe_samples,
            set_reward_attempts_multiplier=set_reward_attempts_multiplier,
            set_reward_quantile=set_reward_quantile,
        )

        if local_search_max_swaps == 0 or local_search_candidates == 0 or not ranked_non_selected_idx:
            return (
                current_graphs,
                current_hashes,
                current_reward,
                current_reward_norm,
                probe_generated,
                probe_generated_graphs,
                0,
            )

        candidate_idx_pool = ranked_non_selected_idx[:local_search_candidates]
        accepted_swaps = 0
        while accepted_swaps < local_search_max_swaps:
            improved = False
            member_scores = self.score_graphs(current_graphs)
            removable_order = np.argsort(member_scores).tolist()[: min(local_search_bottom_m, len(current_graphs))]
            current_hash_set = set(current_hashes)

            for cand_idx in list(candidate_idx_pool):
                candidate_graph = candidate_pool[cand_idx]
                candidate_hash = hash_graph(candidate_graph)
                if candidate_hash in current_hash_set:
                    continue

                for remove_pos in removable_order:
                    trial_graphs = list(current_graphs)
                    trial_graphs[remove_pos] = candidate_graph
                    trial_graphs = dedupe(trial_graphs)
                    if len(trial_graphs) < len(current_graphs):
                        continue
                    trial_graphs = trial_graphs[: len(current_graphs)]
                    (
                        trial_reward,
                        trial_reward_norm,
                        trial_probe_generated,
                        trial_probe_generated_graphs,
                    ) = self._evaluate_set_reward(
                        trial_graphs,
                        iteration,
                        use_context_embedding=use_context_embedding,
                        constraint_level=constraint_level,
                        seed=seed,
                        generation_verbose=generation_verbose,
                        set_reward_probe_samples=set_reward_probe_samples,
                        set_reward_attempts_multiplier=set_reward_attempts_multiplier,
                        set_reward_quantile=set_reward_quantile,
                    )
                    if trial_reward > current_reward + float(local_search_min_improvement):
                        current_graphs = trial_graphs
                        current_hashes = [hash_graph(graph) for graph in current_graphs]
                        current_reward = trial_reward
                        current_reward_norm = trial_reward_norm
                        probe_generated = trial_probe_generated
                        probe_generated_graphs = trial_probe_generated_graphs
                        accepted_swaps += 1
                        improved = True
                        candidate_idx_pool = [idx for idx in candidate_idx_pool if idx != cand_idx]
                        break
                if improved:
                    break

            if not improved:
                break

        return (
            current_graphs,
            current_hashes,
            current_reward,
            current_reward_norm,
            probe_generated,
            probe_generated_graphs,
            accepted_swaps,
        )

    @staticmethod
    def _print_progress_header(generation_backend_eff: str, generation_workers: int) -> None:
        """Print optimization progress table header.

        Args:
            generation_backend_eff: Effective backend.
            generation_workers: Effective worker count.

        Returns:
            None.
        """

        print(f"generation_backend={generation_backend_eff} generation_workers={generation_workers}")
        print(
            f"{'iter':>4} {'generated':>9} {'novel':>6} {'ts_mean':>8} "
            f"{'set_R':>8} {'probe':>6} {'swaps':>5} {'iter_t':>9} {'total_t':>9} "
            f"{'hist_size':>9} {'best':>8} {'mean':>8}"
        )

    @staticmethod
    def _print_progress_row(
        iteration: int,
        generated_count: int,
        novel_count: int,
        avg_ts_sample: float,
        set_reward: float,
        probe_generated: int,
        accepted_swaps: int,
        iteration_elapsed_seconds: float,
        total_elapsed_seconds: float,
        generated_history_size: int,
        best_iter: float,
        mean_iter: float,
    ) -> None:
        """Print one optimization progress row.

        Args:
            iteration: Iteration index.
            generated_count: Number of raw generated graphs.
            novel_count: Number of novel generated graphs.
            avg_ts_sample: Mean TS draw over candidate pool.
            set_reward: Set reward value.
            probe_generated: Probe generation size.
            accepted_swaps: Accepted local swaps.
            iteration_elapsed_seconds: Iteration elapsed time.
            total_elapsed_seconds: Total elapsed time.
            generated_history_size: Count of unique generated hashes.
            best_iter: Best member score in current set.
            mean_iter: Mean member score in current set.

        Returns:
            None.
        """

        print(
            f"{iteration:4d} {generated_count:9d} {novel_count:6d} {avg_ts_sample:8.4f} "
            f"{set_reward:8.4f} {probe_generated:6d} {accepted_swaps:5d} "
            f"{_fmt_elapsed(iteration_elapsed_seconds):>9} {_fmt_elapsed(total_elapsed_seconds):>9} "
            f"{generated_history_size:9d} {best_iter:8.4f} {mean_iter:8.4f}"
        )

    def _initialize_run_state(
        self,
        graphs: list[nx.Graph],
        generator_set_size: int,
    ) -> _OptimizationRunState:
        """Create initial optimization state from top-scoring fitted graphs.

        Args:
            graphs: Fitted source graphs.
            generator_set_size: Desired generator-set size.

        Returns:
            Initialized mutable run state.
        """

        if self.sorted_indices is None:
            raise RuntimeError("GraphOptimizer.fit must be called before optimize.")

        seed_indices = self.sorted_indices[:generator_set_size]
        if not seed_indices:
            raise RuntimeError("No seed graphs available for optimization.")

        generator_set_graphs = [graphs[i] for i in seed_indices]
        generator_set_scores = self.score_graphs(generator_set_graphs)
        history: list[dict[str, Any]] = [
            {
                "iteration": 0,
                "alternatives_tried": 0,
                "best_score": float(np.max(generator_set_scores)),
                "mean_generator_set_score": float(np.mean(generator_set_scores)),
                "set_reward": np.nan,
                "local_swaps": 0,
                "generated": 0,
            }
        ]
        return _OptimizationRunState(
            generator_set_graphs=generator_set_graphs,
            generator_set_scores=generator_set_scores,
            best_generator_set_graphs=list(generator_set_graphs),
            best_generator_set_scores=np.asarray(generator_set_scores, dtype=float).copy(),
            best_generated_from_best_set=[],
            alternatives_tried=0,
            history=history,
            printed_table_header=False,
            best_so_far=float(np.max(generator_set_scores)),
            no_improve_counter=0,
            all_generated_novel=[],
            optimize_t0=time.time(),
        )

    def _run_one_iteration(
        self,
        iteration: int,
        run_state: _OptimizationRunState,
        *,
        graphs: list[nx.Graph],
        dedupe: Any,
        keep_novel: Any,
        generated_history_hashes: set[Any],
        ts_posterior: dict[Any, tuple[float, float]],
        alternatives_per_iteration: int,
        attempts_multiplier: int,
        use_context_embedding: bool,
        constraint_level: int,
        seed: int,
        generation_verbose: bool,
        generator_set_size: int,
        ts_similarity_weight: float,
        ts_diversity_penalty: float,
        ts_prior_alpha: float,
        ts_prior_beta: float,
        ts_posterior_forget: float,
        set_reward_probe_samples: int,
        set_reward_attempts_multiplier: int,
        set_reward_quantile: float,
        local_search_max_swaps: int,
        local_search_candidates: int,
        local_search_bottom_m: int,
        local_search_min_improvement: float,
    ) -> _IterationStepResult:
        """Execute one full optimization iteration and mutate run state.

        Args:
            iteration: 1-based iteration index.
            run_state: Mutable run state updated in place.
            graphs: Fitted source graph pool.
            dedupe: Callable that deduplicates graph lists.
            keep_novel: Callable that removes source-duplicate graphs.
            generated_history_hashes: Set of previously generated graph hashes.
            ts_posterior: TS posterior map updated in place.
            alternatives_per_iteration: Requested generation batch size.
            attempts_multiplier: Main generation attempts multiplier.
            use_context_embedding: Generator fit flag.
            constraint_level: Generator fit constraint level.
            seed: Base random seed.
            generation_verbose: Generator verbosity.
            generator_set_size: Desired generator-set size.
            ts_similarity_weight: Deterministic score weight in TS utility.
            ts_diversity_penalty: Redundancy penalty weight.
            ts_prior_alpha: Beta prior alpha.
            ts_prior_beta: Beta prior beta.
            ts_posterior_forget: Posterior decay factor.
            set_reward_probe_samples: Probe sample count.
            set_reward_attempts_multiplier: Probe attempts multiplier.
            set_reward_quantile: Quantile in `(0, 1]` used for set reward.
            local_search_max_swaps: Max accepted local swaps.
            local_search_candidates: Candidate swap-in count.
            local_search_bottom_m: Weakest members considered for swap-out.
            local_search_min_improvement: Minimum reward gain to accept swap.

        Returns:
            Iteration outputs and termination request (if any).
        """

        iter_t0 = time.time()
        self.generator.fit(
            run_state.generator_set_graphs,
            use_context_embedding=use_context_embedding,
            constraint_level=constraint_level,
            seed=seed + iteration,
            verbose=generation_verbose,
        )

        generated = self.generator.generate(
            n_samples=alternatives_per_iteration,
            max_attempts=alternatives_per_iteration * attempts_multiplier,
            verbose=generation_verbose,
        )

        run_state.alternatives_tried += len(generated)
        if not generated:
            return _IterationStepResult(
                stop_requested=True,
                stop_message=f"Iteration {iteration}: no alternatives generated. Stopping.",
                generated_count=0,
                novel_count=0,
                avg_ts_sample=0.0,
                set_reward=0.0,
                probe_generated=0,
                accepted_swaps=0,
                best_iter=float(np.max(run_state.generator_set_scores)),
                mean_iter=float(np.mean(run_state.generator_set_scores)),
                iteration_elapsed_seconds=float(time.time() - iter_t0),
                total_elapsed_seconds=float(time.time() - run_state.optimize_t0),
            )

        novel_generated = keep_novel(generated)
        novel_generated = dedupe(novel_generated)
        novel_generated = self._filter_unseen_by_history(
            novel_generated,
            generated_history_hashes,
        )

        if not novel_generated:
            fallback_generated = dedupe(generated)
            fallback_generated = self._filter_unseen_by_history(
                fallback_generated,
                generated_history_hashes,
            )
            novel_generated = fallback_generated

        self._update_generated_history_hashes(
            novel_generated,
            generated_history_hashes,
        )
        run_state.all_generated_novel.extend(novel_generated)

        # Reconsider all fitted graphs at every iteration so every graph can
        # be re-selected into the generator set; add novel generated graphs.
        candidate_pool = dedupe(graphs + novel_generated)
        candidate_scores = self.score_graphs(candidate_pool)
        (
            run_state.generator_set_graphs,
            run_state.generator_set_scores,
            avg_ts_sample,
            selected_hashes,
            ranked_non_selected_idx,
        ) = self._select_generator_set_with_ts(
            candidate_pool,
            candidate_scores,
            dedupe=dedupe,
            generator_set_size=generator_set_size,
            ts_similarity_weight=ts_similarity_weight,
            ts_diversity_penalty=ts_diversity_penalty,
            ts_posterior=ts_posterior,
            ts_prior_alpha=ts_prior_alpha,
            ts_prior_beta=ts_prior_beta,
        )
        (
            run_state.generator_set_graphs,
            selected_hashes,
            set_reward,
            set_reward_norm,
            probe_generated,
            probe_generated_graphs,
            accepted_swaps,
        ) = self._refine_generator_set_with_local_swaps(
            candidate_pool=candidate_pool,
            selected_graphs=run_state.generator_set_graphs,
            ranked_non_selected_idx=ranked_non_selected_idx,
            iteration=iteration,
            dedupe=dedupe,
            local_search_max_swaps=local_search_max_swaps,
            local_search_candidates=local_search_candidates,
            local_search_bottom_m=local_search_bottom_m,
            local_search_min_improvement=local_search_min_improvement,
            use_context_embedding=use_context_embedding,
            constraint_level=constraint_level,
            seed=seed,
            generation_verbose=generation_verbose,
            set_reward_probe_samples=set_reward_probe_samples,
            set_reward_attempts_multiplier=set_reward_attempts_multiplier,
            set_reward_quantile=set_reward_quantile,
        )
        run_state.generator_set_scores = self.score_graphs(run_state.generator_set_graphs)
        self._update_posteriors(
            selected_hashes,
            set_reward_norm,
            ts_posterior,
            ts_prior_alpha,
            ts_prior_beta,
            ts_posterior_forget,
        )

        best_iter = float(np.max(run_state.generator_set_scores))
        mean_iter = float(np.mean(run_state.generator_set_scores))
        if best_iter > float(np.max(run_state.best_generator_set_scores)):
            run_state.best_generator_set_graphs = list(run_state.generator_set_graphs)
            run_state.best_generator_set_scores = np.asarray(run_state.generator_set_scores, dtype=float).copy()
            run_state.best_generated_from_best_set = list(probe_generated_graphs)
        iteration_elapsed_seconds = float(time.time() - iter_t0)
        total_elapsed_seconds = float(time.time() - run_state.optimize_t0)
        run_state.history.append(
            {
                "iteration": iteration,
                "alternatives_tried": run_state.alternatives_tried,
                "best_score": best_iter,
                "mean_generator_set_score": mean_iter,
                "set_reward": set_reward,
                "local_swaps": accepted_swaps,
                "generated": len(generated),
                "iteration_elapsed_seconds": iteration_elapsed_seconds,
                "total_elapsed_seconds": total_elapsed_seconds,
            }
        )
        return _IterationStepResult(
            stop_requested=False,
            stop_message=None,
            generated_count=len(generated),
            novel_count=len(novel_generated),
            avg_ts_sample=avg_ts_sample,
            set_reward=set_reward,
            probe_generated=probe_generated,
            accepted_swaps=accepted_swaps,
            best_iter=best_iter,
            mean_iter=mean_iter,
            iteration_elapsed_seconds=iteration_elapsed_seconds,
            total_elapsed_seconds=total_elapsed_seconds,
        )

    def optimize(
        self,
        *,
        generator_set_size: int = 50,
        max_iterations: int = 50,
        alternatives_per_iteration: int = 80,
        score_early_stop_threshold: float = 0.99,
        patience_no_improve: int = 5,
        constraint_level: int = 1,
        use_context_embedding: bool = False,
        attempts_multiplier: int = 8,
        seed: int = 7,
        ts_similarity_weight: float = 0.5,
        ts_diversity_penalty: float = 0.2,
        ts_prior_alpha: float = 1.0,
        ts_prior_beta: float = 1.0,
        ts_posterior_forget: float = 1.0,
        set_reward_probe_samples: int = 40,
        set_reward_attempts_multiplier: int = 4,
        set_reward_quantile: float = 0.75,
        local_search_max_swaps: int = 1,
        local_search_candidates: int = 10,
        local_search_bottom_m: int = 3,
        local_search_min_improvement: float = 1e-4,
        generation_n_jobs: int = -1,
        generation_backend: str = "loky",
        generation_verbose: bool = False,
        checkpoint_path: str | None = None,
        checkpoint_every: int = 1,
        resume_from: str | None = None,
        verbose: bool = True,
    ) -> GraphOptimizationResult:
        """Run iterative optimization with combinatorial Thompson Sampling.

        Args:
            generator_set_size: Number of generator-set graphs kept per iteration.
            max_iterations: Maximum optimization iterations.
            alternatives_per_iteration: Number of generated graphs requested per step.
            score_early_stop_threshold: Early-stop threshold for best generator-set score.
            patience_no_improve: Stop after this many non-improving iterations.
            constraint_level: Generator constraint level.
            use_context_embedding: Whether to use context embeddings in generator fit.
            attempts_multiplier: Multiplier for generation max attempts.
            seed: Base random seed for generator iterations.
            ts_similarity_weight: Weight of known score in TS utility.
            ts_diversity_penalty: Penalty on redundancy during generator-set construction.
            ts_prior_alpha: Beta prior alpha.
            ts_prior_beta: Beta prior beta.
            ts_posterior_forget: Posterior decay factor in `[0, 1]`.
            set_reward_probe_samples: Probe generation size used to estimate set reward.
            set_reward_attempts_multiplier: Attempts multiplier for probe generation.
            set_reward_quantile: Quantile in `(0, 1]` used for set reward.
            local_search_max_swaps: Max accepted swap refinements per iteration.
            local_search_candidates: Number of non-selected candidates tested for swaps.
            local_search_bottom_m: Number of weakest generator-set members considered as swap-out slots.
            local_search_min_improvement: Minimum `R(S)` gain required to accept a swap.
            generation_n_jobs: Generator parallel jobs.
            generation_backend: Generator parallel backend.
            generation_verbose: Generator verbosity.
            checkpoint_path: Optional destination path for periodic checkpoint snapshots.
            checkpoint_every: Save checkpoint every N completed iterations.
            resume_from: Optional checkpoint path to resume from.
            verbose: Whether to print progress and show plot.

        Returns:
            GraphOptimizationResult with generator-set graphs, scores, history, and novel outputs.
        """

        if self.graphs is None or self.sorted_indices is None:
            raise RuntimeError("GraphOptimizer.fit must be called before optimize.")
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be > 0.")
        self._validate_optimize_params(
            generator_set_size=generator_set_size,
            ts_posterior_forget=ts_posterior_forget,
            set_reward_probe_samples=set_reward_probe_samples,
            set_reward_attempts_multiplier=set_reward_attempts_multiplier,
            set_reward_quantile=set_reward_quantile,
            local_search_max_swaps=local_search_max_swaps,
            local_search_candidates=local_search_candidates,
            local_search_bottom_m=local_search_bottom_m,
        )

        graphs = self.graphs
        deduper = GraphHashDeduper(parallel=True)
        source_deduper = GraphHashDeduper(parallel=True).fit(graphs)
        dedupe = deduper.fit_filter
        keep_novel = source_deduper.filter
        source_graph_hashes = [hash_graph(graph) for graph in graphs]
        resume_signature = self._build_resume_signature(
            generator_set_size=generator_set_size,
            alternatives_per_iteration=alternatives_per_iteration,
            constraint_level=constraint_level,
            use_context_embedding=use_context_embedding,
            attempts_multiplier=attempts_multiplier,
            seed=seed,
            ts_similarity_weight=ts_similarity_weight,
            ts_diversity_penalty=ts_diversity_penalty,
            ts_prior_alpha=ts_prior_alpha,
            ts_prior_beta=ts_prior_beta,
            ts_posterior_forget=ts_posterior_forget,
            set_reward_probe_samples=set_reward_probe_samples,
            set_reward_attempts_multiplier=set_reward_attempts_multiplier,
            set_reward_quantile=set_reward_quantile,
            local_search_max_swaps=local_search_max_swaps,
            local_search_candidates=local_search_candidates,
            local_search_bottom_m=local_search_bottom_m,
            local_search_min_improvement=local_search_min_improvement,
        )
        generated_history_hashes: set[Any]
        ts_posterior: dict[Any, tuple[float, float]]
        run_state: _OptimizationRunState
        start_iteration = 1
        (
            generation_workers,
            generation_backend_eff,
        ) = self._configure_generation_runtime(
            alternatives_per_iteration=alternatives_per_iteration,
            generation_n_jobs=generation_n_jobs,
            generation_backend=generation_backend,
            verbose=verbose,
        )
        if resume_from is not None:
            payload = self._load_checkpoint(resume_from)
            self._validate_resume_payload(
                payload=payload,
                expected_signature=resume_signature,
                source_graph_hashes=source_graph_hashes,
            )
            run_state = payload["run_state"]
            generated_history_hashes = set(payload["generated_history_hashes"])
            ts_posterior = dict(payload["ts_posterior"])
            start_iteration = int(payload["completed_iteration"]) + 1
            np.random.set_state(payload["numpy_random_state"])
            random.setstate(payload["python_random_state"])
            if verbose:
                print(f"Resumed optimization from checkpoint: {resume_from}")
                print(f"Continuing from iteration {start_iteration}.")
        else:
            generated_history_hashes = set()
            ts_posterior = {}
            run_state = self._initialize_run_state(
                graphs=graphs,
                generator_set_size=generator_set_size,
            )

        checkpoint_target = checkpoint_path if checkpoint_path is not None else resume_from
        for iteration in range(start_iteration, max_iterations + 1):
            step = self._run_one_iteration(
                iteration=iteration,
                run_state=run_state,
                graphs=graphs,
                dedupe=dedupe,
                keep_novel=keep_novel,
                generated_history_hashes=generated_history_hashes,
                ts_posterior=ts_posterior,
                alternatives_per_iteration=alternatives_per_iteration,
                attempts_multiplier=attempts_multiplier,
                use_context_embedding=use_context_embedding,
                constraint_level=constraint_level,
                seed=seed,
                generation_verbose=generation_verbose,
                generator_set_size=generator_set_size,
                ts_similarity_weight=ts_similarity_weight,
                ts_diversity_penalty=ts_diversity_penalty,
                ts_prior_alpha=ts_prior_alpha,
                ts_prior_beta=ts_prior_beta,
                ts_posterior_forget=ts_posterior_forget,
                set_reward_probe_samples=set_reward_probe_samples,
                set_reward_attempts_multiplier=set_reward_attempts_multiplier,
                set_reward_quantile=set_reward_quantile,
                local_search_max_swaps=local_search_max_swaps,
                local_search_candidates=local_search_candidates,
                local_search_bottom_m=local_search_bottom_m,
                local_search_min_improvement=local_search_min_improvement,
            )
            if step.stop_requested:
                if verbose and step.stop_message:
                    print(step.stop_message)
                break

            if verbose:
                if not run_state.printed_table_header:
                    self._print_progress_header(
                        generation_backend_eff=generation_backend_eff,
                        generation_workers=generation_workers,
                    )
                    run_state.printed_table_header = True
                self._print_progress_row(
                    iteration=iteration,
                    generated_count=step.generated_count,
                    novel_count=step.novel_count,
                    avg_ts_sample=step.avg_ts_sample,
                    set_reward=step.set_reward,
                    probe_generated=step.probe_generated,
                    accepted_swaps=step.accepted_swaps,
                    iteration_elapsed_seconds=step.iteration_elapsed_seconds,
                    total_elapsed_seconds=step.total_elapsed_seconds,
                    generated_history_size=len(generated_history_hashes),
                    best_iter=step.best_iter,
                    mean_iter=step.mean_iter,
                )
            if checkpoint_target is not None and iteration % checkpoint_every == 0:
                self._save_checkpoint(
                    checkpoint_target,
                    {
                        "checkpoint_version": 1,
                        "completed_iteration": int(run_state.history[-1]["iteration"]),
                        "run_state": run_state,
                        "generated_history_hashes": generated_history_hashes,
                        "ts_posterior": ts_posterior,
                        "numpy_random_state": np.random.get_state(),
                        "python_random_state": random.getstate(),
                        "resume_signature": resume_signature,
                        "source_graph_hashes": source_graph_hashes,
                        "saved_at_epoch_seconds": float(time.time()),
                    },
                )

            if step.best_iter >= score_early_stop_threshold:
                if verbose:
                    print(
                        f"Reached score early-stop threshold ({score_early_stop_threshold}). "
                        "Stopping."
                    )
                break

            if step.best_iter <= run_state.best_so_far + 1e-9:
                run_state.no_improve_counter += 1
            else:
                run_state.no_improve_counter = 0
                run_state.best_so_far = step.best_iter

            if run_state.no_improve_counter >= patience_no_improve:
                if verbose:
                    print(f"No improvement for {patience_no_improve} iterations. Stopping.")
                break

        if checkpoint_target is not None:
            self._save_checkpoint(
                checkpoint_target,
                {
                    "checkpoint_version": 1,
                    "completed_iteration": int(run_state.history[-1]["iteration"]),
                    "run_state": run_state,
                    "generated_history_hashes": generated_history_hashes,
                    "ts_posterior": ts_posterior,
                    "numpy_random_state": np.random.get_state(),
                    "python_random_state": random.getstate(),
                    "resume_signature": resume_signature,
                    "source_graph_hashes": source_graph_hashes,
                    "saved_at_epoch_seconds": float(time.time()),
                },
            )

        if verbose:
            print(f"Final generator-set size: {len(run_state.generator_set_graphs)}")
            print(f"Total alternatives tried: {run_state.alternatives_tried}")
            print(f"Best generator-set score: {np.max(run_state.generator_set_scores):.4f}")
            print(f"Generated-history unique count: {len(generated_history_hashes)}")

        result = GraphOptimizationResult(
            generator_set_graphs=run_state.generator_set_graphs,
            generator_set_scores=run_state.generator_set_scores,
            best_generator_set_graphs=run_state.best_generator_set_graphs,
            best_generator_set_scores=run_state.best_generator_set_scores,
            best_generated_from_best_set=run_state.best_generated_from_best_set,
            history=run_state.history,
            all_generated_novel=run_state.all_generated_novel,
        )
        self.last_result = result
        return result
