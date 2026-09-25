"""Edge action ranker fitting and persistence helpers."""

from __future__ import annotations

import copy
import math
import os
import pickle
import random
import re
import tempfile
import time
from itertools import combinations, permutations

import networkx as nx
import numpy as np

from abstractgraph.graphs import is_simple_graph
from abstractgraph_generative._edge_utils import (
    _canonicalize_edge,
    _decomposition_edge_groups,
)

EDGE_RANKER_ARTIFACT_VERSION = 1
_EDGE_RANKER_FEATURE_VERSION = 1


def _sanitize_edge_ranker_name(name: str) -> str:
    """Return a safe, deterministic filename component for a ranker name."""
    if not isinstance(name, str):
        raise TypeError("edge ranker name must be a string")
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError("edge ranker name must contain at least one safe character")
    return sanitized


def _edge_ranker_path(name: str, directory="edge_rankers"):
    safe_name = _sanitize_edge_ranker_name(name)
    return os.path.join(os.fspath(directory), f"edge_ranker__{safe_name}.pkl")


def _stable_edge_ranker_value(value):
    """Make common NetworkX attribute values safe for a feature signature."""
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _edge_ranker_attrs_signature(attrs) -> tuple:
    return tuple(
        sorted(
            (
                (
                    _stable_edge_ranker_value(key),
                    _stable_edge_ranker_value(value),
                )
                for key, value in dict(attrs or {}).items()
            ),
            key=lambda item: (
                repr(type(item[0])),
                repr(item[0]),
                repr(type(item[1])),
                repr(item[1]),
            ),
        )
    )


def _edge_ranker_action_signature(
    partial_graph: nx.Graph, edge, edge_attrs=None
) -> tuple:
    """Describe an edge action without encoding arbitrary node identifiers."""
    u, v = edge
    u_attrs = dict(partial_graph.nodes[u]) if u in partial_graph else {}
    v_attrs = dict(partial_graph.nodes[v]) if v in partial_graph else {}
    u_degree = int(partial_graph.degree(u)) if u in partial_graph else 0
    v_degree = int(partial_graph.degree(v)) if v in partial_graph else 0
    if nx.is_directed(partial_graph):
        u_degree = (int(partial_graph.in_degree(u)), int(partial_graph.out_degree(u)))
        v_degree = (int(partial_graph.in_degree(v)), int(partial_graph.out_degree(v)))
        common_neighbors = (
            len(set(partial_graph.successors(u)) & set(partial_graph.predecessors(v)))
            if u in partial_graph and v in partial_graph
            else 0
        )
    else:
        common_neighbors = (
            len(set(partial_graph.neighbors(u)) & set(partial_graph.neighbors(v)))
            if u in partial_graph and v in partial_graph
            else 0
        )
    return (
        bool(nx.is_directed(partial_graph)),
        _stable_edge_ranker_value(u_attrs.get("label")),
        _stable_edge_ranker_value(v_attrs.get("label")),
        u_degree,
        v_degree,
        int(common_neighbors),
        int(partial_graph.number_of_nodes()),
        int(partial_graph.number_of_edges()),
        _edge_ranker_attrs_signature(edge_attrs),
    )


def _edge_ranker_action_from_input(value):
    if isinstance(value, dict):
        partial_graph = value.get("partial_graph")
        if partial_graph is None:
            parent = value.get("parent")
            partial_graph = parent.get("graph") if parent is not None else None
        edge = value.get("edge", value.get("added_edge"))
        if edge is None:
            raise ValueError("edge ranker inputs must include an edge action")
        edge_attrs = value.get("edge_attrs")
        if edge_attrs is None and value.get("graph") is not None:
            graph = value["graph"]
            if graph.has_edge(*edge):
                edge_attrs = dict(graph.edges[edge])
        if partial_graph is None:
            raise ValueError("edge ranker inputs must include a partial graph")
        return partial_graph, edge, edge_attrs
    if isinstance(value, tuple) and len(value) == 3:
        return value
    raise TypeError(
        "edge ranker inputs must be candidate dictionaries or "
        "(partial_graph, edge, edge_attrs) tuples"
    )


class _EmpiricalEdgeRanker:
    """Small fitted action ranker used when no external estimator is supplied."""

    feature_version = _EDGE_RANKER_FEATURE_VERSION

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = float(smoothing)
        self.positive_counts_ = {}
        self.negative_counts_ = {}
        self.n_positive_ = 0
        self.n_negative_ = 0
        self.is_fitted_ = False

    def fit(self, examples, targets=None):
        examples = list(examples)
        if targets is None:
            targets = [example["label"] for example in examples]
        targets = [int(target) for target in targets]
        if len(examples) != len(targets):
            raise ValueError(
                "edge ranker examples and targets must have the same length"
            )
        if not examples:
            raise ValueError("edge ranker training requires at least one example")
        if any(target not in {0, 1} for target in targets):
            raise ValueError("edge ranker targets must be binary 0/1 labels")
        self.positive_counts_.clear()
        self.negative_counts_.clear()
        self.n_positive_ = 0
        self.n_negative_ = 0
        for example, target in zip(examples, targets):
            signature = _edge_ranker_action_signature(
                *_edge_ranker_action_from_input(example)
            )
            counts = self.positive_counts_ if target == 1 else self.negative_counts_
            counts[signature] = counts.get(signature, 0) + 1
            if target == 1:
                self.n_positive_ += 1
            else:
                self.n_negative_ += 1
        if self.n_positive_ == 0 or self.n_negative_ == 0:
            raise ValueError(
                "edge ranker training requires both positive and negative examples"
            )
        self.is_fitted_ = True
        return self

    def predict(self, examples):
        if not self.is_fitted_:
            raise ValueError("edge ranker is not fitted")
        examples = list(examples)
        smoothing = max(0.0, self.smoothing)
        signatures = [
            _edge_ranker_action_signature(*_edge_ranker_action_from_input(example))
            for example in examples
        ]
        n_signatures = max(
            1, len(set(self.positive_counts_) | set(self.negative_counts_))
        )
        positive_denominator = self.n_positive_ + smoothing * n_signatures
        negative_denominator = self.n_negative_ + smoothing * n_signatures
        global_score = math.log(
            (self.n_positive_ + smoothing) / max(positive_denominator, 1e-12)
        ) - math.log((self.n_negative_ + smoothing) / max(negative_denominator, 1e-12))
        scores = []
        for signature in signatures:
            positive = self.positive_counts_.get(signature, 0)
            negative = self.negative_counts_.get(signature, 0)
            if positive == 0 and negative == 0:
                scores.append(global_score)
                continue
            scores.append(
                math.log((positive + smoothing) / max(positive_denominator, 1e-12))
                - math.log((negative + smoothing) / max(negative_denominator, 1e-12))
            )
        return np.asarray(scores, dtype=float)


def _edge_ranker_missing_edges(graph: nx.Graph, *, allow_self_loops: bool):
    nodes = list(graph.nodes())
    if nx.is_directed(graph):
        candidate_edges = list(permutations(nodes, 2))
    else:
        candidate_edges = list(combinations(nodes, 2))
    if allow_self_loops:
        candidate_edges.extend((node, node) for node in nodes)
    occupied = {_canonicalize_edge(edge, graph) for edge in graph.edges()}
    return [
        edge
        for edge in candidate_edges
        if _canonicalize_edge(edge, graph) not in occupied
    ]


def _build_edge_ranker_examples(
    graphs,
    *,
    n_negative_per_positive: int,
    n_replicates: int,
    seed: int | None,
    allow_self_loops: bool,
    decomposition_function=None,
    nbits: int | None = None,
):
    if n_negative_per_positive < 1:
        raise ValueError("n_negative_per_positive must be >= 1")
    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")
    rng = random.Random(seed)
    examples = []
    for graph in graphs:
        if decomposition_function is None:
            edge_groups = None
        else:
            if nbits is None:
                raise ValueError(
                    "nbits is required when decomposition_function is provided"
                )
            edge_groups = _decomposition_edge_groups(
                graph,
                decomposition_function=decomposition_function,
                nbits=int(nbits),
            )
        for _ in range(n_replicates):
            current_graph = graph.copy()
            groups = (
                None if edge_groups is None else [list(group) for group in edge_groups]
            )
            if groups is not None:
                rng.shuffle(groups)
                ordered_edges = [edge for group in groups for edge in group]
            else:
                ordered_edges = None
            while current_graph.number_of_edges() > 0:
                if ordered_edges is None:
                    edge = rng.choice(list(current_graph.edges()))
                else:
                    available = [
                        candidate
                        for candidate in ordered_edges
                        if _canonicalize_edge(candidate, current_graph)
                        in {
                            _canonicalize_edge(existing, current_graph)
                            for existing in current_graph.edges()
                        }
                    ]
                    if not available:
                        break
                    edge = rng.choice(available)
                    ordered_edges.remove(edge)
                edge = _canonicalize_edge(edge, current_graph)
                edge_attrs = dict(current_graph.edges[edge])
                partial_graph = current_graph.copy()
                partial_graph.remove_edge(*edge)
                examples.append(
                    {
                        "partial_graph": partial_graph,
                        "edge": edge,
                        "edge_attrs": edge_attrs,
                        "label": 1,
                    }
                )
                negative_edges = [
                    candidate
                    for candidate in _edge_ranker_missing_edges(
                        partial_graph, allow_self_loops=allow_self_loops
                    )
                    if _canonicalize_edge(candidate, partial_graph) != edge
                ]
                rng.shuffle(negative_edges)
                for negative_edge in negative_edges[:n_negative_per_positive]:
                    examples.append(
                        {
                            "partial_graph": partial_graph.copy(),
                            "edge": _canonicalize_edge(negative_edge, partial_graph),
                            "edge_attrs": dict(edge_attrs),
                            "label": 0,
                        }
                    )
                current_graph = partial_graph
    return examples


def fit_edge_ranker(
    graphs,
    *,
    dataset_name: str,
    output_dir="edge_rankers",
    ranker=None,
    estimator=None,
    n_negative_per_positive: int = 3,
    n_replicates: int = 1,
    seed: int | None = None,
    allow_self_loops: bool = False,
    decomposition_function=None,
    nbits: int | None = None,
):
    """Fit and persist a reusable domain prior over edge-addition actions.

    The fitted ranker's prediction direction is ``larger = more preferred``.
    A custom ``ranker``/``estimator`` must implement ``fit(examples, targets)``
    and ``predict(candidates)`` using the action dictionaries documented by the
    default ranker.
    """
    if ranker is not None and estimator is not None:
        raise ValueError("provide at most one of ranker and estimator")
    safe_name = _sanitize_edge_ranker_name(dataset_name)
    graph_list = (
        [graphs.copy()]
        if is_simple_graph(graphs)
        else [graph.copy() for graph in graphs]
    )
    if not graph_list:
        raise ValueError("graphs must contain at least one graph")
    examples = _build_edge_ranker_examples(
        graph_list,
        n_negative_per_positive=n_negative_per_positive,
        n_replicates=n_replicates,
        seed=seed,
        allow_self_loops=allow_self_loops,
        decomposition_function=decomposition_function,
        nbits=nbits,
    )
    if not examples:
        raise ValueError("graphs did not produce any edge-ranker training examples")
    fitted_ranker = ranker if ranker is not None else estimator
    if fitted_ranker is None:
        fitted_ranker = _EmpiricalEdgeRanker()
    targets = [example["label"] for example in examples]
    if not hasattr(fitted_ranker, "fit") or not hasattr(fitted_ranker, "predict"):
        raise TypeError(
            "ranker must provide fit(examples, targets) and predict(candidates)"
        )
    fit_result = fitted_ranker.fit(examples, targets)
    if fit_result is not None:
        fitted_ranker = fit_result
    if not hasattr(fitted_ranker, "predict"):
        raise TypeError("fitted edge ranker must provide predict(candidates)")
    try:
        fitted_ranker.dataset_name = dataset_name
        fitted_ranker.artifact_version = EDGE_RANKER_ARTIFACT_VERSION
    except Exception:
        pass
    artifact = {
        "dataset_name": dataset_name,
        "artifact_version": EDGE_RANKER_ARTIFACT_VERSION,
        "ranker": fitted_ranker,
        "feature_version": _EDGE_RANKER_FEATURE_VERSION,
        "n_training_examples": len(examples),
    }
    output_path = os.path.join(os.fspath(output_dir), f"edge_ranker__{safe_name}.pkl")
    os.makedirs(os.fspath(output_dir), exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=os.fspath(output_dir), prefix=".edge_ranker__", delete=False
        ) as handle:
            temporary_path = handle.name
            pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return fitted_ranker


def load_edge_ranker(name: str, *, directory="edge_rankers"):
    """Load and validate ``edge_ranker__{name}.pkl`` from ``directory``."""
    path = _edge_ranker_path(name, directory)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No persisted edge ranker named {name!r} found at {path!r}"
        )
    with open(path, "rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict):
        raise ValueError(
            f"Invalid edge ranker artifact at {path!r}: expected a dictionary"
        )
    if artifact.get("artifact_version") != EDGE_RANKER_ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported edge ranker artifact version {artifact.get('artifact_version')!r}; "
            f"expected {EDGE_RANKER_ARTIFACT_VERSION}"
        )
    if artifact.get("feature_version") != _EDGE_RANKER_FEATURE_VERSION:
        raise ValueError(
            f"Unsupported edge ranker feature version {artifact.get('feature_version')!r}; "
            f"expected {_EDGE_RANKER_FEATURE_VERSION}"
        )
    stored_name = artifact.get("dataset_name")
    if not isinstance(stored_name, str):
        raise ValueError(
            f"Invalid edge ranker artifact at {path!r}: missing dataset_name"
        )
    if _sanitize_edge_ranker_name(stored_name) != _sanitize_edge_ranker_name(name):
        raise ValueError(
            f"Edge ranker artifact at {path!r} was trained for dataset "
            f"{stored_name!r}, not {name!r}"
        )
    ranker = artifact.get("ranker")
    if ranker is None or not hasattr(ranker, "predict"):
        raise ValueError(
            f"Invalid edge ranker artifact at {path!r}: missing usable ranker"
        )
    try:
        ranker.dataset_name = artifact.get("dataset_name")
        ranker.artifact_version = artifact.get("artifact_version")
    except Exception:
        pass
    return ranker


class _OnlineGraphRegressorAdapter:
    """Online adapter for graph regressors with optional replay-backed fitting."""

    def __init__(self, estimator) -> None:
        self.estimator = estimator
        self.estimator_ = copy.deepcopy(estimator)
        self.replay_graphs_ = []
        self.replay_targets_ = []
        self.n_training_examples_ = 0
        self.is_fitted_ = False
        self.supports_partial_fit_ = hasattr(self.estimator_, "partial_fit")
        self.last_fit_time_seconds_ = 0.0

    def partial_fit(self, graphs, targets):
        graph_list = [graph.copy() for graph in graphs]
        target_array = np.asarray(targets, dtype=float).reshape(-1)
        if len(graph_list) != len(target_array):
            raise ValueError("graphs and targets must have the same length")
        if not graph_list:
            return self

        self.n_training_examples_ += len(graph_list)
        fit_start = time.perf_counter()
        if self.supports_partial_fit_:
            self.estimator_.partial_fit(graph_list, target_array)
        else:
            self.replay_graphs_.extend(graph_list)
            self.replay_targets_.extend(target_array.tolist())
            self.estimator_ = copy.deepcopy(self.estimator)
            self.estimator_.fit(self.replay_graphs_, self.replay_targets_)
        self.last_fit_time_seconds_ = time.perf_counter() - fit_start
        self.is_fitted_ = True
        return self

    def predict(self, graphs):
        if not self.is_fitted_:
            return np.zeros(len(graphs), dtype=float)
        predictions = self.estimator_.predict(graphs)
        return np.asarray(predictions, dtype=float).reshape(-1)

    def training_set_size(self) -> int:
        return int(self.n_training_examples_)

    def last_fit_time_seconds(self) -> float:
        return float(self.last_fit_time_seconds_)
