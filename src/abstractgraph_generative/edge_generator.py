from __future__ import annotations

import copy
import math
import random
import time
from itertools import combinations, permutations
from typing import Callable

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from abstractgraph.graphs import graph_to_abstract_graph, is_simple_graph
from abstractgraph.hashing import canonical_bytes, hash_graph
from sklearn.metrics import pairwise_distances
from abstractgraph_generative.interpolate import _build_adjacency


DrawGraphsFn = Callable[..., object]


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


def mix_connected_components(
    graph1: nx.Graph,
    graph2: nx.Graph,
    *,
    target_n_nodes: int | None = None,
    n_trials: int = 128,
    seed: int | None = None,
):
    """Build a graph by mixing connected components from two input graphs.

    Parameters
    ----------
    graph1 : nx.Graph
        First source graph providing candidate connected components.
    graph2 : nx.Graph
        Second source graph providing candidate connected components.
    target_n_nodes : int | None, optional
        Desired node count for the merged graph. If omitted, the midpoint
        between the two source graph sizes is used.
    n_trials : int, optional
        Number of random component-sampling attempts for each component count.
        Larger values improve the chance of matching ``target_n_nodes``.
    seed : int | None, optional
        Random seed controlling component sampling.

    Returns
    -------
    nx.Graph
        A new graph composed of relabeled connected components sampled from
        both inputs.
    """
    if graph1.number_of_nodes() < 1 or graph2.number_of_nodes() < 1:
        raise ValueError("Both input graphs must contain at least one node")
    if nx.is_directed(graph1) != nx.is_directed(graph2):
        raise ValueError("Both input graphs must have the same directedness")
    if graph1.is_multigraph() != graph2.is_multigraph():
        raise ValueError("Both input graphs must both be simple or both multigraphs")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    rng = random.Random(seed)
    component_graphs1 = _connected_component_subgraphs(graph1)
    component_graphs2 = _connected_component_subgraphs(graph2)

    max_pairs = min(len(component_graphs1), len(component_graphs2))
    if max_pairs < 1:
        raise ValueError("Both graphs must expose at least one connected component")

    if target_n_nodes is None:
        target_n_nodes = int(round((graph1.number_of_nodes() + graph2.number_of_nodes()) / 2))
    target_n_nodes = max(1, int(target_n_nodes))

    best_choice = None
    best_score = None

    for n_components_per_graph in range(1, max_pairs + 1):
        trial_count = 1 if (
            n_components_per_graph == len(component_graphs1)
            and n_components_per_graph == len(component_graphs2)
        ) else n_trials
        for _ in range(trial_count):
            selected1 = _sample_components(component_graphs1, n_components_per_graph, rng)
            selected2 = _sample_components(component_graphs2, n_components_per_graph, rng)
            total_nodes = sum(g.number_of_nodes() for g in selected1) + sum(
                g.number_of_nodes() for g in selected2
            )
            score = abs(total_nodes - target_n_nodes)
            tie_break = total_nodes
            if best_score is None or (score, tie_break) < best_score:
                best_score = (score, tie_break)
                best_choice = (selected1, selected2)
                if score == 0:
                    break
        if best_score is not None and best_score[0] == 0:
            break

    if best_choice is None:
        raise ValueError("Could not select connected components from the input graphs")

    selected1, selected2 = best_choice
    return _merge_component_graphs(selected1 + selected2, graph1)


def edge_neighbors(
    G: nx.Graph,
    *,
    n_samples: int = 1,
    seed: int | None = None,
    allow_self_loops: bool = False,
):
    """Generate neighboring graphs by moving one edge to a new location.

    Parameters
    ----------
    G : nx.Graph
        Input simple graph from which neighbors are generated.
    n_samples : int, optional
        Number of independently sampled neighboring graphs to return.
    seed : int | None, optional
        Random seed controlling edge removal and insertion choices.
    allow_self_loops : bool, optional
        Whether candidate destination edges may include ``(node, node)``.

    Returns
    -------
    list[nx.Graph]
        Neighbor graphs obtained by removing one existing edge and adding one
        previously absent edge.
    """
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("random_edge_move_copies supports only simple NetworkX graphs")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if G.number_of_nodes() < 2:
        raise ValueError("Input graph must have at least 2 nodes")
    if G.number_of_edges() < 1:
        raise ValueError("Input graph must have at least 1 edge")

    rng = random.Random(seed)
    nodes = list(G.nodes())
    edges = [tuple(edge) for edge in G.edges()]
    occupied = set(edges)

    if nx.is_directed(G):
        candidate_edges = list(permutations(nodes, 2))
    else:
        candidate_edges = list(combinations(nodes, 2))
    if allow_self_loops:
        candidate_edges += [(node, node) for node in nodes]

    sampled_graphs = []
    for _ in range(n_samples):
        old_edge = rng.choice(edges)
        new_edge_options = [
            edge for edge in candidate_edges if edge != old_edge and edge not in occupied
        ]
        if not new_edge_options:
            raise ValueError("No valid destination edge is available for moving an edge")

        new_edge = rng.choice(new_edge_options)
        H = G.copy()
        H.remove_edge(*old_edge)
        H.add_edge(*new_edge, **dict(G.edges[old_edge]))
        sampled_graphs.append(H)

    return sampled_graphs


def remove_edges(
    G: nx.Graph,
    size=0.1,
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
):
    """Remove a subset of edges from a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph to prune.
    size : float | int, optional
        Number of edges to remove. Values in ``[0, 1)`` are interpreted as a
        fraction of the current edge count; larger values are interpreted as an
        absolute edge count.
    seed : int | None, optional
        Random seed used when ``rng`` is not provided.
    rng : random.Random | None, optional
        Explicit random number generator used for edge sampling. This takes
        precedence over ``seed`` and is useful when a caller wants reproducible
        multi-step workflows.

    Returns
    -------
    tuple[nx.Graph, int]
        The pruned graph and the original edge count before removal.
    """
    H = G.copy()
    n_edges = H.number_of_edges()
    edges = list(H.edges())

    if size < 0:
        raise ValueError("size must be >= 0")
    if size < 1:
        n_remove = int(round(size * n_edges))
    else:
        n_remove = int(size)
    n_remove = max(0, min(n_edges, n_remove))

    if n_remove > 0:
        edge_rng = rng if rng is not None else random.Random(seed)
        removed_edges = edge_rng.sample(edges, k=n_remove)
        H.remove_edges_from(removed_edges)

    return H, n_edges


def make_edge_regression_dataset(
    seed_graph: nx.Graph,
    *,
    n_negative_per_positive: int,
    n_replicates: int = 1,
    seed: int | None = None,
    allow_self_loops: bool = False,
):
    """Build a binary edge-removal dataset from one seed graph.

    Parameters
    ----------
    seed_graph : nx.Graph
        Source graph whose successive edge removals define the positive
        training examples.
    n_negative_per_positive : int
        Number of negative neighbor graphs sampled for each positive example.
    n_replicates : int, optional
        Number of independent edge-removal trajectories to generate.
    seed : int | None, optional
        Random seed controlling edge-removal order and negative sampling.
    allow_self_loops : bool, optional
        Whether negative samples may add self-loops when moving edges.

    Returns
    -------
    tuple[list[nx.Graph], list[nx.Graph], list[tuple[nx.Graph, int]]]
        Positive graphs, negative graphs, and the combined labeled dataset.
    """
    rng = random.Random(seed)
    positives = []
    negatives = []
    dataset = []

    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")

    for _ in range(n_replicates):
        current_graph = seed_graph.copy()
        while current_graph.number_of_edges() > 0:
            edge = rng.choice(list(current_graph.edges()))
            positive_graph = current_graph.copy()
            positive_graph.remove_edge(*edge)

            positives.append(positive_graph)
            dataset.append((positive_graph, 1))

            if positive_graph.number_of_edges() > 0:
                negative_graphs = edge_neighbors(
                    positive_graph,
                    n_samples=n_negative_per_positive,
                    seed=rng.randrange(10**9),
                    allow_self_loops=allow_self_loops,
                )
                negatives.extend(negative_graphs)
                dataset.extend((negative_graph, 0) for negative_graph in negative_graphs)

            current_graph = positive_graph

    return positives, negatives, dataset


def make_edge_regression_dataset_subgraph_ordered(
    seed_graph: nx.Graph,
    *,
    decomposition_function,
    nbits: int,
    n_negative_per_positive: int,
    n_replicates: int = 1,
    seed: int | None = None,
    allow_self_loops: bool = False,
):
    """Build an edge-removal dataset using a decomposition-aware edge order.

    Parameters
    ----------
    seed_graph : nx.Graph
        Source graph whose edges are removed to create training examples.
    decomposition_function : callable
        Function used to decompose the graph into interpretation subgraphs so
        edge removals can respect domain-specific substructure groupings.
    nbits : int
        Bit width passed to the abstract-graph decomposition machinery.
    n_negative_per_positive : int
        Number of negative neighbor graphs sampled for each positive example.
    n_replicates : int, optional
        Number of independent decomposition-group traversal runs to generate.
    seed : int | None, optional
        Random seed controlling group order, edge order within groups, and
        negative sampling.
    allow_self_loops : bool, optional
        Whether negative samples may add self-loops when moving edges.

    Returns
    -------
    tuple[list[nx.Graph], list[nx.Graph], list[tuple[nx.Graph, int]]]
        Positive graphs, negative graphs, and the combined labeled dataset.
    """
    rng = random.Random(seed)
    positives = []
    negatives = []
    dataset = []

    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")

    edge_groups = _decomposition_edge_groups(
        seed_graph,
        decomposition_function=decomposition_function,
        nbits=nbits,
    )

    for _ in range(n_replicates):
        current_graph = seed_graph.copy()
        replicate_groups = [list(group) for group in edge_groups]
        rng.shuffle(replicate_groups)

        for group in replicate_groups:
            while True:
                remaining_group_edges = [
                    edge for edge in group if _graph_has_canonical_edge(current_graph, edge)
                ]
                if not remaining_group_edges:
                    break
                edge = rng.choice(remaining_group_edges)
                positive_graph = current_graph.copy()
                positive_graph.remove_edge(*edge)

                positives.append(positive_graph)
                dataset.append((positive_graph, 1))

                if positive_graph.number_of_edges() > 0:
                    negative_graphs = edge_neighbors(
                        positive_graph,
                        n_samples=n_negative_per_positive,
                        seed=rng.randrange(10**9),
                        allow_self_loops=allow_self_loops,
                    )
                    negatives.extend(negative_graphs)
                    dataset.extend((negative_graph, 0) for negative_graph in negative_graphs)

                current_graph = positive_graph
                if current_graph.number_of_edges() == 0:
                    break

            if current_graph.number_of_edges() == 0:
                break

    return positives, negatives, dataset


def _decomposition_edge_groups(
    graph: nx.Graph,
    *,
    decomposition_function,
    nbits: int,
):
    abstract_graph = graph_to_abstract_graph(
        graph,
        decomposition_function=decomposition_function,
        nbits=nbits,
    )
    groups = []
    seen_groups = set()
    full_graph_edges = frozenset(_canonicalize_edge(edge, graph) for edge in graph.edges())

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        if subgraph is None or subgraph.number_of_edges() == 0:
            continue
        group_edges = frozenset(_canonicalize_edge(edge, graph) for edge in subgraph.edges())
        if not group_edges:
            continue
        # Skip the default full-graph interpretation node; keep only actual decomposition groups.
        if group_edges == full_graph_edges:
            continue
        if group_edges in seen_groups:
            continue
        seen_groups.add(group_edges)
        groups.append(list(group_edges))

    covered_edges = set()
    for group in groups:
        covered_edges.update(group)
    leftover_edges = [edge for edge in full_graph_edges if edge not in covered_edges]
    if leftover_edges:
        groups.append(leftover_edges)

    return groups if groups else [list(full_graph_edges)]


def _canonicalize_edge(edge, graph: nx.Graph):
    u, v = edge[:2]
    if nx.is_directed(graph):
        return (u, v)
    return tuple(sorted((u, v)))


def _graph_has_canonical_edge(graph: nx.Graph, edge) -> bool:
    u, v = edge
    return graph.has_edge(u, v)


def _connected_component_subgraphs(graph: nx.Graph):
    if nx.is_directed(graph):
        components = nx.weakly_connected_components(graph)
    else:
        components = nx.connected_components(graph)
    return [graph.subgraph(nodes).copy() for nodes in components]


def _sample_components(component_graphs, n_components, rng: random.Random):
    if n_components >= len(component_graphs):
        return list(component_graphs)
    indices = rng.sample(range(len(component_graphs)), k=n_components)
    return [component_graphs[idx] for idx in indices]


def _merge_component_graphs(component_graphs, template_graph: nx.Graph):
    merged = template_graph.__class__()
    next_node_id = 0
    for component in component_graphs:
        relabel_map = {}
        for node, node_attrs in component.nodes(data=True):
            relabel_map[node] = next_node_id
            merged.add_node(next_node_id, **dict(node_attrs))
            next_node_id += 1
        relabeled_component = nx.relabel_nodes(component, relabel_map, copy=True)
        if merged.is_multigraph():
            for u, v, key, edge_attrs in relabeled_component.edges(keys=True, data=True):
                merged.add_edge(u, v, key=key, **dict(edge_attrs))
        else:
            for u, v, edge_attrs in relabeled_component.edges(data=True):
                merged.add_edge(u, v, **dict(edge_attrs))
    merged.graph.update(dict(template_graph.graph))
    return merged


class EdgeGenerator:
    """Learn graph-growing policies and generate graphs by adding edges.

    The generator combines:
    - feasibility estimators for partial and final validity checks,
    - a graph estimator that scores promising next states,
    - an optional target estimator used to steer generation toward a label or
      numeric target,
    - retrieval-based pair conditioning through ``store(...)`` and
      ``generate_from_pair(...)``.
    """

    def __init__(
        self,
        feasibility_estimator=None,
        graph_estimator=None,
        target_estimator=None,
        edge_risk_estimator=None,
        target_estimator_mode: str = "classification",
        decomposition_function=None,
        *,
        partial_feasibility_estimator=None,
        final_feasibility_estimator=None,
        n_negative_per_positive: int = 3,
        n_replicates: int = 1,
        beam_size: int = 10,
        max_restarts: int = 3,
        fallback_base_steps: int = 2,
        fallback_growth_factor: float = 2.0,
        beam_growth_factor: float = 1.5,
        max_beam_size: int | None = None,
        enforce_diversity: bool = True,
        use_similarity_repulsion: bool = True,
        repulsion_weight: float = 0.2,
        repulsion_growth_factor: float = 1.5,
        edge_risk_lambda: float = 0.0,
        max_repulsion_memory: int = 256,
        allow_self_loops: bool = False,
        fit_n_jobs: int = -1,
        fit_backend: str = "loky",
        verbose: bool = False,
        seed: int | None = None,
        early_stop_if_final_feasible: bool = True,
        require_single_connected_component: bool = True,
        enforce_repair_label_set_coverage: bool = True,
        max_terminal_completion_lookahead_states: int = 512,
    ):
        """Configure an edge-growing graph generator.

        Parameters
        ----------
        feasibility_estimator : object, optional
            Backward-compatible estimator used for both partial and final
            feasibility if separate estimators are not provided.
        graph_estimator : object
            Estimator scoring which partially built graphs look most promising
            during beam search.
        target_estimator : object, optional
            Optional estimator used to score graphs against a requested target.
        edge_risk_estimator : object, optional
            Optional estimator used to learn online edge-materialization risk.
        target_estimator_mode : str, optional
            ``"classification"`` to match a class label or ``"regression"`` to
            prefer predictions close to a numeric target.
        decomposition_function : callable, optional
            Optional graph decomposition used when building ordered fragment
            training datasets.
        partial_feasibility_estimator : object, optional
            Estimator checked on intermediate states during search.
        final_feasibility_estimator : object, optional
            Estimator checked on terminal states with the requested edge count.
        n_negative_per_positive : int, optional
            Number of negative graphs sampled per positive fragment during
            training-set construction.
        n_replicates : int, optional
            Number of independent fragment-generation runs per training graph.
        beam_size : int, optional
            Base number of candidates retained at each search step.
        max_restarts : int, optional
            Maximum number of fallback/backtracking phases allowed when search
            gets stuck.
        fallback_base_steps : int, optional
            Initial rollback depth used by fallback search repair.
        fallback_growth_factor : float, optional
            Multiplicative growth applied to rollback depth after each fallback.
        beam_growth_factor : float, optional
            Multiplicative growth applied to beam size after each fallback.
        max_beam_size : int | None, optional
            Optional cap on the expanded fallback beam size.
        enforce_diversity : bool, optional
            Whether previously seen graphs are filtered to encourage diverse
            search trajectories.
        use_similarity_repulsion : bool, optional
            Whether failed states contribute a similarity penalty in later
            fallback phases.
        repulsion_weight : float, optional
            Base weight of the similarity-repulsion penalty.
        repulsion_growth_factor : float, optional
            Multiplicative growth of repulsion during fallback phases.
        edge_risk_lambda : float, optional
            Weight of the edge-risk penalty subtracted from selection scores.
        max_repulsion_memory : int, optional
            Maximum number of failed graph embeddings retained for repulsion.
        allow_self_loops : bool, optional
            Whether candidate added edges and negative samples may include
            self-loops.
        fit_n_jobs : int, optional
            Parallelism used when building fragment datasets during fitting.
        fit_backend : str, optional
            Joblib backend used for parallel fragment dataset construction.
        verbose : bool, optional
            Whether fitting and generation emit progress logs.
        seed : int | None, optional
            Seed for the internal random number generator controlling sampling,
            fallback randomness, and pair-conditioned generation.
        early_stop_if_final_feasible : bool, optional
            Whether search may terminate before reaching ``n_edges`` when the
            current graph already satisfies the final feasibility estimator and
            every feasible one-edge expansion scores worse.
        require_single_connected_component : bool, optional
            Whether final-feasible graphs with more than one connected
            component should continue growing past ``n_edges`` until a single
            connected component is reached, within a bounded extra-edge window.
        enforce_repair_label_set_coverage : bool, optional
            Whether ``repair(...)`` should fail early when the query graph has
            node labels that are absent from the selected repair neighborhood.
            Extra labels in the neighborhood are tolerated.
        max_terminal_completion_lookahead_states : int, optional
            Maximum number of future completion states explored per candidate
            when estimating whether the remaining edge budget can still reach a
            final-feasible graph. If the cap is reached, the candidate is kept
            because impossibility was not proven. Set to ``0`` to disable this
            lookahead.

        Returns
        -------
        None
            Initializes the generator in-place.
        """
        # Estimator roles: search feasibility, graph scoring, and optional
        # target steering can be configured independently.
        (
            self.partial_feasibility_estimator,
            self.final_feasibility_estimator,
        ) = self._resolve_feasibility_estimators(
            feasibility_estimator=feasibility_estimator,
            partial_feasibility_estimator=partial_feasibility_estimator,
            final_feasibility_estimator=final_feasibility_estimator,
        )
        if graph_estimator is None:
            raise ValueError("graph_estimator is required")
        # Backward-compatible alias for code that still expects one search-time estimator.
        self.feasibility_estimator = self.partial_feasibility_estimator
        self.graph_estimator = graph_estimator
        self.target_estimator = target_estimator
        self.edge_risk_estimator = edge_risk_estimator
        if target_estimator_mode not in {"classification", "regression"}:
            raise ValueError(
                "target_estimator_mode must be 'classification' or 'regression'"
        )
        self.target_estimator_mode = target_estimator_mode
        self.decomposition_function = decomposition_function

        # Training-set construction and search-shape parameters.
        self.n_negative_per_positive = n_negative_per_positive
        self.n_replicates = n_replicates
        self.beam_size = beam_size
        self.max_restarts = max_restarts
        self.fallback_base_steps = fallback_base_steps
        self.fallback_growth_factor = fallback_growth_factor
        self.beam_growth_factor = beam_growth_factor
        self.max_beam_size = max_beam_size
        self.enforce_diversity = enforce_diversity
        self.use_similarity_repulsion = use_similarity_repulsion
        self.repulsion_weight = repulsion_weight
        self.repulsion_growth_factor = repulsion_growth_factor
        self.edge_risk_lambda = edge_risk_lambda
        self.max_repulsion_memory = max_repulsion_memory
        self.allow_self_loops = allow_self_loops
        self.fit_n_jobs = fit_n_jobs
        self.fit_backend = fit_backend
        self.verbose = verbose
        self.seed = seed
        self.early_stop_if_final_feasible = early_stop_if_final_feasible
        self.require_single_connected_component = require_single_connected_component
        self.enforce_repair_label_set_coverage = bool(enforce_repair_label_set_coverage)
        self.max_terminal_completion_lookahead_states = int(
            max_terminal_completion_lookahead_states
        )
        self.rng = random.Random(seed)

        # Learned datasets, search bookkeeping, and retrieval caches are
        # populated after fitting and/or storing graphs.
        self.seed_graphs_ = None
        self.positives_ = None
        self.negatives_ = None
        self.dataset_ = None
        self.targets_ = None
        self.target_graphs_ = None
        self.target_values_ = None
        self.edge_attribute_templates_ = None
        self.top_k_ = beam_size
        self.n_tried_ = 0
        self.max_depth_ = 0
        self.embedding_cache_ = {}
        self.diversity_memory_hashes_ = []
        self.diversity_memory_hash_set_ = set()
        self.failed_memory_hashes_ = []
        self.failed_memory_hash_set_ = set()
        self.edge_risk_model_ = (
            None
            if self.edge_risk_estimator is None
            else _OnlineGraphRegressorAdapter(self.edge_risk_estimator)
        )
        self.edge_risk_trace_ = None
        self.stored_graphs_ = None
        self.stored_targets_ = None
        self.stored_graph_hash_to_index_ = {}
        self.retrieval_transformer_ = None
        self.stored_retrieval_vectors_ = None
        self.stored_distance_matrix_ = None
        self.surgical_backtracking_ = True
        self.last_pair_session_ = None
        self.last_repair_training_graphs_ = None
        self.last_repair_training_targets_ = None
        self.last_repair_label_set_mismatch_ = None

    def _resolve_feasibility_estimators(
        self,
        *,
        feasibility_estimator,
        partial_feasibility_estimator,
        final_feasibility_estimator,
    ):
        resolved_partial = (
            partial_feasibility_estimator
            if partial_feasibility_estimator is not None
            else feasibility_estimator
        )
        resolved_final = (
            final_feasibility_estimator
            if final_feasibility_estimator is not None
            else feasibility_estimator
        )
        if resolved_partial is None or resolved_final is None:
            raise ValueError(
                "Provide feasibility_estimator or both partial_feasibility_estimator "
                "and final_feasibility_estimator"
            )
        if resolved_partial is resolved_final:
            resolved_final = copy.deepcopy(resolved_final)
        return resolved_partial, resolved_final

    def fit(
        self,
        graphs,
        targets=None,
        *,
        deduplicate_feasibility_graphs: bool = True,
        partial_feasibility_extra_graphs=None,
    ):
        """Fit the generator from one or more training graphs.

        Parameters
        ----------
        graphs : nx.Graph | iterable[nx.Graph]
            Training graphs used to construct fragment datasets and fit the
            internal estimators.
        targets : object | iterable[object], optional
            Optional per-graph targets used to fit the target estimator on graph
            fragments.
        deduplicate_feasibility_graphs : bool, optional
            Whether to deduplicate graphs before fitting partial/final
            feasibility estimators. Repair-local fits can disable this because
            their graph selection is already ID-based and only one neighbor
            expansion is used.
        partial_feasibility_extra_graphs : iterable[nx.Graph], optional
            Extra graphs used only to fit the partial feasibility estimator.
            This is useful during repair to make node-only bootstrap states
            admissible without teaching the final feasibility estimator to
            accept the repaired query graph.

        Returns
        -------
        EdgeGenerator
            The fitted generator instance.
        """
        graph_list = self._as_graph_list(graphs)
        self.seed_graphs_ = [graph.copy() for graph in graph_list]
        dataset_parts = self._build_fragment_datasets(self.seed_graphs_)
        self._store_graph_estimator_training_data(dataset_parts)

        partial_fit_graphs = (
            list(self.seed_graphs_)
            + [graph for graph in self.positives_ if graph.number_of_edges() > 0]
        )
        final_fit_graphs = list(self.seed_graphs_)
        if deduplicate_feasibility_graphs:
            partial_fit_graphs = self._unique_graphs(partial_fit_graphs)
            final_fit_graphs = self._unique_graphs(final_fit_graphs)
        if partial_feasibility_extra_graphs is not None:
            partial_fit_graphs.extend(
                graph.copy() for graph in partial_feasibility_extra_graphs
            )
        partial_feasibility_fit_start = time.perf_counter()
        self.partial_feasibility_estimator.fit(partial_fit_graphs)
        partial_feasibility_fit_time = time.perf_counter() - partial_feasibility_fit_start
        final_feasibility_fit_start = time.perf_counter()
        self.final_feasibility_estimator.fit(final_fit_graphs)
        final_feasibility_fit_time = time.perf_counter() - final_feasibility_fit_start
        if self.verbose:
            partial_fit_min = int(partial_feasibility_fit_time // 60)
            partial_fit_sec = partial_feasibility_fit_time - 60 * partial_fit_min
            final_fit_min = int(final_feasibility_fit_time // 60)
            final_fit_sec = final_feasibility_fit_time - 60 * final_fit_min
            print(
                f"[fit] partial_feasibility_graphs={len(partial_fit_graphs)} "
                f"final_feasibility_graphs={len(final_fit_graphs)} "
                f"positives={len(self.positives_)} negatives={len(self.negatives_)} "
                f"dataset={len(self.dataset_)} "
                f"partial_time={partial_fit_min}m {partial_fit_sec:.1f}s "
                f"final_time={final_fit_min}m {final_fit_sec:.1f}s"
            )

        self.targets_ = np.array([label for graph, label in self.dataset_], dtype=int)
        train_graphs = [graph for graph, label in self.dataset_]
        graph_estimator_fit_start = time.perf_counter()
        self.graph_estimator.fit(train_graphs, self.targets_)
        graph_estimator_fit_time = time.perf_counter() - graph_estimator_fit_start
        if self.verbose:
            n_positive = int(np.sum(self.targets_ == 1))
            n_negative = int(np.sum(self.targets_ == 0))
            graph_estimator_fit_min = int(graph_estimator_fit_time // 60)
            graph_estimator_fit_sec = graph_estimator_fit_time - 60 * graph_estimator_fit_min
            print(
                f"[fit] graph_estimator_graphs={len(train_graphs)} "
                f"positive_labels={n_positive} negative_labels={n_negative} "
                f"time={graph_estimator_fit_min}m {graph_estimator_fit_sec:.1f}s"
            )

        if targets is not None:
            target_list = self._coerce_optional_per_graph_argument(
                targets,
                self.seed_graphs_,
                name="targets",
            )
            target_graphs, target_values = self._build_target_fragment_dataset(
                dataset_parts,
                target_list,
            )
            self._fit_target_estimator_from_fragments(
                target_graphs,
                target_values,
                verbose_prefix="[fit]",
            )
        else:
            self.target_graphs_ = None
            self.target_values_ = None

        self.edge_attribute_templates_ = self._collect_edge_attribute_templates(
            self.seed_graphs_
        )
        self.embedding_cache_ = {}
        self._initialize_diversity_memory(self.seed_graphs_)
        self.failed_memory_hashes_ = []
        self.failed_memory_hash_set_ = set()
        return self

    def fit_target_estimator(self, graphs, targets):
        """Fit only the optional target estimator from graphs and targets.

        Parameters
        ----------
        graphs : nx.Graph | iterable[nx.Graph]
            Graphs used to build fragment-level target-estimator training data.
        targets : object | iterable[object]
            Per-graph target values aligned with ``graphs``.

        Returns
        -------
        EdgeGenerator
            The generator instance with an updated target estimator.
        """
        if self.target_estimator is None:
            raise ValueError("target_estimator is None; provide one before fitting")
        graph_list = [graph.copy() for graph in self._as_graph_list(graphs)]
        target_list = self._coerce_optional_per_graph_argument(
            targets,
            graph_list,
            name="targets",
        )
        dataset_parts = self._build_fragment_datasets(graph_list)
        target_graphs, target_values = self._build_target_fragment_dataset(
            dataset_parts,
            target_list,
        )
        self._fit_target_estimator_from_fragments(
            target_graphs,
            target_values,
            verbose_prefix="[fit_target_estimator]",
        )
        return self

    def store(self, graphs, targets=None):
        """Store a graph corpus for retrieval-based pair generation.

        Parameters
        ----------
        graphs : iterable[nx.Graph]
            Graph corpus used as the retrieval space for
            ``generate_from_pair(...)``.
        targets : object | iterable[object], optional
            Optional per-graph targets stored alongside the retrieval corpus.

        Returns
        -------
        EdgeGenerator
            The generator instance with an initialized retrieval index.
        """
        graph_list = [graph.copy() for graph in self._as_graph_list(graphs)]
        if len(graph_list) < 2:
            raise ValueError("store(graphs, ...) requires at least two graphs")

        self.stored_graphs_ = graph_list
        if targets is None:
            self.stored_targets_ = None
        else:
            self.stored_targets_ = list(
                self._coerce_optional_per_graph_argument(
                    targets,
                    self.stored_graphs_,
                    name="targets",
                )
            )

        self.stored_graph_hash_to_index_ = {}
        for idx, graph in enumerate(self.stored_graphs_):
            graph_hash = hash_graph(graph)
            if graph_hash not in self.stored_graph_hash_to_index_:
                self.stored_graph_hash_to_index_[graph_hash] = idx

        transformer = self._make_retrieval_transformer()
        self.retrieval_transformer_ = transformer
        self.stored_retrieval_vectors_ = self._vectorize_retrieval_graphs(
            transformer,
            self.stored_graphs_,
            fit=True,
        )
        self.stored_distance_matrix_ = pairwise_distances(self.stored_retrieval_vectors_)
        np.fill_diagonal(self.stored_distance_matrix_, 0.0)
        return self

    def generate(
        self,
        graphs,
        n_edges,
        *,
        target=None,
        target_lambda: float = 1.0,
        return_path: bool = True,
        draw_graphs_fn: DrawGraphsFn | None = None,
        verbose: bool | None = None,
    ):
        """Generate graphs by adding edges until a requested edge count is met.

        Parameters
        ----------
        graphs : nx.Graph | iterable[nx.Graph]
            Starting graph or graphs from which generation begins.
        n_edges : int | iterable[int]
            Desired final edge count for each input graph.
        target : object | iterable[object], optional
            Optional target value used by the target estimator to steer search.
        target_lambda : float, optional
            Weight applied to the target score when combining it with the graph
            estimator score.
        return_path : bool, optional
            Whether to return the full graph-growth path or only the final
            generated graph for each input.
        draw_graphs_fn : callable, optional
            Optional visualization callback used when verbose logging is
            enabled.
        verbose : bool | None, optional
            Overrides the instance-level verbosity for this call.

        Returns
        -------
        list[nx.Graph] | nx.Graph | list[list[nx.Graph]] | list[nx.Graph] | None
            Generated path(s) or final graph(s), matching the single-graph vs
            multi-graph input shape and the ``return_path`` setting.
        """
        if self.edge_attribute_templates_ is None:
            raise ValueError("EdgeGenerator must be fit before calling generate")
        verbose = self.verbose if verbose is None else verbose

        graph_list, edge_counts, target_values = self._normalize_generate_inputs(
            graphs,
            n_edges,
            target=target,
        )

        paths = []
        for i, (graph, target_n_edges, target_value) in enumerate(
            zip(graph_list, edge_counts, target_values)
        ):
            try:
                path = self._generate_one(
                    graph,
                    target_n_edges,
                    target=target_value,
                    target_lambda=target_lambda,
                    draw_graphs_fn=draw_graphs_fn,
                    verbose=verbose,
                    graph_index=i,
                )
            except ValueError as exc:
                if verbose:
                    print(
                        f"[graph {i}] failed target_edges={target_n_edges} "
                        f"reason={exc}"
                    )
                continue
            paths.append(path)

        if self._is_single_graph_input(graphs):
            if not paths:
                return [] if return_path else None
            return paths[0] if return_path else paths[0][-1]
        if return_path:
            return paths
        return [path[-1] for path in paths]

    def generate_from_pair(
        self,
        graph_a=None,
        graph_b=None,
        *,
        size_of_edge_removal=None,
        n_paths: int = 3,
        path_k: int = 3,
        n_neighbors_per_path_graph: int = 3,
        target=None,
        target_lambda: float = 1.0,
        return_path: bool = True,
        draw_graphs_fn: DrawGraphsFn | None = None,
        verbose: bool | None = None,
    ):
        """Generate a graph conditioned on a pair of endpoint graphs.

        Parameters
        ----------
        graph_a : nx.Graph | None, optional
            First endpoint graph. If both ``graph_a`` and ``graph_b`` are
            omitted, the last cached pair session is reused.
        graph_b : nx.Graph | None, optional
            Second endpoint graph paired with ``graph_a``.
        size_of_edge_removal : float | int | None, optional
            Amount of edge pruning applied to each endpoint graph before
            component mixing starts the generation process. When reusing a
            cached pair session, the cached value is kept unless an explicit
            override is provided here.
        n_paths : int, optional
            Number of shortest retrieval paths to extract between the endpoint
            graphs in the stored corpus.
        path_k : int, optional
            Neighborhood size used when sparsifying the retrieval graph for
            shortest-path computation.
        n_neighbors_per_path_graph : int, optional
            Number of extra nearest neighbors added around each graph selected
            from the retrieved paths.
        target : object, optional
            Optional explicit target for the generated graph. If omitted, a
            target may be inferred from stored endpoint targets.
        target_lambda : float, optional
            Weight applied to target steering during graph generation.
        return_path : bool, optional
            Whether to return the full generation path or only the final graph.
        draw_graphs_fn : callable, optional
            Optional visualization callback used when verbose logging is
            enabled.
        verbose : bool | None, optional
            Overrides the instance-level verbosity for this call.

        Returns
        -------
        list[nx.Graph] | nx.Graph | None
            Generated path or final graph, depending on ``return_path``.
        """
        verbose = self.verbose if verbose is None else verbose
        if (graph_a is None) != (graph_b is None):
            raise ValueError("graph_a and graph_b must either both be provided or both be None")

        if graph_a is None and graph_b is None:
            session = self._require_cached_pair_session()
            if size_of_edge_removal is not None:
                session = dict(session)
                session["size_of_edge_removal"] = float(size_of_edge_removal)
                self.last_pair_session_ = session
            if verbose:
                print("[pair] reusing cached pair session and fitted estimators")
            return self._generate_from_cached_pair_session(
                session,
                target_lambda=target_lambda,
                return_path=return_path,
                draw_graphs_fn=draw_graphs_fn,
                verbose=verbose,
            )

        self._require_stored_dataset()
        resolved_size_of_edge_removal = (
            0.5 if size_of_edge_removal is None else float(size_of_edge_removal)
        )
        pair_context = self._prepare_pair_training_context(
            graph_a,
            graph_b,
            n_paths=n_paths,
            path_k=path_k,
            n_neighbors_per_path_graph=n_neighbors_per_path_graph,
        )
        self._log_pair_training_context(
            pair_context,
            draw_graphs_fn=draw_graphs_fn,
            verbose=verbose,
        )
        self._fit_pair_training_graphs(pair_context["fit_graphs"], pair_context["fit_targets"])

        resolved_target = self._resolve_pair_target(pair_context["query"], target)
        self._cache_pair_session(
            graph_a=graph_a,
            graph_b=graph_b,
            size_of_edge_removal=resolved_size_of_edge_removal,
            target=resolved_target,
        )

        return self._generate_from_cached_pair_session(
            self.last_pair_session_,
            target_lambda=target_lambda,
            return_path=return_path,
            draw_graphs_fn=draw_graphs_fn,
            verbose=verbose,
        )

    def repair(
        self,
        graph,
        *,
        n_neighbors: int = 1,
        target=None,
        target_lambda: float = 0.5,
        return_path: bool = True,
        draw_graphs_fn: DrawGraphsFn | None = None,
        verbose: bool | None = None,
    ):
        """Repair one graph by refitting on stored nearest neighbors.

        The method reuses the stored retrieval corpus to select the nearest
        ``n_neighbors`` training graphs whose node-label sets cover the input
        graph labels, refits the generator on that local neighborhood, keeps
        the original graph edge count as the repair target, and, when the input
        graph is final-infeasible, seeds generation from one or more surgically
        repaired rollback states derived from the estimator's violating-edge
        sets.

        Parameters
        ----------
        graph : nx.Graph
            Graph to repair.
        n_neighbors : int, optional
            Number of nearest stored graphs used to refit the local generator.
        target : object, optional
            Optional target value forwarded to ``generate(...)``.
        target_lambda : float, optional
            Weight applied to target steering during repair generation.
        return_path : bool, optional
            Whether to return the full repair path or only the final graph.
        draw_graphs_fn : callable, optional
            Optional visualization callback used when verbose logging is
            enabled.
        verbose : bool | None, optional
            Overrides the instance-level verbosity for this call.

        Returns
        -------
        list[nx.Graph] | nx.Graph | None
            Repair path or final repaired graph, matching ``return_path``.
        """
        verbose = self.verbose if verbose is None else verbose
        self._require_stored_dataset()

        repair_context = self._prepare_repair_training_context(
            graph,
            n_neighbors=n_neighbors,
        )
        label_set_mismatch = None
        if self.enforce_repair_label_set_coverage:
            label_set_mismatch = self._repair_label_set_mismatch(repair_context)
        self.last_repair_label_set_mismatch_ = label_set_mismatch
        if label_set_mismatch is not None:
            if verbose:
                print(
                    "[repair] label-set mismatch between input graph and repair neighborhood; "
                    f"missing_from_neighbors={label_set_mismatch['missing_from_neighbors']} "
                    f"extra_in_neighbors={label_set_mismatch['extra_in_neighbors']}"
                )
            return [] if return_path else None
        self.last_repair_training_graphs_ = [
            fit_graph.copy() for fit_graph in repair_context["fit_graphs"]
        ]
        if repair_context["fit_targets"] is None:
            self.last_repair_training_targets_ = None
        else:
            self.last_repair_training_targets_ = list(repair_context["fit_targets"])
        self._log_repair_training_context(
            repair_context,
            draw_graphs_fn=draw_graphs_fn,
            verbose=verbose,
        )
        partial_feasibility_extra_graphs = (
            self._repair_partial_feasibility_bootstrap_graphs(
                repair_context["graph"],
                repair_context["fit_graphs"],
            )
        )
        self._fit_pair_training_graphs(
            repair_context["fit_graphs"],
            repair_context["fit_targets"],
            deduplicate_feasibility_graphs=False,
            partial_feasibility_extra_graphs=partial_feasibility_extra_graphs,
        )

        start_graph = graph.copy()
        target_n_edges = int(start_graph.number_of_edges())
        resolved_target = self._resolve_repair_target(repair_context, target)

        if bool(self.final_feasibility_estimator.predict([start_graph])[0]):
            if verbose:
                print("[repair] input graph is already final-feasible; returning unchanged graph")
            return [start_graph] if return_path else start_graph

        repaired_states = self._build_repair_start_states(
            start_graph,
            target=resolved_target,
            target_lambda=target_lambda,
        )
        if not repaired_states:
            if verbose:
                print("[repair] no surgical repair starts could be constructed")
            return [] if return_path else None

        for repair_index, repaired_state in enumerate(repaired_states, start=1):
            if verbose:
                removed_edges = repaired_state.get("repair_removed_edges", ())
                print(
                    f"[repair] attempt={repair_index}/{len(repaired_states)} "
                    f"start_edges={repaired_state['graph'].number_of_edges()} "
                    f"target_edges={target_n_edges} removed_edges={list(removed_edges)}"
                )
                self._draw_graphs(
                    draw_graphs_fn,
                    [start_graph, repaired_state["graph"]],
                    n_graphs_per_line=2,
                    titles=[
                        (
                            "current repair input\n"
                            f"edges={start_graph.number_of_edges()} target_edges={target_n_edges}"
                        ),
                        (
                            "pruned repair start\n"
                            f"edges={repaired_state['graph'].number_of_edges()} "
                            f"removed_edges={len(removed_edges)}"
                        ),
                    ],
                )
            repaired_path = self.generate(
                repaired_state["graph"],
                target_n_edges,
                target=resolved_target,
                target_lambda=target_lambda,
                return_path=True,
                draw_graphs_fn=draw_graphs_fn,
                verbose=verbose,
            )
            if not repaired_path:
                continue
            if hash_graph(repaired_path[0]) == hash_graph(start_graph):
                full_path = repaired_path
            else:
                full_path = [start_graph] + repaired_path
            return full_path if return_path else full_path[-1]

        return [] if return_path else None

    def _graph_unique_node_labels(self, graph) -> set:
        labels = set()
        for _, attrs in graph.nodes(data=True):
            label = attrs.get("label")
            if label is not None:
                labels.add(label)
        return labels

    def _repair_label_set_mismatch(self, repair_context):
        graph_labels = self._graph_unique_node_labels(repair_context["graph"])
        neighbor_labels = set()
        for fit_graph in repair_context["fit_graphs"]:
            neighbor_labels.update(self._graph_unique_node_labels(fit_graph))
        missing_from_neighbors = sorted(graph_labels - neighbor_labels)
        extra_in_neighbors = sorted(neighbor_labels - graph_labels)
        if not missing_from_neighbors:
            return None
        return {
            "graph_labels": sorted(graph_labels),
            "neighbor_labels": sorted(neighbor_labels),
            "missing_from_neighbors": missing_from_neighbors,
            "extra_in_neighbors": extra_in_neighbors,
        }

    def _node_only_graph_copy(self, graph):
        node_only_graph = graph.__class__()
        node_only_graph.graph.update(graph.graph)
        for node, attrs in graph.nodes(data=True):
            node_only_graph.add_node(node, **dict(attrs))
        return node_only_graph

    def _repair_partial_feasibility_bootstrap_graphs(self, graph, fit_graphs):
        return [self._node_only_graph_copy(graph)]

    def _cache_pair_session(
        self,
        *,
        graph_a,
        graph_b,
        size_of_edge_removal,
        target,
    ) -> None:
        self.last_pair_session_ = {
            "graph_a": None if graph_a is None else graph_a.copy(),
            "graph_b": None if graph_b is None else graph_b.copy(),
            "size_of_edge_removal": float(size_of_edge_removal),
            "target": target,
        }

    def _require_cached_pair_session(self):
        if self.last_pair_session_ is None:
            raise ValueError(
                "No cached pair session is available; call generate_from_pair(graph_a, graph_b, ...)"
            )
        return self.last_pair_session_

    def _generate_from_cached_pair_session(
        self,
        session,
        *,
        target_lambda: float,
        return_path: bool,
        draw_graphs_fn: DrawGraphsFn | None,
        verbose: bool,
    ):
        graph_a = session["graph_a"]
        graph_b = session["graph_b"]
        size_of_edge_removal = session["size_of_edge_removal"]
        resolved_target = session["target"]

        start_graph_a, target_n_edges_a = remove_edges(
            graph_a,
            size=size_of_edge_removal,
            rng=self.rng,
        )
        start_graph_b, target_n_edges_b = remove_edges(
            graph_b,
            size=size_of_edge_removal,
            rng=self.rng,
        )
        mixed_graph = mix_connected_components(
            start_graph_a,
            start_graph_b,
            seed=self.rng.randrange(10**9),
        )
        mixed_target_n_edges = int(round(np.mean([target_n_edges_a, target_n_edges_b])))
        return self.generate(
            mixed_graph,
            mixed_target_n_edges,
            target=resolved_target,
            target_lambda=target_lambda,
            return_path=return_path,
            draw_graphs_fn=draw_graphs_fn,
            verbose=verbose,
        )

    def _augment_indices_with_nearest_neighbors(
        self,
        distance_matrix,
        selected_indices,
        *,
        k: int,
    ):
        if k < 0:
            raise ValueError("n_neighbors_per_path_graph must be >= 0")

        selected = []
        seen = set()
        for idx in selected_indices:
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(int(idx))

        if k == 0 or not selected:
            return selected

        distances = np.asarray(distance_matrix, dtype=float)
        n_nodes = distances.shape[0]
        for idx in list(selected):
            if idx < 0 or idx >= n_nodes:
                continue
            row = distances[idx]
            neighbor_order = np.argsort(row, kind="stable")
            n_added = 0
            for neighbor_idx in neighbor_order:
                neighbor_idx = int(neighbor_idx)
                if neighbor_idx == idx or neighbor_idx in seen:
                    continue
                if not np.isfinite(row[neighbor_idx]):
                    continue
                seen.add(neighbor_idx)
                selected.append(neighbor_idx)
                n_added += 1
                if n_added >= k:
                    break
        return sorted(selected)

    # Search setup and orchestration.

    def _normalize_generate_inputs(self, graphs, n_edges, *, target):
        graph_list = self._as_graph_list(graphs)
        edge_counts = [n_edges] if isinstance(n_edges, int) else list(n_edges)
        if len(graph_list) != len(edge_counts):
            raise ValueError("graphs and n_edges must have the same length")
        target_values = self._coerce_optional_per_graph_argument(
            target,
            graph_list,
            name="target",
        )
        if target_values is None:
            target_values = [None] * len(graph_list)
        return graph_list, edge_counts, target_values

    # Pair-conditioned retrieval and fitting.

    def _prepare_pair_training_context(
        self,
        graph_a,
        graph_b,
        *,
        n_paths: int,
        path_k: int,
        n_neighbors_per_path_graph: int,
    ):
        training_set_start = time.perf_counter()
        query = self._build_pair_query_corpus(graph_a, graph_b)
        path_matrix = self._path_matrix_from_distance_matrix(
            query["distance_matrix"],
            k=path_k,
        )
        paths = self._shortest_paths_from_matrix(
            path_matrix,
            query["source_idx"],
            query["dest_idx"],
            n_paths=n_paths,
        )
        if not paths:
            raise ValueError("Could not find shortest paths between the requested graphs")

        selected_indices = sorted({idx for path in paths for idx in path})
        selected_indices = self._augment_indices_with_nearest_neighbors(
            query["distance_matrix"],
            selected_indices,
            k=n_neighbors_per_path_graph,
        )
        fit_graphs = [query["graphs"][idx].copy() for idx in selected_indices]
        fit_targets = self._select_pair_targets(query, selected_indices)
        return {
            "query": query,
            "paths": paths,
            "selected_indices": selected_indices,
            "fit_graphs": fit_graphs,
            "fit_targets": fit_targets,
            "path_k": path_k,
            "training_set_elapsed": time.perf_counter() - training_set_start,
        }

    def _prepare_repair_training_context(
        self,
        graph,
        *,
        n_neighbors: int,
    ):
        if int(n_neighbors) < 1:
            raise ValueError("n_neighbors must be >= 1")

        graph_copy = graph.copy()
        graph_hash = hash_graph(graph_copy)
        stored_idx = self.stored_graph_hash_to_index_.get(graph_hash)
        if stored_idx is not None:
            distances = np.asarray(self.stored_distance_matrix_[stored_idx], dtype=float).copy()
        else:
            query_vector = self._vectorize_retrieval_graphs(
                self.retrieval_transformer_,
                [graph_copy],
                fit=False,
            )[0]
            distances = pairwise_distances(
                np.asarray(query_vector, dtype=float).reshape(1, -1),
                np.asarray(self.stored_retrieval_vectors_, dtype=float),
            ).ravel()

        sorted_candidate_indices = []
        for idx in np.argsort(distances, kind="stable"):
            idx = int(idx)
            if stored_idx is not None and idx == stored_idx:
                continue
            if not np.isfinite(distances[idx]):
                continue
            sorted_candidate_indices.append(idx)

        neighbor_indices = []
        if self.enforce_repair_label_set_coverage:
            missing_labels = set(self._graph_unique_node_labels(graph_copy))
            for idx in sorted_candidate_indices:
                candidate_labels = self._graph_unique_node_labels(self.stored_graphs_[idx])
                if not missing_labels.intersection(candidate_labels):
                    continue
                neighbor_indices.append(idx)
                missing_labels.difference_update(candidate_labels)
                if not missing_labels:
                    break

        for idx in sorted_candidate_indices:
            if idx in neighbor_indices:
                continue
            if len(neighbor_indices) >= int(n_neighbors):
                break
            neighbor_indices.append(idx)
            if len(neighbor_indices) >= int(n_neighbors):
                break

        if not neighbor_indices and stored_idx is not None:
            neighbor_indices = [int(stored_idx)]
        if not neighbor_indices:
            raise ValueError("Could not resolve any stored neighbors for repair")

        fit_graphs = [self.stored_graphs_[idx].copy() for idx in neighbor_indices]
        fit_targets = None
        if self.stored_targets_ is not None:
            fit_targets = [self.stored_targets_[idx] for idx in neighbor_indices]

        return {
            "graph": graph_copy,
            "query_index": None if stored_idx is None else int(stored_idx),
            "neighbor_indices": neighbor_indices,
            "neighbor_distances": [float(distances[idx]) for idx in neighbor_indices],
            "fit_graphs": fit_graphs,
            "fit_targets": fit_targets,
        }

    def _resolve_repair_target(self, repair_context, requested_target):
        if requested_target is not None:
            return requested_target
        query_index = repair_context.get("query_index")
        if query_index is None or self.stored_targets_ is None:
            return None
        return self.stored_targets_[query_index]

    def _log_repair_training_context(
        self,
        repair_context,
        *,
        draw_graphs_fn: DrawGraphsFn | None,
        verbose: bool,
    ) -> None:
        if not verbose:
            return
        print(
            f"[repair] query_index={repair_context['query_index']} "
            f"n_neighbors={len(repair_context['neighbor_indices'])} "
            f"neighbor_indices={repair_context['neighbor_indices']} "
            f"neighbor_distances={[round(d, 4) for d in repair_context['neighbor_distances']]}"
        )
        self._draw_graphs(
            draw_graphs_fn,
            [repair_context["graph"]],
            n_graphs_per_line=1,
            titles=["query"],
        )
        self._draw_graphs(
            draw_graphs_fn,
            repair_context["fit_graphs"],
            n_graphs_per_line=min(len(repair_context["fit_graphs"]), 7),
            titles=[f"nn:{idx}" for idx in repair_context["neighbor_indices"]],
        )

    def _build_repair_start_states(
        self,
        graph,
        *,
        target,
        target_lambda: float,
    ):
        start_graph = graph.copy()
        start_score = float(self._positive_scores([start_graph])[0])
        start_target_score = float(self._target_scores([start_graph], target=target)[0])
        start_state = self._make_state(
            start_graph,
            parent=None,
            score=start_score,
            depth=start_graph.number_of_edges(),
        )
        start_state["target_score"] = float(start_target_score)
        start_state["selection_score"] = float(start_score + target_lambda * start_target_score)

        infeasible_candidate = {
            "graph": start_graph,
            "score": float(start_score),
            "target_score": float(start_target_score),
            "selection_score": float(start_state["selection_score"]),
            "feasibility_stage": "final",
        }
        self._annotate_infeasible_candidates_with_violating_edge_sets([infeasible_candidate])

        repaired_states = []
        seen_hashes = set()
        n_repair_attempts = max(1, int(self.max_restarts))
        for fallback_index in range(n_repair_attempts):
            rollback_steps = self._rollback_steps_for_fallback(fallback_index)
            removed_edges, repair_score = self._select_edges_for_surgical_repair(
                start_state,
                [infeasible_candidate],
                rollback_steps=rollback_steps,
            )
            if not removed_edges:
                removed_edges = self._random_repair_removed_edge(start_state)
                repair_score = 0.0
            if not removed_edges:
                continue
            repaired_state = self._make_repaired_state(
                start_state,
                removed_edges,
                score=repair_score,
            )
            repaired_state = self._repair_state_until_partial_feasible(repaired_state)
            if repaired_state is None:
                continue
            repaired_hash = repaired_state["graph_hash"]
            if repaired_hash in seen_hashes:
                continue
            seen_hashes.add(repaired_hash)
            repaired_states.append(repaired_state)
        return repaired_states

    def _repair_state_until_partial_feasible(self, state):
        current_state = state
        removed_edges = list(current_state.get("repair_removed_edges", ()))
        while not bool(self.partial_feasibility_estimator.predict([current_state["graph"]])[0]):
            if current_state["graph"].number_of_edges() == 0:
                return current_state
            candidate = {
                "graph": current_state["graph"],
                "score": float(current_state.get("score") or 0.0),
                "selection_score": float(current_state.get("selection_score") or 0.0),
                "feasibility_stage": "partial",
            }
            self._annotate_infeasible_candidates_with_violating_edge_sets([candidate])
            next_removed_edges, repair_score = self._select_edges_for_surgical_repair(
                current_state,
                [candidate],
                rollback_steps=1,
            )
            if not next_removed_edges:
                next_removed_edges = self._fallback_repair_removed_edges(
                    current_state,
                    rollback_steps=1,
                )
                repair_score = 0.0
            if not next_removed_edges:
                return None
            score = float(current_state.get("selection_score") or 0.0) + float(repair_score)
            current_state = self._make_repaired_state(
                current_state,
                next_removed_edges,
                score=score,
            )
            removed_edges.extend(next_removed_edges)
            current_state["repair_removed_edges"] = tuple(removed_edges)
        return current_state

    def _fallback_repair_removed_edges(self, state, *, rollback_steps: int = 1):
        n_remove = max(0, min(int(rollback_steps), state["graph"].number_of_edges()))
        if n_remove < 1:
            return []
        edge_order = tuple(state.get("edge_order", ()))
        if edge_order:
            return list(edge_order[-n_remove:])
        graph_edges = self._canonical_graph_edges(state["graph"])
        if graph_edges:
            return list(graph_edges[-n_remove:])
        return []

    def _random_repair_removed_edge(self, state):
        graph_edges = list(self._canonical_graph_edges(state["graph"]))
        if not graph_edges:
            return []
        return [self.rng.choice(graph_edges)]

    def _select_pair_targets(self, query, selected_indices):
        if query["targets"] is None:
            return None
        return [query["targets"][idx] for idx in selected_indices]

    def _log_pair_training_context(
        self,
        pair_context,
        *,
        draw_graphs_fn: DrawGraphsFn | None,
        verbose: bool,
    ) -> None:
        if not verbose:
            return
        query = pair_context["query"]
        paths = pair_context["paths"]
        fit_graphs = pair_context["fit_graphs"]
        selected_indices = pair_context["selected_indices"]
        path_lengths = [len(path) for path in paths]
        print(
            f"[pair] source_idx={query['source_idx']} dest_idx={query['dest_idx']} "
            f"n_paths={len(paths)} selected_graphs={len(fit_graphs)} "
            f"path_k={pair_context['path_k']} "
            f"path_lengths={path_lengths} "
            f"training_set_time={self._format_minutes_seconds(pair_context['training_set_elapsed'])}"
        )
        print(f"[pair] selected_indices={selected_indices}")
        for path_idx, path in enumerate(paths, start=1):
            row_indices = list(path)
            row_graphs = [query["graphs"][idx] for idx in row_indices]
            row_titles = self._pair_graph_titles(
                query,
                row_indices,
                source_idx=row_indices[0],
                dest_idx=row_indices[-1],
            )
            print(f"[pair] path {path_idx}/{len(paths)} indices={path}")
            self._draw_graphs(
                draw_graphs_fn,
                row_graphs,
                n_graphs_per_line=min(len(row_graphs), 7),
                titles=row_titles,
            )
        print(f"[pair] training_set_indices={selected_indices}")
        self._draw_graphs(
            draw_graphs_fn,
            fit_graphs,
            n_graphs_per_line=min(len(fit_graphs), 7),
            titles=self._pair_graph_titles(query, selected_indices),
        )

    def _pair_graph_titles(self, query, indices, *, source_idx=None, dest_idx=None):
        titles = []
        for idx in indices:
            label = f"idx={idx}"
            if source_idx is not None and idx == source_idx:
                label = f"src\n{label}"
            elif dest_idx is not None and idx == dest_idx:
                label = f"dest\n{label}"
            target_value = (
                query["targets"][idx]
                if query["targets"] is not None and idx < len(query["targets"])
                else None
            )
            if target_value is not None:
                label = f"{label}\ntgt={target_value}"
            titles.append(label)
        return titles

    def _fit_pair_training_graphs(
        self,
        fit_graphs,
        fit_targets,
        *,
        deduplicate_feasibility_graphs: bool = True,
        partial_feasibility_extra_graphs=None,
    ) -> None:
        if fit_targets is not None and all(target_value is not None for target_value in fit_targets):
            self.fit(
                fit_graphs,
                targets=fit_targets,
                deduplicate_feasibility_graphs=deduplicate_feasibility_graphs,
                partial_feasibility_extra_graphs=partial_feasibility_extra_graphs,
            )
            return

        self.fit(
            fit_graphs,
            deduplicate_feasibility_graphs=deduplicate_feasibility_graphs,
            partial_feasibility_extra_graphs=partial_feasibility_extra_graphs,
        )
        if self.target_estimator is None or fit_targets is None:
            return
        labeled_pairs = [
            (graph, target_value)
            for graph, target_value in zip(fit_graphs, fit_targets)
            if target_value is not None
        ]
        if not labeled_pairs:
            return
        labeled_graphs, labeled_targets = zip(*labeled_pairs)
        self.fit_target_estimator(list(labeled_graphs), list(labeled_targets))

    def _resolve_pair_target(self, query, requested_target):
        if requested_target is not None:
            return requested_target
        return self._infer_pair_target(
            query["targets"][query["source_idx"]] if query["targets"] is not None else None,
            query["targets"][query["dest_idx"]] if query["targets"] is not None else None,
        )

    def _generate_one(
        self,
        graph: nx.Graph,
        n_edges: int,
        *,
        target=None,
        target_lambda: float = 1.0,
        draw_graphs_fn: DrawGraphsFn | None = None,
        verbose: bool = False,
        graph_index: int = 0,
    ):
        start_graph = graph.copy()
        self._reset_edge_risk_attempt_trace()
        if start_graph.number_of_edges() > n_edges:
            raise ValueError("Input graph already has more edges than n_edges")

        self.n_tried_ = 0
        self.max_depth_ = 0
        n_fallbacks = max(0, self.max_restarts)
        total_phases = n_fallbacks + 1
        max_total_edges = self._max_total_edges_for_generation(start_graph, n_edges)
        start_time = time.perf_counter()
        if verbose:
            remaining_edges = n_edges - start_graph.number_of_edges()
            start_parts = [
                f"[graph {graph_index}] start",
                f"start_edges={start_graph.number_of_edges()}",
                f"target_edges={n_edges}",
                f"remaining_edges={remaining_edges}",
            ]
            if target is not None:
                start_parts.append(f"target={target}")
                start_parts.append(f"target_lambda={target_lambda:.3f}")
            print(" ".join(start_parts))
            self._draw_graphs(draw_graphs_fn, [start_graph])

        if (
            start_graph.number_of_edges() == n_edges
            and self._is_terminal_solution_graph(start_graph, n_edges=n_edges)
        ):
            return self._finish_if_start_graph_is_solution(
                start_graph,
                verbose=verbose,
                graph_index=graph_index,
            )

        search = self._initialize_search_state(start_graph)
        beam_limit = self._beam_limit_for_fallback(search["fallback_index"])
        self.top_k_ = beam_limit
        if verbose and total_phases > 1:
            self._print_phase_banner(
                graph_index=graph_index,
                fallback_index=search["fallback_index"],
                total_phases=total_phases,
                beam_limit=beam_limit,
                n_fallbacks=n_fallbacks,
            )

        # Main search loop: expand, score, retain, then optionally repair/backtrack.
        while search["beam"]:
            expandable_beam = [
                state
                for state in search["beam"]
                if state["graph"].number_of_edges() < max_total_edges
            ]
            if not expandable_beam:
                self._mark_unexpandable_beam_as_completion_infeasible(search)
                break

            generated = self._expand_beam(expandable_beam)
            scored = self._score_generated_candidates(
                generated,
                n_edges=n_edges,
                max_total_edges=max_total_edges,
                target=target,
                target_lambda=target_lambda,
                fallback_index=search["fallback_index"],
            )
            early_stop_path = self._find_early_stop_in_beam(
                search["beam"],
                scored=scored,
                n_edges=n_edges,
                target=target,
                target_lambda=target_lambda,
                fallback_index=search["fallback_index"],
                graph_index=graph_index,
                total_phases=total_phases,
                start_time=start_time,
                verbose=verbose,
            )
            if early_stop_path is not None:
                return early_stop_path
            retained = self._retain_unseen_candidates(
                scored["feasible_candidates"],
                search=search,
                next_depth=search["depth"] + 1,
                beam_limit=beam_limit,
            )

            self._log_search_step(
                retained,
                scored,
                start_graph=start_graph,
                n_edges=n_edges,
                next_depth=search["depth"] + 1,
                target=target,
                target_lambda=target_lambda,
                graph_index=graph_index,
                total_phases=total_phases,
                fallback_index=search["fallback_index"],
                beam_limit=beam_limit,
                step_start_time=search["step_start_time"],
                draw_graphs_fn=draw_graphs_fn,
                verbose=verbose,
            )

            if retained:
                path = self._advance_search_with_retained(
                    retained,
                    search=search,
                    n_edges=n_edges,
                    graph_index=graph_index,
                    total_phases=total_phases,
                    start_time=start_time,
                    verbose=verbose,
                )
                if path is not None:
                    return path
                continue

            beam_limit = self._apply_search_fallback(
                search,
                infeasible_candidates=scored["infeasible_candidates"],
                n_fallbacks=n_fallbacks,
                total_phases=total_phases,
                graph_index=graph_index,
                verbose=verbose,
            )
            if beam_limit is None:
                break

        self._close_edge_risk_training_states(open_state_ids=set())
        raise ValueError("Could not generate a feasible graph with the requested number of edges")

    def _finish_if_start_graph_is_solution(self, start_graph, *, verbose: bool, graph_index: int):
        if not bool(self.final_feasibility_estimator.predict([start_graph])[0]):
            raise ValueError("Start graph does not satisfy the final feasibility estimator")
        if verbose:
            print(
                f"[graph {graph_index}] solved depth=0 max_depth=0 "
                f"edges={start_graph.number_of_edges()} remaining_edges=0 "
                f"tried=0 elapsed=0m 0.0s eta=0m 0.0s"
            )
        return [start_graph]

    def _graph_component_count(self, graph: nx.Graph) -> int:
        if graph.number_of_nodes() <= 0:
            return 0
        if nx.is_directed(graph):
            return int(nx.number_weakly_connected_components(graph))
        return int(nx.number_connected_components(graph))

    def _is_connectivity_satisfied(self, graph: nx.Graph) -> bool:
        if not self.require_single_connected_component:
            return True
        return self._graph_component_count(graph) <= 1

    def _is_terminal_solution_graph(self, graph: nx.Graph, *, n_edges: int) -> bool:
        if graph.number_of_edges() < int(n_edges):
            return False
        return self._is_connectivity_satisfied(graph)

    def _minimum_edges_needed_for_connectivity(self, graph: nx.Graph) -> int:
        if not self.require_single_connected_component:
            return 0
        return max(0, self._graph_component_count(graph) - 1)

    def _node_violation_sets(self, graphs) -> list[list[frozenset]]:
        estimator = self.final_feasibility_estimator
        if not hasattr(estimator, "violating_node_labels_sets"):
            return [[] for _ in graphs]
        return estimator.violating_node_labels_sets(graphs)

    def _minimum_node_violation_hitting_set_size(
        self,
        node_sets,
        *,
        max_size: int | None = None,
    ) -> int:
        node_sets = [frozenset(node_set) for node_set in node_sets if node_set]
        if not node_sets:
            return 0
        node_sets.sort(key=len)
        candidate_nodes = sorted(set().union(*node_sets), key=repr)
        max_exact_size = len(candidate_nodes) if max_size is None else min(int(max_size), len(candidate_nodes))
        for size in range(1, max_exact_size + 1):
            for candidate in combinations(candidate_nodes, size):
                candidate_set = set(candidate)
                if all(candidate_set.intersection(node_set) for node_set in node_sets):
                    return size
        if max_size is not None and max_exact_size < len(candidate_nodes):
            return int(max_size) + 1
        return len(candidate_nodes)

    def _minimum_edges_needed_for_node_violations(
        self,
        graph: nx.Graph,
        node_sets,
        *,
        max_total_edges: int,
    ) -> int:
        remaining_edge_budget = max(0, int(max_total_edges) - graph.number_of_edges())
        return self._minimum_node_violation_hitting_set_size(
            node_sets,
            max_size=remaining_edge_budget,
        )

    def _minimum_edges_needed_for_completion(
        self,
        graph: nx.Graph,
        node_sets,
        *,
        max_total_edges: int,
    ) -> int:
        return max(
            self._minimum_edges_needed_for_connectivity(graph),
            self._minimum_edges_needed_for_node_violations(
                graph,
                node_sets,
                max_total_edges=max_total_edges,
            ),
        )

    def _completion_slack(self, graph: nx.Graph, *, max_total_edges: int, node_sets=None) -> int:
        remaining_edge_budget = int(max_total_edges) - graph.number_of_edges()
        completion_edges_needed = self._minimum_edges_needed_for_completion(
            graph,
            [] if node_sets is None else node_sets,
            max_total_edges=max_total_edges,
        )
        return remaining_edge_budget - completion_edges_needed

    def _is_completion_possible(self, graph: nx.Graph, *, max_total_edges: int, node_sets=None) -> bool:
        return self._completion_slack(
            graph,
            max_total_edges=max_total_edges,
            node_sets=node_sets,
        ) >= 0

    def _has_final_feasible_completion_within_budget(
        self,
        graph: nx.Graph,
        *,
        n_edges: int,
        max_total_edges: int,
    ) -> bool | None:
        state_cap = int(self.max_terminal_completion_lookahead_states)
        if state_cap <= 0:
            return None
        if self.edge_attribute_templates_ is None:
            return None
        remaining_budget = int(max_total_edges) - graph.number_of_edges()
        if remaining_budget < 0:
            return False

        frontier = [graph.copy()]
        seen_hashes = {hash_graph(graph)}
        explored_states = 1
        for depth in range(remaining_budget + 1):
            terminal_graphs = [
                candidate
                for candidate in frontier
                if self._is_terminal_solution_graph(candidate, n_edges=n_edges)
            ]
            if terminal_graphs:
                final_mask = np.asarray(
                    self.final_feasibility_estimator.predict(terminal_graphs),
                    dtype=bool,
                )
                if bool(np.any(final_mask)):
                    return True

            if depth >= remaining_budget:
                break

            next_frontier = []
            for candidate in frontier:
                if candidate.number_of_edges() >= int(max_total_edges):
                    continue
                for edge in self._missing_edges(candidate):
                    for edge_attrs in self.edge_attribute_templates_:
                        completion_graph = candidate.copy()
                        completion_graph.add_edge(*edge, **edge_attrs)
                        graph_hash = hash_graph(completion_graph)
                        if graph_hash in seen_hashes:
                            continue
                        seen_hashes.add(graph_hash)
                        next_frontier.append(completion_graph)
                        explored_states += 1
                        if explored_states > state_cap:
                            return None

            if not next_frontier:
                break

            partial_mask = np.asarray(
                self.partial_feasibility_estimator.predict(next_frontier),
                dtype=bool,
            )
            frontier = [
                candidate
                for candidate, is_partial_feasible in zip(next_frontier, partial_mask)
                if is_partial_feasible
            ]
            if not frontier:
                break

        return False

    def _max_total_edges_for_generation(self, start_graph: nx.Graph, n_edges: int) -> int:
        base_edges = int(n_edges)
        if not self.require_single_connected_component:
            return base_edges
        start_components = self._graph_component_count(start_graph)
        if start_components <= 1:
            return base_edges
        extra_edges = max(0, start_components - 1)
        return base_edges + extra_edges

    def _initialize_search_state(self, start_graph):
        beam = [self._make_state(start_graph, parent=None, score=1.0, depth=0)]
        beam_history = [self._copy_beam(beam)]
        return {
            "beam": beam,
            "beam_history": beam_history,
            "blocked_state_keys_by_depth": {},
            "tabu_path_signatures": set(),
            "visited": self._rebuild_visited_from_history(beam_history),
            "depth": 0,
            "fallback_index": -1,
            "step_start_time": time.perf_counter(),
        }

    # Beam expansion and candidate scoring.

    def _expand_beam(self, beam):
        generated = []
        for state in beam:
            generated.extend(self._expand_state(state))
        self.n_tried_ += len(generated)
        return generated

    def _score_generated_candidates(
        self,
        generated,
        *,
        n_edges: int,
        max_total_edges: int,
        target,
        target_lambda: float,
        fallback_index: int,
    ):
        feasible_candidates = []
        infeasible_candidates = []
        repulsion_lambda = 0.0
        if generated:
            self._partition_candidates_by_feasibility(
                generated,
                n_edges=n_edges,
                max_total_edges=max_total_edges,
                target=target,
                target_lambda=target_lambda,
                feasible_candidates=feasible_candidates,
                infeasible_candidates=infeasible_candidates,
            )
            repulsion_lambda = self._rank_feasible_candidates(
                feasible_candidates,
                fallback_index=fallback_index,
            )
            self._rank_infeasible_candidates(infeasible_candidates)
        return {
            "generated": generated,
            "feasible_candidates": feasible_candidates,
            "infeasible_candidates": infeasible_candidates,
            "repulsion_lambda": repulsion_lambda,
        }

    def _partition_candidates_by_feasibility(
        self,
        generated,
        *,
        n_edges: int,
        max_total_edges: int,
        target,
        target_lambda: float,
        feasible_candidates,
        infeasible_candidates,
    ) -> None:
        generated_graphs = [cand["graph"] for cand in generated]
        partial_feasibility_mask = np.asarray(
            self.partial_feasibility_estimator.predict(generated_graphs),
            dtype=bool,
        )
        positive_scores = self._positive_scores(generated_graphs)
        target_scores = self._target_scores(generated_graphs, target=target)
        risk_scores = self._edge_risk_scores(generated)
        node_violation_sets = [[] for _ in generated]
        partial_feasible_indices = [
            idx
            for idx, is_partial_feasible in enumerate(partial_feasibility_mask)
            if is_partial_feasible
        ]
        if partial_feasible_indices:
            partial_feasible_graphs = [generated_graphs[idx] for idx in partial_feasible_indices]
            for idx, graph_node_sets in zip(
                partial_feasible_indices,
                self._node_violation_sets(partial_feasible_graphs),
            ):
                node_violation_sets[idx] = graph_node_sets
        partial_terminal_candidates = []
        for cand, is_partial_feasible, score, target_score, risk_score, cand_node_sets in zip(
            generated,
            partial_feasibility_mask,
            positive_scores,
            target_scores,
            risk_scores,
            node_violation_sets,
        ):
            cand["score"] = float(score)
            cand["target_score"] = float(target_score)
            cand["risk_score"] = float(risk_score)
            cand["selection_score"] = float(
                cand["score"]
                + target_lambda * cand["target_score"]
                - self.edge_risk_lambda * cand["risk_score"]
            )
            node_violation_completion_edges = self._minimum_edges_needed_for_node_violations(
                cand["graph"],
                cand_node_sets,
                max_total_edges=max_total_edges,
            )
            completion_edges_needed = max(
                self._minimum_edges_needed_for_connectivity(cand["graph"]),
                node_violation_completion_edges,
            )
            remaining_edge_budget = int(max_total_edges) - cand["graph"].number_of_edges()
            cand["node_violation_completion_edges"] = node_violation_completion_edges
            cand["completion_slack"] = remaining_edge_budget - completion_edges_needed
            terminal_completion_possible = None
            if (
                is_partial_feasible
                and cand["completion_slack"] >= 0
                and cand["graph"].number_of_edges() < int(n_edges)
            ):
                terminal_completion_possible = (
                    self._has_final_feasible_completion_within_budget(
                        cand["graph"],
                        n_edges=n_edges,
                        max_total_edges=max_total_edges,
                    )
                )
            if not is_partial_feasible:
                cand["feasibility_stage"] = "partial"
                self._mark_trace_state_status(cand, "partial_infeasible")
                infeasible_candidates.append(cand)
            elif cand["completion_slack"] < 0:
                cand["feasibility_stage"] = "completion"
                self._mark_trace_state_status(cand, "completion_infeasible")
                infeasible_candidates.append(cand)
            elif terminal_completion_possible is False:
                cand["feasibility_stage"] = "completion"
                cand["terminal_completion_infeasible"] = True
                self._mark_trace_state_status(cand, "completion_infeasible")
                infeasible_candidates.append(cand)
            elif terminal_completion_possible is None:
                cand["terminal_completion_unknown"] = True
                if cand["graph"].number_of_edges() >= n_edges:
                    partial_terminal_candidates.append(cand)
                else:
                    feasible_candidates.append(cand)
            else:
                if cand["graph"].number_of_edges() >= n_edges:
                    partial_terminal_candidates.append(cand)
                else:
                    feasible_candidates.append(cand)
        self._promote_final_feasible_candidates(
            partial_terminal_candidates,
            feasible_candidates=feasible_candidates,
            infeasible_candidates=infeasible_candidates,
        )

    def _promote_final_feasible_candidates(
        self,
        partial_terminal_candidates,
        *,
        feasible_candidates,
        infeasible_candidates,
    ) -> None:
        if not partial_terminal_candidates:
            return
        final_mask = np.asarray(
            self.final_feasibility_estimator.predict(
                [cand["graph"] for cand in partial_terminal_candidates]
            ),
            dtype=bool,
        )
        for cand, is_final_feasible in zip(partial_terminal_candidates, final_mask):
            if is_final_feasible:
                cand["connected_components"] = self._graph_component_count(cand["graph"])
                feasible_candidates.append(cand)
            else:
                cand["feasibility_stage"] = "final"
                self._mark_trace_state_status(cand, "final_infeasible")
                infeasible_candidates.append(cand)

    def _rank_feasible_candidates(self, feasible_candidates, *, fallback_index: int):
        if not feasible_candidates:
            return 0.0
        repulsions, repulsion_lambda = self._repulsion_values(
            [cand["graph"] for cand in feasible_candidates],
            fallback_index=fallback_index,
        )
        for cand, repulsion in zip(feasible_candidates, repulsions):
            cand["repulsion"] = float(repulsion)
            cand["selection_score"] = float(
                cand["selection_score"] - repulsion_lambda * cand["repulsion"]
            )
        feasible_candidates.sort(
            key=lambda cand: (
                cand["selection_score"],
                cand.get("completion_slack", 0),
            ),
            reverse=True,
        )
        return repulsion_lambda

    def _rank_infeasible_candidates(self, infeasible_candidates) -> None:
        if not infeasible_candidates:
            return
        self._annotate_infeasible_candidates_with_violations(infeasible_candidates)
        infeasible_candidates.sort(
            key=lambda cand: (
                cand.get("selection_score", cand["score"]),
                -cand.get("violation_count", 0.0),
            ),
            reverse=True,
        )

    def _retain_unseen_candidates(self, feasible_candidates, *, search, next_depth: int, beam_limit: int):
        blocked_state_keys = search["blocked_state_keys_by_depth"].get(next_depth, set())
        unseen_candidates = []
        for cand in feasible_candidates:
            state_key = cand["key"]
            if state_key in search["visited"] or state_key in blocked_state_keys:
                continue
            if cand["path_signature"] in search["tabu_path_signatures"]:
                continue
            if self.enforce_diversity and cand["graph_hash"] in self.diversity_memory_hash_set_:
                continue
            unseen_candidates.append(cand)
        retained = self._select_beam_candidates(unseen_candidates, beam_limit=beam_limit)
        retained_ids = {cand.get("state_id") for cand in retained}
        for cand in unseen_candidates:
            if cand.get("state_id") in retained_ids:
                self._mark_trace_state_status(cand, "retained")
            else:
                self._mark_trace_state_status(cand, "pruned")
        return retained

    # Search progress logging.

    def _log_search_step(
        self,
        retained,
        scored,
        *,
        start_graph,
        n_edges: int,
        next_depth: int,
        target,
        target_lambda: float,
        graph_index: int,
        total_phases: int,
        fallback_index: int,
        beam_limit: int,
        step_start_time: float,
        draw_graphs_fn: DrawGraphsFn | None,
        verbose: bool,
    ) -> None:
        if not verbose:
            return
        repulsion_lambda = scored["repulsion_lambda"]
        target_active = target is not None
        repulsion_active = repulsion_lambda > 0.0
        best_score = retained[0]["score"] if retained else None
        best_selection_score = (
            retained[0]["selection_score"] if retained else None
        ) if (target_active or repulsion_active) else None
        best_target_score = (
            retained[0].get("target_score") if retained and target_active else None
        )
        best_repulsion = retained[0].get("repulsion", 0.0) if retained else 0.0
        best_risk = retained[0].get("risk_score", 0.0) if retained else 0.0
        step_elapsed = time.perf_counter() - step_start_time
        current_edges = (
            retained[0]["graph"].number_of_edges()
            if retained
            else start_graph.number_of_edges() + next_depth
        )
        remaining_edges = max(0, n_edges - current_edges)
        eta_str = self._format_minutes_seconds(remaining_edges * step_elapsed)
        line1 = (
            f"[graph {graph_index}] fallback={fallback_index + 2}/{total_phases} "
            f"depth={next_depth} remaining_edges={remaining_edges} "
            f"step_time={self._format_minutes_seconds(step_elapsed)} eta={eta_str}"
        )
        partial_infeasible = sum(
            1
            for cand in scored.get("infeasible_candidates", [])
            if cand.get("feasibility_stage") == "partial"
        )
        completion_infeasible = sum(
            1
            for cand in scored.get("infeasible_candidates", [])
            if cand.get("feasibility_stage") == "completion"
        )
        terminal_completion_infeasible = sum(
            1
            for cand in scored.get("infeasible_candidates", [])
            if cand.get("terminal_completion_infeasible")
        )
        terminal_completion_unknown = sum(
            1
            for cand in scored.get("feasible_candidates", [])
            if cand.get("terminal_completion_unknown")
        )
        final_infeasible = sum(
            1
            for cand in scored.get("infeasible_candidates", [])
            if cand.get("feasibility_stage") == "final"
        )
        partial_feasible = max(0, len(scored["generated"]) - partial_infeasible)
        line2 = (
            f"generated={len(scored['generated'])} partial_feasible={partial_feasible} "
            f"viable={len(scored['feasible_candidates'])} retained={len(retained)} "
            f"tried={self.n_tried_}"
        )
        if partial_infeasible:
            line2 = f"{line2} partial_infeasible={partial_infeasible}"
        if completion_infeasible:
            line2 = f"{line2} completion_infeasible={completion_infeasible}"
        if terminal_completion_infeasible:
            line2 = (
                f"{line2} terminal_completion_infeasible={terminal_completion_infeasible}"
            )
        if terminal_completion_unknown:
            line2 = f"{line2} terminal_completion_unknown={terminal_completion_unknown}"
        if final_infeasible:
            line2 = f"{line2} final_infeasible={final_infeasible}"
        line3_parts = [f"best_score={self._format_optional_score(best_score)}"]
        if target_active:
            line3_parts.append(
                f"best_target_score={self._format_optional_score(best_target_score)}"
            )
        if target_active or repulsion_active:
            line3_parts.append(
                f"best_selection_score={self._format_optional_score(best_selection_score)}"
            )
        if repulsion_active:
            line3_parts.append(f"best_repulsion={best_repulsion:.3f}")
        if self.edge_risk_lambda > 0.0:
            line3_parts.append(f"best_risk={best_risk:.3f}")
        line4_parts = []
        if target_active:
            line4_parts.append(f"target_lambda={target_lambda:.3f}")
        if repulsion_active:
            line4_parts.append(f"repulsion_lambda={repulsion_lambda:.3f}")
        if self.edge_risk_lambda > 0.0 and self.edge_risk_estimator is not None:
            line4_parts.append(f"edge_risk_lambda={self.edge_risk_lambda:.3f}")
        line4_parts.append(f"beam_limit={beam_limit}")
        log_lines = [line1, line2, " ".join(line3_parts), " ".join(line4_parts)]
        if len(scored["feasible_candidates"]) == 0:
            remaining_fallbacks = max(0, total_phases - (fallback_index + 2))
            if remaining_fallbacks > 0:
                log_lines.append(
                    f"[graph {graph_index}] BACKTRACK no feasible candidates remain; "
                    f"{remaining_fallbacks} fallback phase(s) left"
                )
            else:
                log_lines.append(
                    f"[graph {graph_index}] FAILED no feasible candidates remain; "
                    "no fallback phases left"
                )
        print("\n".join(log_lines))
        self._draw_retained_candidates(
            retained,
            target_active=target_active,
            repulsion_active=repulsion_active,
            draw_graphs_fn=draw_graphs_fn,
        )

    def _format_optional_score(self, value):
        return f"{value:.3f}" if value is not None else "None"

    def _draw_retained_candidates(
        self,
        retained,
        *,
        target_active: bool,
        repulsion_active: bool,
        draw_graphs_fn: DrawGraphsFn | None,
    ) -> None:
        if not retained:
            return
        retained_graphs = [cand["graph"] for cand in retained]
        retained_titles = []
        for cand in retained:
            title_line1_parts = []
            title_line2_parts = [f"clf={cand['score']:.3f}"]
            if target_active or repulsion_active:
                title_line1_parts.append(
                    f"sel={cand.get('selection_score', cand['score']):.3f}"
                )
            if repulsion_active:
                title_line1_parts.append(f"rep={cand.get('repulsion', 0.0):.3f}")
            if self.edge_risk_lambda > 0.0:
                title_line1_parts.append(f"risk={cand.get('risk_score', 0.0):.3f}")
            if cand.get("completion_slack") is not None:
                title_line1_parts.append(f"slack={cand['completion_slack']}")
            if target_active:
                title_line2_parts.append(f"tgt={cand.get('target_score', 0.0):.3f}")
            if title_line1_parts:
                retained_titles.append(
                    " ".join(title_line1_parts) + "\n" + " ".join(title_line2_parts)
                )
            else:
                retained_titles.append(" ".join(title_line2_parts))
        self._draw_graphs(
            draw_graphs_fn,
            retained_graphs,
            n_graphs_per_line=min(len(retained_graphs), 7),
            titles=retained_titles,
        )

    # Beam retention and fallback transitions.

    def _advance_search_with_retained(
        self,
        retained,
        *,
        search,
        n_edges: int,
        graph_index: int,
        total_phases: int,
        start_time: float,
        verbose: bool,
    ):
        for cand in retained:
            search["visited"].add(cand["key"])
        search["depth"] += 1
        self.max_depth_ = max(self.max_depth_, search["depth"])
        search["beam"] = retained
        if len(search["beam_history"]) > search["depth"]:
            search["beam_history"][search["depth"]] = self._copy_beam(retained)
            del search["beam_history"][search["depth"] + 1 :]
        else:
            search["beam_history"].append(self._copy_beam(retained))
        search["step_start_time"] = time.perf_counter()
        return self._find_solution_in_beam(
            retained,
            n_edges=n_edges,
            graph_index=graph_index,
            total_phases=total_phases,
            fallback_index=search["fallback_index"],
            start_time=start_time,
            verbose=verbose,
        )

    def _find_solution_in_beam(
        self,
        beam,
        *,
        n_edges: int,
        graph_index: int,
        total_phases: int,
        fallback_index: int,
        start_time: float,
        verbose: bool,
    ):
        for state in beam:
            if not self._is_terminal_solution_graph(state["graph"], n_edges=n_edges):
                continue
            self._mark_trace_state_status(state, "solved")
            path = self._reconstruct_path(state)
            if verbose:
                elapsed_str = self._format_minutes_seconds(time.perf_counter() - start_time)
                current_edges = state["graph"].number_of_edges()
                edge_shortfall = max(0, n_edges - current_edges)
                print(
                    f"[graph {graph_index}] solved fallback={fallback_index + 2}/{total_phases} "
                    f"depth={state['depth']} max_depth={self.max_depth_} "
                    f"edges={current_edges} edge_shortfall={edge_shortfall} remaining_edges=0 "
                    f"tried={self.n_tried_} elapsed={elapsed_str} eta=0m 0.0s"
                )
            return path
        return None

    def _find_early_stop_in_beam(
        self,
        beam,
        *,
        scored,
        n_edges: int,
        target,
        target_lambda: float,
        fallback_index: int,
        graph_index: int,
        total_phases: int,
        start_time: float,
        verbose: bool,
    ):
        if not self.early_stop_if_final_feasible or not beam:
            return None

        current_states = self._score_current_beam_states_for_early_stop(
            beam,
            target=target,
            target_lambda=target_lambda,
            fallback_index=fallback_index,
        )
        if not current_states:
            return None

        best_current = current_states[0]
        best_expansion_score = None
        if scored["feasible_candidates"]:
            best_expansion_score = scored["feasible_candidates"][0].get(
                "selection_score",
                scored["feasible_candidates"][0]["score"],
            )

        current_score = best_current.get("selection_score", best_current["score"])
        if best_expansion_score is not None and current_score < best_expansion_score:
            return None

        self._mark_trace_state_status(best_current, "solved")
        path = self._reconstruct_path(best_current)
        if verbose:
            elapsed_str = self._format_minutes_seconds(time.perf_counter() - start_time)
            current_edges = best_current["graph"].number_of_edges()
            edge_shortfall = max(0, n_edges - current_edges)
            print(
                f"[graph {graph_index}] early_stop fallback={fallback_index + 2}/{total_phases} "
                f"depth={best_current['depth']} max_depth={self.max_depth_} "
                f"edges={current_edges} edge_shortfall={edge_shortfall} remaining_edges=0 "
                f"tried={self.n_tried_} elapsed={elapsed_str} eta=0m 0.0s"
            )
            print(
                f"[graph {graph_index}] early_stop_selection_score="
                f"{current_score:.3f} best_expansion_selection_score="
                f"{self._format_optional_score(best_expansion_score)}"
            )
        return path

    def _score_current_beam_states_for_early_stop(
        self,
        beam,
        *,
        target,
        target_lambda: float,
        fallback_index: int,
    ):
        beam_graphs = [state["graph"] for state in beam]
        final_mask = np.asarray(
            self.final_feasibility_estimator.predict(beam_graphs),
            dtype=bool,
        )
        if not np.any(final_mask):
            return []

        positive_scores = self._positive_scores(beam_graphs)
        target_scores = self._target_scores(beam_graphs, target=target)
        risk_scores = self._edge_risk_scores(beam)
        final_states = []
        for state, is_final_feasible, score, target_score, risk_score in zip(
            beam,
            final_mask,
            positive_scores,
            target_scores,
            risk_scores,
        ):
            if not is_final_feasible:
                continue
            if not self._is_connectivity_satisfied(state["graph"]):
                continue
            state["score"] = float(score)
            state["target_score"] = float(target_score)
            state["risk_score"] = float(risk_score)
            state["selection_score"] = float(
                state["score"]
                + target_lambda * state["target_score"]
                - self.edge_risk_lambda * state["risk_score"]
            )
            final_states.append(state)

        if not final_states:
            return []

        repulsions, repulsion_lambda = self._repulsion_values(
            [state["graph"] for state in final_states],
            fallback_index=fallback_index,
        )
        for state, repulsion in zip(final_states, repulsions):
            state["repulsion"] = float(repulsion)
            state["selection_score"] = float(
                state["selection_score"] - repulsion_lambda * state["repulsion"]
            )
        final_states.sort(
            key=lambda state: state.get("selection_score", state["score"]),
            reverse=True,
        )
        return final_states

    def _apply_search_fallback(
        self,
        search,
        *,
        infeasible_candidates,
        n_fallbacks: int,
        total_phases: int,
        graph_index: int,
        verbose: bool,
    ):
        self._mark_blocked_beam(search)
        if search["fallback_index"] + 1 >= n_fallbacks:
            self._close_edge_risk_training_states(open_state_ids=set())
            return None
        search["fallback_index"] += 1
        rollback_steps = self._rollback_steps_for_fallback(search["fallback_index"])
        beam_limit = self._beam_limit_for_fallback(search["fallback_index"])
        self.top_k_ = beam_limit
        repaired_beam = self._repair_beam_from_infeasible_candidates(
            search["beam"],
            infeasible_candidates,
            rollback_steps=rollback_steps,
            beam_limit=beam_limit,
        )
        if repaired_beam:
            self._restore_repaired_beam(
                repaired_beam,
                search=search,
                beam_limit=beam_limit,
                rollback_steps=rollback_steps,
                n_fallbacks=n_fallbacks,
                total_phases=total_phases,
                graph_index=graph_index,
                verbose=verbose,
            )
            return beam_limit
        self._rollback_search_without_repair(
            search,
            rollback_steps=rollback_steps,
            beam_limit=beam_limit,
            n_fallbacks=n_fallbacks,
            total_phases=total_phases,
            graph_index=graph_index,
            verbose=verbose,
        )
        return beam_limit

    def _mark_blocked_beam(self, search) -> None:
        search["blocked_state_keys_by_depth"].setdefault(search["depth"], set()).update(
            state["key"] for state in search["beam"]
        )
        search["tabu_path_signatures"].update(
            state["path_signature"] for state in search["beam"]
        )
        for state in search["beam"]:
            self._mark_trace_state_status(state, "blocked")
        self._remember_failed_graphs([state["graph"] for state in search["beam"]])

    def _mark_unexpandable_beam_as_completion_infeasible(self, search) -> None:
        for state in search["beam"]:
            if self._is_connectivity_satisfied(state["graph"]):
                self._mark_trace_state_status(state, "blocked")
            else:
                self._mark_trace_state_status(state, "completion_infeasible")
        self._remember_failed_graphs([state["graph"] for state in search["beam"]])

    def _restore_repaired_beam(
        self,
        repaired_beam,
        *,
        search,
        beam_limit: int,
        rollback_steps: int,
        n_fallbacks: int,
        total_phases: int,
        graph_index: int,
        verbose: bool,
    ) -> None:
        repaired_depth = repaired_beam[0]["depth"]
        search["beam_history"] = search["beam_history"][: repaired_depth + 1]
        search["beam_history"][repaired_depth] = self._copy_beam(repaired_beam)
        search["beam"] = repaired_beam
        search["depth"] = repaired_depth
        search["visited"] = self._rebuild_visited_from_history(search["beam_history"])
        search["step_start_time"] = time.perf_counter()
        self._close_edge_risk_training_states(
            open_state_ids=self._trace_open_state_ids(search["beam"])
        )
        if verbose:
            edge_risk_training_size = self._edge_risk_training_set_size()
            edge_risk_fit_time = self._edge_risk_last_fit_time_seconds()
            removed_descriptions = [
                ",".join(str(edge) for edge in state.get("repair_removed_edges", ()))
                for state in repaired_beam
            ]
            fallback_parts = [
                f"[graph {graph_index}] fallback={search['fallback_index'] + 1}/{n_fallbacks}",
                f"rollback_steps={rollback_steps}",
                f"surgical_repairs={len(repaired_beam)}",
                f"to_depth={repaired_depth}",
                f"beam_limit={beam_limit}",
            ]
            if edge_risk_training_size is not None:
                fallback_parts.append(
                    f"edge_risk_training_set_size={edge_risk_training_size}"
                )
            if edge_risk_fit_time is not None:
                fallback_parts.append(
                    f"edge_risk_fit_time={self._format_minutes_seconds(edge_risk_fit_time)}"
                )
            print(" ".join(fallback_parts))
            print(f"[graph {graph_index}] surgical_removed_edges={removed_descriptions}")
        if verbose and total_phases > 1:
            self._print_phase_banner(
                graph_index=graph_index,
                fallback_index=search["fallback_index"],
                total_phases=total_phases,
                beam_limit=beam_limit,
                n_fallbacks=n_fallbacks,
            )

    def _rollback_search_without_repair(
        self,
        search,
        *,
        rollback_steps: int,
        beam_limit: int,
        n_fallbacks: int,
        total_phases: int,
        graph_index: int,
        verbose: bool,
    ) -> None:
        fallback_depth = max(0, search["depth"] - rollback_steps)
        search["beam_history"] = search["beam_history"][: fallback_depth + 1]
        search["beam"] = self._copy_beam(search["beam_history"][fallback_depth])
        search["depth"] = fallback_depth
        search["visited"] = self._rebuild_visited_from_history(search["beam_history"])
        search["step_start_time"] = time.perf_counter()
        self._close_edge_risk_training_states(
            open_state_ids=self._trace_open_state_ids(search["beam"])
        )
        if verbose:
            fallback_parts = [
                f"[graph {graph_index}] fallback={search['fallback_index'] + 1}/{n_fallbacks}",
                f"rollback_steps={rollback_steps}",
                f"to_depth={fallback_depth}",
                f"beam_limit={beam_limit}",
            ]
            edge_risk_training_size = self._edge_risk_training_set_size()
            edge_risk_fit_time = self._edge_risk_last_fit_time_seconds()
            if edge_risk_training_size is not None:
                fallback_parts.append(
                    f"edge_risk_training_set_size={edge_risk_training_size}"
                )
            if edge_risk_fit_time is not None:
                fallback_parts.append(
                    f"edge_risk_fit_time={self._format_minutes_seconds(edge_risk_fit_time)}"
                )
            print(" ".join(fallback_parts))
        if verbose and total_phases > 1:
            self._print_phase_banner(
                graph_index=graph_index,
                fallback_index=search["fallback_index"],
                total_phases=total_phases,
                beam_limit=beam_limit,
                n_fallbacks=n_fallbacks,
            )

    def _print_phase_banner(
        self,
        *,
        graph_index: int,
        fallback_index: int,
        total_phases: int,
        beam_limit: int,
        n_fallbacks: int,
    ) -> None:
        print(
            f"[graph {graph_index}] fallback={fallback_index + 2}/{total_phases} "
            f"beam_limit={beam_limit}"
        )

    def _edge_risk_training_set_size(self) -> int | None:
        if self.edge_risk_model_ is None or self.edge_risk_estimator is None:
            return None
        return self.edge_risk_model_.training_set_size()

    def _edge_risk_last_fit_time_seconds(self) -> float | None:
        if self.edge_risk_model_ is None or self.edge_risk_estimator is None:
            return None
        return self.edge_risk_model_.last_fit_time_seconds()

    def _expand_state(self, state):
        candidates = []
        graph = state["graph"]
        for edge in self._missing_edges(graph):
            for edge_attrs in self.edge_attribute_templates_:
                candidate_graph = graph.copy()
                candidate_graph.add_edge(*edge, **edge_attrs)
                candidates.append(
                    self._make_state(
                        candidate_graph,
                        parent=state,
                        score=None,
                        added_edge=edge,
                        depth=state["depth"] + 1,
                    )
                )
        self.rng.shuffle(candidates)
        return candidates

    def _make_state(self, graph, parent, score, *, added_edge=None, depth: int | None = None, edge_order=None):
        state_key = self._state_key(graph)
        graph_hash = hash_graph(graph)
        if parent is None:
            path_signature = (state_key,)
        else:
            path_signature = parent["path_signature"] + (state_key,)
        if edge_order is None:
            if parent is not None and added_edge is not None:
                edge_order = parent["edge_order"] + (
                    _canonicalize_edge(added_edge, graph),
                )
            else:
                edge_order = self._canonical_graph_edges(graph)
        if depth is None:
            depth = 0 if parent is None else parent.get("depth", 0)
            if added_edge is not None:
                depth += 1
        state = {
            "graph": graph,
            "graph_hash": graph_hash,
            "parent": parent,
            "score": score,
            "selection_score": score,
            "key": state_key,
            "path_signature": path_signature,
            "added_edge": None if added_edge is None else _canonicalize_edge(added_edge, graph),
            "repair_removed_edges": (),
            "edge_order": tuple(edge_order),
            "depth": int(depth),
        }
        self._register_trace_state(state)
        return state

    def _repair_beam_from_infeasible_candidates(
        self,
        beam,
        infeasible_candidates,
        *,
        rollback_steps: int,
        beam_limit: int,
    ):
        if not self.surgical_backtracking_ or rollback_steps < 1 or not infeasible_candidates:
            return []

        selected_candidates = self._select_infeasible_candidates_for_repair(
            beam,
            infeasible_candidates,
            beam_limit=beam_limit,
        )
        if not selected_candidates:
            return []

        self._annotate_infeasible_candidates_with_violating_edge_sets(selected_candidates)

        grouped_candidates: dict[tuple, list] = {}
        for cand in selected_candidates:
            grouped_candidates.setdefault(cand["parent"]["key"], []).append(cand)

        repaired_states = []
        for state in beam:
            state_candidates = grouped_candidates.get(state["key"], [])
            removed_edges, repair_score = self._select_edges_for_surgical_repair(
                state,
                state_candidates,
                rollback_steps=rollback_steps,
            )
            if not removed_edges:
                continue
            repaired_states.append(
                self._make_repaired_state(
                    state,
                    removed_edges,
                    score=repair_score,
                )
            )

        if not repaired_states:
            return []

        repaired_states.sort(
            key=lambda state: state.get("selection_score", 0.0),
            reverse=True,
        )
        repaired_states = self._deduplicate_retained_candidates(
            repaired_states,
            fallback_candidates=[],
            target_size=min(beam_limit, len(repaired_states)),
        )
        return repaired_states

    def _select_infeasible_candidates_for_repair(self, beam, infeasible_candidates, *, beam_limit: int):
        if not infeasible_candidates:
            return []
        by_parent = {state["key"]: [] for state in beam}
        for cand in infeasible_candidates:
            parent = cand.get("parent")
            if parent is None:
                continue
            if parent["key"] not in by_parent:
                continue
            by_parent[parent["key"]].append(cand)
        per_parent_limit = max(1, beam_limit)
        selected = []
        for state in beam:
            state_candidates = by_parent.get(state["key"], [])
            state_candidates.sort(
                key=lambda cand: (
                    cand.get("selection_score", cand.get("score", 0.0)),
                    -cand.get("violation_count", 0.0),
                ),
                reverse=True,
            )
            selected.extend(state_candidates[:per_parent_limit])
        return selected

    def _feasibility_estimator_for_stage(self, stage: str):
        if stage == "final":
            return self.final_feasibility_estimator
        return self.partial_feasibility_estimator

    def _annotate_infeasible_candidates_with_violations(self, candidates) -> None:
        if not candidates:
            return
        grouped_candidates = {}
        for cand in candidates:
            stage = cand.get("feasibility_stage", "partial")
            grouped_candidates.setdefault(stage, []).append(cand)
        for stage, stage_candidates in grouped_candidates.items():
            estimator = self._feasibility_estimator_for_stage(stage)
            violation_counts = np.asarray(
                estimator.number_of_violations(
                    [cand["graph"] for cand in stage_candidates]
                ),
                dtype=float,
            ).reshape(-1)
            for cand, violation_count in zip(stage_candidates, violation_counts):
                cand["violation_count"] = float(violation_count)

    def _annotate_infeasible_candidates_with_violating_edge_sets(self, candidates) -> None:
        if not candidates:
            return
        grouped_candidates = {}
        for cand in candidates:
            stage = cand.get("feasibility_stage", "partial")
            grouped_candidates.setdefault(stage, []).append(cand)
        for stage, stage_candidates in grouped_candidates.items():
            estimator = self._feasibility_estimator_for_stage(stage)
            if not hasattr(estimator, "violating_edge_sets"):
                for cand in stage_candidates:
                    cand["violating_edge_sets"] = []
                continue
            violating_edge_sets = estimator.violating_edge_sets(
                [cand["graph"] for cand in stage_candidates]
            )
            for cand, edge_sets in zip(stage_candidates, violating_edge_sets):
                cand["violating_edge_sets"] = edge_sets

    def _select_edges_for_surgical_repair(self, state, candidates, *, rollback_steps: int):
        if rollback_steps < 1:
            return [], 0.0

        parent_edges = set(self._canonical_graph_edges(state["graph"]))
        if not parent_edges:
            return [], 0.0

        edge_counts = {}
        edge_weights = {}
        recency = self._edge_recency_map(state)
        for cand in candidates:
            candidate_score = max(
                0.0,
                float(cand.get("selection_score", cand.get("score", 0.0))),
            )
            for edge_set in cand.get("violating_edge_sets", []):
                relevant_edges = [edge for edge in edge_set if edge in parent_edges]
                if not relevant_edges:
                    continue
                increment = 1.0 / float(len(relevant_edges))
                for edge in relevant_edges:
                    edge_counts[edge] = edge_counts.get(edge, 0.0) + 1.0
                    edge_weights[edge] = edge_weights.get(edge, 0.0) + candidate_score * increment

        ranked_edges = sorted(
            parent_edges,
            key=lambda edge: (
                edge_counts.get(edge, 0.0),
                edge_weights.get(edge, 0.0),
                recency.get(edge, 0),
            ),
            reverse=True,
        )
        if not any(edge_counts.get(edge, 0.0) > 0.0 for edge in ranked_edges):
            return [], 0.0
        selected = [
            edge for edge in ranked_edges if edge_counts.get(edge, 0.0) > 0.0
        ][:rollback_steps]

        if len(selected) < min(rollback_steps, len(parent_edges)):
            for edge in sorted(parent_edges, key=lambda edge: recency.get(edge, 0), reverse=True):
                if edge in selected:
                    continue
                selected.append(edge)
                if len(selected) >= min(rollback_steps, len(parent_edges)):
                    break

        repair_score = float(
            sum(edge_counts.get(edge, 0.0) for edge in selected)
            + sum(edge_weights.get(edge, 0.0) for edge in selected)
        )
        return selected, repair_score

    def _make_repaired_state(self, state, removed_edges, *, score: float):
        repaired_graph = state["graph"].copy()
        for edge in removed_edges:
            if repaired_graph.has_edge(*edge):
                repaired_graph.remove_edge(*edge)
        removed_edge_set = set(removed_edges)
        repaired_edge_order = tuple(
            edge for edge in state["edge_order"] if edge not in removed_edge_set
        )
        repaired_state = self._make_state(
            repaired_graph,
            parent=state,
            score=score,
            depth=max(0, state["depth"] - len(removed_edges)),
            edge_order=repaired_edge_order,
        )
        repaired_state["repair_removed_edges"] = tuple(removed_edges)
        repaired_state["selection_score"] = float(score)
        return repaired_state

    def _select_beam_candidates(self, candidates, *, beam_limit: int | None = None):
        if not candidates:
            return []

        current_beam_limit = self.top_k_ if beam_limit is None else beam_limit
        n_keep = min(current_beam_limit, len(candidates))
        n_top = 1 if n_keep == 1 else max(1, n_keep // 2)
        n_random = n_keep - n_top

        top_candidates = candidates[:n_top]
        remaining_candidates = candidates[n_top:]
        if len(remaining_candidates) <= n_random:
            random_candidates = remaining_candidates
        else:
            random_candidates = self.rng.sample(remaining_candidates, k=n_random)

        retained = top_candidates + random_candidates
        retained = self._deduplicate_retained_candidates(
            retained,
            fallback_candidates=remaining_candidates,
            target_size=n_keep,
        )
        retained.sort(
            key=lambda cand: cand.get("selection_score", cand["score"]),
            reverse=True,
        )
        return retained

    def _deduplicate_retained_candidates(
        self,
        retained,
        *,
        fallback_candidates,
        target_size: int,
    ):
        unique_retained = []
        seen_hashes = set()

        def add_candidate(candidate):
            graph_hash = candidate.get("graph_hash", hash_graph(candidate["graph"]))
            if graph_hash in seen_hashes:
                return False
            seen_hashes.add(graph_hash)
            unique_retained.append(candidate)
            return True

        for candidate in retained:
            add_candidate(candidate)

        if len(unique_retained) >= target_size:
            return unique_retained[:target_size]

        for candidate in fallback_candidates:
            if len(unique_retained) >= target_size:
                break
            add_candidate(candidate)

        return unique_retained

    def _require_stored_dataset(self):
        if self.stored_graphs_ is None or self.stored_distance_matrix_ is None:
            raise ValueError("Call store(graphs, targets=...) before generate_from_pair")

    def _make_retrieval_transformer(self):
        transformer = getattr(self.graph_estimator, "transformer", None)
        if transformer is None:
            raise ValueError("graph_estimator must expose a transformer for store(...)")
        return copy.deepcopy(transformer)

    def _vectorize_retrieval_graphs(self, transformer, graphs, *, fit: bool):
        if fit and hasattr(transformer, "fit_transform"):
            features = transformer.fit_transform(graphs)
        elif fit and hasattr(transformer, "fit") and hasattr(transformer, "transform"):
            transformer.fit(graphs)
            features = transformer.transform(graphs)
        elif hasattr(transformer, "transform"):
            features = transformer.transform(graphs)
        else:
            raise ValueError(
                "retrieval transformer must provide fit_transform(...) or transform(...)"
            )
        if hasattr(features, "toarray"):
            features = features.toarray()
        return np.asarray(features, dtype=float)

    def _build_pair_query_corpus(self, graph_a, graph_b):
        graphs = [graph.copy() for graph in self.stored_graphs_]
        targets = None if self.stored_targets_ is None else list(self.stored_targets_)
        vectors = np.asarray(self.stored_retrieval_vectors_, dtype=float)
        distance_matrix = np.asarray(self.stored_distance_matrix_, dtype=float).copy()

        source_idx, graphs, targets, vectors, distance_matrix = self._resolve_or_append_query_graph(
            graph_a,
            graphs=graphs,
            targets=targets,
            vectors=vectors,
            distance_matrix=distance_matrix,
        )
        dest_idx, graphs, targets, vectors, distance_matrix = self._resolve_or_append_query_graph(
            graph_b,
            graphs=graphs,
            targets=targets,
            vectors=vectors,
            distance_matrix=distance_matrix,
        )
        if source_idx == dest_idx:
            raise ValueError("graph_a and graph_b resolve to the same stored/query graph")
        return {
            "graphs": graphs,
            "targets": targets,
            "vectors": vectors,
            "distance_matrix": distance_matrix,
            "source_idx": source_idx,
            "dest_idx": dest_idx,
        }

    def _resolve_or_append_query_graph(
        self,
        graph,
        *,
        graphs,
        targets,
        vectors,
        distance_matrix,
    ):
        graph_hash = hash_graph(graph)
        stored_idx = self.stored_graph_hash_to_index_.get(graph_hash)
        if stored_idx is not None:
            return stored_idx, graphs, targets, vectors, distance_matrix

        graph_copy = graph.copy()
        query_vector = self._vectorize_retrieval_graphs(
            self.retrieval_transformer_,
            [graph_copy],
            fit=False,
        )[0]
        idx = len(graphs)
        graphs.append(graph_copy)
        if targets is not None:
            targets.append(None)
        distance_matrix = self._append_distance_row(distance_matrix, vectors, query_vector)
        vectors = np.vstack([vectors, query_vector])
        return idx, graphs, targets, vectors, distance_matrix

    def _append_distance_row(self, distance_matrix, vectors, query_vector):
        if vectors.size == 0:
            return np.zeros((1, 1), dtype=float)
        distances = pairwise_distances(
            np.asarray(query_vector, dtype=float).reshape(1, -1),
            np.asarray(vectors, dtype=float),
        ).ravel()
        old_n = distance_matrix.shape[0]
        expanded = np.zeros((old_n + 1, old_n + 1), dtype=float)
        expanded[:old_n, :old_n] = distance_matrix
        expanded[-1, :-1] = distances
        expanded[:-1, -1] = distances
        expanded[-1, -1] = 0.0
        return expanded

    def _path_matrix_from_distance_matrix(self, distance_matrix, *, k: int):
        if k < 0:
            raise ValueError("path_k must be >= 0")
        distances = np.asarray(distance_matrix, dtype=float)
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError("distance_matrix must be a square matrix")
        if distances.shape[0] <= 1:
            return distances.copy()
        adjacency = _build_adjacency(distances, k=max(1, int(k)), degree_penalty=0.0)
        path_matrix = adjacency.toarray().astype(float, copy=False)
        missing_mask = path_matrix == 0.0
        np.fill_diagonal(missing_mask, False)
        path_matrix[missing_mask] = np.inf
        np.fill_diagonal(path_matrix, 0.0)
        return path_matrix

    def _shortest_paths_from_matrix(self, distance_matrix, source_idx, dest_idx, *, n_paths: int):
        if n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        working_matrix = np.asarray(distance_matrix, dtype=float).copy()
        paths = []
        for _ in range(n_paths):
            path = self._dense_shortest_path(working_matrix, source_idx, dest_idx)
            if not path:
                break
            paths.append(path)
            self._remove_path_edges_from_matrix(working_matrix, path)
        return paths

    def _dense_shortest_path(self, distance_matrix: np.ndarray, source_idx: int, dest_idx: int):
        n_nodes = distance_matrix.shape[0]
        inf = float("inf")
        dist = np.full(n_nodes, inf, dtype=float)
        prev = np.full(n_nodes, -1, dtype=int)
        visited = np.zeros(n_nodes, dtype=bool)
        dist[source_idx] = 0.0

        for _ in range(n_nodes):
            current = -1
            current_dist = inf
            for node_idx in range(n_nodes):
                if visited[node_idx]:
                    continue
                if dist[node_idx] < current_dist:
                    current = node_idx
                    current_dist = dist[node_idx]
            if current < 0 or not np.isfinite(current_dist):
                break
            if current == dest_idx:
                break
            visited[current] = True

            row = distance_matrix[current]
            for neighbor_idx, weight in enumerate(row):
                if visited[neighbor_idx] or neighbor_idx == current:
                    continue
                if not np.isfinite(weight):
                    continue
                candidate_dist = current_dist + weight
                if candidate_dist < dist[neighbor_idx]:
                    dist[neighbor_idx] = candidate_dist
                    prev[neighbor_idx] = current

        if not np.isfinite(dist[dest_idx]):
            return []

        path = [dest_idx]
        cursor = dest_idx
        while cursor != source_idx:
            cursor = prev[cursor]
            if cursor < 0:
                return []
            path.append(int(cursor))
        path.reverse()
        return path

    def _remove_path_edges_from_matrix(self, distance_matrix: np.ndarray, path):
        for left, right in zip(path[:-1], path[1:]):
            distance_matrix[left, right] = np.inf
            distance_matrix[right, left] = np.inf

    def _infer_pair_target(self, target_a, target_b):
        if target_a is None or target_b is None:
            return None
        if not isinstance(target_a, (int, float, np.integer, np.floating)):
            return None
        if not isinstance(target_b, (int, float, np.integer, np.floating)):
            return None
        if self.target_estimator_mode == "regression":
            return 0.5 * (float(target_a) + float(target_b))
        return self.rng.choice([target_a, target_b])

    def _build_fragment_datasets(self, graphs):
        dataset_seeds = [self.rng.randrange(10**9) for _ in graphs]
        dataset_builder = self._dataset_builder()
        return Parallel(n_jobs=self.fit_n_jobs, backend=self.fit_backend)(
            delayed(dataset_builder)(
                graph,
                **self._dataset_builder_kwargs(dataset_seed),
            )
            for graph, dataset_seed in zip(graphs, dataset_seeds)
        )

    def _dataset_builder(self):
        if self.decomposition_function is None:
            return make_edge_regression_dataset
        return make_edge_regression_dataset_subgraph_ordered

    def _dataset_builder_kwargs(self, dataset_seed: int):
        kwargs = dict(
            n_negative_per_positive=self.n_negative_per_positive,
            n_replicates=self.n_replicates,
            seed=dataset_seed,
            allow_self_loops=self.allow_self_loops,
        )
        if self.decomposition_function is not None:
            kwargs["decomposition_function"] = self.decomposition_function
            kwargs["nbits"] = self._decomposition_nbits()
        return kwargs

    def _decomposition_nbits(self) -> int:
        transformer = getattr(self.graph_estimator, "transformer", None)
        nbits = getattr(transformer, "nbits", None)
        if nbits is None:
            raise ValueError(
                "decomposition_function requires graph_estimator.transformer.nbits"
            )
        return int(nbits)

    def _store_graph_estimator_training_data(self, dataset_parts):
        self.positives_ = []
        self.negatives_ = []
        self.dataset_ = []
        for positives, negatives, dataset in dataset_parts:
            self.positives_.extend(positives)
            self.negatives_.extend(negatives)
            self.dataset_.extend(dataset)

    def _build_target_fragment_dataset(self, dataset_parts, targets):
        target_graphs = []
        target_values = []
        for target_value, (positives, _, _) in zip(targets, dataset_parts):
            target_graphs.extend(positives)
            target_values.extend([target_value] * len(positives))
        return target_graphs, target_values

    def _fit_target_estimator_from_fragments(
        self,
        target_graphs,
        target_values,
        *,
        verbose_prefix: str,
    ):
        if self.target_estimator is None:
            raise ValueError("target_estimator is None; provide one before fitting")
        self.target_graphs_ = list(target_graphs)
        self.target_values_ = list(target_values)
        target_fit_start = time.perf_counter()
        self.target_estimator.fit(self.target_graphs_, self.target_values_)
        target_fit_time = time.perf_counter() - target_fit_start
        if self.verbose:
            target_fit_min = int(target_fit_time // 60)
            target_fit_sec = target_fit_time - 60 * target_fit_min
            target_label = (
                "target_classes"
                if self.target_estimator_mode == "classification"
                else "target_values"
            )
            print(
                f"{verbose_prefix} target_estimator_graphs={len(self.target_graphs_)} "
                f"{target_label}={len(set(self.target_values_))} "
                f"mode={self.target_estimator_mode} "
                f"time={target_fit_min}m {target_fit_sec:.1f}s"
            )

    def _rollback_steps_for_fallback(self, fallback_index: int) -> int:
        steps = self.fallback_base_steps * (self.fallback_growth_factor**fallback_index)
        return max(1, int(math.ceil(steps)))

    def _beam_limit_for_fallback(self, fallback_index: int) -> int:
        if fallback_index < 0:
            beam_limit = self.beam_size
        else:
            beam_limit = int(
                math.ceil(
                    self.beam_size * (self.beam_growth_factor ** (fallback_index + 1))
                )
            )
        beam_limit = max(1, beam_limit)
        if self.max_beam_size is not None:
            beam_limit = min(beam_limit, self.max_beam_size)
        return beam_limit

    def _copy_beam(self, beam):
        return list(beam)

    def _rebuild_visited_from_history(self, beam_history):
        visited = set()
        for beam in beam_history:
            for state in beam:
                visited.add(state["key"])
        return visited

    def _format_minutes_seconds(self, elapsed: float) -> str:
        elapsed_min = int(elapsed // 60)
        elapsed_sec = elapsed - 60 * elapsed_min
        return f"{elapsed_min}m {elapsed_sec:.1f}s"

    def _repulsion_values(self, graphs, *, fallback_index: int):
        memory_hashes = self._repulsion_memory_hashes(fallback_index)
        if not memory_hashes or self.repulsion_weight <= 0:
            return np.zeros(len(graphs), dtype=float), 0.0
        candidate_embeddings = self._graph_embeddings(graphs)
        memory_embeddings = self._graph_embeddings_from_hashes(memory_hashes)
        candidate_norm = self._normalize_rows(candidate_embeddings)
        memory_norm = self._normalize_rows(memory_embeddings)
        similarities = candidate_norm @ memory_norm.T
        repulsion = np.maximum(0.0, np.max(similarities, axis=1))
        if fallback_index < 0:
            lam = float(self.repulsion_weight)
        else:
            lam = float(
                self.repulsion_weight
                * (self.repulsion_growth_factor ** max(0, fallback_index))
            )
        return repulsion, lam

    def _repulsion_memory_hashes(self, fallback_index: int):
        memory_hashes = []
        if self.enforce_diversity and self.diversity_memory_hashes_:
            memory_hashes.extend(self.diversity_memory_hashes_)
        if (
            self.use_similarity_repulsion
            and fallback_index >= 0
            and self.failed_memory_hashes_
        ):
            memory_hashes.extend(self.failed_memory_hashes_)
        deduped_hashes = []
        seen_hashes = set()
        for graph_hash in memory_hashes:
            if graph_hash in seen_hashes:
                continue
            seen_hashes.add(graph_hash)
            deduped_hashes.append(graph_hash)
        return deduped_hashes

    def _remember_failed_graphs(self, graphs) -> None:
        if not self.use_similarity_repulsion or self.max_repulsion_memory <= 0:
            return
        graph_hashes = self._graph_hashes(graphs)
        self._graph_embeddings(graphs)
        for graph_hash in graph_hashes:
            if graph_hash in self.failed_memory_hash_set_:
                continue
            self.failed_memory_hashes_.append(graph_hash)
            self.failed_memory_hash_set_.add(graph_hash)
        while len(self.failed_memory_hashes_) > self.max_repulsion_memory:
            old_hash = self.failed_memory_hashes_.pop(0)
            self.failed_memory_hash_set_.discard(old_hash)

    def _graph_hashes(self, graphs):
        return [hash_graph(graph) for graph in graphs]

    def _unique_graphs(self, graphs):
        unique_graphs = []
        seen_keys = set()
        for graph in graphs:
            graph_key = self._wl_graph_key(graph)
            if graph_key in seen_keys:
                continue
            seen_keys.add(graph_key)
            unique_graphs.append(graph)
        return unique_graphs

    def _wl_graph_key(self, graph: nx.Graph):
        node_attr = "_abstractgraph_wl_node_label"
        edge_attr = "_abstractgraph_wl_edge_label"
        wl_graph = graph.__class__()
        for node, attrs in graph.nodes(data=True):
            wl_graph.add_node(
                node,
                **{node_attr: canonical_bytes(dict(attrs)).decode("utf-8")},
            )
        for u, v, attrs in graph.edges(data=True):
            wl_graph.add_edge(
                u,
                v,
                **{edge_attr: canonical_bytes(dict(attrs)).decode("utf-8")},
            )
        return (
            "directed" if nx.is_directed(graph) else "undirected",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            weisfeiler_lehman_graph_hash(
                wl_graph,
                node_attr=node_attr,
                edge_attr=edge_attr,
            ),
        )

    def _graph_embeddings(self, graphs):
        graph_hashes = self._graph_hashes(graphs)
        return self._graph_embeddings_from_hashes(graph_hashes, graphs=graphs)

    def _graph_embeddings_from_hashes(self, graph_hashes, *, graphs=None):
        graph_hashes = list(graph_hashes)
        missing_graphs = []
        missing_hashes = []
        if graphs is not None:
            for graph, graph_hash in zip(graphs, graph_hashes):
                if graph_hash in self.embedding_cache_:
                    continue
                missing_graphs.append(graph)
                missing_hashes.append(graph_hash)
        if missing_graphs:
            raw_features = self.graph_estimator._transform_raw(missing_graphs)
            rows = self._matrix_to_rows(raw_features)
            for graph_hash, row in zip(missing_hashes, rows):
                self.embedding_cache_[graph_hash] = row
        if not graph_hashes:
            return np.empty((0, 0), dtype=float)
        return np.vstack([self.embedding_cache_[graph_hash] for graph_hash in graph_hashes])

    def _initialize_diversity_memory(self, graphs):
        self.diversity_memory_hashes_ = []
        self.diversity_memory_hash_set_ = set()
        if not self.enforce_diversity:
            return
        diversity_hashes = self._graph_hashes(graphs)
        self._graph_embeddings(graphs)
        for graph_hash in diversity_hashes:
            if graph_hash in self.diversity_memory_hash_set_:
                continue
            self.diversity_memory_hashes_.append(graph_hash)
            self.diversity_memory_hash_set_.add(graph_hash)

    def _matrix_to_rows(self, features):
        if hasattr(features, "toarray"):
            features = features.toarray()
        rows = []
        for row in features:
            if hasattr(row, "toarray"):
                row = row.toarray()
            rows.append(np.asarray(row, dtype=float).ravel())
        return rows

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms > 0.0, norms, 1.0)
        return matrix / norms

    def _reconstruct_path(self, state):
        path = []
        current = state
        while current is not None:
            path.append(current["graph"])
            current = current["parent"]
        path.reverse()
        return path

    def _positive_scores(self, graphs):
        return self._class_probability(
            self.graph_estimator,
            graphs,
            target=1,
            estimator_name="graph_estimator",
        )

    def _target_scores(self, graphs, *, target):
        if target is None:
            return np.zeros(len(graphs), dtype=float)
        if self.target_estimator is None:
            raise ValueError(
                "generate(..., target=...) requires a fitted target_estimator"
            )
        if self.target_estimator_mode == "classification":
            return self._class_probability(
                self.target_estimator,
                graphs,
                target=target,
                estimator_name="target_estimator",
            )
        return self._regression_target_score(
            self.target_estimator,
            graphs,
            target=target,
            estimator_name="target_estimator",
        )

    def _edge_risk_scores(self, candidates):
        if (
            self.edge_risk_model_ is None
            or self.edge_risk_lambda == 0.0
            or not candidates
        ):
            return np.zeros(len(candidates), dtype=float)
        pair_graphs = []
        for cand in candidates:
            parent = cand.get("parent")
            if parent is None:
                pair_graphs.append(cand["graph"].copy())
            else:
                pair_graphs.append(
                    self._make_edge_risk_graph_pair(parent["graph"], cand["graph"])
                )
        return self.edge_risk_model_.predict(pair_graphs)

    def _make_edge_risk_graph_pair(self, parent_graph: nx.Graph, child_graph: nx.Graph):
        return nx.disjoint_union(parent_graph.copy(), child_graph.copy())

    def _reset_edge_risk_attempt_trace(self) -> None:
        if self.edge_risk_model_ is None:
            self.edge_risk_trace_ = None
            return
        self.edge_risk_trace_ = {
            "next_state_id": 0,
            "states": {},
            "trained_state_ids": set(),
        }

    def _register_trace_state(self, state) -> None:
        trace = self.edge_risk_trace_
        if trace is None:
            state["state_id"] = None
            state["parent_state_id"] = None if state.get("parent") is None else state["parent"].get("state_id")
            state["root_decision_id"] = None
            return

        state_id = int(trace["next_state_id"])
        trace["next_state_id"] += 1
        parent = state.get("parent")
        parent_state_id = None if parent is None else parent.get("state_id")
        state["state_id"] = state_id
        state["parent_state_id"] = parent_state_id
        state["root_decision_id"] = state_id if parent_state_id is not None else None
        trace["states"][state_id] = {
            "state": state,
            "parent_state_id": parent_state_id,
            "child_state_ids": [],
            "status": "created",
        }
        if parent_state_id is not None and parent_state_id in trace["states"]:
            trace["states"][parent_state_id]["child_state_ids"].append(state_id)

    def _mark_trace_state_status(self, state, status: str) -> None:
        trace = self.edge_risk_trace_
        if trace is None:
            return
        state_id = state.get("state_id")
        if state_id is None or state_id not in trace["states"]:
            return
        trace["states"][state_id]["status"] = status

    def _trace_open_state_ids(self, beam):
        trace = self.edge_risk_trace_
        if trace is None:
            return set()
        open_state_ids = set()
        for state in beam:
            current = state
            while current is not None:
                state_id = current.get("state_id")
                if state_id is None or state_id in open_state_ids:
                    break
                open_state_ids.add(state_id)
                current = current.get("parent")
        return open_state_ids

    def _collect_trace_descendant_ids(self, state_id: int):
        trace = self.edge_risk_trace_
        if trace is None or state_id not in trace["states"]:
            return []
        descendants = []
        stack = list(trace["states"][state_id]["child_state_ids"])
        while stack:
            child_id = stack.pop()
            descendants.append(child_id)
            stack.extend(trace["states"][child_id]["child_state_ids"])
        return descendants

    def _trace_failure_ratio_for_state(self, state_id: int) -> float:
        trace = self.edge_risk_trace_
        if trace is None or state_id not in trace["states"]:
            return 0.0
        descendant_ids = [state_id] + self._collect_trace_descendant_ids(state_id)
        total_descendants = len(descendant_ids)
        if total_descendants == 0:
            return 0.0
        failure_statuses = {
            "partial_infeasible",
            "final_infeasible",
            "completion_infeasible",
            "blocked",
        }
        infeasible_descendants = sum(
            1
            for descendant_id in descendant_ids
            if trace["states"][descendant_id]["status"] in failure_statuses
        )
        return infeasible_descendants / float(total_descendants)

    def _close_edge_risk_training_states(self, *, open_state_ids) -> None:
        trace = self.edge_risk_trace_
        if trace is None or self.edge_risk_model_ is None:
            return

        training_graphs = []
        training_targets = []
        for state_id, trace_state in trace["states"].items():
            if state_id in open_state_ids or state_id in trace["trained_state_ids"]:
                continue
            state = trace_state["state"]
            parent = state.get("parent")
            if parent is None:
                trace["trained_state_ids"].add(state_id)
                continue
            training_graphs.append(
                self._make_edge_risk_graph_pair(parent["graph"], state["graph"])
            )
            training_targets.append(self._trace_failure_ratio_for_state(state_id))
            trace["trained_state_ids"].add(state_id)

        if training_graphs:
            self.edge_risk_model_.partial_fit(training_graphs, training_targets)
            if self.verbose:
                fit_time = self.edge_risk_model_.last_fit_time_seconds()
                fit_min = int(fit_time // 60)
                fit_sec = fit_time - 60 * fit_min
                print(
                    f"[edge_risk_fit] graphs={len(training_graphs)} "
                    f"training_set_size={self.edge_risk_model_.training_set_size()} "
                    f"time={fit_min}m {fit_sec:.1f}s"
                )

    def _class_probability(self, estimator, graphs, *, target, estimator_name: str):
        probs = estimator.predict_proba(graphs)
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            wrapped_estimator = getattr(estimator, "estimator_", None)
            classes = getattr(wrapped_estimator, "classes_", None)
        if classes is None:
            raise ValueError(f"{estimator_name} does not expose fitted classes_")
        classes = list(classes)
        if target not in classes:
            raise ValueError(
                f"{estimator_name} was not trained with requested class {target!r}; "
                f"available classes: {classes!r}"
            )
        class_idx = classes.index(target)
        return probs[:, class_idx]

    def _regression_target_score(self, estimator, graphs, *, target, estimator_name: str):
        if not hasattr(estimator, "predict"):
            raise ValueError(f"{estimator_name} does not expose predict()")
        predictions = np.asarray(estimator.predict(graphs), dtype=float).reshape(-1)
        target_value = float(target)
        return 1.0 / (1.0 + np.abs(predictions - target_value))

    def _missing_edges(self, graph: nx.Graph):
        nodes = list(graph.nodes())
        if nx.is_directed(graph):
            candidate_edges = list(permutations(nodes, 2))
        else:
            candidate_edges = list(combinations(nodes, 2))
        if self.allow_self_loops:
            candidate_edges += [(node, node) for node in nodes]
        occupied = {tuple(edge) for edge in graph.edges()}
        return [edge for edge in candidate_edges if edge not in occupied]

    def _canonical_graph_edges(self, graph: nx.Graph):
        return tuple(
            sorted(_canonicalize_edge(edge, graph) for edge in graph.edges())
        )

    def _edge_recency_map(self, state):
        return {
            edge: idx + 1
            for idx, edge in enumerate(state.get("edge_order", ()))
        }

    def _collect_edge_attribute_templates(self, graphs):
        templates = []
        seen = set()
        for graph in graphs:
            for _, _, edge_attrs in graph.edges(data=True):
                key = tuple(sorted(edge_attrs.items()))
                if key in seen:
                    continue
                seen.add(key)
                templates.append(dict(edge_attrs))
        return templates or [{}]

    def _as_graph_list(self, graphs):
        return [graphs] if self._is_single_graph_input(graphs) else list(graphs)

    def _coerce_optional_per_graph_argument(self, values, graphs, *, name: str):
        if values is None:
            return None
        if self._is_single_graph_input(graphs):
            return [values]
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be a scalar or a sequence matching graphs")
        try:
            value_list = list(values)
        except TypeError:
            return [values] * len(graphs)
        if len(value_list) != len(graphs):
            raise ValueError(f"{name} and graphs must have the same length")
        return value_list

    def _is_single_graph_input(self, graphs):
        return is_simple_graph(graphs)

    def _state_key(self, graph: nx.Graph):
        if nx.is_directed(graph):
            edge_key = tuple(
                sorted(
                    (u, v, tuple(sorted(graph.edges[u, v].items())))
                    for u, v in graph.edges()
                )
            )
        else:
            edge_key = tuple(
                sorted(
                    (
                        tuple(sorted((u, v))),
                        tuple(sorted(graph.edges[u, v].items())),
                    )
                    for u, v in graph.edges()
                )
            )
        return (tuple(sorted(graph.nodes())), graph.number_of_edges(), edge_key)

    def _draw_graphs(
        self,
        draw_graphs_fn: DrawGraphsFn | None,
        graphs: list[nx.Graph],
        *,
        n_graphs_per_line: int | None = None,
        titles: list[str] | None = None,
    ) -> None:
        if draw_graphs_fn is not None:
            try:
                kwargs = {}
                if n_graphs_per_line is not None:
                    kwargs["n_graphs_per_line"] = n_graphs_per_line
                if titles is not None:
                    kwargs["titles"] = titles
                draw_graphs_fn(graphs, **kwargs)
            except TypeError:
                draw_graphs_fn(graphs)
