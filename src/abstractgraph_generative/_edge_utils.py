"""Graph and dataset helpers used by edge generation."""

from __future__ import annotations

import random
from itertools import combinations, permutations

import networkx as nx

from abstractgraph.graphs import graph_to_abstract_graph


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
        target_n_nodes = int(
            round((graph1.number_of_nodes() + graph2.number_of_nodes()) / 2)
        )
    target_n_nodes = max(1, int(target_n_nodes))

    best_choice = None
    best_score = None

    for n_components_per_graph in range(1, max_pairs + 1):
        trial_count = (
            1
            if (
                n_components_per_graph == len(component_graphs1)
                and n_components_per_graph == len(component_graphs2)
            )
            else n_trials
        )
        for _ in range(trial_count):
            selected1 = _sample_components(
                component_graphs1, n_components_per_graph, rng
            )
            selected2 = _sample_components(
                component_graphs2, n_components_per_graph, rng
            )
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
            edge
            for edge in candidate_edges
            if edge != old_edge and edge not in occupied
        ]
        if not new_edge_options:
            raise ValueError(
                "No valid destination edge is available for moving an edge"
            )

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
                dataset.extend(
                    (negative_graph, 0) for negative_graph in negative_graphs
                )

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
                    edge
                    for edge in group
                    if _graph_has_canonical_edge(current_graph, edge)
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
                    dataset.extend(
                        (negative_graph, 0) for negative_graph in negative_graphs
                    )

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
    full_graph_edges = frozenset(
        _canonicalize_edge(edge, graph) for edge in graph.edges()
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        if subgraph is None or subgraph.number_of_edges() == 0:
            continue
        group_edges = frozenset(
            _canonicalize_edge(edge, graph) for edge in subgraph.edges()
        )
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
            for u, v, key, edge_attrs in relabeled_component.edges(
                keys=True, data=True
            ):
                merged.add_edge(u, v, key=key, **dict(edge_attrs))
        else:
            for u, v, edge_attrs in relabeled_component.edges(data=True):
                merged.add_edge(u, v, **dict(edge_attrs))
    merged.graph.update(dict(template_graph.graph))
    return merged
