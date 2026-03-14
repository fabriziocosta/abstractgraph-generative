"""Self-contained GRAN-style graph generator for labeled NetworkX graphs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Hashable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


@dataclass
class _EdgeStats:
    """Sufficient statistics for Bernoulli edge probabilities."""

    edge_count: int = 0
    total_count: int = 0

    def update(self, has_edge: bool) -> None:
        """
        Update counts for one candidate edge.

        Args:
            has_edge: Whether the candidate edge exists in the training graph.

        Returns:
            None.
        """
        self.total_count += 1
        if has_edge:
            self.edge_count += 1

    def probability(self, alpha: float = 1.0, beta: float = 1.0) -> float:
        """
        Return smoothed Bernoulli estimate.

        Args:
            alpha: Beta prior alpha pseudo-count.
            beta: Beta prior beta pseudo-count.

        Returns:
            Smoothed probability in [0, 1].
        """
        return (self.edge_count + alpha) / (self.total_count + alpha + beta)


class GRANGraphGenerator:
    """Class-conditional GRAN-style block autoregressive generator for NetworkX graphs."""

    def __init__(
        self,
        *,
        block_size: int = 4,
        ratio_bins: int = 4,
        smoothing_alpha: float = 1.0,
        smoothing_beta: float = 1.0,
        enforce_connected: bool = True,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        """
        Initialize generator.

        Args:
            block_size: Number of new nodes added per autoregressive step.
            ratio_bins: Number of bins for generation-progress conditioning.
            smoothing_alpha: Beta prior alpha for Bernoulli edge probabilities.
            smoothing_beta: Beta prior beta for Bernoulli edge probabilities.
            enforce_connected: If True, connect generated disconnected components.
            random_state: Default random seed.
            verbose: If True, print simple fitting diagnostics.

        Returns:
            None.
        """
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if ratio_bins <= 0:
            raise ValueError("ratio_bins must be positive.")
        if smoothing_alpha <= 0.0 or smoothing_beta <= 0.0:
            raise ValueError("smoothing_alpha and smoothing_beta must be positive.")

        self.block_size = int(block_size)
        self.ratio_bins = int(ratio_bins)
        self.smoothing_alpha = float(smoothing_alpha)
        self.smoothing_beta = float(smoothing_beta)
        self.enforce_connected = bool(enforce_connected)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self._states: Dict[Hashable, Dict[str, object]] = {}
        self._is_fitted = False

    def _log(self, message: str) -> None:
        """
        Print a verbose log line when enabled.

        Args:
            message: Message payload without model prefix.

        Returns:
            None.
        """
        if self.verbose:
            print(f"[GRAN] {message}")

    @staticmethod
    def _order_nodes_for_training(graph: nx.Graph) -> List[Hashable]:
        """
        Compute deterministic node order used by autoregressive decomposition.

        Args:
            graph: Input graph.

        Returns:
            Ordered list of node ids.
        """
        nodes = list(graph.nodes())
        return sorted(nodes, key=lambda n: (-int(graph.degree(n)), repr(n)))

    def _ratio_bin(self, existing_nodes: int, total_nodes: int) -> int:
        """
        Discretize generation progress into bins.

        Args:
            existing_nodes: Number of nodes already in the partial graph.
            total_nodes: Target final number of nodes.

        Returns:
            Bin index in [0, ratio_bins - 1].
        """
        if total_nodes <= 0:
            return 0
        ratio = float(existing_nodes) / float(max(total_nodes, 1))
        idx = int(np.floor(ratio * self.ratio_bins))
        return int(np.clip(idx, 0, self.ratio_bins - 1))

    @staticmethod
    def _canonical_label_pair(label_u: Hashable, label_v: Hashable) -> Tuple[Hashable, Hashable]:
        """
        Return stable unordered pair key for undirected label pairs.

        Args:
            label_u: First label.
            label_v: Second label.

        Returns:
            Ordered tuple key.
        """
        if repr(label_u) <= repr(label_v):
            return label_u, label_v
        return label_v, label_u

    def _fit_one_type(self, graphs: Sequence[nx.Graph], type_key: Hashable) -> Dict[str, object]:
        """
        Fit one class-conditional state.

        Args:
            graphs: Graphs for one class.
            type_key: Class key.

        Returns:
            State dictionary used by generation.
        """
        if len(graphs) == 0:
            raise ValueError("Cannot fit empty graph list for a class.")

        label_counter: Counter[Hashable] = Counter()
        size_counter: Counter[int] = Counter()
        edge_stats: DefaultDict[Tuple[object, ...], _EdgeStats] = defaultdict(_EdgeStats)

        total_edges = 0
        total_edge_slots = 0
        progress_every = max(1, len(graphs) // 10)

        self._log(
            f"[type={type_key}] fit_start "
            f"graphs={len(graphs)} block_size={self.block_size} ratio_bins={self.ratio_bins}"
        )

        for graph_idx, graph in enumerate(graphs, start=1):
            n = int(graph.number_of_nodes())
            if n <= 0:
                continue

            order = self._order_nodes_for_training(graph)
            ordered_labels = [graph.nodes[node].get("label", 0) for node in order]

            for label in ordered_labels:
                label_counter[label] += 1
            size_counter[n] += 1

            index_of: Dict[Hashable, int] = {node: idx for idx, node in enumerate(order)}
            adjacency = np.zeros((n, n), dtype=np.uint8)
            for u, v in graph.edges():
                if u not in index_of or v not in index_of:
                    continue
                i = index_of[u]
                j = index_of[v]
                adjacency[i, j] = 1
                adjacency[j, i] = 1

            total_edges += int(np.sum(np.triu(adjacency, k=1)))
            total_edge_slots += int(n * (n - 1) // 2)

            for start in range(0, n, self.block_size):
                stop = min(start + self.block_size, n)
                ratio_bin = self._ratio_bin(existing_nodes=start, total_nodes=n)

                for i in range(start, stop):
                    for j in range(0, start):
                        lu, lv = self._canonical_label_pair(ordered_labels[i], ordered_labels[j])
                        key = ("cross", ratio_bin, lu, lv)
                        edge_stats[key].update(bool(adjacency[i, j]))

                        key_fallback = ("cross", None, lu, lv)
                        edge_stats[key_fallback].update(bool(adjacency[i, j]))

                        key_dense = ("cross", ratio_bin, None, None)
                        edge_stats[key_dense].update(bool(adjacency[i, j]))

                        key_global_cross = ("cross", None, None, None)
                        edge_stats[key_global_cross].update(bool(adjacency[i, j]))

                for i in range(start, stop):
                    for j in range(i + 1, stop):
                        lu, lv = self._canonical_label_pair(ordered_labels[i], ordered_labels[j])
                        key = ("intra", ratio_bin, lu, lv)
                        edge_stats[key].update(bool(adjacency[i, j]))

                        key_fallback = ("intra", None, lu, lv)
                        edge_stats[key_fallback].update(bool(adjacency[i, j]))

                        key_dense = ("intra", ratio_bin, None, None)
                        edge_stats[key_dense].update(bool(adjacency[i, j]))

                        key_global_intra = ("intra", None, None, None)
                        edge_stats[key_global_intra].update(bool(adjacency[i, j]))

            if graph_idx % progress_every == 0 or graph_idx == len(graphs):
                running_density = float(total_edges) / float(total_edge_slots) if total_edge_slots > 0 else 0.0
                self._log(
                    f"[type={type_key}] fit_progress "
                    f"graph={graph_idx:03d}/{len(graphs):03d} "
                    f"running_density={running_density:.4f}"
                )

        if not label_counter:
            label_counter[0] = 1
        if not size_counter:
            size_counter[max(1, self.block_size)] = 1

        labels = list(label_counter.keys())
        label_probs = np.array([label_counter[label] for label in labels], dtype=np.float64)
        label_probs = label_probs / np.sum(label_probs)

        size_values = np.array(sorted(size_counter.keys()), dtype=np.int64)
        size_probs = np.array([size_counter[int(size)] for size in size_values], dtype=np.float64)
        size_probs = size_probs / np.sum(size_probs)

        if total_edge_slots <= 0:
            global_density = 0.2
        else:
            global_density = float(total_edges) / float(total_edge_slots)

        state: Dict[str, object] = {
            "type_key": type_key,
            "label_values": labels,
            "label_probs": label_probs,
            "size_values": size_values,
            "size_probs": size_probs,
            "edge_stats": edge_stats,
            "global_density": global_density,
            "default_num_nodes": int(np.round(np.average(size_values, weights=size_probs))),
        }

        mean_nodes = float(np.average(size_values, weights=size_probs))
        self._log(
            f"[type={type_key}] fit_done "
            f"graphs={len(graphs)} labels={len(labels)} "
            f"avg_nodes={mean_nodes:.2f} density={global_density:.4f} "
            f"size_support={len(size_values)} edge_patterns={len(edge_stats)}"
        )

        return state

    def fit(
        self,
        graphs: Sequence[nx.Graph],
        targets: Optional[Sequence[Hashable]] = None,
    ) -> "GRANGraphGenerator":
        """
        Fit class-conditional GRAN-style generator.

        Args:
            graphs: Input graph list.
            targets: Optional class labels. If provided, one state is trained per class key.

        Returns:
            Self.
        """
        if len(graphs) == 0:
            raise ValueError("graphs must not be empty.")
        if targets is not None and len(targets) != len(graphs):
            raise ValueError("targets must have the same length as graphs.")

        if targets is None:
            self._states = {"default": self._fit_one_type(graphs=graphs, type_key="default")}
        else:
            buckets: Dict[Hashable, List[nx.Graph]] = defaultdict(list)
            for graph, target in zip(graphs, targets):
                buckets[target].append(graph)
            self._log(f"fit_start classes={len(buckets)} total_graphs={len(graphs)}")
            self._states = {key: self._fit_one_type(graphs=value, type_key=key) for key, value in buckets.items()}

        self._log(f"fit_done classes={len(self._states)} keys={list(self._states.keys())}")
        self._is_fitted = True
        return self

    def _resolve_state_for_generation(self, graph_type: Optional[Hashable]) -> Tuple[Hashable, Dict[str, object]]:
        """
        Resolve which fitted state to use.

        Args:
            graph_type: Optional requested class key.

        Returns:
            Tuple of resolved key and state dictionary.
        """
        if len(self._states) == 1 and graph_type is None:
            graph_type = next(iter(self._states.keys()))
        if graph_type is None:
            raise ValueError("graph_type is required when multiple type models are fitted.")
        if graph_type not in self._states:
            raise KeyError(f"Unknown graph_type {graph_type!r}. Known keys: {list(self._states.keys())}")
        return graph_type, self._states[graph_type]

    def _edge_probability(
        self,
        *,
        state: Dict[str, object],
        kind: str,
        ratio_bin: int,
        label_u: Hashable,
        label_v: Hashable,
    ) -> float:
        """
        Get edge probability from fitted statistics with fallbacks.

        Args:
            state: Class state dictionary.
            kind: Edge type, either "cross" or "intra".
            ratio_bin: Progress bin.
            label_u: First node label.
            label_v: Second node label.

        Returns:
            Bernoulli probability in [0, 1].
        """
        edge_stats = state["edge_stats"]
        assert isinstance(edge_stats, dict)

        lu, lv = self._canonical_label_pair(label_u, label_v)
        candidates = [
            (kind, ratio_bin, lu, lv),
            (kind, None, lu, lv),
            (kind, ratio_bin, None, None),
            (kind, None, None, None),
        ]
        for key in candidates:
            stat = edge_stats.get(key)
            if stat is not None and stat.total_count > 0:
                return float(stat.probability(alpha=self.smoothing_alpha, beta=self.smoothing_beta))

        return float(state["global_density"])

    def _connect_components(self, graph: nx.Graph, rng: np.random.Generator) -> None:
        """
        Connect graph components by adding one random bridge per merge step.

        Args:
            graph: Graph modified in place.
            rng: Random generator.

        Returns:
            None.
        """
        if graph.number_of_nodes() <= 1:
            return
        components = [list(component) for component in nx.connected_components(graph)]
        while len(components) > 1:
            base = components[0]
            other = components[1]
            u = int(rng.choice(base))
            v = int(rng.choice(other))
            graph.add_edge(u, v)
            components = [list(component) for component in nx.connected_components(graph)]

    def generate(
        self,
        *,
        num_graphs: int,
        graph_type: Optional[Hashable] = None,
        num_nodes: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[nx.Graph]:
        """
        Generate graphs from fitted class-conditional model.

        Args:
            num_graphs: Number of graphs to generate.
            graph_type: Class key to sample from.
            num_nodes: Optional fixed graph size. If None, sample from empirical size distribution.
            seed: Optional random seed override.

        Returns:
            List of generated labeled NetworkX graphs.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit(...) before generate(...).")
        if num_graphs <= 0:
            raise ValueError("num_graphs must be positive.")

        resolved_graph_type, state = self._resolve_state_for_generation(graph_type)

        label_values = list(state["label_values"])
        label_probs = np.asarray(state["label_probs"], dtype=np.float64)
        size_values = np.asarray(state["size_values"], dtype=np.int64)
        size_probs = np.asarray(state["size_probs"], dtype=np.float64)

        if seed is None:
            seed = self.random_state
        rng = np.random.default_rng(int(seed))

        self._log(
            f"[type={resolved_graph_type}] generate_start "
            f"num_graphs={num_graphs} num_nodes={num_nodes} seed={int(seed)}"
        )

        outputs: List[nx.Graph] = []
        edge_counts: List[int] = []
        node_counts: List[int] = []
        for _ in range(num_graphs):
            if num_nodes is None:
                sampled_nodes = int(rng.choice(size_values, p=size_probs))
            else:
                sampled_nodes = int(num_nodes)
            if sampled_nodes <= 0:
                raise ValueError("num_nodes must be positive.")

            labels = list(rng.choice(label_values, size=sampled_nodes, p=label_probs))
            graph = nx.Graph()
            for node_id, label in enumerate(labels):
                graph.add_node(node_id, label=label)

            for start in range(0, sampled_nodes, self.block_size):
                stop = min(start + self.block_size, sampled_nodes)
                ratio_bin = self._ratio_bin(existing_nodes=start, total_nodes=sampled_nodes)

                for i in range(start, stop):
                    for j in range(0, start):
                        probability = self._edge_probability(
                            state=state,
                            kind="cross",
                            ratio_bin=ratio_bin,
                            label_u=labels[i],
                            label_v=labels[j],
                        )
                        if rng.random() < probability:
                            graph.add_edge(i, j)

                for i in range(start, stop):
                    for j in range(i + 1, stop):
                        probability = self._edge_probability(
                            state=state,
                            kind="intra",
                            ratio_bin=ratio_bin,
                            label_u=labels[i],
                            label_v=labels[j],
                        )
                        if rng.random() < probability:
                            graph.add_edge(i, j)

            if self.enforce_connected and graph.number_of_nodes() > 1 and not nx.is_connected(graph):
                self._connect_components(graph=graph, rng=rng)

            outputs.append(graph)
            edge_counts.append(int(graph.number_of_edges()))
            node_counts.append(int(graph.number_of_nodes()))

        if outputs:
            mean_nodes = float(np.mean(node_counts))
            mean_edges = float(np.mean(edge_counts))
            densities: List[float] = []
            for nodes, edges in zip(node_counts, edge_counts):
                denom = nodes * (nodes - 1) / 2.0
                densities.append((edges / denom) if denom > 0 else 0.0)
            self._log(
                f"[type={resolved_graph_type}] generate_done "
                f"graphs={len(outputs)} avg_nodes={mean_nodes:.2f} "
                f"avg_edges={mean_edges:.2f} avg_density={float(np.mean(densities)):.4f}"
            )

        return outputs
