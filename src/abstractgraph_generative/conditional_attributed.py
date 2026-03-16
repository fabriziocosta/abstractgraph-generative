"""Context-aware conditional autoregressive generation for attributed graphs.

This module extends the simplified conditional autoregressive generator with a
base-graph context scorer. Each fitted component caches an embedding computed
by vectorizing either the union of radius-limited neighborhoods around its
anchor nodes in the original training base graph, or the full graph when
``preimage_context_radius=None``. During generation, a bounded set of legal
rewiring branches is materialized on the current partial graph, embedded
through the same context vectorizer, and sampled with probability proportional
to their cosine similarity to the stored component context. When
``context_vectorizer=None``, the generator cleanly falls back to the base
conditional autoregressive search with no probabilistic context scoring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Optional, Sequence

import networkx as nx

from abstractgraph_generative.conditional import (
    ConditionalAutoregressiveGenerator,
    _CandidateAssignment,
    _GenerationState,
)
from abstractgraph_generative.rewrite import (
    _cosine_similarity,
    _transform_context_graphs,
)
from abstractgraph.graphs import get_mapped_subgraph, graph_to_abstract_graph
from abstractgraph.hashing import hash_graph


@dataclass
class _ScoredBranch:
    """One legal branch scored by attribute-context similarity.

    Args:
        state: Materialized next generation state.
        weight: Non-negative branch sampling weight.
        similarity: Raw cosine similarity used to derive ``weight``.
        component_id: Selected component instance id.

    Returns:
        None.
    """

    state: _GenerationState
    weight: float
    similarity: Optional[float]
    component_id: int


class AttributedConditionalAutoregressiveGenerator(ConditionalAutoregressiveGenerator):
    """Conditional generator that biases rewiring by attributed anchor context."""

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
        context_vectorizer=None,
        preimage_context_radius: Optional[int] = None,
        num_context_rewirings: int = 16,
        n_jobs: int = -1,
        debug: bool = False,
        debug_level: int = 1,
    ):
        """Initialize the attributed conditional generator.

        Args:
            decomposition_function: Decomposition function for AbstractGraph conversion.
            nbits: Hash bit width for hashing base and interpretation neighborhoods.
            feasibility_estimator: Optional final-graph feasibility estimator.
            base_cut_radius: Canonical radius for anchor neighborhood hashes.
            interpretation_cut_radius: Canonical radius for interpretation-node neighborhood hashes.
            preimage_cut_radius: Deprecated alias for ``base_cut_radius``.
            image_cut_radius: Deprecated alias for ``interpretation_cut_radius``.
            context_vectorizer: Graph vectorizer used for context embeddings.
            preimage_context_radius: Radius of the anchor-context neighborhood union.
                If None, embed the whole graph directly instead of extracting a
                union of anchor-centered neighborhoods.
            num_context_rewirings: Number of legal rewiring branches to score
                before sampling a preferred branch.
            n_jobs: Reserved for API compatibility.
            debug: If True, print generation diagnostics.
            debug_level: Debug verbosity level.

        Returns:
            None.
        """
        super().__init__(
            decomposition_function=decomposition_function,
            nbits=nbits,
            feasibility_estimator=feasibility_estimator,
            base_cut_radius=base_cut_radius,
            interpretation_cut_radius=interpretation_cut_radius,
            preimage_cut_radius=preimage_cut_radius,
            image_cut_radius=image_cut_radius,
            n_jobs=n_jobs,
            debug=debug,
            debug_level=debug_level,
        )
        self.context_vectorizer = context_vectorizer
        self.preimage_context_radius = (
            None if preimage_context_radius is None else int(preimage_context_radius)
        )
        self.num_context_rewirings = max(1, int(num_context_rewirings))
        self._component_context_embeddings: dict[int, object] = {}
        self._full_graph_context_embeddings: dict[int, object] = {}

    def fit(self, graphs: Sequence[nx.Graph], **_) -> "AttributedConditionalAutoregressiveGenerator":
        """Fit components and cache per-component context embeddings.

        Args:
            graphs: Training base graphs.
            **_: Ignored compatibility kwargs from previous versions.

        Returns:
            AttributedConditionalAutoregressiveGenerator: Self.
        """
        graph_list = list(graphs) if graphs is not None else None
        super().fit(graph_list, **_)

        self._component_context_embeddings = {}
        self._full_graph_context_embeddings = {}
        if self.context_vectorizer is None:
            return self

        abstract_graphs = [
            graph_to_abstract_graph(
                g,
                decomposition_function=self.decomposition_function,
                nbits=self.nbits,
                label_mode=self.label_mode,
            )
            for g in graph_list
        ]

        full_graph_embeddings = {}
        if self.preimage_context_radius is None:
            for graph in graph_list:
                cache_key = hash_graph(graph)
                full_graph_embeddings[cache_key] = self._embed_context_graph(graph)

        comp_id = 0
        for ag in abstract_graphs:
            graph_embedding = None
            if self.preimage_context_radius is None:
                graph_embedding = full_graph_embeddings.get(hash_graph(ag.base_graph))
            for image_node in ag.interpretation_graph.nodes():
                if comp_id in self._components:
                    if self.preimage_context_radius is None:
                        self._component_context_embeddings[comp_id] = graph_embedding
                    else:
                        anchor_nodes = self._training_anchor_nodes(ag.interpretation_graph, image_node)
                        self._component_context_embeddings[comp_id] = self._embed_anchor_context(
                            ag.base_graph,
                            anchor_nodes,
                        )
                comp_id += 1
        return self

    def _context_scoring_enabled(self) -> bool:
        """Return whether attributed context scoring can run.

        Args:
            None.

        Returns:
            bool: True when a context vectorizer is available.
        """
        return self.context_vectorizer is not None

    @staticmethod
    def _similarity_to_weight(similarity: Optional[float]) -> Optional[float]:
        """Convert cosine similarity into a non-negative sampling weight.

        Args:
            similarity: Cosine similarity in ``[-1, 1]``.

        Returns:
            Optional[float]: Weight in ``[0, 1]``, or None if unavailable.
        """
        if similarity is None:
            return None
        return max(0.0, (float(similarity) + 1.0) * 0.5)

    def _training_anchor_nodes(self, interpretation_graph: nx.Graph, image_node) -> list:
        """Collect training-time anchor nodes for one interpretation-node mapping.

        Args:
            interpretation_graph: Training interpretation graph.
            image_node: Interpretation node whose anchors are requested.

        Returns:
            list: Sorted anchor nodes in the original base graph.
        """
        mapped_subgraph = get_mapped_subgraph(interpretation_graph.nodes[image_node])
        if not isinstance(mapped_subgraph, nx.Graph):
            return []
        anchor_nodes = set()
        mapped_nodes = set(mapped_subgraph.nodes())
        for neighbor in interpretation_graph.neighbors(image_node):
            neighbor_mapped_subgraph = get_mapped_subgraph(interpretation_graph.nodes[neighbor])
            if not isinstance(neighbor_mapped_subgraph, nx.Graph):
                continue
            anchor_nodes.update(mapped_nodes & set(neighbor_mapped_subgraph.nodes()))
        return sorted(anchor_nodes, key=lambda node: self._preimage_node_order_key(mapped_subgraph, node))

    def _materialized_anchor_nodes(
        self,
        state: _GenerationState,
        node,
        candidate: _CandidateAssignment,
    ) -> list:
        """Collect currently instantiated anchor nodes for one legal branch.

        Args:
            state: Materialized state after committing the branch.
            node: Expanded interpretation node.
            candidate: Candidate assignment committed for ``node``.

        Returns:
            list: Sorted materialized anchor node ids present in ``state.graph``.
        """
        node_map = state.node_maps.get(node, {})
        if not isinstance(node_map, dict):
            return []
        anchor_globals = set()
        for port in candidate.component.ports:
            for local_node in port.anchor_local_nodes:
                global_node = node_map.get(local_node)
                if global_node in state.graph:
                    anchor_globals.add(global_node)
        return sorted(anchor_globals, key=lambda global_node: self._current_anchor_order_key(state.graph, global_node))

    def _extract_anchor_context_subgraph(
        self,
        graph: nx.Graph,
        anchor_nodes: Sequence,
    ) -> Optional[nx.Graph]:
        """Extract the union of anchor-centered neighborhoods.

        Args:
            graph: Graph providing the current structural context.
            anchor_nodes: Anchor nodes used as neighborhood centers.

        Returns:
            Optional[nx.Graph]: Context subgraph, or None when anchors are unavailable.
        """
        if self.preimage_context_radius is None:
            return graph
        centers = [node for node in anchor_nodes if node in graph]
        if not centers:
            return None

        included_nodes = set()
        for center in centers:
            lengths = nx.single_source_shortest_path_length(
                graph,
                center,
                cutoff=int(self.preimage_context_radius),
            )
            included_nodes.update(lengths.keys())
        if not included_nodes:
            return None
        return graph.subgraph(included_nodes).copy()

    def _embed_context_graph(self, graph: Optional[nx.Graph]):
        """Vectorize one context graph into a dense embedding.

        Args:
            graph: Context graph to embed.

        Returns:
            Optional[object]: Dense embedding vector, or None on failure.
        """
        if graph is None or self.context_vectorizer is None:
            return None
        cache_key = None
        if self.preimage_context_radius is None:
            cache_key = hash_graph(graph)
            if cache_key in self._full_graph_context_embeddings:
                return self._full_graph_context_embeddings[cache_key]
        vectors = _transform_context_graphs(self.context_vectorizer, [graph])
        if vectors is None or len(vectors) == 0:
            return None
        embedding = vectors[0]
        if cache_key is not None:
            self._full_graph_context_embeddings[cache_key] = embedding
        return embedding

    def _embed_anchor_context(self, graph: nx.Graph, anchor_nodes: Sequence):
        """Embed the requested anchor context or the full graph.

        Args:
            graph: Graph containing the anchor nodes.
            anchor_nodes: Anchor nodes used to build the context union.

        Returns:
            Optional[object]: Context embedding vector, or None when unavailable.
        """
        context_graph = self._extract_anchor_context_subgraph(graph, anchor_nodes)
        return self._embed_context_graph(context_graph)

    def _context_branch_weight(
        self,
        *,
        component_id: int,
        current_context_embedding,
    ) -> tuple[float, Optional[float]]:
        """Score one legal branch against the stored training context.

        Args:
            component_id: Selected component instance id.
            current_context_embedding: Embedding from the current partial graph.

        Returns:
            tuple[float, Optional[float]]: Sampling weight and raw similarity.
        """
        reference_embedding = self._component_context_embeddings.get(component_id)
        similarity = _cosine_similarity(current_context_embedding, reference_embedding)
        weight = self._similarity_to_weight(similarity)
        if weight is None:
            return 1.0, None
        return float(weight), similarity

    def _weighted_branch_order(
        self,
        branches: list[_ScoredBranch],
        rng: random.Random,
    ) -> list[_ScoredBranch]:
        """Sample a backtracking order with weighted preference and no replacement.

        Args:
            branches: Legal scored branches for the current expansion step.
            rng: Random generator.

        Returns:
            list[_ScoredBranch]: Weighted random order of the branches.
        """
        remaining = list(branches)
        ordered: list[_ScoredBranch] = []
        while remaining:
            weights = [
                branch.weight
                if math.isfinite(branch.weight) and float(branch.weight) > 0.0
                else 0.0
                for branch in remaining
            ]
            total = float(sum(weights))
            if total <= 0.0:
                index = int(rng.randrange(len(remaining)))
            else:
                threshold = float(rng.random()) * total
                cumulative = 0.0
                index = len(remaining) - 1
                for i, weight in enumerate(weights):
                    cumulative += float(weight)
                    if cumulative >= threshold:
                        index = i
                        break
            ordered.append(remaining.pop(index))
        return ordered

    def _search(
        self,
        state: _GenerationState,
        rng: random.Random,
        counters: dict,
        max_backtracks: int,
    ) -> Optional[_GenerationState]:
        """Run backtracking search with context-aware rewiring preference.

        Args:
            state: Current generation state.
            rng: Random number generator.
            counters: Mutable recursion counters.
            max_backtracks: Maximum branching attempts.

        Returns:
            Optional[_GenerationState]: Solved state when successful.
        """
        if not self._context_scoring_enabled():
            return super()._search(state, rng, counters, max_backtracks)

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

        legal_branches: list[_ScoredBranch] = []
        for candidate in candidates:
            future_assignments = self._enumerate_future_port_assignments(
                state,
                node,
                requirements,
                candidate,
                rng,
            )
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
                counters["branches"] += 1
                if counters["branches"] > max_backtracks:
                    counters["backtrack_limit_hits"] = int(counters.get("backtrack_limit_hits", 0)) + 1
                    return None

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

                anchor_nodes = self._materialized_anchor_nodes(next_state, node, candidate)
                current_context_embedding = self._embed_anchor_context(next_state.graph, anchor_nodes)
                weight, similarity = self._context_branch_weight(
                    component_id=candidate.component.comp_id,
                    current_context_embedding=current_context_embedding,
                )
                legal_branches.append(
                    _ScoredBranch(
                        state=next_state,
                        weight=weight,
                        similarity=similarity,
                        component_id=candidate.component.comp_id,
                    )
                )
                if len(legal_branches) >= self.num_context_rewirings:
                    break
            if len(legal_branches) >= self.num_context_rewirings:
                break

        if not legal_branches:
            return None

        self._dbg(
            2,
            "context_scored_branches",
            node=node,
            branches=[
                {
                    "component_id": branch.component_id,
                    "similarity": branch.similarity,
                    "weight": branch.weight,
                }
                for branch in legal_branches
            ],
        )

        for branch in self._weighted_branch_order(legal_branches, rng):
            solved = self._search(branch.state, rng, counters, max_backtracks)
            if solved is not None:
                return solved
        return None
