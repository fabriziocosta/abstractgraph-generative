from __future__ import annotations

import networkx as nx
import random

from abstractgraph import node as node_operator
from abstractgraph_generative.conditional import (
    ComponentInstance,
    ConditionalAutoregressiveGenerator,
    _GenerationState,
)


def test_generate_sequential_filters_interpretation_mismatches(monkeypatch) -> None:
    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=node_operator(),
        nbits=6,
        n_jobs=1,
    )
    generator._is_fitted = True

    target = nx.path_graph(2)
    first_output = nx.path_graph(3)
    second_output = nx.path_graph(4)
    generated_outputs = iter([first_output, second_output])
    match_results = iter([False, True])

    monkeypatch.setattr(
        generator,
        "_generate_one",
        lambda target_graph, rng, *, max_backtracks, attempt_trace=None, **kwargs: next(
            generated_outputs
        ),
    )
    monkeypatch.setattr(
        generator,
        "_matches_target_interpretation",
        lambda graph, target_graph: next(match_results),
    )
    monkeypatch.setattr(generator, "_filter_feasible_graphs", lambda graphs: graphs)

    outputs = generator.generate(
        n_samples=1,
        interpretation_graphs=[target],
        max_total_attempts=2,
    )

    assert outputs == [second_output]


def test_retrieve_candidates_softly_avoids_seed_subgraph_hashes() -> None:
    generator = ConditionalAutoregressiveGenerator(
        decomposition_function=node_operator(),
        nbits=6,
        n_jobs=1,
    )
    seed_component = ComponentInstance(
        comp_id=0,
        interpretation_type=1,
        deg=0,
        subgraph=nx.path_graph(2),
        ports=(),
    )
    alternative_component = ComponentInstance(
        comp_id=1,
        interpretation_type=1,
        deg=0,
        subgraph=nx.path_graph(3),
        ports=(),
    )
    generator._components = {
        0: seed_component,
        1: alternative_component,
    }
    generator._bucket = {(1, 0): [0, 1]}
    generator._inv = {}
    generator._inv_freq = {}
    generator._component_subgraph_hash_by_id = {
        0: generator._component_subgraph_hash(seed_component),
        1: generator._component_subgraph_hash(alternative_component),
    }
    generator._component_signature_by_id = {
        0: generator._component_signature(seed_component),
        1: generator._component_signature(alternative_component),
    }
    state = _GenerationState(
        target_interpretation=nx.empty_graph(1),
        target_signatures={0: (1, 0)},
        graph=nx.Graph(),
        assigned={0: False},
        comp_of={},
        node_maps={},
        edge_bindings={},
        avoid_component_subgraph_hashes=frozenset(
            {generator._component_subgraph_hash(seed_component)}
        ),
    )

    _requirements, candidates = generator._retrieve_candidates(
        state,
        0,
        rng=random.Random(0),
    )

    assert [candidate.component.comp_id for candidate in candidates] == [1]
