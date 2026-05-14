from __future__ import annotations

import networkx as nx

from abstractgraph import node as node_operator
from abstractgraph_generative.conditional import ConditionalAutoregressiveGenerator


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
        lambda target_graph, rng, *, max_backtracks, attempt_trace=None: next(
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
