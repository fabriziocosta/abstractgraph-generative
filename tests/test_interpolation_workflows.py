from __future__ import annotations

import numpy as np

from abstractgraph import node as node_operator
from abstractgraph_generative.interpolation_generation import (
    InterpolationGenerator,
    make_pairs,
)
from abstractgraph_generative.interpolation_path import InterpolationEstimator


class _SizeTransformer:
    decomposition_function = node_operator()
    nbits = 6

    def transform(self, graphs):
        return np.asarray([[g.number_of_nodes(), g.number_of_edges()] for g in graphs])


def test_interpolation_estimator_empty_donor_contract():
    assert InterpolationGenerator.__module__ == (
        "abstractgraph_generative.interpolation_generation"
    )
    estimator = InterpolationEstimator(graph_transformer=_SizeTransformer())
    assert estimator.fit([]) is estimator
    assert estimator.interpolate(None, None) == []


def test_make_pairs_is_seeded_and_returns_requested_disjoint_pairs():
    first = make_pairs(6, 3, np.random.default_rng(23))
    second = make_pairs(6, 3, np.random.default_rng(23))

    assert first == second
    assert len(first) == 3
    assert all(left != right for left, right in first)
    assert len({index for pair in first for index in pair}) == 6
    assert make_pairs(1, 3, np.random.default_rng(23)) == []


def test_legacy_interpolation_modules_are_removed():
    from importlib.util import find_spec

    assert find_spec("abstractgraph_generative.interpolate") is None
    assert find_spec("abstractgraph_generative.interpolation") is None
