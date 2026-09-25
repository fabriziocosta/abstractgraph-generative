from __future__ import annotations

import numpy as np

from abstractgraph_generative.generative_performance import (
    bootstrap,
    expected_gain_weights,
)


def test_bootstrap_is_deterministic_for_seed():
    instances = np.arange(12).reshape(6, 2)
    targets = np.arange(6)

    first_x, first_y = bootstrap(instances, targets, seed=19)
    second_x, second_y = bootstrap(instances, targets, seed=19)

    assert np.array_equal(first_x, second_x)
    assert np.array_equal(first_y, second_y)
    assert len(first_x) == len(instances)


def test_expected_gain_weights_normalize_and_handle_zero_gain():
    weights = expected_gain_weights(np.asarray([4, 8, 16]), (0.0, 1.0, 1.0))
    zero_gain = expected_gain_weights(np.asarray([4, 8]), (0.5, 0.0, 1.0))

    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
    assert np.array_equal(zero_gain, np.zeros(2))
