"""Regression tolerances must never conceal structural changes or invalid results."""

import numpy as np
import pytest

from benchmarks.run import compare_series


def test_comparison_accepts_roundoff_and_near_zero_noise():
    compare_series(
        np.array([0.0, 6009.140820042374]), np.array([1e-10, 6009.140820042354]), label="demand"
    )


@pytest.mark.parametrize(
    "actual,expected",
    [
        ([1.01], [1.0]),
        ([np.nan], [np.nan]),
        ([1.0, 1.0], [1.0]),
        (["2023-01-02"], ["2023-01-01"]),
        ([True], [False]),
    ],
)
def test_comparison_rejects_real_regressions(actual, expected):
    with pytest.raises(AssertionError):
        compare_series(np.asarray(actual), np.asarray(expected), label="channel")
