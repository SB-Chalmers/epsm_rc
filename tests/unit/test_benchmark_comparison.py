# File: test_benchmark_comparison.py
# Purpose: Regression tolerances must never conceal structural changes or invalid results.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-10-29

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
