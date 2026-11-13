# File: _validation.py
# Purpose: Shared input checks whose messages name the field, the value and the valid range.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-10-29

"""Shared input checks whose messages name the field, the value and the valid range.

Notes
-----
Each check raises ``ValueError`` (or ``TypeError`` for non-numbers and
booleans) with a message naming the field, the value received and the valid
range, so errors from settings and public calls point at the input to fix.

Examples
--------
Example::

    from epsm_rc._validation import finite, fraction

    finite("floor_area_m2", 100.0, positive=True)
    fraction("solar_shading_factor", 0.7)
    try:
        finite("floor_area_m2", -1.0, positive=True)
    except ValueError as error:
        print(error)  # floor_area_m2 must be finite and positive, not -1.0
"""

from math import isfinite
from numbers import Real


def number(name, value):
    """Return ``value`` as a float, rejecting booleans and non-numeric input by name."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, not {type(value).__name__} {value!r}")
    return float(value)


def finite(name, value, *, positive=False, nonnegative=False, minimum=None, maximum=None):
    """Return a finite float within the stated bounds, or raise naming the constraint."""
    result = number(name, value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite, not {value!r}")
    if positive and result <= 0:
        raise ValueError(f"{name} must be finite and positive, not {value!r}")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be finite and nonnegative, not {value!r}")
    if (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        raise ValueError(
            f"{name} must be finite and between {minimum} and {maximum}, not {value!r}"
        )
    return result


def fraction(name, value):
    """Return a finite fraction in [0, 1]."""
    try:
        return finite(name, value, minimum=0, maximum=1)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite fraction between 0 and 1, not {value!r}") from (
            error
        )


def choice(name, value, options):
    """Return ``value`` if it is one of ``options``; otherwise list every valid option."""
    if value not in options:
        valid = ", ".join(repr(option) for option in sorted(options))
        raise ValueError(f"Unknown {name} {value!r}; choose one of {valid}")
    return value
