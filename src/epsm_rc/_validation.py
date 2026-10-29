# SPDX-License-Identifier: MIT
"""Shared input checks whose messages name the field, the value and the valid range."""

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
