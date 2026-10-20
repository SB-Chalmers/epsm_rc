# File: preparation.py
# Purpose: Caller-owned bounded reuse of RC preparation, never thermal states.
# SPDX-License-Identifier: MIT
# License: MIT; see the project-root LICENSE.
# Date created: 2026-10-18
"""Explicit sequential preparation reuse for independent RC simulations."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from types import MappingProxyType
from typing import TypeVar

import numpy as np

T = TypeVar("T")


class PreparationCache:
    """Bounded caller-owned cache of weather and radiation preparation.

    Parameters
    ----------
    max_entries : int
        Positive LRU entry limit, default 64. Geometry has a separate cache with
        the same limit. Entry sizes depend on calendar and geometry complexity.

    Notes
    -----
    Construction performs no preparation or file access. Every simulation still
    verifies weather bytes. Installed calculation identity is captured lazily once
    per process; restart Python after changing installed code or dependencies.
    Keys include that identity, the weather checksum and calculation settings.
    The adapter shares immutable numeric preparation; ``get_or_compute`` retains
    detached writable copies. No Building5R1C state is ever retained.
    Use one instance per sequential caller, and one per worker in parallel work.
    Cache contents are runtime accelerators, not portable scientific artifacts.
    """

    def __init__(self, max_entries: int = 64):
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._geometry_cache = None
        self.hits = self.misses = 0

    @contextmanager
    def batch(self):
        """Yield this cache for compatibility with existing sequential batches.

        Preparation entries have the same lifetime inside and outside this
        context. Calculation identity is captured once per Python process.
        This context performs no file access and does not monitor source edits.
        """
        yield self

    @property
    def geometry_cache(self):
        """Return the lazily created Core GeometryCache; no geometry is resolved."""
        if self._geometry_cache is None:
            from epsm_core.pipeline import GeometryCache

            self._geometry_cache = GeometryCache(max_entries=self.max_entries)
        return self._geometry_cache

    def __len__(self):
        return len(self._entries)

    def __repr__(self):
        return f"PreparationCache(entries={len(self)}, max_entries={self.max_entries})"

    def clear(self) -> None:
        """Release preparation and geometry entries and reset counters; no I/O."""
        self._entries.clear()
        if self._geometry_cache is not None:
            self._geometry_cache.clear()
        self.hits = self.misses = 0

    def get_or_compute(self, namespace: str, inputs: dict, compute: Callable[[], T]) -> T:
        """Return a detached value for explicit JSON-compatible calculation inputs.

        The caller supplies all dependencies in ``inputs``. A miss executes
        ``compute``, whose file access or calculations belong to the requesting
        operation. Exceptions are propagated and never cached.
        """
        return self._get_or_compute(namespace, inputs, compute, readonly=False)

    def _get_or_compute_readonly(self, namespace, inputs, compute):
        """Share immutable numeric preparation; never share caller-owned buffers."""
        return self._get_or_compute(namespace, inputs, compute, readonly=True)

    def _get_or_compute(self, namespace, inputs, compute, *, readonly):
        key = (
            namespace,
            hashlib.sha256(
                json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            readonly,
        )
        if key in self._entries:
            self.hits += 1
            self._entries.move_to_end(key)
            value = self._entries[key]
            return value if readonly else deepcopy(value)
        self.misses += 1
        value = compute()
        if readonly:
            value = _readonly(value)
        self._entries[key] = value if readonly else deepcopy(value)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return value


def _readonly(value):
    """Freeze numeric arrays onto bytes, mappings and sequences recursively.

    A writable owner cannot be recovered through ``array.base`` or ``setflags``.
    Pandas/custom objects and object arrays require the ordinary detached route.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Readonly preparation cannot contain object arrays")
        return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _readonly(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_readonly(item) for item in value)
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex, np.number)):
        return value
    raise TypeError(f"Unsupported readonly preparation type: {type(value).__name__}")
