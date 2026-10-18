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
from collections.abc import Callable
from copy import deepcopy
from typing import TypeVar

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
    verifies weather bytes and installed source identity. Keys include that
    identity, the weather checksum and every relevant calculation setting.
    Returned values are detached copies; no Building5R1C state is ever retained.
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
        key = (
            namespace,
            hashlib.sha256(
                json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
        )
        if key in self._entries:
            self.hits += 1
            self._entries.move_to_end(key)
            return deepcopy(self._entries[key])
        self.misses += 1
        value = compute()
        self._entries[key] = deepcopy(value)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return value
