# File: _identity.py
# Purpose: Capture installed calculation identity once per Python process.
# SPDX-License-Identifier: MIT
# License: MIT; see the project-root LICENSE.
"""Lazy process identity; restart after editing installed sources or dependencies.

Identity is provenance for the loaded calculation, not a live source watcher.
No file or dependency inspection happens at import or on subsequent evaluations.
"""

from __future__ import annotations

import hashlib
import sys
from copy import deepcopy
from functools import lru_cache
from importlib import import_module, metadata
from pathlib import Path


@lru_cache(maxsize=1)
def _process_identity():
    sources = {}
    for package in ("epsm_rc", "epsm_core", "epsm_data"):
        root = Path(import_module(package).__file__).parent
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        sources[package] = digest.hexdigest()
    native = Path(import_module("epsm_rc._core_rs").__file__)
    return {
        "source_sha256": sources,
        "native_sha256": {"epsm_rc._core_rs": hashlib.sha256(native.read_bytes()).hexdigest()},
        "dependencies": {
            package: metadata.version(package)
            for package in ("numpy", "pandas", "pvlib", "shapely", "pydantic")
        },
        "python": sys.version.split()[0],
    }


def implementation_identity():
    """Return detached provenance for this process's installed calculation.

    The first call reads installed Python sources, the native extension and numerical dependency
    versions. Later calls copy the captured metadata without filesystem access.
    Restart Python after changing source or installed dependencies.
    """
    return deepcopy(_process_identity())
