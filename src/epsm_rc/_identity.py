# File: _identity.py
# Purpose: Content fingerprints with explicit batch-scoped source reuse.
# SPDX-License-Identifier: MIT
# License: MIT; see the project-root LICENSE.
# Date created: 2026-10-19
"""Inspect installed implementations without retaining a process-global identity.

Batch reuse checks every previously discovered source file and directory on each
evaluation. Edits, replacements and directory changes trigger a full rescan and
content hash. Weather verification remains the adapter's responsibility.
"""

from __future__ import annotations

import hashlib
import os
import sys
from importlib import import_module, metadata
from pathlib import Path


def _stamp(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class SourceFingerprintCache:
    """Reuse source digests while file and directory metadata remain unchanged.

    This is a sequential, runtime accelerator for an explicitly scoped batch.
    Filesystem metadata must report edits normally; it is not a defence against
    a filesystem that hides content changes or concurrent source installation.
    """

    def __init__(self):
        self._entries = {}

    def clear(self):
        """Forget reused digests without inspecting files."""
        self._entries.clear()

    def fingerprint(self, root):
        previous = self._entries.get(root)
        if previous is not None:
            digest, stamps = previous
            try:
                if all(_stamp(path) == stamp for path, stamp in stamps):
                    return digest
            except FileNotFoundError:
                pass
        digest, stamps = _fingerprint(root)
        self._entries[root] = digest, stamps
        return digest


def _fingerprint(root):
    paths, directories = [], []

    def failed(error):
        raise error

    for directory, _, filenames in os.walk(root, onerror=failed):
        parent = Path(directory)
        directories.append(parent)
        paths.extend(parent / name for name in filenames if name.endswith(".py"))
    digest = hashlib.sha256()
    # Preserve the original sorted relative-path/content identity exactly.
    stamps = [(path, _stamp(path)) for path in directories]
    for path in sorted(paths):
        before = _stamp(path)
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        if _stamp(path) != before:
            raise RuntimeError("Installed Python source changed while fingerprinting")
        stamps.append((path, before))
    if any(_stamp(path) != stamp for path, stamp in stamps):
        raise RuntimeError("Installed Python sources changed while fingerprinting")
    return digest.hexdigest(), stamps


def implementation_identity(source_cache=None):
    """Read installed source identity and current numerical dependency versions.

    Without a source cache, read every Python source. With a batch-owned cache,
    stat all known sources and directories and rehash changed package trees.
    Dependency versions and package locations are resolved on every call.
    """
    sources = {}
    for package in ("epsm_rc", "epsm_core", "epsm_data"):
        root = Path(import_module(package).__file__).parent
        sources[package] = (
            source_cache.fingerprint(root) if source_cache is not None else _fingerprint(root)[0]
        )
    return {
        "source_sha256": sources,
        "dependencies": {
            package: metadata.version(package)
            for package in ("numpy", "pandas", "pvlib", "shapely", "pydantic")
        },
        "python": sys.version.split()[0],
    }
