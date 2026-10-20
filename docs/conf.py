# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
# License: See LICENSE at the project root.
"""Build the installed RC package reference; no sibling source paths are inserted."""

from epsm_rc._version import __version__

project = "EPSM RC"
copyright = "2026, EPSM Development contributors"
release = __version__
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.doctest"]
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autoclass_content = "class"
autodoc_typehints = "none"
autodoc_member_order = "bysource"
root_doc = "index"
exclude_patterns = ["_build", "history"]
html_theme = "alabaster"
