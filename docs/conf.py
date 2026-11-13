# File: conf.py
# Purpose: Build the installed RC package reference; no sibling source paths are inserted.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-10-29

"""Build the installed RC package reference; no sibling source paths are inserted."""

from epsm_rc._version import __version__

project = "EPSM RC"
copyright = "2026, Sanjay Somanath"
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
