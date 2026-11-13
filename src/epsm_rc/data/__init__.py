# File: __init__.py
# Purpose: Packaged data files (bundled EPW weather) for epsm_rc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-10-29

"""Packaged data files (bundled EPW weather) for epsm_rc.

Notes
-----
Bundled EPW files (for example ``goteborg.epw``) are teaching inputs, read
only when a caller selects them explicitly.

Examples
--------
Example::

    from importlib.resources import files

    path = files("epsm_rc") / "data/epw/goteborg.epw"
"""
