# File: _version.py
# Purpose: Release version shared by runtime imports and package metadata.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-10-29

"""Release version shared by runtime imports and package metadata.

Notes
-----
The single version string is read by ``epsm_rc.__version__`` and by the
build metadata, so the two cannot disagree.

Examples
--------
Example::

    import epsm_rc

    print(epsm_rc.__version__)
"""

__version__ = "0.12.2"
