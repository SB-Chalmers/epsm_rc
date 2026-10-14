# File: 11_core_engine_comparison.py
# Purpose: Run compilation and RC calculation from explicit shared input files.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# Author: Sanjay Somanath
# Research attribution: Numerical solver credits remain in iso13790.py.
# Date created: 2026-10-14
# Date modified: 2026-10-14
# Changelog:
#   2026-10-14 - Demonstrate the explicit shared Core comparison CLI.
"""Compile EnergyPlus inputs and simulate RC from explicit shared input files.

Requires epsm_rc[platform] and installed EnergyPlus 24.2 with its adjacent schema.
Run with --help for required snapshot, weather, RC settings and output arguments.
The snapshot must contain caller-supplied geometry, fabric, glazing, operation
and HVAC. EnergyPlus is compiled only; this does not run or calibrate an engine year.
"""

from epsm_rc.parametric import main

if __name__ == "__main__":
    main()
