# File: src/epsm_rc/__init__.py
# Purpose: Deliberate public imports with lazy advanced and optional APIs.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
# License: See LICENSE at the project root.
# Date modified: 2026-10-20
"""Hourly reduced thermal calculation from explicit physical or numerical inputs.

Use ``simulate_building`` with a Core Building, explicit weather and ``RCSettings``
for the recommended shared physical interface. ``Building5R1C`` and the named
standalone modules retain numerical experiments without Core. Advanced APIs load
only when requested; import performs no file access or model selection.

The numerical solver retains its DIBS (Dynamic ISO Building Simulator) attribution
and MIT terms in ``iso13790.py``. Radiation correlation notices remain packaged.

Examples
--------
>>> from epsm_rc import Building5R1C
>>> Building5R1C(floor_area_m2=100).floor_area_m2
100
"""

import logging as _stdlib_logging

from ._version import __version__ as __version__
from .iso13790 import (
    Building5R1C,
    BuildingClass,
    Construction,
    ConstructionLayer,
    ThermalBridgeClass,
    VentilationSystem,
    calculate_thermal_mass_from_constructions,
    create_swedish_apartment,
    create_swedish_villa,
    estimate_ground_temps_from_air,
)
from .solar import SolarGainsCalculator, estimate_window_orientation_split, simplified_solar_gains
from .weather import EPWData, get_irradiance, get_temperature, load_epw, parse_ground_temperatures

__all__ = [
    "simulate_building",
    "preflight",
    "output_capabilities",
    "RCSettings",
    "PreparationCache",
    # Core model
    "Building5R1C",
    "BuildingClass",
    "VentilationSystem",
    "ThermalBridgeClass",
    "ConstructionLayer",
    "Construction",
    "calculate_thermal_mass_from_constructions",
    "estimate_ground_temps_from_air",
    # Swedish factories
    "create_swedish_villa",
    "create_swedish_apartment",
    # Solar
    "SolarGainsCalculator",
    "simplified_solar_gains",
    "estimate_window_orientation_split",
    # Weather
    "EPWData",
    "load_epw",
    "get_temperature",
    "get_irradiance",
    "parse_ground_temperatures",
    # TABULA
    "BuildingTABULA",
    "ClimateData",
    "ThermalMassClass",
    # Geometry
    "BuildingGeometry",
    # GIS importer (optional, requires pyproj)
    "FootprintImporter",
    # Parametric building generator
    "ParametricBuilding",
    "generate_idf",
    "get_5r1c_params",
    "EngineComparison",
    "prepare_engine_comparison",
]


# Platform integration imports Core only when explicitly requested.
def __getattr__(name):
    """Load advanced and optional public APIs only when explicitly requested."""
    from importlib import import_module

    modules = {
        "PreparationCache": "preparation",
        "EngineComparison": "comparison",
        "prepare_engine_comparison": "comparison",
        "simulate_building": "platform",
        "preflight": "platform",
        "output_capabilities": "outputs",
        "RCSettings": "platform",
        "ParametricBuilding": "parametric",
        "generate_idf": "parametric",
        "get_5r1c_params": "parametric",
        "BuildingGeometry": "building_geometry",
        "BuildingTABULA": "tabula",
        "ClimateData": "tabula",
        "ThermalMassClass": "tabula",
        "FootprintImporter": "gis_footprint",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(import_module(f".{modules[name]}", __name__), name)
    except ImportError as error:
        if name == "FootprintImporter":
            raise ImportError(
                "FootprintImporter requires the geometry dependencies. "
                "Install epsm_rc[platform,gis] before importing GIS footprints."
            ) from error
        raise
    globals()[name] = value
    return value


# A library emits no output until its host selects handlers.

_stdlib_logging.getLogger("epsm_rc").addHandler(_stdlib_logging.NullHandler())
