"""epsm_rc — ISO 13790:2008 5R1C Building Thermal Model.

A standalone package for hourly building heating/cooling demand simulation
using the simplified 5-resistance, 1-capacitance thermal network from ISO 13790.

Core components:
    - Building5R1C: Main thermal model class
    - SolarGainsCalculator: Solar irradiance → window gains (pvlib)
    - HeatingSystem family: HeatPump, DistrictHeating, DirectElectric, etc.
    - BuildingGeometry: Polygon-based envelope geometry
    - BuildingTABULA: TABULA seasonal energy balance method

Quick start::

    from epsm_rc import Building5R1C, create_swedish_villa

    villa = create_swedish_villa(floor_area_m2=150, era="1976_1990")
    results = villa.simulate_with_epw("goteborg.epw")
    print(f"Annual heating: {results['heating_demand_kwh']:.0f} kWh")

Adapted from DIBS (Dynamic ISO Building Simulator) by IWU, MIT License.
"""

import logging as _stdlib_logging

from .building_geometry import BuildingGeometry
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
from .parametric import ParametricBuilding, generate_idf, get_5r1c_params
from .solar import SolarGainsCalculator, estimate_window_orientation_split, simplified_solar_gains
from .tabula import BuildingTABULA, ClimateData, ThermalMassClass
from .weather import EPWData, get_irradiance, get_temperature, load_epw, parse_ground_temperatures

try:
    from .gis_footprint import FootprintImporter
except ImportError:
    FootprintImporter = None  # type: ignore[assignment,misc]

__all__ = [
    "simulate_building",
    "RCSettings",
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

__version__ = "0.4.0"


# Platform integration imports Core only when explicitly requested.
def __getattr__(name):
    if name in {"EngineComparison", "prepare_engine_comparison"}:
        from . import comparison
        return getattr(comparison, name)
    if name in {"simulate_building", "RCSettings"}:
        from . import platform
        return getattr(platform, name)
    raise AttributeError(name)


# A library emits no output until its host selects handlers.

_stdlib_logging.getLogger("epsm_rc").addHandler(_stdlib_logging.NullHandler())
