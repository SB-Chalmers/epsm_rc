# -*- coding: utf-8 -*-
"""
ISO 13790:2008 5R1C Simplified Hourly Thermal Building Model
=============================================================

Implements the simplified hourly method from ISO 13790:2008 for calculating
building heating/cooling demand. The 5R1C model uses 5 thermal resistances and
1 thermal capacitance to represent building thermal dynamics.

ISO 13790 Standard Structure
----------------------------
The implementation follows these key sections of ISO 13790:2008:

**Section 7 - Building Energy Need for Space Heating and Cooling**
    - Section 7.2.2: Simple hourly method (5R1C model description)
    - Section 7.2.2.2: Main variables (H_tr,is, H_tr,w, H_ve, C_m, A_t)

**Section 8 - Heat Transfer by Transmission**
    - Section 8.3: Transmission heat transfer coefficients (H_tr)
    - Eq. (18): H_tr = Sum(U_i * A_i) for envelope elements

**Section 9 - Heat Transfer by Ventilation**
    - Section 9.3: Ventilation heat transfer coefficient
    - Eq. (21): H_ve = ρ.c_p.V̇ = ρ.c_p.V.n/3600
    - Eq. (27): Temperature adjustment factor b_ve for heat recovery

**Section 12 - Dynamic Parameters**
    - Section 12.2.2: Coupling to thermal mass (H_tr,ms, H_tr,em)
    - Section 12.3.1: Internal heat capacity C_m
    - Table 12: Default kappa values by building class [kJ/(m^2.K)]

**Annex C - Full Set of Equations for Simple Hourly Method (Normative)**
    - C.1: Introduction and Figure C.1 (RC network diagram)
    - C.2: Heat flow distribution (Eq. C.1, C.2, C.3)
    - C.3: Temperature calculations (Eq. C.4-C.12)
    - C.4: Heating/cooling demand calculation procedure

Key Equations Summary
---------------------
Network Parameters (Section 7.2.2.2, Section 12.2.2):
    - Eq. (9):  H_tr,is = h_is * A_tot        where h_is = 3.45 W/(m^2.K)
    - Eq. (64): H_tr,em = 1/(1/H_tr,op - 1/H_tr,ms)
    - Eq. (65): H_tr,ms = h_ms * A_m          where h_ms = 9.1 W/(m^2.K)

Heat Flow Distribution (Annex C.2):
    - Eq. (C.1): Φ_ia = 0.5 * Φ_int
    - Eq. (C.2): Φ_m = (A_m/A_t) * 0.5 * (Φ_int + Φ_sol)
    - Eq. (C.3): Φ_st = (1 - A_m/A_t - H_tr,w/(9.1*A_t)) * 0.5 * (Φ_int + Φ_sol)

Temperature Calculations (Annex C.3):
    - Eq. (C.4): θ_m,t - Crank-Nicolson mass temperature update
    - Eq. (C.5): Φ_m,tot - total heat flow to mass node
    - Eq. (C.6-C.8): Combined conductances H_tr,1, H_tr,2, H_tr,3
    - Eq. (C.10): θ_s - surface temperature
    - Eq. (C.11): θ_air - air temperature
    - Eq. (C.12): θ_op = 0.3*θ_air + 0.7*θ_s - operative temperature

Demand Calculation (Annex C.4.2):
    - Eq. (C.13): Linear interpolation for required heating/cooling power

Network Topology (ISO 13790 Figure C.1)
---------------------------------------
::

    θ_e ────[H_tr,em]────┬────[H_tr,ms]────┬────[H_tr,is]──── θ_air
          (exterior)     │    (mass-surf)  │    (surf-air)      │
                        [C_m]            θ_s                 [H_ve]
                         │               (surface)              │
                        θ_m                                  θ_sup
                       (mass)                              (supply)

    Heat flows: Φ_m → mass node, Φ_st → surface node, Φ_ia → air node

Model Capabilities
------------------
    - Thermal mass dynamics (concrete, masonry, timber)
    - Solar gains through windows (simplified or detailed pvlib/Perez)
    - Solar gains through opaque elements (ISO 13790 Eq. 43)
    - Internal gains (occupants, appliances, lighting)
    - Ventilation with optional heat recovery
    - Infiltration losses
    - Transmission losses through envelope
    - Sky radiation losses (Annex E)
    - Thermal bridges (ΔU_tb methodology)
    - Dynamic ground coupling (monthly ground temperatures)

ISO 13790 Compliance Features (2026-02)
---------------------------------------
The following features provide strict ISO 13790 compliance:

**Sensible Recovery in Reduced Ventilation Conductance (Section 9.3)**
    Mechanical conductance is reduced once by (1 - η_hr); infiltration and
    natural flow are unrecovered. calc_supply_air_temperature() returns outdoor
    temperature as the reduced node's driving temperature, not the physical
    post-exchanger supply state. Standalone behavior changed to remove its
    historical second recovery adjustment.

**Intermittent Heating/Cooling (Section 13.2.3)**
    Parameters t_set_heating_setback, t_set_cooling_setback, and occupancy_schedule
    enable night/absence setback operation. get_active_setpoints() returns the
    appropriate setpoint based on the occupancy schedule.

**Detailed Shading Factors (Section 11.3.3)**
    Parameters f_shading_horizon, f_shading_overhang, and f_shading_fins provide
    component-level shading factors. calc_detailed_shading_factor() computes:
        F_sh,ob = F_hor × F_ov × F_fin

**Strict ISO Heat Flow Distribution (Annex C.2)**
    calc_heat_flows(strict_iso=True) uses exact ISO 13790 equations:
        Eq. (C.1): Φ_ia = 0.5 × Φ_int
        Eq. (C.2): Φ_m = (A_m/A_t) × 0.5 × (Φ_int + Φ_sol)
        Eq. (C.3): Φ_st = (1 - A_m/A_t - H_tr,w/(9.1×A_t)) × 0.5 × (Φ_int + Φ_sol)
    The default flexible mode allows convective_fraction_internal calibration.

References
----------
    - ISO 13790:2008 "Energy performance of buildings - Calculation of energy
      use for space heating and cooling"
    - Adapted from DIBS (Dynamic ISO Building Simulator)
      Copyright 2023 Institut Wohnen und Umwelt (IWU)
      https://github.com/IWUGERMANY/DIBS---Dynamic-ISO-Building-Simulator
      MIT License
    - RC_BuildingSimulator, Copyright 2016 Architecture and Building Systems, ETH Zurich

License: MIT (consistent with DIBS source)

Attribution: Adapted from DIBS by IWU (Bischof, Knoll, Hörner); maintained in EPSM RC.
"""

from __future__ import annotations

import bisect
import random
import warnings
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .solar import SolarGainsCalculator


def _finite_nonnegative(name, value, *, positive=False):
    """Check one scalar physical input without changing the supplied value."""
    if not np.isscalar(value) or not np.isfinite(value) or value < 0 or (positive and value == 0):
        constraint = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {constraint}")


def _input_series(name, values, *, length=None, nonnegative=False):
    """Validate a complete one-dimensional series before thermal state advances."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or (length is not None and len(array) != length):
        raise ValueError(f"{name} must be a nonempty one-dimensional series of the required length")
    if not np.isfinite(array).all() or (nonnegative and (array < 0).any()):
        raise ValueError(
            f"{name} must contain finite{' nonnegative' if nonnegative else ''} values"
        )
    return array


def _refresh_standalone(function):
    """Refresh editable standalone inputs once per outer public calculation.

    An owned Core-adapter model has already been prepared and validated. Nested
    solver trials use the same coefficients and must not rebuild them each time.
    """

    @wraps(function)
    def calculate(self, *args, **kwargs):
        if self._prepared_hourly:
            return function(self, *args, **kwargs)
        depth = getattr(self, "_calculation_depth", 0)
        if depth == 0:
            self._refresh_inputs()
        self._calculation_depth = depth + 1
        try:
            return function(self, *args, **kwargs)
        finally:
            self._calculation_depth = depth

    return calculate


class VentilationSystem(Enum):
    """Ventilation system types with typical heat recovery efficiencies.

    ISO 13790:2008 Section 9.3 defines the ventilation heat transfer coefficient:
        H_ve = ρ_a * c_a * q_ve,k,mn * b_ve,k  [W/K]

    Heat recovery efficiency η_H,hr reduces effective ventilation loss:
        H_ve,adj = H_ve * (1 - η_H,hr * f_ve,t)

    Based on TABULA "Performance of ventilation systems" and EN 15251.
    """

    NATURAL = 1  # Självdrag - natural ventilation, η = 0
    EXHAUST = 2  # Frånluft - exhaust only, η = 0
    BALANCED = 3  # FTX utan VVX - balanced without heat recovery, η = 0
    BALANCED_HR = 4  # FTX med VVX - balanced with heat recovery, η = 0.62-0.70
    BALANCED_HR_HIGH = 5  # FTX med VVX hög effektivitet - high efficiency HR, η = 0.75-0.83
    GROUND_COUPLED = 6  # FTX med jordvärmeväxlare - ground-coupled, η = 0.80-0.90


# Default heat recovery efficiencies by ventilation system type
VENTILATION_HR_EFFICIENCY = {
    VentilationSystem.NATURAL: 0.0,
    VentilationSystem.EXHAUST: 0.0,
    VentilationSystem.BALANCED: 0.0,
    VentilationSystem.BALANCED_HR: 0.65,  # Average of 0.62-0.70
    VentilationSystem.BALANCED_HR_HIGH: 0.80,  # Average of 0.75-0.83
    VentilationSystem.GROUND_COUPLED: 0.85,  # Average of 0.80-0.90
}


class ThermalBridgeClass(Enum):
    """Thermal bridge classification per ISO 13790:2008.

    ISO 13790 Section 8.3.1 defines total transmission coefficient:
        H_tr,adj = H_D + H_g + H_U + H_A  [W/K]

    Where linear thermal bridges (ψ-values) contribute to H_D:
        H_D = Sum(A_i * U_i) + Sum(l_k * ψ_k) + Sum(χ_j)

    ΔU_tb values represent an overall surcharge [W/(m^2.K)] to account for
    thermal bridging when detailed ψ/χ values are unavailable.
    """

    MINIMAL = 1  # Thermally optimized construction (new passive houses)
    LOW = 2  # Standard new construction with thermal break details
    MEDIUM = 3  # Existing buildings with some thermal bridging
    HIGH = 4  # Significant thermal bridges (concrete penetrations, etc.)


# Thermal bridge surcharge by classification [W/(m^2.K)]
THERMAL_BRIDGE_DELTA_U = {
    ThermalBridgeClass.MINIMAL: 0.02,  # Thermally optimized
    ThermalBridgeClass.LOW: 0.05,  # Standard existing buildings
    ThermalBridgeClass.MEDIUM: 0.10,  # Moderate thermal bridging
    ThermalBridgeClass.HIGH: 0.15,  # Significant thermal bridges
}


# Swedish default ground temperatures [°C] for soil under conditioned buildings
# These represent ground temps under a heated slab-on-grade, NOT undisturbed soil.
# Accounts for "thermal pillow" effect where building warms soil in winter.
# Based on typical Swedish climate (annual mean air ~7°C).
# Do NOT use raw EPW ground temps - they are for undisturbed soil, too extreme.
# Reference: EnergyPlus Slab preprocessor methodology (Bahnfleth 1989, Clements 2004)
SWEDISH_DEFAULT_GROUND_TEMPS = [4.0, 3.5, 4.0, 6.0, 9.0, 12.0, 14.5, 14.0, 12.0, 9.0, 6.0, 4.5]


# Internal gains limits for activity-based calculations [W/m^2]
# Per Sveby Swedish standard: residential typically 4-6 W/m^2
INTERNAL_GAINS_MIN_W_M2 = 2.0  # Minimum reasonable (unoccupied)
INTERNAL_GAINS_MAX_W_M2 = 15.0  # Maximum reasonable for residential
# Note: Commercial/office may reach 20-25 W/m^2, but residential rarely exceeds 10 W/m^2


# Hour index at month start for non-leap years (Jan..Dec)
# Used for precise month lookup from hour_of_year (0..8759).
_MONTH_START_HOURS = (0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016)


def estimate_ground_temps_from_air(
    annual_mean_air_temp: float,
    amplitude: float = 5.47,
    phase_shift_months: float = 1.5,
) -> list[float]:
    """Estimate monthly ground temperatures from annual mean air temperature.

    Uses sinusoidal approximation with damping and phase shift to estimate
    soil temperatures under a conditioned building. The building's thermal
    influence (thermal pillow effect) warms the soil relative to air temps.

    This is a simplified alternative to running the E+ Slab preprocessor.

    Defaults calibrated against EnergyPlus Slab preprocessor for 5 Swedish
    climates (Malmö, Göteborg, Stockholm, Sundsvall, Kiruna):
      amplitude=5.47, building_offset=1.21, phase_shift=0.98
      → RMS error 0.64°C (vs 2.16°C with original defaults)

    Parameters
    ----------
    annual_mean_air_temp : float
        Annual mean outdoor air temperature [°C]
    amplitude : float
        Temperature swing amplitude [°C] (default 5.47, calibrated)
    phase_shift_months : float
        Phase lag from air temp peak [months] (default 1.5)

    Returns
    -------
    list[float]
        List of 12 monthly ground temperatures [°C] (Jan-Dec)

    Examples
    --------
    >>> from epsm_rc.iso13790 import estimate_ground_temps_from_air
    >>> temps = estimate_ground_temps_from_air(7.0)  # Swedish climate
    >>> len(temps)
    12
    """
    # Ground under building is warmer than undisturbed soil in winter
    # Calibrated against E+ Slab preprocessor for Swedish climates
    building_offset = 1.21

    # Base temp is slightly above annual mean due to building influence
    base_temp = annual_mean_air_temp + building_offset

    # Generate sinusoidal profile with phase shift
    # Air temp minimum is ~Feb (month 1), maximum is ~Jul-Aug (month 7)
    # Ground temp lags air by phase_shift months
    # Use negative cosine so minimum is at phase_shift after Feb
    # Phase calibrated to 0.98 months (shift from default peak)
    temps = []
    for month in range(12):
        angle = 2 * np.pi * (month - 7.5 + 0.98) / 12  # Calibrated phase
        temp = base_temp + amplitude * np.cos(angle)
        temps.append(round(float(temp), 1))

    return temps


@dataclass
class ConstructionLayer:
    """A single layer in a building construction.

    Used for calculating thermal mass from actual material properties.
    Only the internal layers (exposed to interior) contribute to effective
    thermal mass for daily heating/cooling cycles.

    Parameters
    ----------
    thickness_m : float
        Layer thickness [m]
    density_kg_m3 : float
        Material density [kg/m³]
    specific_heat_j_kgk : float
        Specific heat capacity [J/(kg.K)]
    name : str
        Optional descriptive name

    Common material properties:
        - Concrete: ρ=2400, c=880 → volumetric capacity 2112 kJ/(m³.K)
        - Brick: ρ=1800, c=840 → volumetric capacity 1512 kJ/(m³.K)
        - Timber: ρ=500, c=1600 → volumetric capacity 800 kJ/(m³.K)
        - Gypsum board: ρ=800, c=1000 → volumetric capacity 800 kJ/(m³.K)
        - Insulation: ρ=30, c=1000 → volumetric capacity 30 kJ/(m³.K)
    """

    thickness_m: float
    density_kg_m3: float
    specific_heat_j_kgk: float
    name: str = ""

    def __post_init__(self):
        """Reject negative or nonfinite material properties at entry."""
        self._validate_inputs()

    def _validate_inputs(self):
        for name in ("thickness_m", "density_kg_m3", "specific_heat_j_kgk"):
            _finite_nonnegative(name, getattr(self, name))

    def __repr__(self):
        return (
            f"ConstructionLayer(thickness_m={self.thickness_m:.6g}, "
            f"density_kg_m3={self.density_kg_m3:.6g}, "
            f"specific_heat_j_kgk={self.specific_heat_j_kgk:.6g})"
        )

    def summary(self):
        """Return supplied material properties and areal heat capacity, without I/O."""
        self._validate_inputs()
        return {
            "name": self.name,
            "thickness_m": self.thickness_m,
            "density_kg_m3": self.density_kg_m3,
            "specific_heat_j_kgk": self.specific_heat_j_kgk,
            "thermal_mass_j_m2_k": self.thermal_mass_per_area,
        }

    @property
    def thermal_mass_per_area(self) -> float:
        """Thermal mass per unit area [J/(m^2.K)]."""
        return self.thickness_m * self.density_kg_m3 * self.specific_heat_j_kgk

    @property
    def thermal_mass_kj_m2k(self) -> float:
        """Thermal mass per unit area [kJ/(m^2.K)]."""
        return self.thermal_mass_per_area / 1000


# Common construction layers for convenience
COMMON_LAYERS = {
    "concrete_200mm": ConstructionLayer(0.20, 2400, 880, "200mm concrete"),
    "concrete_150mm": ConstructionLayer(0.15, 2400, 880, "150mm concrete"),
    "concrete_100mm": ConstructionLayer(0.10, 2400, 880, "100mm concrete"),
    "brick_200mm": ConstructionLayer(0.20, 1800, 840, "200mm brick"),
    "brick_120mm": ConstructionLayer(0.12, 1800, 840, "120mm brick"),
    "timber_frame": ConstructionLayer(0.045, 500, 1600, "45mm timber stud"),
    "gypsum_13mm": ConstructionLayer(0.013, 800, 1000, "13mm gypsum board"),
    "gypsum_26mm": ConstructionLayer(0.026, 800, 1000, "2x13mm gypsum board"),
    "screed_50mm": ConstructionLayer(0.05, 2000, 1000, "50mm cement screed"),
    "screed_75mm": ConstructionLayer(0.075, 2000, 1000, "75mm cement screed"),
}


@dataclass
class Construction:
    """Building construction assembly for thermal mass calculation.

    Only the internal layer(s) contribute to effective thermal mass.
    ISO 13790 limits the effective thickness based on thermal penetration depth,
    which is ~10cm for concrete with daily temperature cycles.

    Parameters
    ----------
    name : str
        Descriptive name for the construction
    internal_layers : List[ConstructionLayer]
        List of layers exposed to interior (inside-out order)
    area_m2 : float
        Total area of this construction type [m^2]
    effective_depth_limit_m : float
        Maximum depth that contributes to thermal mass
        (default 0.10m based on daily cycle penetration depth)

    Examples
    --------
    >>> from epsm_rc import Construction, ConstructionLayer
    >>> # Concrete wall with gypsum finish
    >>> wall = Construction(
    ...     name="Heavy wall",
    ...     internal_layers=[
    ...         ConstructionLayer(0.013, 800, 1000, "gypsum"),
    ...         ConstructionLayer(0.20, 2400, 880, "concrete"),
    ...     ],
    ...     area_m2=150.0,
    ... )
    >>> print(f"Wall Cm: {wall.thermal_mass_kj_m2k:.0f} kJ/(m^2.K)")
    Wall Cm: 194 kJ/(m^2.K)
    """

    name: str
    internal_layers: List[ConstructionLayer]
    area_m2: float
    effective_depth_limit_m: float = 0.10  # ~10cm penetration depth for daily cycles

    def __post_init__(self):
        self._validate_inputs()

    def _validate_inputs(self):
        _finite_nonnegative("area_m2", self.area_m2)
        _finite_nonnegative("effective_depth_limit_m", self.effective_depth_limit_m, positive=True)
        if any(not isinstance(layer, ConstructionLayer) for layer in self.internal_layers):
            raise ValueError("internal_layers must contain ConstructionLayer objects")
        for layer in self.internal_layers:
            layer._validate_inputs()

    def __repr__(self):
        return f"Construction(layers={len(self.internal_layers)}, area_m2={self.area_m2!r})"

    def summary(self):
        """Return construction inputs and effective storage, without changing layers."""
        self._validate_inputs()
        return {
            "name": self.name,
            "area_m2": self.area_m2,
            "effective_depth_limit_m": self.effective_depth_limit_m,
            "layers": [layer.summary() for layer in self.internal_layers],
            "thermal_mass_j_k": self.total_thermal_mass_j_k,
        }

    @property
    def thermal_mass_per_area(self) -> float:
        """Effective thermal mass per unit area [J/(m^2.K)].

        Sums layer contributions up to the penetration depth limit.
        """
        total_depth = 0.0
        total_cm = 0.0

        for layer in self.internal_layers:
            remaining_depth = self.effective_depth_limit_m - total_depth
            if remaining_depth <= 0:
                break

            effective_thickness = min(layer.thickness_m, remaining_depth)
            layer_cm = effective_thickness * layer.density_kg_m3 * layer.specific_heat_j_kgk
            total_cm += layer_cm
            total_depth += layer.thickness_m

        return total_cm

    @property
    def thermal_mass_kj_m2k(self) -> float:
        """Effective thermal mass per unit area [kJ/(m^2.K)]."""
        return self.thermal_mass_per_area / 1000

    @property
    def total_thermal_mass_j_k(self) -> float:
        """Total thermal mass for this construction [J/K]."""
        return self.thermal_mass_per_area * self.area_m2


def calculate_thermal_mass_from_constructions(
    constructions: List[Construction],
    floor_area_m2: float,
) -> tuple[float, float]:
    """Calculate total thermal capacitance from construction assemblies.

    ISO 13790:2008 Section 12.3.1.1 Eq. (67) - Internal heat capacity:

        C_m = Sum(kappa_j * A_j)  [J/K]

    Where kappa_j is the internal heat capacity per area from ISO 13786:2007,
    considering only the thermally active portion of each construction
    (typically 100mm depth from internal surface).

    Parameters
    ----------
    constructions : List[Construction]
        List of Construction objects (walls, floor, ceiling, partitions)
    floor_area_m2 : float
        Total floor area for normalization [m^2]

    Returns
    -------
    tuple[float, float]
        Tuple of (C_m in J/K, kappa in J/(m^2.K) per floor area)

    Examples
    --------
    >>> from epsm_rc.iso13790 import Construction, COMMON_LAYERS, calculate_thermal_mass_from_constructions
    >>> constructions = [
    ...     Construction("External walls", [COMMON_LAYERS["gypsum_13mm"], COMMON_LAYERS["concrete_200mm"]], 150),
    ...     Construction("Floor slab", [COMMON_LAYERS["screed_50mm"], COMMON_LAYERS["concrete_150mm"]], 100),
    ...     Construction("Internal walls", [COMMON_LAYERS["gypsum_26mm"]], 80),
    ... ]
    >>> Cm, kappa = calculate_thermal_mass_from_constructions(constructions, 100)
    >>> print(f"Total Cm: {Cm/1e6:.1f} MJ/K, kappa: {kappa/1000:.0f} kJ/(m^2.K)")
    Total Cm: 51.3 MJ/K, kappa: 513 kJ/(m^2.K)
    """
    total_cm = sum(c.total_thermal_mass_j_k for c in constructions)
    kappa = total_cm / floor_area_m2 if floor_area_m2 > 0 else 0.0
    return total_cm, kappa


class BuildingClass(Enum):
    """ISO 13790:2008 Section 12.3.1.2 Table 12 - Building thermal mass classes.

    Determines the internal heat capacity per floor area (kappa) and effective
    mass area factor (A_m/A_f) for the 5R1C simplified hourly model.

    DEPRECATED: Use Construction-based thermal mass calculation instead.
    These discrete categories are retained for backward compatibility.

    ISO 13790 Table 12 defines five classes based on construction type:
        - Very Light: Suspended ceiling/floor, light partitions (kappa <= 80)
        - Light: Suspended ceiling/raised floor, gypsum partitions (80 < kappa <= 110)
        - Medium: Normal concrete ceiling, light partition (110 < kappa <= 165)
        - Heavy: Concrete ceiling, brick walls (165 < kappa <= 260)
        - Very Heavy: Exposed concrete, massive walls (kappa > 260)

    The effective mass area A_m is calculated per ISO 13790 Section 12.2.2 Eq. (66):
        A_m = C_m^2 / Sum(A_j * kappa_j^2)
    For default values, this simplifies to A_m = factor * A_f.
    """

    VERY_LIGHT = 1  # kappa = 80 kJ/(m^2.K), A_m = 2.5 * A_f
    LIGHT = 2  # kappa = 110 kJ/(m^2.K), A_m = 2.5 * A_f
    MEDIUM = 3  # kappa = 165 kJ/(m^2.K), A_m = 2.5 * A_f
    HEAVY = 4  # kappa = 260 kJ/(m^2.K), A_m = 3.0 * A_f
    VERY_HEAVY = 5  # kappa = 370 kJ/(m^2.K), A_m = 3.5 * A_f


# =============================================================================
# ISO 13790:2008 Section 12.3.1.2 Table 12 - Default Thermal Capacitance Values
# =============================================================================
# Internal heat capacity per floor area C_m/A_f = kappa [J/(m^2.K)]
# These are default values; actual calculation per Section 12.3.1.1 Eq. (67)/(68):
#     C_m = Sum(kappa_j * A_j)  where kappa_j is from ISO 13786:2007
#
# DEPRECATED: Use construction-based calculation for accurate results
THERMAL_CAPACITANCE = {
    BuildingClass.VERY_LIGHT: 80_000,  # 80 kJ/(m^2.K) - suspended ceilings, light partitions
    BuildingClass.LIGHT: 110_000,  # 110 kJ/(m^2.K) - timber frame, gypsum partitions
    BuildingClass.MEDIUM: 165_000,  # 165 kJ/(m^2.K) - concrete ceiling, light partitions
    BuildingClass.HEAVY: 260_000,  # 260 kJ/(m^2.K) - concrete + brick partitions
    BuildingClass.VERY_HEAVY: 370_000,  # 370 kJ/(m^2.K) - exposed concrete, massive walls
}

# =============================================================================
# ISO 13790:2008 Section 12.2.2 - Effective Mass Area Factor A_m/A_f
# =============================================================================
# Per Eq. (66): A_m = C_m^2 / Sum(A_j * kappa_j^2)
# For default kappa values, this simplifies to A_m = factor * A_f
# The factor increases with thermal mass as heavy buildings have more
# thermally active surface area participating in heat storage.
MASS_AREA_FACTOR = {
    BuildingClass.VERY_LIGHT: 2.5,  # Light construction
    BuildingClass.LIGHT: 2.5,  # Light construction
    BuildingClass.MEDIUM: 2.5,  # Medium construction
    BuildingClass.HEAVY: 3.0,  # Heavy construction
    BuildingClass.VERY_HEAVY: 3.5,  # Very heavy construction
}


@dataclass
class Building5R1C:
    """ISO 13790 5R1C simplified hourly building thermal model.

    Calculates heating/cooling demand using the resistance-capacitance network
    from ISO 13790 Annex C. Suitable for hourly or sub-hourly simulation.

    ``simulate``, ``step`` and ``calc_heating_demand`` validate editable model
    inputs and refresh derived coefficients before advancing thermal state.
    ``summary()`` refreshes coefficients without advancing the temperatures.
    The low-level network equation helpers use those current coefficients;
    call ``summary()`` before inspecting them after editing physical inputs.

    This implementation focuses on residential buildings and provides:
    - Heating demand calculation with thermal mass dynamics
    - Solar and internal gain integration
    - Ventilation with optional heat recovery

    Parameters
    ----------
    floor_area_m2 : float
        Heated floor area [m^2]
    building_class : Optional[BuildingClass]
        Thermal mass class (affects time constant)
    u_walls : float
        U-value of opaque walls [W/(m^2.K)]
    u_roof : float
        U-value of roof [W/(m^2.K)]
    u_floor : float
        U-value of floor/ground [W/(m^2.K)]
    u_windows : float
        U-value of windows [W/(m^2.K)]
    wall_area_m2 : float
        Total external wall area [m^2]
    roof_area_m2 : float
        Roof area [m^2]
    floor_area_ground_m2 : float
        Floor area in contact with ground [m^2]
    window_area_m2 : float
        Total window area [m^2]
    building_height_m : float
        Average building height [m]
    ach_infiltration : float
        Air changes per hour from infiltration [1/h]
    ach_ventilation : float
        Air changes per hour from mechanical ventilation [1/h]
    heat_recovery_efficiency : float
        Heat recovery efficiency (0-1)
    g_value : float
        Solar heat gain coefficient of windows
    t_set_heating : float
        Heating setpoint temperature [°C]
    t_set_cooling : float
        Cooling setpoint temperature [°C]
    internal_gains_w_per_m2 : float
        Average internal gains [W/m^2]

    Examples
    --------
    >>> from epsm_rc import Building5R1C, BuildingClass
    >>> building = Building5R1C(
    ...     floor_area_m2=120,
    ...     building_class=BuildingClass.MEDIUM,
    ...     u_walls=0.25,
    ...     u_roof=0.15,
    ...     u_floor=0.20,
    ...     u_windows=1.2,
    ...     wall_area_m2=150,
    ...     roof_area_m2=70,
    ...     floor_area_ground_m2=70,
    ...     window_area_m2=25,
    ... )
    >>> # Simulate one hour
    >>> t_out = -5.0  # Outdoor temp [°C]
    >>> solar_gains = 500  # Solar gains through windows [W]
    >>> q_heat = building.calc_heating_demand(t_out, solar_gains)
    """

    # Building geometry
    floor_area_m2: float = 120.0
    building_class: Optional[BuildingClass] = (
        BuildingClass.MEDIUM
    )  # DEPRECATED: use thermal_capacitance_j_m2k

    # Construction-based thermal mass (preferred over building_class)
    # If set, overrides building_class for thermal capacitance calculation
    # Units: J/(m^2.K) per floor area (same as ISO 13790 kappa)
    # Typical values: 80,000 (light timber) to 370,000 (heavy concrete)
    thermal_capacitance_j_m2k: Optional[float] = None  # e.g., 165_000 for medium

    # List of Construction objects for detailed thermal mass calculation
    # If provided, thermal_capacitance_j_m2k is calculated automatically
    constructions: Optional[List[Construction]] = None

    # U-values [W/(m^2.K)]
    u_walls: float = 0.25
    u_roof: float = 0.15
    u_floor: float = 0.20
    u_windows: float = 1.2

    # Areas [m^2]
    wall_area_m2: float = 150.0
    roof_area_m2: float = 70.0
    floor_area_ground_m2: float = 70.0
    window_area_m2: float = 25.0

    # Building volume
    building_height_m: float = 2.5  # External building height (for wall area context)
    ceiling_height_m: float = 2.5  # Interior ceiling height per floor (for volume)

    # Ventilation [1/h]
    ach_infiltration: float = 0.2
    ach_ventilation: float = 0.5
    heat_recovery_efficiency: float = 0.0

    # Solar properties
    g_value: float = 0.6  # Solar heat gain coefficient at normal incidence
    frame_factor: float = 0.7  # Glass fraction of window (1 - frame)
    shading_factor: float = 1.0  # External shading (1 = no shading)

    # Non-perpendicular incidence correction factor (TABULA/ISO usage)
    # Applied when no explicit angle-dependent g-value model is configured.
    # Typical value: 0.9 (accounts for average incidence losses).
    non_perpendicular_incidence_factor: float = 0.9

    # Angle-dependent g-value polynomial coefficients (optional)
    # Models how SHGC varies with incidence angle: g(θ) = g_0 × Σ(coef[i] × θ^i)
    # where θ is incidence angle in radians. Applied to direct beam only.
    # If None, g_value is used uniformly (hemispherical average per ISO 13790).
    # Example for double-pane clear: (1.0, -0.189, 0.644, -0.596) from Modelica
    # EnergyPlus has category-specific coefficients (see Engineering Reference).
    g_value_angle_coefficients: Optional[Tuple[float, ...]] = None

    # Orientation-specific window areas [m^2] (optional - enables detailed solar)
    # If None, falls back to simplified solar using total window_area_m2
    window_area_south: Optional[float] = None
    window_area_north: Optional[float] = None
    window_area_east: Optional[float] = None
    window_area_west: Optional[float] = None

    # Site location for solar calculations
    latitude: float = 57.7  # Default: Gothenburg, Sweden
    longitude: float = 12.0  # Default: Gothenburg, Sweden

    # EPW weather file path (if set, enables detailed solar from EPW)
    epw_path: Optional[Path] = None

    # Solar calculation method: 'simplified', 'detailed', or 'auto'
    # 'auto' (default): Uses orientation-aware pvlib when EPW or orientation data is available
    # 'detailed': Uses Perez with supplied DNI/DHI; GHI-only decomposition uses isotropic sky
    # 'simplified': Uses GHI * window_area * g_value * 0.5 (no orientation)
    solar_method: Literal["simplified", "detailed", "auto"] = "auto"

    # Solar gains through opaque elements (ISO 13790 enhancement)
    wall_absorptivity: float = 0.6  # αS,c - solar absorptivity of walls (dark=0.9, light=0.3)
    roof_absorptivity: float = 0.6  # αS,c - solar absorptivity of roof
    r_se_wall: float = 0.04  # External surface resistance walls [m^2K/W]
    r_se_roof: float = 0.04  # External surface resistance roof [m^2K/W]

    # Sky radiation parameters (ISO 13790 Annex E)
    # Φr = Rse * Uc * Ac * hr * Δθer
    form_factor_roof: float = 1.0  # F_r - view factor to sky for roof (horizontal=1.0)
    form_factor_wall: float = 0.5  # F_r - view factor to sky for walls (vertical=0.5)
    delta_theta_er: float = 11.0  # Δθer - sky-air temperature difference [K] (EN 13790 default)
    h_r: float = 5.0  # Radiative heat transfer coefficient [W/(m^2.K)]

    # =========================================================================
    # ISO 13790 Network Parameters (advanced overrides)
    # =========================================================================
    # These allow matching other simulation tools like Modelica Buildings Library

    # Surface-air heat transfer coefficient [W/(m²K)]
    # ISO 13790 default: 3.45 (combined convective + radiative)
    # BESTEST/Modelica: 2.1 (convective only, radiation handled separately)
    h_is: float = 3.45
    h_ms: float = 9.1  # Mass-surface coefficient [W/(m²K)]
    air_volumetric_heat_capacity_j_m3_k: float = 1200.0

    # Total internal surface area [m²] (optional override)
    # If None, uses ISO 13790 default: A_tot = 4.5 × A_f
    # Modelica uses actual surface areas: sum(A_walls) + A_windows + A_roof + A_floor
    a_tot_surfaces_m2: Optional[float] = None

    # Effective mass area factor (A_m / A_f) override
    # If None, derived from building class or thermal capacitance
    # Modelica uses explicit facMas parameter (e.g., 2.95 for BESTEST)
    mass_area_factor_override: Optional[float] = None

    # Setpoints [°C]
    t_set_heating: float = 21.0
    t_set_cooling: float = 26.0

    # Temperature control mode
    use_operative_temperature: bool = False  # If True, control to T_op instead of T_air

    # Internal gains [W/m^2]
    # Sveby Swedish standard for residential: 4-6 W/m^2 typical
    # TABULA uses 3.0 W/m^2 as default
    # Activity-based calculations may produce higher values - capped at reasonable maximum
    internal_gains_w_per_m2: float = 5.0  # Sveby typical for Swedish residential

    # Convective fraction of internal gains
    # ISO 13790 default: 0.5 (50% convective, 50% radiative)
    # EnergyPlus typical: ~0.67 weighted average:
    #   - People: 70% convective, 30% radiative
    #   - Lights: 60% convective, 40% radiative
    #   - Equipment: 70% convective, 30% radiative
    # Higher convective fraction means more direct heating of air, less to thermal mass
    # This significantly affects heating demand for high internal gain buildings
    # A weighted 0.67 is an illustrative alternative, not a validated universal correction.
    convective_fraction_internal: float = 0.5

    # Temperature adjustment for ground coupling
    temp_adj_ground: float = 0.5  # b_gw factor for ground floor (static default)

    # Monthly ground temperatures for dynamic ground coupling [°C]
    # If provided, ground losses are calculated using actual ground temp instead of temp_adj_ground factor
    # Format: list of 12 monthly average ground temps (Jan-Dec)
    #
    # IMPORTANT: Do NOT use raw EPW "undisturbed" ground temperatures!
    # EPW ground temps represent soil far from buildings - too cold in winter, too warm in summer.
    # Per EnergyPlus documentation: "Do not use undisturbed ground temperatures from weather data.
    # These values are too extreme for soil under a conditioned building."
    #
    # Proper options:
    # 1. Use E+ Slab/Basement preprocessor (Bahnfleth 1989, Clements 2004)
    # 2. Use these Swedish defaults (estimated for conditioned buildings):
    #    [4.0, 3.5, 4.0, 6.0, 9.0, 12.0, 14.5, 14.0, 12.0, 9.0, 6.0, 4.5]
    # 3. Use simplified rule: annual_mean_air_temp ± 2-3°C damped seasonal variation
    #
    # Swedish defaults below represent ground under heated slab-on-grade building,
    # accounting for thermal pillow effect where building warms soil in winter.
    ground_temps_monthly: Optional[List[float]] = None

    # Thermal bridge surcharge [W/(m^2.K)]
    # Added to all envelope U-values per ISO 13790 / TABULA methodology
    # Can be set directly or via thermal_bridge_class
    # Typical values: MINIMAL=0.02, LOW=0.05, MEDIUM=0.10, HIGH=0.15
    delta_u_thermal_bridge: float = 0.0
    thermal_bridge_class: Optional[ThermalBridgeClass] = None  # If set, overrides delta_u

    # Ventilation system type (optional - sets default heat recovery)
    ventilation_system: Optional[VentilationSystem] = (
        None  # If set, overrides heat_recovery_efficiency
    )

    # Window opening behaviour (stochastic ventilation for cooling)
    # 0.0 = never open windows, 1.0 = always open when beneficial
    # Controls probability of occupant opening windows when T_air > T_set_cooling and T_out < T_air
    ventilation_behaviour: float = 0.0
    ach_window_open: float = 3.0  # Air changes when windows are open [1/h] (Sveby: 2-4 ACH)

    # =========================================================================
    # ISO 13790 Section 13.2.3 - Intermittent Heating/Cooling
    # =========================================================================
    # Setback temperatures for reduced operation periods (night/absence)
    # When None, continuous operation at t_set_heating/cooling is assumed
    t_set_heating_setback: Optional[float] = None  # e.g., 16.0°C for night setback
    t_set_cooling_setback: Optional[float] = None  # e.g., 28.0°C for night setback

    # Occupancy/setback schedule (optional)
    # If provided, should be array of 8760 booleans (True=occupied/normal, False=setback)
    # If None, continuous operation is assumed
    occupancy_schedule: Optional[np.ndarray] = None

    # =========================================================================
    # ISO 13790 Section 11.3.3 - Detailed Shading Factors
    # =========================================================================
    # External obstacle shading factors (F_sh,ob = F_hor × F_ov × F_fin)
    # F_hor: horizon shading (distant obstacles, hills) - default 1.0 (no obstruction)
    # F_ov: overhang shading (horizontal projections above window) - default 1.0
    # F_fin: fin shading (vertical projections beside window) - default 1.0
    # These are more detailed than the simple shading_factor above
    f_shading_horizon: float = 1.0  # F_hor - horizon obstructions
    f_shading_overhang: float = 1.0  # F_ov - overhang/balcony above window
    f_shading_fins: float = 1.0  # F_fin - side fins/projections

    # Internal state (updated during simulation)
    t_m: float = field(default=20.0, init=False)  # Mass temperature [°C]
    t_s: float = field(default=20.0, init=False)  # Surface temperature [°C]
    t_air: float = field(default=20.0, init=False)  # Air temperature [°C]
    _windows_open: bool = field(default=False, init=False)  # Track if windows were opened last step

    # Derived properties (calculated in __post_init__)
    _c_m: float = field(default=0.0, init=False, repr=False)
    _h_tr_op: float = field(default=0.0, init=False, repr=False)
    _h_tr_w: float = field(default=0.0, init=False, repr=False)
    _h_tr_tb: float = field(default=0.0, init=False, repr=False)  # Thermal bridges
    _h_tr_ms: float = field(default=0.0, init=False, repr=False)
    _h_tr_is: float = field(default=0.0, init=False, repr=False)
    _h_tr_em: float = field(default=0.0, init=False, repr=False)
    _h_ve: float = field(default=0.0, init=False, repr=False)
    _mass_area: float = field(default=0.0, init=False, repr=False)
    _a_tot: float = field(default=0.0, init=False, repr=False)
    _building_vol: float = field(default=0.0, init=False, repr=False)

    # Supply air temperature state (for heat recovery)
    _t_supply: float = field(default=20.0, init=False, repr=False)

    # Solar calculator (lazy initialized)
    _solar_calc: Optional["SolarGainsCalculator"] = field(default=None, init=False, repr=False)
    _window_areas_by_orientation: Optional[dict] = field(default=None, init=False, repr=False)

    # Enabled only on a fresh deterministic Core-adapter model. Ordinary mutable
    # standalone instances retain uncached property evaluation and public API.
    _prepared_hourly = False

    def _prepare_deterministic_hourly(self):
        """Prepare fixed coefficients after Core's declared U-value uplift.

        The adapter owns this model for one run, uses one-hour steps, and disables
        stochastic windows. Only airflow, gains and setpoints change thereafter.
        Rebuilding physical properties disables this specialization again.
        """
        self._calculate_thermal_properties()
        self._prepared_sky = self.calc_sky_radiation_loss()
        self._prepared_mass_ratio = self._mass_area / self._a_tot if self._a_tot > 0 else 0.0
        self._prepared_window_ratio = (
            self._h_tr_w / (self.h_ms * self._a_tot) if self._a_tot > 0 else 0.0
        )
        self._prepared_hourly = True
        self._prepared_airflow = None
        self._hour_heat_flows = None
        self._update_ventilation_coefficient()

    def __post_init__(self):
        """Calculate derived thermal properties."""
        self._validate_inputs()
        # Apply thermal bridge class if specified
        if self.thermal_bridge_class is not None:
            self.delta_u_thermal_bridge = THERMAL_BRIDGE_DELTA_U[self.thermal_bridge_class]

        # Apply ventilation system if specified
        if self.ventilation_system is not None:
            self.heat_recovery_efficiency = VENTILATION_HR_EFFICIENCY[self.ventilation_system]

        # Validate internal gains are within reasonable range
        # Activity-based calculations can sometimes produce unrealistic values
        if self.internal_gains_w_per_m2 > INTERNAL_GAINS_MAX_W_M2:
            import warnings

            warnings.warn(
                f"Internal gains {self.internal_gains_w_per_m2:.1f} W/m^2 exceeds maximum "
                f"reasonable value {INTERNAL_GAINS_MAX_W_M2} W/m^2 for residential. "
                f"Capping to maximum. Check activity-based calculation inputs.",
                UserWarning,
            )
            self.internal_gains_w_per_m2 = INTERNAL_GAINS_MAX_W_M2

        if self.ground_temps_monthly is not None and len(self.ground_temps_monthly) != 12:
            raise ValueError(
                f"ground_temps_monthly must contain 12 monthly values, got {len(self.ground_temps_monthly)}"
            )

        if self.non_perpendicular_incidence_factor < 0.0:
            raise ValueError("non_perpendicular_incidence_factor must be >= 0")

        self._calculate_thermal_properties()

    def _validate_inputs(self):
        """Check editable physical inputs before deriving coefficients or advancing state."""
        for name in (
            "floor_area_m2",
            "ceiling_height_m",
            "building_height_m",
            "h_is",
            "h_ms",
            "air_volumetric_heat_capacity_j_m3_k",
        ):
            _finite_nonnegative(name, getattr(self, name), positive=True)
        for name in (
            "u_walls",
            "u_roof",
            "u_floor",
            "u_windows",
            "wall_area_m2",
            "roof_area_m2",
            "floor_area_ground_m2",
            "window_area_m2",
            "ach_infiltration",
            "ach_ventilation",
            "internal_gains_w_per_m2",
            "r_se_wall",
            "r_se_roof",
            "delta_theta_er",
            "h_r",
            "delta_u_thermal_bridge",
            "ach_window_open",
        ):
            _finite_nonnegative(name, getattr(self, name))
        for name in (
            "heat_recovery_efficiency",
            "g_value",
            "frame_factor",
            "shading_factor",
            "non_perpendicular_incidence_factor",
            "wall_absorptivity",
            "roof_absorptivity",
            "form_factor_roof",
            "form_factor_wall",
            "convective_fraction_internal",
            "temp_adj_ground",
            "ventilation_behaviour",
            "f_shading_horizon",
            "f_shading_overhang",
            "f_shading_fins",
        ):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be >= 0 and <= 1, and finite")
        for name in ("thermal_capacitance_j_m2k", "a_tot_surfaces_m2", "mass_area_factor_override"):
            if getattr(self, name) is not None:
                _finite_nonnegative(name, getattr(self, name), positive=True)
        for name in (
            "window_area_north",
            "window_area_south",
            "window_area_east",
            "window_area_west",
        ):
            if getattr(self, name) is not None:
                _finite_nonnegative(name, getattr(self, name))
        for name in (
            "t_set_heating",
            "t_set_cooling",
            "t_set_heating_setback",
            "t_set_cooling_setback",
        ):
            if getattr(self, name) is not None and not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.t_set_heating > self.t_set_cooling:
            raise ValueError("Heating setpoint must not exceed cooling setpoint")
        if not all(np.isfinite(value) for value in (self.t_air, self.t_s, self.t_m)):
            raise ValueError("Current thermal-state temperatures must be finite")
        setback_heat = (
            self.t_set_heating if self.t_set_heating_setback is None else self.t_set_heating_setback
        )
        setback_cool = (
            self.t_set_cooling if self.t_set_cooling_setback is None else self.t_set_cooling_setback
        )
        if setback_heat > setback_cool:
            raise ValueError("Setback heating setpoint must not exceed setback cooling setpoint")
        if self.ground_temps_monthly is not None:
            if len(self.ground_temps_monthly) != 12:
                raise ValueError("ground_temps_monthly must contain 12 monthly values")
            _input_series("ground_temps_monthly", self.ground_temps_monthly, length=12)
        for name, expected in (
            ("building_class", BuildingClass),
            ("thermal_bridge_class", ThermalBridgeClass),
            ("ventilation_system", VentilationSystem),
        ):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), expected):
                raise ValueError(f"{name} must be a {expected.__name__}")
        if self.occupancy_schedule is not None:
            schedule = _input_series("occupancy_schedule", self.occupancy_schedule)
            if not np.isin(schedule, [0, 1]).all():
                raise ValueError("occupancy_schedule must contain boolean or 0/1 values")
        if self.g_value_angle_coefficients is not None:
            _input_series("g_value_angle_coefficients", self.g_value_angle_coefficients)
        if self.solar_method not in {"auto", "simplified", "detailed"}:
            raise ValueError("solar_method must be auto, simplified or detailed")
        if not np.isfinite(self.latitude) or not -90 <= self.latitude <= 90:
            raise ValueError("latitude must be finite and in [-90, 90]")
        if not np.isfinite(self.longitude) or not -180 <= self.longitude <= 180:
            raise ValueError("longitude must be finite and in [-180, 180]")
        if self.constructions is not None:
            for construction in self.constructions:
                if not isinstance(construction, Construction):
                    raise ValueError("constructions must contain Construction objects")
                construction._validate_inputs()
            if (
                not self.constructions
                or sum(c.total_thermal_mass_j_k for c in self.constructions) <= 0
            ):
                raise ValueError("constructions require positive effective thermal storage")

    def _refresh_inputs(self):
        self._validate_inputs()
        if self.thermal_bridge_class is not None:
            self.delta_u_thermal_bridge = THERMAL_BRIDGE_DELTA_U[self.thermal_bridge_class]
        if self.ventilation_system is not None:
            self.heat_recovery_efficiency = VENTILATION_HR_EFFICIENCY[self.ventilation_system]
        solar_key = (
            self.epw_path,
            self.latitude,
            self.longitude,
            self.window_area_m2,
            self.window_area_north,
            self.window_area_south,
            self.window_area_east,
            self.window_area_west,
        )
        if solar_key != getattr(self, "_solar_inputs_key", None):
            self._solar_calc = None
            self._window_areas_by_orientation = None
            self._solar_inputs_key = solar_key
        self._calculate_thermal_properties()

    @staticmethod
    def _validate_timestep_and_capacity(dt_hours, max_heating_power, max_cooling_power):
        _finite_nonnegative("dt_hours", dt_hours, positive=True)
        if (
            not np.isscalar(max_heating_power)
            or np.isnan(max_heating_power)
            or max_heating_power < 0
        ):
            raise ValueError("max_heating_power must be nonnegative; +inf means unlimited")
        if not np.isscalar(max_cooling_power) or np.isnan(max_cooling_power):
            raise ValueError(
                "max_cooling_power must be a scalar capacity; either infinity means unlimited"
            )

    @staticmethod
    def _validate_forcing(t_out, solar_gains, internal_gains, solar_gains_opaque):
        if not np.isscalar(t_out) or not np.isfinite(t_out):
            raise ValueError("t_out must be finite")
        for name, value in (
            ("solar_gains", solar_gains),
            ("solar_gains_opaque", solar_gains_opaque),
        ):
            _finite_nonnegative(name, value)
        if internal_gains is not None:
            _finite_nonnegative("internal_gains", internal_gains)

    def __repr__(self):
        """Show the physical scale and current air temperature without expanding arrays."""
        return f"Building5R1C(floor_area_m2={self.floor_area_m2!r}, t_air_c={self.t_air!r})"

    @_refresh_standalone
    def summary(self):
        """Return current model scale and thermal state without reading weather files.

        Returns
        -------
        dict
            Areas, volume, effective thermal capacitance and current temperatures.
            Editable physical inputs are validated and derived coefficients refreshed.

        Examples
        --------
        >>> from epsm_rc import Building5R1C
        >>> Building5R1C(floor_area_m2=100).summary()["floor_area_m2"]
        100
        """
        return {
            "floor_area_m2": self.floor_area_m2,
            "volume_m3": self._building_vol,
            "thermal_capacitance_j_k": self._c_m,
            "heat_loss_coefficient_w_k": self.heat_loss_coefficient,
            "air_temperature_c": self.t_air,
            "surface_temperature_c": self.t_s,
            "mass_temperature_c": self.t_m,
        }

    def _month_index_from_hour_of_year(self, hour_of_year: int) -> int:
        """Return month index (0=Jan..11=Dec) for a non-leap-year hour index."""
        # Wrap safely for long simulations while keeping non-leap-year mapping.
        hour_wrapped = int(hour_of_year) % 8760
        return bisect.bisect_right(_MONTH_START_HOURS, hour_wrapped) - 1

    def _effective_incidence_factor(self) -> float:
        """Return incidence factor; disable constant factor when angular model is active."""
        if self.g_value_angle_coefficients is not None:
            return 1.0
        return self.non_perpendicular_incidence_factor

    def _calculate_thermal_properties(self):
        """Calculate the 5R1C network parameters from building geometry.

        Implements ISO 13790:2008 Section 7.2.2.2 and Section 12.2.2 for the simplified
        hourly method network parameters.

        Calculates:
            - A_tot: Total internal surface area [Section 7.2.2.2, λat = 4.5]
            - C_m: Internal heat capacity [Section 12.3.1, Eq. (67)/(68)]
            - A_m: Effective mass area [Section 12.2.2, Eq. (66)]
            - H_tr,op: Opaque transmission [Section 8.3, Eq. (18)]
            - H_tr,w: Window transmission [Section 8.3, Eq. (18)]
            - H_tr,ms: Mass-surface coupling [Section 12.2.2, Eq. (65)]
            - H_tr,is: Surface-air coupling [Section 7.2.2.2, Eq. (9)]
            - H_tr,em: Mass-exterior coupling [Section 12.2.2, Eq. (64)]
        """
        self._prepared_hourly = False
        A_f = self.floor_area_m2

        # Building volume [m³]
        # Use ceiling_height_m (interior height per floor), NOT building_height_m
        # This is critical for multi-story buildings where floor_area is total across all floors
        self._building_vol = A_f * self.ceiling_height_m

        # =========================================================================
        # ISO 13790:2008 Section 7.2.2.2 - Total Internal Surface Area
        # =========================================================================
        # A_tot = λ_at * A_f  where λ_at = 4.5 (dimensionless ratio)
        # This represents all surfaces facing the building zone.
        #
        # Note: The Modelica Buildings Library uses actual surface areas instead:
        #   A_tot = Sum(A_wall) + Sum(A_win) + A_roof + A_floor + internal_floors
        # Testing showed this shifts mean bias from +2.8% to -2.1% vs EnergyPlus
        # (60-sample GSA) but increases scatter (MAE 5.98%→6.48%, RMSE 7.36%→8.50%).
        # The 4.5 factor implicitly accounts for furniture and partition surfaces
        # that the pure geometric calculation misses. Keeping ISO default.
        #
        # Override: Use explicit a_tot_surfaces_m2 if provided (for Modelica parity)
        if self.a_tot_surfaces_m2 is not None:
            self._a_tot = self.a_tot_surfaces_m2
        else:
            self._a_tot = 4.5 * A_f

        # =========================================================================
        # ISO 13790:2008 Section 12.3.1 - Internal Heat Capacity C_m
        # =========================================================================
        # Priority order for thermal capacitance:
        # 1. Explicit constructions list (most accurate, per Eq. 67/68)
        # 2. Explicit thermal_capacitance_j_m2k value
        # 3. Fallback to building_class lookup (Table 12, deprecated)
        if self.constructions is not None:
            # Calculate from actual construction properties
            total_cm, kappa = calculate_thermal_mass_from_constructions(self.constructions, A_f)
            self._c_m = total_cm
            # Estimate mass area from kappa (reverse of ISO 13790 Table 12)
            if kappa >= 260_000:
                self._mass_area = 3.5 * A_f
            elif kappa >= 165_000:
                self._mass_area = 3.0 * A_f
            else:
                self._mass_area = 2.5 * A_f
        elif self.thermal_capacitance_j_m2k is not None:
            # Use explicit value
            kappa = self.thermal_capacitance_j_m2k
            self._c_m = kappa * A_f
            # Estimate mass area from kappa
            if kappa >= 260_000:
                self._mass_area = 3.5 * A_f
            elif kappa >= 165_000:
                self._mass_area = 3.0 * A_f
            else:
                self._mass_area = 2.5 * A_f
        elif self.building_class is not None:
            # Fallback to deprecated building class lookup
            self._mass_area = MASS_AREA_FACTOR[self.building_class] * A_f
            kappa = THERMAL_CAPACITANCE[self.building_class]
            self._c_m = kappa * A_f
        else:
            # Default to MEDIUM if nothing specified
            self._mass_area = 2.5 * A_f
            kappa = 165_000
            self._c_m = kappa * A_f

        # Override mass area factor if explicitly provided (for Modelica parity)
        if self.mass_area_factor_override is not None:
            self._mass_area = self.mass_area_factor_override * A_f

        # =========================================================================
        # ISO 13790:2008 Section 8.3 - Transmission Heat Transfer Coefficients
        # =========================================================================
        # Eq. (18): H_tr = Sum(U_i * A_i * b_tr,i)
        # H_tr_op: opaque elements (walls, roof, ground floor)
        # Note: Ground floor losses handled separately if ground_temps_monthly is set
        self._h_tr_op_walls_roof = (
            self.u_walls * self.wall_area_m2 + self.u_roof * self.roof_area_m2
        )

        # Ground floor coefficient [W/K] - used for dynamic or static ground coupling
        self._h_tr_ground = self.u_floor * self.floor_area_ground_m2

        # Combined H_tr_op: ALWAYS includes ground to maintain ISO 13790 RC topology.
        # Per ISO 13790 Section 8.3 Eq. (18): H_tr,op = Sum(U_i * A_i * b_tr,i)
        # Ground coupling handled via weighted equivalent outdoor temperature
        # in calc_phi_m_tot(), NOT as a separate term outside the network.
        if self.ground_temps_monthly is not None:
            # Monthly ground temps: include full ground in H_tr_op.
            # Driving temperature handled as weighted θ_e_opaque in calc_phi_m_tot.
            self._h_tr_op = self._h_tr_op_walls_roof + self._h_tr_ground
        else:
            # Static: use temp_adj_ground factor (ISO 13790 default b_tr,g)
            self._h_tr_op = self._h_tr_op_walls_roof + self._h_tr_ground * self.temp_adj_ground

        # H_tr_w: windows/glazing (ISO 13790 Section 8.3 Eq. 18)
        self._h_tr_w = self.u_windows * self.window_area_m2

        # Thermal bridge surcharge (ISO 13790 Section 8.3 / TABULA ΔU_tb methodology)
        # H_tr_tb = ΔU_tb * SumA_env (applied to total envelope area)
        if self.delta_u_thermal_bridge > 0:
            total_envelope_area = (
                self.wall_area_m2
                + self.roof_area_m2
                + self.floor_area_ground_m2
                + self.window_area_m2
            )
            self._h_tr_tb = self.delta_u_thermal_bridge * total_envelope_area
            # Add thermal bridge losses to opaque transmission
            self._h_tr_op += self._h_tr_tb
        else:
            self._h_tr_tb = 0.0

        # =========================================================================
        # ISO 13790:2008 Section 12.2.2 - Coupling to Thermal Mass
        # =========================================================================
        # Eq. (65): H_tr,ms = h_ms * A_m  where h_ms = 9.1 W/(m^2.K)
        # This couples the thermal mass node (θ_m) to the surface node (θ_s)
        self._h_tr_ms = self.h_ms * self._mass_area

        # =========================================================================
        # ISO 13790:2008 Section 7.2.2.2 Eq. (9) - Surface-Air Coupling
        # =========================================================================
        # H_tr,is = h_is * A_tot  where h_is = 3.45 W/(m^2.K) (ISO default)
        # This couples the surface node (θ_s) to the air node (θ_air)
        # h_is represents combined convective heat transfer at internal surfaces
        # Note: For BESTEST/Modelica validation, use h_is=2.1 (convective only)
        self._h_tr_is = self.h_is * self._a_tot

        # =========================================================================
        # ISO 13790:2008 Section 12.2.2 Eq. (64) - Mass-Exterior Coupling
        # =========================================================================
        # H_tr,em = 1 / (1/H_tr,op - 1/H_tr,ms)
        # This ensures energy balance in the RC network by coupling the
        # thermal mass directly to the exterior through opaque elements.
        if self._h_tr_op > 0 and self._h_tr_ms > 0:
            inv_h_tr_em = (1.0 / self._h_tr_op) - (1.0 / self._h_tr_ms)
            if inv_h_tr_em > 0:
                self._h_tr_em = 1.0 / inv_h_tr_em
            else:
                # Fallback for very high mass area
                self._h_tr_em = self._h_tr_op
        else:
            self._h_tr_em = 0.0

        # =========================================================================
        # ISO 13790:2008 Section 9.3 - Ventilation Heat Transfer Coefficient
        # =========================================================================
        self._update_ventilation_coefficient()

    def _update_ventilation_coefficient(self, ach_total: Optional[float] = None):
        """Update ventilation heat transfer coefficient.

        ISO 13790:2008 Section 9.3, Eq. (21) and Eq. (27).

        Eq. (21): H_ve = ρ.c_p.V̇ = ρ.c_p.V.n/3600  [W/K]
        Eq. (27): b_ve = 1 - (n_mech/n_tot) * η_hr   (heat recovery adjustment)

        Parameters
        ----------
        ach_total : Optional[float]
            Total flow [1/h], including the declared mechanical share.
            Added flow is unrecovered; totals below mechanical flow fail.
        """
        if ach_total is None:
            ach_total = self.ach_infiltration + self.ach_ventilation
        if self._prepared_hourly:
            airflow_key = (ach_total, self.ach_ventilation, self.heat_recovery_efficiency)
            if airflow_key == self._prepared_airflow:
                return

        if not np.isfinite(ach_total) or ach_total < 0:
            raise ValueError("Total air change rate must be finite and nonnegative")
        if ach_total < self.ach_ventilation:
            raise ValueError("Total air change rate cannot omit the declared mechanical flow")
        # Only mechanical flow passes the heat exchanger. Infiltration and any
        # additional natural/window flow keep their full outdoor conductance.
        ach_unrecovered = ach_total - self.ach_ventilation
        ach_effective = ach_unrecovered + self.ach_ventilation * (
            1.0 - self.heat_recovery_efficiency
        )
        self._h_ve = (
            self.air_volumetric_heat_capacity_j_m3_k * self._building_vol * ach_effective / 3600.0
        )
        if self._prepared_hourly:
            self._prepared_airflow = airflow_key
            h1 = (
                1.0 / (1.0 / self._h_ve + 1.0 / self._h_tr_is)
                if self._h_ve > 0 and self._h_tr_is > 0
                else 0.0
            )
            h2 = h1 + self._h_tr_w
            h3 = 1.0 / (1.0 / h2 + 1.0 / self._h_tr_ms) if h2 > 0 and self._h_tr_ms > 0 else 0.0
            self._prepared_network = h1, h2, h3
            c_dt = self._c_m / 3600.0
            h_sum = h3 + self._h_tr_em
            self._prepared_cn = c_dt - 0.5 * h_sum, c_dt + 0.5 * h_sum

    def _handle_window_opening(self, t_out: float) -> float:
        """Determine additional ventilation from window opening (stochastic).

        Implements occupant behaviour for natural ventilation cooling.
        Windows may be opened when:
        1. Indoor air temperature exceeds cooling setpoint (T_air > T_set_cooling)
        2. Outdoor temperature provides cooling potential (T_out < T_air)
        3. Random probability check passes (based on ventilation_behaviour)

        Window flow is unrecovered; declared mechanical recovery remains applied.

        Parameters
        ----------
        t_out : float
            Outdoor temperature [°C]

        Returns
        -------
        float
            Additional air change rate from window opening [1/h]
        """
        self._windows_open = False

        # Check if window opening is disabled
        if self.ventilation_behaviour <= 0.0:
            return 0.0

        # Condition 1: Indoor temperature above cooling setpoint
        if self.t_air <= self.t_set_cooling:
            return 0.0

        # Condition 2: Outdoor provides cooling potential
        if t_out >= self.t_air:
            return 0.0

        # Condition 3: Probabilistic occupant behaviour check
        if self.ventilation_behaviour >= 1.0 or random.random() < self.ventilation_behaviour:
            self._windows_open = True
            return self.ach_window_open

        return 0.0

    def _update_ventilation_with_window_opening(self, t_out: float):
        """Add unrecovered window flow using the common ventilation coefficient.

        Mechanical recovery remains applied only to mechanical flow. Infiltration
        and the additional window flow retain full outdoor conductance.
        """
        ach_window = self._handle_window_opening(t_out)
        self._update_ventilation_coefficient(
            self.ach_infiltration + self.ach_ventilation + ach_window,
        )

    def calc_supply_air_temperature(self, t_out: float) -> float:
        """Return the outdoor driving temperature for the reduced ventilation node.

        Sensible recovery is already applied once in mechanical conductance:
        H_ve = rho_cp * V * (n_inf + n_mech * (1 - eta) + n_window) / 3600.
        Warming this driving temperature would apply recovery again and also
        recover infiltration. This method therefore always returns ``t_out``;
        it does not report the physical post-exchanger mechanical supply state.
        This corrects the historical standalone numerical behavior. The shared
        Building path and standalone solver now use the same calculation.

        Parameters
        ----------
        t_out : float
            Outdoor temperature [degrees Celsius].

        Returns
        -------
        float
            Unmodified outdoor driving temperature [degrees Celsius].
        """
        return t_out

    def get_active_setpoints(self, hour_of_year: Optional[int] = None) -> tuple[float, float]:
        """Get active heating and cooling setpoints for current hour.

        ISO 13790:2008 Section 13.2.3 - Intermittent Heating/Cooling

        Returns the appropriate setpoint based on occupancy schedule:
        - During occupied periods: normal setpoints (t_set_heating, t_set_cooling)
        - During unoccupied/setback: reduced setpoints if defined

        Parameters
        ----------
        hour_of_year : Optional[int]
            Hour index (0-8759) for schedule lookup

        Returns
        -------
        tuple[float, float]
            Tuple of (heating_setpoint, cooling_setpoint) [°C]
        """
        # Default to normal setpoints
        t_heat = self.t_set_heating
        t_cool = self.t_set_cooling

        # Check if we should use setback temperatures
        use_setback = False

        if self.occupancy_schedule is not None and hour_of_year is not None:
            # Use schedule to determine occupied/unoccupied
            if 0 <= hour_of_year < len(self.occupancy_schedule):
                use_setback = not self.occupancy_schedule[hour_of_year]

        # Apply setback temperatures if in setback period
        if use_setback:
            if self.t_set_heating_setback is not None:
                t_heat = self.t_set_heating_setback
            if self.t_set_cooling_setback is not None:
                t_cool = self.t_set_cooling_setback

        return t_heat, t_cool

    def calc_detailed_shading_factor(self) -> float:
        """Calculate combined external shading factor from component factors.

        ISO 13790:2008 Section 11.3.3 - Shading Reduction Factor

        The shading reduction factor F_sh,ob accounts for external obstacles:

            F_sh,ob = F_hor × F_ov × F_fin

        Where:
            F_hor = reduction factor for horizon shading (distant obstacles)
            F_ov = reduction factor for overhangs above window
            F_fin = reduction factor for fins beside window

        This is combined with the simple shading_factor for total shading:

            F_sh,total = shading_factor × F_sh,ob

        Note: Per ISO 13790, these factors are typically monthly values
        varying with sun angle. This implementation uses constant values
        as a simplification. For detailed calculation, see EN ISO 52016-1.

        Returns
        -------
        float
            Combined shading factor (0-1)
        """
        # Calculate F_sh,ob from component factors
        f_sh_ob = self.f_shading_horizon * self.f_shading_overhang * self.f_shading_fins

        # Combine with simple shading factor for total
        return self.shading_factor * f_sh_ob

    @property
    def h_tr_1(self) -> float:
        """Combined conductance H_tr,1 [W/K].

        ISO 13790:2008 Annex C.3, Eq. (C.6):
            H_tr,1 = 1 / (1/H_ve + 1/H_tr,is)

        This represents the parallel combination of ventilation and
        surface-air heat transfer paths.
        """
        if self._prepared_hourly:
            return self._prepared_network[0]
        if self._h_ve > 0 and self._h_tr_is > 0:
            return 1.0 / (1.0 / self._h_ve + 1.0 / self._h_tr_is)
        return 0.0

    @property
    def h_tr_2(self) -> float:
        """Combined conductance H_tr,2 [W/K].

        ISO 13790:2008 Annex C.3, Eq. (C.7):
            H_tr,2 = H_tr,1 + H_tr,w

        This adds window transmission to the air-side conductance path.
        """
        if self._prepared_hourly:
            return self._prepared_network[1]
        return self.h_tr_1 + self._h_tr_w

    @property
    def h_tr_3(self) -> float:
        """Combined conductance H_tr,3 [W/K].

        ISO 13790:2008 Annex C.3, Eq. (C.8):
            H_tr,3 = 1 / (1/H_tr,2 + 1/H_tr,ms)

        This represents the parallel combination of window/air and
        mass-surface heat transfer paths.
        """
        if self._prepared_hourly:
            return self._prepared_network[2]
        if self.h_tr_2 > 0 and self._h_tr_ms > 0:
            return 1.0 / (1.0 / self.h_tr_2 + 1.0 / self._h_tr_ms)
        return 0.0

    @property
    def time_constant_hours(self) -> float:
        """Building thermal time constant τ [hours].

        ISO 13790:2008 Section 12.2.1.1:
            τ = C_m / (H_tr,em + H_tr,3)

        The time constant indicates how quickly the building responds
        to changes in heat gains and outdoor temperature. Larger values
        indicate heavier buildings with more thermal inertia.
        """
        h_total = self._h_tr_em + self.h_tr_3
        if h_total > 0:
            return (self._c_m / h_total) / 3600.0
        return 0.0

    @property
    def heat_loss_coefficient(self) -> float:
        """Total heat loss coefficient [W/K].

        Sum of transmission and ventilation losses.
        """
        return self._h_tr_op + self._h_tr_w + self._h_ve

    def calc_sky_radiation_loss(self) -> float:
        """Calculate extra heat flow due to thermal radiation to the sky [W].

        ISO 13790:2008 Annex E, Eq. (E.1):
            Φ_r = R_se * U_c * A_c * h_r * Δθ_er * F_r

        Where:
            R_se = external surface resistance [m^2K/W]
            U_c = thermal transmittance [W/(m^2K)]
            A_c = area of element [m^2]
            h_r = radiative coefficient (default 5 W/(m^2K))
            Δθ_er = difference between external air and sky temp (default 11 K)
            F_r = form factor to sky (horizontal=1.0, vertical=0.5)

        This heat loss is additional to transmission losses and accounts for
        the fact that surfaces (especially roofs) radiate to a cold sky.

        Returns
        -------
        float
            Sky radiation heat loss [W] (positive = heat loss)
        """
        if self._prepared_hourly:
            return self._prepared_sky
        # Roof contribution (horizontal surface, Fr = 1.0)
        phi_r_roof = (
            self.r_se_roof
            * self.u_roof
            * self.roof_area_m2
            * self.h_r
            * self.delta_theta_er
            * self.form_factor_roof
        )

        # Wall contribution (vertical surfaces, Fr = 0.5 typical)
        phi_r_wall = (
            self.r_se_wall
            * self.u_walls
            * self.wall_area_m2
            * self.h_r
            * self.delta_theta_er
            * self.form_factor_wall
        )

        return phi_r_roof + phi_r_wall

    def calc_solar_gains_opaque(
        self,
        i_sol_wall: float = 0.0,
        i_sol_roof: float = 0.0,
    ) -> float:
        """Calculate solar gains through opaque elements [W].

        ISO 13790 §11.3.4 Eq. (46): A_sol = αS,c * Rse * Uc * Ac
        Combined with §11.3.2 Eq. (44): Φsol,k = F_sh,ob * A_sol * I_sol

        Solar radiation absorbed by opaque surfaces (walls, roof) contributes
        to internal gains, especially for dark or poorly insulated surfaces.

        Parameters
        ----------
        i_sol_wall : float
            Average solar irradiance on walls [W/m^2]
        i_sol_roof : float
            Solar irradiance on roof [W/m^2] (typically horizontal global)

        Returns
        -------
        float
            Solar gains through opaque elements [W]
        """
        # Effective solar collecting area for walls
        # A_sol,op = α * Rse * U * A
        a_sol_wall = self.wall_absorptivity * self.r_se_wall * self.u_walls * self.wall_area_m2

        # Effective solar collecting area for roof
        a_sol_roof = self.roof_absorptivity * self.r_se_roof * self.u_roof * self.roof_area_m2

        # Total solar gains through opaque elements
        phi_sol_op = a_sol_wall * i_sol_wall + a_sol_roof * i_sol_roof

        return phi_sol_op

    def _get_solar_calculator(self) -> "SolarGainsCalculator":
        """Get or create the solar gains calculator (lazy initialization).

        If epw_path is set, creates calculator from EPW file (includes location).
        Otherwise uses latitude/longitude from building parameters.
        """
        if self._solar_calc is None:
            from .solar import SolarGainsCalculator

            if self.epw_path is not None:
                self._solar_calc = SolarGainsCalculator.from_epw(self.epw_path)
                # Update lat/lon from EPW
                self.latitude = self._solar_calc.latitude
                self.longitude = self._solar_calc.longitude
            else:
                self._solar_calc = SolarGainsCalculator(
                    latitude=self.latitude,
                    longitude=self.longitude,
                )
        return self._solar_calc

    def _resolve_solar_method(self) -> str:
        """Resolve 'auto' solar method to actual method based on available data."""
        if self.solar_method != "auto":
            return self.solar_method

        # Auto-detect: use detailed if we have orientation data or EPW
        has_orientation = any(
            a is not None
            for a in [
                self.window_area_south,
                self.window_area_north,
                self.window_area_east,
                self.window_area_west,
            ]
        )

        if has_orientation or self.epw_path is not None:
            return "detailed"
        else:
            return "simplified"

    def _get_window_areas_by_orientation(self) -> dict[str, float]:
        """Get window areas by orientation, estimating if not specified."""
        if self._window_areas_by_orientation is None:
            # Check if orientation-specific areas are provided
            if any(
                a is not None
                for a in [
                    self.window_area_south,
                    self.window_area_north,
                    self.window_area_east,
                    self.window_area_west,
                ]
            ):
                # Use provided values, defaulting to 0 for unspecified
                self._window_areas_by_orientation = {
                    "south": self.window_area_south or 0.0,
                    "north": self.window_area_north or 0.0,
                    "east": self.window_area_east or 0.0,
                    "west": self.window_area_west or 0.0,
                }
            else:
                # Estimate from total window area
                from .solar import estimate_window_orientation_split

                self._window_areas_by_orientation = estimate_window_orientation_split(
                    self.window_area_m2,
                    aspect_ratio=1.0,  # Default assumption
                    south_bias=1.5,  # Slightly more south glazing
                )
        return self._window_areas_by_orientation

    @_refresh_standalone
    def calc_solar_gains(
        self,
        ghi: float,
        hour_of_year: Optional[int] = None,
    ) -> float:
        """Calculate solar gains through windows [W].

        Uses either simplified (0.5 factor) or detailed (geometry-based)
        calculation depending on solar_method setting.

        Parameters
        ----------
        ghi : float
            Global horizontal irradiance [W/m^2]
        hour_of_year : Optional[int]
            Hour index (0-8759), required for detailed method

        Returns
        -------
        float
            Solar gains through windows [W]
        """
        if ghi <= 0:
            return 0.0

        # Calculate combined shading factor (ISO 13790 Section 11.3.3)
        f_shading = self.calc_detailed_shading_factor()
        f_incidence = self._effective_incidence_factor()

        method = self._resolve_solar_method()

        if method == "detailed" and hour_of_year is not None:
            # Detailed calculation with solar geometry (pvlib Perez model)
            calc = self._get_solar_calculator()
            window_areas = self._get_window_areas_by_orientation()

            return (
                calc.calculate_single_hour(
                    ghi=ghi,
                    hour_of_year=hour_of_year,
                    window_areas=window_areas,
                    g_value=self.g_value,
                    frame_factor=self.frame_factor,
                    g_value_angle_coefficients=self.g_value_angle_coefficients,
                )
                * f_shading
                * f_incidence
            )
        else:
            # Simplified calculation (ISO 13790 default)
            # Effective solar factor: 0.5 accounts for average orientation/incidence
            solar_factor = (
                self.window_area_m2
                * self.g_value
                * self.frame_factor
                * f_shading
                * f_incidence
                * 0.5  # Average orientation/incidence factor
            )
            return ghi * solar_factor

    @_refresh_standalone
    def calc_solar_gains_annual(
        self,
        ghi: np.ndarray,
        dni: Optional[np.ndarray] = None,
        dhi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate annual hourly solar gains [W].

        Efficient batch calculation for full-year simulation.
        With the detailed method, supplied DNI/DHI use Perez sky irradiance.
        Omitting both channels uses Erbs decomposition and isotropic diffuse
        irradiance. The simplified method uses the documented scalar GHI factor.
        All supplied arrays are checked before initializing a solar calculator.

        Parameters
        ----------
        ghi : np.ndarray
            Global horizontal irradiance [W/m^2] for 8760 hours
        dni : Optional[np.ndarray]
            Direct normal irradiance [W/m^2] (optional, decomposed from GHI if not provided)
        dhi : Optional[np.ndarray]
            Diffuse horizontal irradiance [W/m^2] (optional)

        Returns
        -------
        np.ndarray
            Hourly solar gains [W] for 8760 hours
        """
        ghi = _input_series("ghi", ghi, length=8760, nonnegative=True)
        if (dni is None) != (dhi is None):
            raise ValueError("Supply both dni and dhi, or omit both for decomposition")
        if dni is not None:
            dni = _input_series("dni", dni, length=8760, nonnegative=True)
            dhi = _input_series("dhi", dhi, length=8760, nonnegative=True)

        # Calculate combined shading factor (ISO 13790 Section 11.3.3)
        f_shading = self.calc_detailed_shading_factor()
        f_incidence = self._effective_incidence_factor()

        method = self._resolve_solar_method()

        if method == "detailed":
            calc = self._get_solar_calculator()
            window_areas = self._get_window_areas_by_orientation()

            return (
                calc.calculate_hourly_solar_gains(
                    ghi=ghi,
                    dni=dni,
                    dhi=dhi,
                    window_areas=window_areas,
                    g_value=self.g_value,
                    frame_factor=self.frame_factor,
                    g_value_angle_coefficients=self.g_value_angle_coefficients,
                )
                * f_shading
                * f_incidence
            )
        else:
            # Simplified calculation
            solar_factor = (
                self.window_area_m2
                * self.g_value
                * self.frame_factor
                * f_shading
                * f_incidence
                * 0.5
            )
            return ghi * solar_factor

    def calc_heat_flows(
        self,
        t_out: float,
        internal_gains: float,
        solar_gains: float,
        energy_demand: float = 0.0,
        solar_gains_opaque: float = 0.0,
        include_sky_radiation: bool = True,
        strict_iso: bool = False,
    ) -> tuple[float, float, float]:
        """Calculate heat flows to each node.

        ISO 13790:2008 Annex C.2 - Heat Flow Distribution
        ==================================================

        The internal (Φ_int) and solar (Φ_sol) heat gains are distributed
        to the three temperature nodes according to Equations C.1-C.3:

        Eq. (C.1): Φ_ia = 0.5 × Φ_int
            → 50% of internal gains go directly to air node (convective)

        Eq. (C.2): Φ_m = (A_m/A_t) × 0.5 × (Φ_int + Φ_sol)
            → Fraction absorbed by thermal mass surfaces

        Eq. (C.3): Φ_st = (1 - A_m/A_t - H_tr,w/(9.1×A_t)) × 0.5 × (Φ_int + Φ_sol)
            → Remaining radiative fraction to surface node

        The coefficient h_ms (default 9.1) in Eq. (C.3) is also used in
        mass-surface coupling, including when explicitly overridden.

        Note on strict_iso mode:
            When strict_iso=True, the exact ISO 13790 equations are used with
            the 0.5 convective/radiative split for internal gains.
            When strict_iso=False (default), convective_fraction_internal can
            be adjusted to match EnergyPlus (~0.67) for calibration.

        Parameters
        ----------
        t_out : float
            Outdoor temperature [°C]
        internal_gains : float
            Internal heat gains Φ_int [W]
        solar_gains : float
            Solar heat gains through windows Φ_sol [W]
        energy_demand : float
            Heating (+) or cooling (-) power Φ_HC,nd [W]
        solar_gains_opaque : float
            Solar gains through opaque elements [W]
        include_sky_radiation : bool
            If True, subtract sky radiation losses [W]
        strict_iso : bool
            If True, use exact ISO 13790 C.1-C.3 equations with 0.5 factor

        Returns
        -------
        tuple[float, float, float]
            Tuple of (Φ_ia, Φ_st, Φ_m) heat flows to air, surface, mass nodes [W]
        """
        if self._prepared_hourly and self._hour_heat_flows is not None:
            phi_ia, phi_st, phi_m = self._hour_heat_flows
            return (
                (phi_ia + energy_demand, phi_st, phi_m)
                if energy_demand
                else (phi_ia, phi_st, phi_m)
            )
        # Total solar gains (windows + opaque)
        phi_sol_total = solar_gains + solar_gains_opaque

        if strict_iso:
            # =====================================================================
            # STRICT ISO 13790 Equations C.1, C.2, C.3
            # =====================================================================
            # Eq. (C.1): Φ_ia = 0.5 × Φ_int
            phi_ia = 0.5 * internal_gains

            if self._a_tot > 0:
                # Area ratios per ISO 13790
                ratio_mass = (
                    self._prepared_mass_ratio
                    if self._prepared_hourly
                    else self._mass_area / self._a_tot
                )
                ratio_window = (
                    self._prepared_window_ratio
                    if self._prepared_hourly
                    else self._h_tr_w / (self.h_ms * self._a_tot)
                )
                ratio_surface = max(0.0, 1.0 - ratio_mass - ratio_window)

                # Eq. (C.2): Φ_m = (A_m/A_t) × 0.5 × (Φ_int + Φ_sol)
                phi_m = ratio_mass * 0.5 * (internal_gains + phi_sol_total)

                # Eq. (C.3): Φ_st = (1 - A_m/A_t - H_tr,w/(9.1×A_t)) × 0.5 × (Φ_int + Φ_sol)
                phi_st = ratio_surface * 0.5 * (internal_gains + phi_sol_total)
            else:
                phi_st = 0.0
                phi_m = 0.0
        else:
            # =====================================================================
            # FLEXIBLE MODE (default) - allows calibration of convective fraction
            # =====================================================================
            # Heat flow to air node (φ_ia) - Modified Eq. C.1
            # Allows adjustment from ISO default 0.5 to EnergyPlus typical ~0.67
            phi_ia = self.convective_fraction_internal * internal_gains

            # Distribute remaining gains to surface and mass nodes
            radiant_fraction = 1.0 - self.convective_fraction_internal
            phi_gains_radiant = radiant_fraction * internal_gains + phi_sol_total

            if self._a_tot > 0:
                # ISO 13790 Eq. C.2 and C.3 area ratios
                ratio_mass = (
                    self._prepared_mass_ratio
                    if self._prepared_hourly
                    else self._mass_area / self._a_tot
                )
                ratio_window = (
                    self._prepared_window_ratio
                    if self._prepared_hourly
                    else self._h_tr_w / (self.h_ms * self._a_tot)
                )
                ratio_surface = max(0.0, 1.0 - ratio_mass - ratio_window)

                # Heat flow to surface node (φ_st)
                phi_st = ratio_surface * phi_gains_radiant

                # Heat flow to mass node (φ_m)
                phi_m = ratio_mass * phi_gains_radiant
            else:
                phi_st = 0.0
                phi_m = 0.0

        # Subtract sky radiation losses (extra heat flow to exterior)
        # This is added to the surface node as it represents radiative loss
        if include_sky_radiation and self.delta_theta_er > 0:
            phi_r_sky = self.calc_sky_radiation_loss()
            phi_st -= phi_r_sky  # Negative because it's a loss

        if self._prepared_hourly:
            self._hour_heat_flows = phi_ia + 0.0, phi_st, phi_m
        # Add heating/cooling to air node (assumes radiator/convective heating)
        phi_ia += energy_demand

        return phi_ia, phi_st, phi_m

    def calc_phi_m_tot(
        self,
        t_out: float,
        phi_ia: float,
        phi_st: float,
        phi_m: float,
        hour_of_year: Optional[int] = None,
    ) -> float:
        """Calculate total heat flow to mass node.

        ISO 13790:2008 Annex C.3, Eq. (C.5):
            Φ_m,tot = Φ_m + H_tr,em.θ_e + H_tr,3.{Φ_st + H_tr,w.θ_e +
                      H_tr,1.[(Φ_ia + Φ_HC,nd)/H_ve + θ_sup]} / H_tr,2

        This equation combines all heat flows entering the mass node for
        use in the Crank-Nicolson mass temperature update (Eq. C.4).

        If ground_temps_monthly is set, ground floor heat loss is calculated
        separately using monthly ground temperature instead of outdoor air.

        Parameters
        ----------
        t_out : float
            Outdoor temperature θ_e [°C]
        phi_ia : float
            Heat flow to air node Φ_ia [W]
        phi_st : float
            Heat flow to surface node Φ_st [W]
        phi_m : float
            Heat flow to mass node Φ_m [W]
        hour_of_year : Optional[int]
            Hour of year (0-8759) for monthly ground temp lookup

        Returns
        -------
        float
            Total heat flow to mass node Φ_m,tot [W]
        """
        # Recovery is already applied to mechanical conductance; drive from outdoor air.
        t_supply = self.calc_supply_air_temperature(t_out)
        self._t_supply = t_supply  # Store for diagnostics

        # =====================================================================
        # Equivalent opaque exterior temperature (weighted by conductance)
        # =====================================================================
        # ISO 13790 H_tr,op includes all opaque elements. When the ground
        # floor faces a different temperature (monthly ground temp) than
        # walls/roof (outdoor air), we compute a weighted driving temperature:
        #   θ_e_opaque = (H_walls_roof × θ_e + H_ground × θ_ground) / H_tr_op
        # This maintains the RC network topology (H_tr_em uses full H_tr_op)
        # and dynamically couples ground losses to mass temperature, so gains
        # that warm the mass automatically increase ground heat flow.
        if self.ground_temps_monthly is not None and hour_of_year is not None:
            month = self._month_index_from_hour_of_year(hour_of_year)
            t_ground = self.ground_temps_monthly[month]
            if self._h_tr_op > 0:
                t_e_opaque = (
                    self._h_tr_op_walls_roof * t_out
                    + self._h_tr_ground * t_ground
                    + self._h_tr_tb * t_out
                ) / self._h_tr_op
            else:
                t_e_opaque = t_out
        elif self.ground_temps_monthly is not None:
            # If hour index is unavailable, retain dynamic coupling using annual mean ground temp.
            t_ground = float(np.mean(self.ground_temps_monthly))
            if self._h_tr_op > 0:
                t_e_opaque = (
                    self._h_tr_op_walls_roof * t_out
                    + self._h_tr_ground * t_ground
                    + self._h_tr_tb * t_out
                ) / self._h_tr_op
            else:
                t_e_opaque = t_out
        else:
            # Static case: b_tr,g already folded into H_tr_op, use θ_e directly
            t_e_opaque = t_out

        if self._h_ve <= 0:
            # No ventilation - simplified calculation
            return phi_m + self._h_tr_em * t_e_opaque

        phi_m_tot = (
            phi_m
            + self._h_tr_em * t_e_opaque
            + self.h_tr_3
            * (phi_st + self._h_tr_w * t_out + self.h_tr_1 * ((phi_ia / self._h_ve) + t_supply))
            / self.h_tr_2
        )
        return phi_m_tot

    def calc_t_m_next(self, t_m_prev: float, phi_m_tot: float, dt_hours: float = 1.0) -> float:
        """Calculate mass temperature for next timestep.

        ISO 13790:2008 Annex C.3, Eq. (C.4) - Crank-Nicolson discretization:

            θ_m,t = {θ_m,t-1.(C_m/Δt - 0.5.(H_tr,3 + H_tr,em)) + Φ_m,tot}
                    / {C_m/Δt + 0.5.(H_tr,3 + H_tr,em)}

        The Crank-Nicolson scheme uses the average of current and next timestep
        values for stability. Mass temperature is an instantaneous value at
        the end of the timestep (t), not an average over the hour.

        Parameters
        ----------
        t_m_prev : float
            Previous mass temperature θ_m,t-1 [°C]
        phi_m_tot : float
            Total heat flow to mass node Φ_m,tot [W]
        dt_hours : float
            Timestep Δt [hours]

        Returns
        -------
        float
            Mass temperature at end of timestep θ_m,t [°C]
        """
        if self._prepared_hourly and dt_hours == 1.0:
            a, b = self._prepared_cn
            return (t_m_prev * a + phi_m_tot) / b
        dt_seconds = dt_hours * 3600.0

        # C.4: T_m(t+1) = [T_m(t).(C_m/dt - 0.5.(H3+H_em)) + φ_m_tot] /
        #                 [C_m/dt + 0.5.(H3+H_em)]
        c_dt = self._c_m / dt_seconds
        h_sum = self.h_tr_3 + self._h_tr_em

        t_m_next = (t_m_prev * (c_dt - 0.5 * h_sum) + phi_m_tot) / (c_dt + 0.5 * h_sum)
        return t_m_next

    def calc_t_s(self, t_m: float, t_out: float, phi_ia: float, phi_st: float) -> float:
        """Calculate surface temperature.

        ISO 13790:2008 Annex C.3, Eq. (C.10):

            θ_s = {H_tr,ms.θ_m + Φ_st + H_tr,w.θ_e + H_tr,1.(θ_sup + Φ_ia/H_ve)}
                  / (H_tr,ms + H_tr,w + H_tr,1)

        The surface temperature θ_s is a weighted average between air and
        mean radiant temperature. It represents the internal surfaces of
        the building zone.

        Parameters
        ----------
        t_m : float
            Mass temperature θ_m [°C]
        t_out : float
            Outdoor temperature θ_e [°C]
        phi_ia : float
            Heat flow to air node Φ_ia [W]
        phi_st : float
            Heat flow to surface node Φ_st [W]

        Returns
        -------
        float
            Surface temperature θ_s [°C]
        """
        # Use outdoor driving temperature with already reduced mechanical conductance.
        t_supply = self.calc_supply_air_temperature(t_out)

        if self._h_ve <= 0:
            # Simplified for no ventilation
            h_sum = self._h_tr_ms + self._h_tr_w + self._h_tr_is
            if h_sum > 0:
                return (self._h_tr_ms * t_m + phi_st + self._h_tr_w * t_out) / h_sum
            return t_m

        numerator = (
            self._h_tr_ms * t_m
            + phi_st
            + self._h_tr_w * t_out
            + self.h_tr_1 * (t_supply + phi_ia / self._h_ve)
        )
        denominator = self._h_tr_ms + self._h_tr_w + self.h_tr_1

        if denominator > 0:
            return numerator / denominator
        return t_m

    def calc_t_air(self, t_s: float, t_out: float, phi_ia: float) -> float:
        """Calculate air temperature.

        ISO 13790:2008 Annex C.3, Eq. (C.11):

            θ_air = (H_tr,is.θ_s + H_ve.θ_sup + Φ_ia) / (H_tr,is + H_ve)

        The air temperature node is coupled to the surface node via the
        convective/radiative surface coefficient H_tr,is, and to outside
        via ventilation H_ve. It receives the convective fraction of
        internal gains and the applied heating/cooling power (Φ_ia).

        Parameters
        ----------
        t_s : float
            Surface temperature θ_s [°C]
        t_out : float
            Outdoor temperature θ_e [°C]
        phi_ia : float
            Heat flow to air node Φ_ia [W]

        Returns
        -------
        float
            Air temperature θ_air [°C]
        """
        # Use outdoor driving temperature with already reduced mechanical conductance.
        t_supply = self.calc_supply_air_temperature(t_out)

        h_sum = self._h_tr_is + self._h_ve
        if h_sum > 0:
            return (self._h_tr_is * t_s + self._h_ve * t_supply + phi_ia) / h_sum
        return t_s

    @property
    def t_operative(self) -> float:
        """Operative temperature.

        ISO 13790:2008 Annex C.3, Eq. (C.12):

            θ_op = 0.3.θ_air + 0.7.θ_s

        The operative temperature is a comfort metric that weights air and
        mean radiant temperature. The 30/70 weighting accounts for the
        typical seated occupant's convective/radiative heat exchange ratio.

        Note: Some implementations use 50/50 weighting. The ISO 13790
        standard uses 30/70 for the simplified hourly method.

        Returns
        -------
        float
            Operative temperature θ_op [°C]
        """
        return 0.3 * self.t_air + 0.7 * self.t_s

    @property
    def windows_open(self) -> bool:
        """Whether windows were opened in the last timestep for natural cooling."""
        return self._windows_open

    @_refresh_standalone
    def step(
        self,
        t_out: float,
        solar_gains: float = 0.0,
        internal_gains: Optional[float] = None,
        energy_demand: float = 0.0,
        dt_hours: float = 1.0,
        solar_gains_opaque: float = 0.0,
        include_sky_radiation: bool = True,
        enable_window_opening: bool = True,
        hour_of_year: Optional[int] = None,
    ) -> tuple[float, float, float]:
        """Advance simulation by one timestep.

        Updates internal temperatures based on outdoor conditions and gains.
        Invalid inputs fail before temperatures change. Public physical edits
        refresh derived coefficients at the start of this operation.

        Parameters
        ----------
        t_out : float
            Outdoor temperature [°C]
        solar_gains : float
            Solar heat gains through windows [W]
        internal_gains : Optional[float]
            Internal gains [W], defaults to internal_gains_w_per_m2 * A_f
        energy_demand : float
            Heating (+) or cooling (-) power applied [W]
        dt_hours : float
            Timestep [hours]
        solar_gains_opaque : float
            Solar gains through opaque elements [W]
        include_sky_radiation : bool
            Include sky radiation losses [bool]
        enable_window_opening : bool
            Allow stochastic window opening for cooling [bool]
        hour_of_year : Optional[int]
            Hour of year (0-8759) for monthly ground temp lookup

        Returns
        -------
        tuple[float, float, float]
            Tuple of (t_air, t_s, t_m) node temperatures [°C]
        """
        if not self._prepared_hourly and self._calculation_depth == 1:
            self._validate_timestep_and_capacity(dt_hours, np.inf, -np.inf)
            self._validate_forcing(t_out, solar_gains, internal_gains, solar_gains_opaque)
            if not np.isscalar(energy_demand) or not np.isfinite(energy_demand):
                raise ValueError("energy_demand must be finite")
        if internal_gains is None:
            internal_gains = self.internal_gains_w_per_m2 * self.floor_area_m2

        # Handle window opening behaviour for natural cooling
        if enable_window_opening and self.ventilation_behaviour > 0:
            self._update_ventilation_with_window_opening(t_out)

        t_m_prev = self.t_m

        # Calculate heat flows
        phi_ia, phi_st, phi_m = self.calc_heat_flows(
            t_out,
            internal_gains,
            solar_gains,
            energy_demand,
            solar_gains_opaque=solar_gains_opaque,
            include_sky_radiation=include_sky_radiation,
        )

        # Calculate total heat flow to mass (with optional ground temp)
        phi_m_tot = self.calc_phi_m_tot(t_out, phi_ia, phi_st, phi_m, hour_of_year=hour_of_year)

        # Update mass temperature
        t_m_next = self.calc_t_m_next(t_m_prev, phi_m_tot, dt_hours)

        # Average mass temperature for this timestep (Crank-Nicolson)
        self.t_m = (t_m_next + t_m_prev) / 2.0

        # Calculate surface and air temperatures
        self.t_s = self.calc_t_s(self.t_m, t_out, phi_ia, phi_st)
        self.t_air = self.calc_t_air(self.t_s, t_out, phi_ia)

        # Restore normal ventilation coefficient for next timestep
        if enable_window_opening and self.ventilation_behaviour > 0 and self._windows_open:
            self._update_ventilation_coefficient()

        # Store next mass temp for next iteration
        self.t_m = t_m_next

        return self.t_air, self.t_s, self.t_m

    @_refresh_standalone
    def calc_heating_demand(
        self,
        t_out: float,
        solar_gains: float = 0.0,
        internal_gains: Optional[float] = None,
        dt_hours: float = 1.0,
        max_heating_power: float = np.inf,
        max_cooling_power: float = 0.0,
        solar_gains_opaque: float = 0.0,
        hour_of_year: Optional[int] = None,
        include_sky_radiation: bool = True,
    ) -> float:
        """Calculate heating/cooling demand to maintain setpoint.

        This operation advances the internal thermal state by one timestep.
        ``include_sky_radiation=False`` disables sky loss in every control trial.

        ISO 13790:2008 Annex C.4.2 - Iterative Calculation Procedure:

        The procedure determines the HVAC system energy required to maintain
        the zone at the desired setpoint temperature through linear interpolation:

        Step 1: Calculate free-running temperature θ_0 with Φ_HC,nd = 0
        Step 2: If θ_0 ≠ θ_set, calculate θ_10 with reference power Φ_HC,10
                (typically 10 W/m^2 * floor area)
        Step 3: Interpolate required power using Thales' theorem, Eq. (C.13):

            Φ_HC,nd = Φ_HC,10 * (θ_set - θ_0) / (θ_10 - θ_0)

        Step 4: Apply system capacity limitations
        Step 5: Final simulation step with calculated Φ_HC,nd

        When `use_operative_temperature=True`, the setpoint is maintained for
        operative temperature (θ_op = 0.3*θ_air + 0.7*θ_s per Eq. C.12) rather
        than air temperature alone. This better represents thermal comfort.

        Parameters
        ----------
        t_out : float
            Outdoor temperature θ_e [°C]
        solar_gains : float
            Solar heat gains through windows Φ_sol [W]
        internal_gains : Optional[float]
            Internal gains Φ_int [W]
        dt_hours : float
            Timestep Δt [hours]
        max_heating_power : float
            Maximum heating capacity [W]
        max_cooling_power : float
            Cooling capacity [W], accepted as either signed power or magnitude
        solar_gains_opaque : float
            Solar gains through opaque elements [W]
        hour_of_year : Optional[int]
            Hour of year (0-8759) for monthly ground temp lookup
        include_sky_radiation : bool
            Include sky loss in free, reference and final steps

        Returns
        -------
        float
            Heating demand Φ_H,nd [W] (positive) or cooling demand Φ_C,nd [W] (negative)
        """
        if not self._prepared_hourly and self._calculation_depth == 1:
            self._validate_timestep_and_capacity(dt_hours, max_heating_power, max_cooling_power)
            self._validate_forcing(t_out, solar_gains, internal_gains, solar_gains_opaque)
        if self._prepared_hourly:
            self._hour_heat_flows = None
        if internal_gains is None:
            internal_gains = self.internal_gains_w_per_m2 * self.floor_area_m2

        # Get active setpoints (may be reduced during setback periods)
        # ISO 13790 Section 13.2.3 - Intermittent Heating/Cooling
        t_set_heat, t_set_cool = self.get_active_setpoints(hour_of_year)

        # Step 1: Calculate temperature with no heating/cooling
        t_m_prev = self.t_m

        # Temporarily step with zero energy
        self.step(
            t_out,
            solar_gains,
            internal_gains,
            energy_demand=0.0,
            dt_hours=dt_hours,
            solar_gains_opaque=solar_gains_opaque,
            hour_of_year=hour_of_year,
            include_sky_radiation=include_sky_radiation,
        )

        free_state = self.t_air, self.t_s, self.t_m
        # Get control temperature based on mode
        if self.use_operative_temperature:
            t_ctrl_0 = self.t_operative
        else:
            t_ctrl_0 = self.t_air

        # Reset mass temperature
        self.t_m = t_m_prev

        # Check if heating or cooling needed (using active setpoints)
        if t_ctrl_0 < t_set_heat:
            # Heating needed
            if self._prepared_hourly and self.ventilation_behaviour <= 0 and max_heating_power == 0:
                self.t_air, self.t_s, self.t_m = free_state
                return 0.0
            t_ctrl_set = t_set_heat

            # Step 2: Calculate with reference heating (10 W/m^2)
            q_ref = 10.0 * self.floor_area_m2
            self.step(
                t_out,
                solar_gains,
                internal_gains,
                energy_demand=q_ref,
                dt_hours=dt_hours,
                solar_gains_opaque=solar_gains_opaque,
                hour_of_year=hour_of_year,
                include_sky_radiation=include_sky_radiation,
            )

            if self.use_operative_temperature:
                t_ctrl_10 = self.t_operative
            else:
                t_ctrl_10 = self.t_air
            self.t_m = t_m_prev

            # Interpolate required heating (Thales theorem, Eq. C.13)
            if abs(t_ctrl_10 - t_ctrl_0) > 1e-6:
                q_demand = q_ref * (t_ctrl_set - t_ctrl_0) / (t_ctrl_10 - t_ctrl_0)
            else:
                q_demand = 0.0

            # Apply capacity limit
            q_demand = min(q_demand, max_heating_power)
            q_demand = max(q_demand, 0.0)

        elif t_ctrl_0 > t_set_cool:
            # Cooling needed
            if self._prepared_hourly and self.ventilation_behaviour <= 0 and max_cooling_power == 0:
                self.t_air, self.t_s, self.t_m = free_state
                return 0.0
            t_ctrl_set = t_set_cool

            # Step 2: Calculate with reference cooling
            q_ref = -10.0 * self.floor_area_m2
            self.step(
                t_out,
                solar_gains,
                internal_gains,
                energy_demand=q_ref,
                dt_hours=dt_hours,
                solar_gains_opaque=solar_gains_opaque,
                hour_of_year=hour_of_year,
                include_sky_radiation=include_sky_radiation,
            )

            if self.use_operative_temperature:
                t_ctrl_10 = self.t_operative
            else:
                t_ctrl_10 = self.t_air
            self.t_m = t_m_prev

            # Interpolate required cooling
            if abs(t_ctrl_10 - t_ctrl_0) > 1e-6:
                q_demand = q_ref * (t_ctrl_set - t_ctrl_0) / (t_ctrl_10 - t_ctrl_0)
            else:
                q_demand = 0.0

            # Apply capacity limit (cooling is negative)
            q_demand = max(q_demand, -abs(max_cooling_power))
            q_demand = min(q_demand, 0.0)

        else:
            # No heating or cooling needed
            q_demand = 0.0

        if self._prepared_hourly and q_demand == 0.0:
            self.t_air, self.t_s, self.t_m = free_state
            return q_demand
        # Step 3/4: Apply the calculated demand and update state
        self.step(
            t_out,
            solar_gains,
            internal_gains,
            energy_demand=q_demand,
            dt_hours=dt_hours,
            solar_gains_opaque=solar_gains_opaque,
            hour_of_year=hour_of_year,
            include_sky_radiation=include_sky_radiation,
        )

        return q_demand

    @_refresh_standalone
    def simulate(
        self,
        t_outdoor: np.ndarray,
        solar_gains: Optional[np.ndarray] = None,
        internal_gains: Optional[np.ndarray] = None,
        solar_gains_opaque: Optional[np.ndarray] = None,
        dt_hours: float = 1.0,
        max_heating_power: float = np.inf,
        max_cooling_power: float = 0.0,
        include_sky_radiation: bool = True,
    ) -> dict[str, np.ndarray]:
        """Run a fresh thermal simulation over supplied time series.

        Parameters
        ----------
        t_outdoor : array-like
            Nonempty one-dimensional outdoor temperatures in degrees Celsius.
        solar_gains, internal_gains, solar_gains_opaque : array-like, optional
            Finite nonnegative heat gains in W, each matching ``t_outdoor``.
            Missing solar series are zero; missing internal gains use the model's
            constant internal gains per floor area.
        dt_hours : float, default 1
            Positive finite timestep in hours. Controls follow elapsed whole
            hours: two half-hour steps use one hourly occupancy entry. Monthly
            ground temperature selection uses the same independent clock.
        max_heating_power : float, default inf
            Nonnegative heating capacity in W; infinity means unlimited.
        max_cooling_power : float, default 0
            Cooling capacity in W, accepted as a magnitude or signed power.
            Zero disables cooling; either infinity means unlimited.
        include_sky_radiation : bool, default True
            Apply sky radiation loss in every free, reference and final step.

        Returns
        -------
        dict[str, numpy.ndarray]
            ``heating_demand`` is signed HVAC power in W. ``t_air``, ``t_m``,
            ``t_s`` and ``t_operative`` are temperatures in degrees Celsius.
            ``windows_open`` records stochastic opening decisions.

        Raises
        ------
        ValueError
            Physical inputs, array shapes, lengths, finite values, schedule
            coverage, timestep or capacities are invalid. Temperature state is
            unchanged when input validation fails.

        Notes
        -----
        No files are read or written. Public physical edits refresh derived
        coefficients once, before calculation. After validation, all three
        temperatures reset to the heating setpoint; this method mutates the model.

        Examples
        --------
        >>> from epsm_rc import Building5R1C
        >>> result = Building5R1C().simulate([-5.0, -4.0])
        >>> result["heating_demand"].shape
        (2,)
        """
        self._validate_timestep_and_capacity(dt_hours, max_heating_power, max_cooling_power)
        t_outdoor = _input_series("t_outdoor", t_outdoor)
        n = len(t_outdoor)

        if solar_gains is None:
            solar_gains = np.zeros(n)
        if internal_gains is None:
            internal_gains = np.full(n, self.internal_gains_w_per_m2 * self.floor_area_m2)
        if solar_gains_opaque is None:
            solar_gains_opaque = np.zeros(n)

        solar_gains = _input_series("solar_gains", solar_gains, length=n, nonnegative=True)
        internal_gains = _input_series("internal_gains", internal_gains, length=n, nonnegative=True)
        solar_gains_opaque = _input_series(
            "solar_gains_opaque", solar_gains_opaque, length=n, nonnegative=True
        )
        last_hour = int((n - 1) * dt_hours)
        if self.occupancy_schedule is not None and len(self.occupancy_schedule) <= last_hour:
            raise ValueError("occupancy_schedule does not cover the simulation elapsed hours")

        # Pre-allocate output arrays
        q_heat = np.zeros(n)
        t_air = np.zeros(n)
        t_m = np.zeros(n)
        t_s = np.zeros(n)
        t_op = np.zeros(n)
        windows_open = np.zeros(n, dtype=bool)

        # Initialize mass temperature to setpoint
        self.reset(self.t_set_heating)

        for i in range(n):
            q_heat[i] = self.calc_heating_demand(
                t_out=t_outdoor[i],
                solar_gains=solar_gains[i],
                internal_gains=internal_gains[i],
                dt_hours=dt_hours,
                max_heating_power=max_heating_power,
                max_cooling_power=max_cooling_power,
                solar_gains_opaque=solar_gains_opaque[i],
                hour_of_year=int(i * dt_hours),
                include_sky_radiation=include_sky_radiation,
            )
            t_air[i] = self.t_air
            t_m[i] = self.t_m
            t_s[i] = self.t_s
            t_op[i] = self.t_operative
            windows_open[i] = self._windows_open

        return {
            "heating_demand": q_heat,
            "t_air": t_air,
            "t_m": t_m,
            "t_s": t_s,
            "t_operative": t_op,
            "windows_open": windows_open,
        }

    @_refresh_standalone
    def simulate_with_epw(
        self,
        epw_path: Optional[Path] = None,
        internal_gains: Optional[np.ndarray] = None,
        dt_hours: float = 1.0,
        max_heating_power: float = np.inf,
        max_cooling_power: float = 0.0,
        include_sky_radiation: bool = True,
    ) -> dict[str, np.ndarray | float]:
        """Run annual simulation using EPW weather file with detailed solar (pvlib Perez).

        Reads and validates the selected local EPW before setting location or
        ground inputs. This operation changes thermal state and the selected
        location. When no ground temperatures are supplied, it installs the
        documented historical Swedish ground-temperature assumptions.

        This method:
        - Uses actual DNI/DHI from EPW (no decomposition needed)
        - Uses Perez sky model for accurate diffuse on tilted surfaces
        - Calculates orientation-specific solar gains
        - Automatically extracts location from EPW

        Parameters
        ----------
        epw_path : Optional[Path]
            Path to EPW weather file. If None, uses self.epw_path
        internal_gains : Optional[np.ndarray]
            Internal gains series [W], defaults to constant from
            internal_gains_w_per_m2 * floor_area_m2
        dt_hours : float
            Must be 1 hour; EPW records are not resampled
        max_heating_power : float
            Maximum heating capacity [W]
        max_cooling_power : float
            Cooling capacity [W], accepted as either signed power or magnitude
        include_sky_radiation : bool
            Include sky radiation losses (default True)

        Returns
        -------
        Dictionary with

                - 'heating_demand': Heating power [W]
                - 'heating_demand_kwh': Total annual heating [kWh]
                - 'cooling_demand_kwh': Total annual cooling [kWh]
                - 'solar_gains': Hourly solar gains [W]
                - 't_air': Indoor air temperature [°C]
                - 't_outdoor': Outdoor temperature [°C] (from EPW)
                - 't_m': Mass temperature [°C]
                - 't_operative': Operative temperature [°C]


        Example with a caller-supplied complete local EPW and the platform extra:

        .. code-block:: python

            from epsm_rc import Building5R1C

            building = Building5R1C(
                floor_area_m2=150,
                window_area_south=15, window_area_north=5,
                window_area_east=8, window_area_west=8,
                thermal_capacitance_j_m2k=200_000,
            )
            results = building.simulate_with_epw("weather.epw")
            print(results["heating_demand_kwh"])
        """
        self._validate_timestep_and_capacity(dt_hours, max_heating_power, max_cooling_power)
        if dt_hours != 1:
            raise ValueError("EPW simulation uses hourly records; dt_hours must be 1")
        if internal_gains is not None:
            internal_gains = _input_series(
                "internal_gains", internal_gains, length=8760, nonnegative=True
            )
        from .solar import SolarGainsCalculator

        # Determine EPW path
        if epw_path is None:
            epw_path = self.epw_path
        if epw_path is None:
            raise ValueError("No EPW path provided. Set epw_path argument or self.epw_path")

        # Validate weather before changing the solver's ground or location state.
        solar_calc = SolarGainsCalculator.from_epw(epw_path)

        # Set default ground temperatures if not already specified
        # NOTE: We do NOT auto-load EPW ground temps because they are "undisturbed" soil temps
        # which are too extreme for soil under a conditioned building (per E+ documentation).
        # Instead, use Swedish defaults that account for thermal pillow effect.
        if self.ground_temps_monthly is None:
            # Swedish defaults: represents ground under heated slab-on-grade
            # Based on typical Swedish climate with damped seasonal variation
            # and ~2-3°C warming from building thermal pillow effect in winter
            self.ground_temps_monthly = SWEDISH_DEFAULT_GROUND_TEMPS.copy()

        # Create solar calculator from EPW
        self._solar_calc = solar_calc

        # Update lat/lon from EPW
        self.latitude = solar_calc.latitude
        self.longitude = solar_calc.longitude

        # Get weather data from EPW (ensure numpy arrays)
        if solar_calc._epw_data is None:
            raise ValueError("EPW data not loaded properly")
        t_outdoor = np.asarray(solar_calc._epw_data["temp_air"].values)

        # Calculate solar gains using detailed method (pvlib Perez)
        window_areas = self._get_window_areas_by_orientation()

        # Non-perpendicular incidence correction (ISO 13790 §11.4.2)
        # When angle-dependent coefficients are provided, they handle incidence
        # reduction internally per orientation. Otherwise apply flat F_w factor.
        f_incidence = self._effective_incidence_factor()

        solar_gains = (
            solar_calc.calculate_from_epw(
                window_areas=window_areas,
                g_value=self.g_value,
                frame_factor=self.frame_factor,
                g_value_angle_coefficients=self.g_value_angle_coefficients,
            )
            * self.shading_factor
            * f_incidence
        )

        # =====================================================================
        # Opaque surface solar gains (ISO 13790 §11.3.4 Eq. 46 + §11.3.2 Eq. 44)
        # =====================================================================
        # Solar radiation absorbed by external opaque surfaces (walls, roof)
        # contributes heat to the building through the mass node.
        # A_sol = α_S × R_se × U × A  (Eq. 46), Φ_sol = A_sol × I_sol  (Eq. 44)
        # E+ handles this automatically in its surface heat balance.
        solar_gains_opaque = solar_calc.calculate_opaque_solar_gains(
            wall_area=self.wall_area_m2,
            roof_area=self.roof_area_m2,
            u_walls=self.u_walls,
            u_roof=self.u_roof,
            wall_absorptivity=self.wall_absorptivity,
            roof_absorptivity=self.roof_absorptivity,
            r_se_wall=self.r_se_wall,
            r_se_roof=self.r_se_roof,
        )

        # Run simulation
        results = self.simulate(
            t_outdoor=t_outdoor,
            solar_gains=solar_gains,
            internal_gains=internal_gains,
            solar_gains_opaque=solar_gains_opaque,
            dt_hours=dt_hours,
            max_heating_power=max_heating_power,
            max_cooling_power=max_cooling_power,
            include_sky_radiation=include_sky_radiation,
        )

        # Add convenience totals
        heating = np.maximum(results["heating_demand"], 0)
        cooling = np.minimum(results["heating_demand"], 0)
        results["heating_demand_kwh"] = np.sum(heating) * dt_hours / 1000
        results["cooling_demand_kwh"] = abs(np.sum(cooling) * dt_hours / 1000)
        results["solar_gains"] = solar_gains
        results["solar_gains_opaque"] = solar_gains_opaque
        results["t_outdoor"] = t_outdoor

        # Add ground temperature info to results
        if self.ground_temps_monthly is not None:
            results["ground_temps_monthly"] = np.array(self.ground_temps_monthly)

        return results

    def reset(self, t_initial: Optional[float] = None):
        """Reset internal state to initial temperature.

        Parameters
        ----------
        t_initial : Optional[float]
            Initial temperature [°C], defaults to heating setpoint
        """
        if t_initial is None:
            t_initial = self.t_set_heating
        if not np.isscalar(t_initial) or not np.isfinite(t_initial):
            raise ValueError("Initial temperature must be finite")
        self.t_m = t_initial
        self.t_s = t_initial
        self.t_air = t_initial
        self._windows_open = False


def create_swedish_villa(
    floor_area_m2: float = 150.0,
    era: Literal["pre_1960", "1960_1975", "1976_1990", "1991_2010", "post_2010"] = "1976_1990",
    use_config: Optional[bool] = None,
    **kwargs,
) -> Building5R1C:
    """Create a Building5R1C configured for typical Swedish villa.

    U-values based on Swedish building regulations and typical construction
    by era. Reference: Boverket building regulations (BBR).

    Parameters
    ----------
    floor_area_m2 : float
        Heated floor area [m^2]
    era : Literal['pre_1960', '1960_1975', '1976_1990', '1991_2010', 'post_2010']
        Construction era for U-value defaults
    use_config : Optional[bool]
        Deprecated. Kept for backward compatibility. Override
               parameters via **kwargs instead.
    **kwargs
        Override any Building5R1C parameter (e.g., u_walls=0.3)

    Returns
    -------
    Building5R1C
        Configured Building5R1C instance
    """
    if use_config is not None:
        warnings.warn(
            "use_config is deprecated and has no effect; pass explicit operating inputs. "
            "It will remain accepted until at least 0.9.",
            DeprecationWarning,
            stacklevel=2,
        )

    # U-values by Swedish building era [W/(m^2.K)]
    # Reference: Boverket, Swedish Energy Agency studies, TABULA
    u_values_by_era = {
        "pre_1960": {"u_walls": 0.60, "u_roof": 0.40, "u_floor": 0.40, "u_windows": 2.8},
        "1960_1975": {"u_walls": 0.40, "u_roof": 0.25, "u_floor": 0.30, "u_windows": 2.5},
        "1976_1990": {"u_walls": 0.25, "u_roof": 0.15, "u_floor": 0.25, "u_windows": 1.8},
        "1991_2010": {"u_walls": 0.20, "u_roof": 0.12, "u_floor": 0.20, "u_windows": 1.4},
        "post_2010": {"u_walls": 0.15, "u_roof": 0.10, "u_floor": 0.15, "u_windows": 1.0},
    }

    u_values = u_values_by_era[era]

    # Building class by era (thermal mass)
    building_classes = {
        "pre_1960": BuildingClass.HEAVY,  # Solid construction
        "1960_1975": BuildingClass.MEDIUM,  # Mixed construction
        "1976_1990": BuildingClass.LIGHT,  # Lightweight timber frame
        "1991_2010": BuildingClass.LIGHT,  # Timber frame
        "post_2010": BuildingClass.MEDIUM,  # Concrete + insulation
    }

    building_class = building_classes[era]

    # Typical geometry for Swedish villa
    # Assume 2-story, roughly square
    stories = 2
    footprint = floor_area_m2 / stories
    side_length = np.sqrt(footprint)
    height = 2.5 * stories

    params = {
        "floor_area_m2": floor_area_m2,
        "building_class": building_class,
        "wall_area_m2": 4 * side_length * height,
        "roof_area_m2": footprint * 1.2,  # Pitched roof factor
        "floor_area_ground_m2": footprint,
        "window_area_m2": floor_area_m2 * 0.15,  # ~15% window-to-floor ratio
        "building_height_m": height,
        "ach_infiltration": 0.3 if era in ["pre_1960", "1960_1975"] else 0.15,
        "ach_ventilation": 0.5,
        "heat_recovery_efficiency": 0.0 if era in ["pre_1960", "1960_1975", "1976_1990"] else 0.7,
        **u_values,
    }

    # Apply any user overrides
    params.update(kwargs)

    return Building5R1C(**params)


def create_swedish_apartment(
    floor_area_m2: float = 70.0,
    era: Literal["pre_1960", "1960_1975", "1976_1990", "1991_2010", "post_2010"] = "1976_1990",
    use_config: Optional[bool] = None,
    **kwargs,
) -> Building5R1C:
    """Create a legacy illustrative apartment; use explicit geometry for new work.

    This compatibility helper uses coarse, unsourced era values and assumes two
    exposed walls plus partial roof and floor exposure. It does not resolve an
    epsm_data LoD1 archetype or describe a particular apartment. For new work,
    resolve construction with epsm_data, provide measured/declared geometry to
    epsm_core, and run the resulting Building through the RC platform adapter.

    Parameters
    ----------
    floor_area_m2 : float
        Heated floor area [m^2]
    era : Literal['pre_1960', '1960_1975', '1976_1990', '1991_2010', 'post_2010']
        Construction era for U-value defaults
    use_config : Optional[bool]
        Deprecated. Kept for backward compatibility. Override
               parameters via **kwargs instead.
    **kwargs
        Override any Building5R1C parameter (e.g., u_walls=0.2)

    Returns
    -------
    Building5R1C
        Configured Building5R1C instance
    """
    warnings.warn(
        "create_swedish_apartment is deprecated: resolve epsm_data LoD1 "
        "construction and explicit epsm_core geometry instead",
        DeprecationWarning,
        stacklevel=2,
    )

    if use_config is not None:
        warnings.warn(
            "use_config is deprecated and has no effect; pass explicit operating inputs. "
            "It will remain accepted until at least 0.9.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Legacy illustrative U-values by era; not an epsm_data catalogue lookup.
    u_values_by_era = {
        "pre_1960": {"u_walls": 0.50, "u_roof": 0.35, "u_floor": 0.35, "u_windows": 2.5},
        "1960_1975": {"u_walls": 0.35, "u_roof": 0.20, "u_floor": 0.25, "u_windows": 2.2},
        "1976_1990": {"u_walls": 0.22, "u_roof": 0.12, "u_floor": 0.20, "u_windows": 1.6},
        "1991_2010": {"u_walls": 0.18, "u_roof": 0.10, "u_floor": 0.18, "u_windows": 1.2},
        "post_2010": {"u_walls": 0.12, "u_roof": 0.08, "u_floor": 0.12, "u_windows": 0.9},
    }

    u_values = u_values_by_era[era]

    # Apartments typically heavy (concrete) - building class by era
    building_classes = {
        "pre_1960": BuildingClass.VERY_HEAVY,  # Pre-war masonry
        "1960_1975": BuildingClass.VERY_HEAVY,  # Miljonprogrammet concrete
        "1976_1990": BuildingClass.HEAVY,  # Concrete
        "1991_2010": BuildingClass.HEAVY,  # Concrete
        "post_2010": BuildingClass.HEAVY,  # Concrete + CLT
    }

    building_class = building_classes[era]

    # Apartment geometry - less external surface per area
    # Assume middle unit (2 external walls)
    height = 2.6
    width = np.sqrt(floor_area_m2)  # Roughly square

    # Apartment envelope: assume corner/end unit with 2 external walls
    # Also include partial roof/floor transmission through thermal bridges
    external_wall_area = 2 * width * height

    # Mid-floor apartments still have some transmission through
    # party walls/floors due to thermal bridges and edge effects
    # Use ~20% of floor area as effective roof/floor transmission area
    effective_roof_floor_area = floor_area_m2 * 0.2

    params = {
        "floor_area_m2": floor_area_m2,
        "building_class": building_class,
        "wall_area_m2": external_wall_area,
        "roof_area_m2": effective_roof_floor_area,  # Thermal bridge/edge losses
        "floor_area_ground_m2": effective_roof_floor_area,  # Thermal bridge/edge losses
        "window_area_m2": floor_area_m2 * 0.12,  # ~12% window ratio
        "building_height_m": height,
        "ach_infiltration": 0.3 if era in ["pre_1960", "1960_1975"] else 0.15,
        "ach_ventilation": 0.5,
        "heat_recovery_efficiency": 0.0 if era in ["pre_1960", "1960_1975", "1976_1990"] else 0.7,
        "temp_adj_ground": 0.5,  # Party floor has ~50% temp diff vs outside
        **u_values,
    }

    params.update(kwargs)

    return Building5R1C(**params)
