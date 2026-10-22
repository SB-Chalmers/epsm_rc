# File: src/epsm_rc/heating.py
# Purpose: Standalone thermal demand and explicit heating-system calculations.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
# License: See LICENSE at the project root.
# Research attribution: hplib and Carnot references are retained below.
# Date modified: 2026-10-20
# Changelog:
#   2026-10-20 - Report model selection and reject failed hplib calculations.
"""Standalone heating systems with explicitly chosen heat-pump calculations.

``HeatPumpCarnot`` uses an adopted Carnot-efficiency approximation;
``HeatPumpHPLib`` uses the requested hplib model and its published correlation.
Their prediction accuracy is not established by this package's software checks.
The deprecated ``HeatPump`` factory reports which concrete model was selected.

Thermal demand uses an attached, mutable ``Building5R1C`` when supplied; otherwise
it uses the documented steady-state heat-loss approximation. Calls receiving a
building model reset and advance its state. Electrical loads are in watts at the
caller-selected slot duration; they are not the Core adapter's delivered-energy
accounting. Core owns that shared physical Building delivery boundary.

Import performs no database read or model selection. Explicit hplib construction
loads its installed model database; failures propagate with the requested model
and design conditions. No external services or result files are used here.

Examples
--------
>>> import numpy as np
>>> from epsm_rc.heating import HeatPumpCarnot
>>> model = HeatPumpCarnot(supply_temp_c=45, carnot_efficiency=0.4)
>>> model.cop(np.array([-7.0, 7.0])).shape
(2,)

References
----------
* hplib: https://github.com/FZJ-IEK3-VSA/hplib (MIT).
* Tjaden et al. (2021): https://doi.org/10.5281/zenodo.5521597.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import numpy as np

# Import Building5R1C for type hints (avoid circular import)
if TYPE_CHECKING:
    from .iso13790 import Building5R1C

# Try to import hplib - if not available, fall back to simplified model
try:
    import hplib.hplib as hpl

    HPLIB_AVAILABLE = True
except ImportError:
    hpl = None
    HPLIB_AVAILABLE = False


class HeatingSystem(ABC):
    """Abstract base class for heating strategies."""

    def __repr__(self):
        """Show the selected model and scalar configuration without running it."""
        details = f"indoor_temp_c={self.indoor_temp_c:g}"
        coefficient = getattr(self, "u_total_w_per_k", None)
        if coefficient is not None:
            details += f", u_total_w_per_k={coefficient:g}"
        return f"{type(self).__name__}({details})"

    def summary(self):
        """Return the selected model and its demand source without I/O.

        Returns
        -------
        dict
            Model class, indoor temperature in Celsius, heat-loss coefficient
            in W/K, attached thermal-model class and requested hplib model, if any.
            This is inspection data, not a saved calculation snapshot.
        """
        return {
            "model": type(self).__name__,
            "indoor_temperature_c": self.indoor_temp_c,
            "building_heat_loss_w_k": getattr(self, "u_total_w_per_k", None),
            "thermal_model": (
                type(self.building_model).__name__ if self.building_model is not None else None
            ),
            "hplib_model": getattr(self, "hp_model", None),
        }

    @abstractmethod
    def calculate_load(self, outdoor_temp_c: np.ndarray) -> np.ndarray:
        """Return electrical load [W] for given outdoor temperature series."""
        raise NotImplementedError


def _compute_5r1c_heat_demand(
    building_model,
    outdoor_temp_c: np.ndarray,
    solar_gains: Optional[np.ndarray],
    internal_gains: Optional[np.ndarray],
    dt_hours: float,
) -> np.ndarray:
    """Compute 5R1C heat demand series in one batch Rust call when possible."""
    import random

    from .iso13790 import _can_use_native

    outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=np.float64)
    if outdoor_temp_c.ndim != 1:
        raise ValueError("outdoor_temp_c must be one-dimensional")
    n = len(outdoor_temp_c)
    if solar_gains is None:
        solar_gains = np.zeros(n, dtype=np.float64)
    if internal_gains is None:
        internal_gains = np.full(
            n,
            building_model.internal_gains_w_per_m2 * building_model.floor_area_m2,
            dtype=np.float64,
        )
    # Validate before resetting state and before crossing the native boundary.
    sol_arr = np.ascontiguousarray(solar_gains, dtype=np.float64)
    int_arr = np.ascontiguousarray(internal_gains, dtype=np.float64)
    out_arr = np.ascontiguousarray(outdoor_temp_c, dtype=np.float64)
    for name, array in (
        ("outdoor_temp_c", out_arr),
        ("solar_gains", sol_arr),
        ("internal_gains", int_arr),
    ):
        if array.shape != (n,):
            raise ValueError(f"{name} must be one-dimensional and match outdoor_temp_c length")
        if not np.isfinite(array).all() or (name != "outdoor_temp_c" and (array < 0).any()):
            raise ValueError(
                f"{name} must contain finite{' nonnegative' if name != 'outdoor_temp_c' else ''} values"
            )
    if (
        hasattr(building_model, "_refresh_inputs")
        and _can_use_native(building_model, needs_demand=True)
        and n > 0
    ):
        from ._core_rs import simulate_5r1c_hourly

        building_model._refresh_inputs()
        building_model._validate_timestep_and_capacity(dt_hours, np.inf, 0.0)
        opq_arr = np.zeros(n, dtype=np.float64)
        res = simulate_5r1c_hourly(
            building_model,
            out_arr,
            sol_arr,
            int_arr,
            opq_arr,
            None,
            float(dt_hours),
            float(np.inf),
            0.0,
            True,
            random.random if 0.0 < building_model.ventilation_behaviour < 1.0 else None,
            False,
        )
        return np.maximum(0.0, res["heating_demand"])

    building_model.reset()
    q_demand = np.zeros(n)
    for i in range(n):
        q_demand[i] = building_model.calc_heating_demand(
            t_out=outdoor_temp_c[i],
            solar_gains=solar_gains[i] if solar_gains is not None else 0.0,
            internal_gains=internal_gains[i] if internal_gains is not None else None,
            dt_hours=dt_hours,
        )
    return np.maximum(0.0, q_demand)


# =============================================================================
# hplib-based Heat Pump Model (Primary - Empirically Validated)
# =============================================================================


@dataclass(repr=False)
class HeatPumpHPLib(HeatingSystem):
    """Calculate the explicitly requested hplib heat-pump correlation.

    Uses parametric fit equations derived from public Keymark test data:
        COP = p1*T_in + p2*T_out + p3 + p4*T_amb
        P_el = P_el_ref * (p1*T_in + p2*T_out + p3 + p4*T_amb)

    Published aggregate fit metrics reported by hplib (not RC certification):
        - MAPE COP: 9.8% average
        - MAPE P_el: 16.3% average
        - MAPE P_th: 19.7% average

    Supports:
        - Generic heat pumps (Group 1-6 averages)
        - Specific manufacturer models (5812 air/water HPs in database)
        - Inverter (regulated) and on-off subtypes
        - ISO 13790 5R1C building model for dynamic heat demand


    Parameters
    ----------
    indoor_temp_c : float
        Target indoor temperature [°C]
    u_total_w_per_k : float
        Building heat loss coefficient H_tr [W/K].
        This is NOT a U-value! It's the total heat transfer coefficient:
        H_tr = Σ(U_i × A_i) summed over walls, roof, floor, windows.
        Used only for heat pump SIZING. If building_model is attached,
        the 5R1C model calculates actual heat demand dynamically.
    hp_model : str
        hplib model name or 'Generic' for average performance
    group_id : int
        Heat pump type (1=air/water regulated, 4=air/water on-off, etc.)
    design_temp_c : float
        Outdoor design temperature for aux heater sizing [°C]
    supply_temp_c : float
        Heating supply water temperature [°C] (affects COP)
    thermal_capacity_w : Optional[float]
        Desired thermal output at design conditions [W]
        If None, auto-sized from building heat loss
    min_cop : float, default 1
        Lower bound applied to calculated COP, dimensionless.
    thermal_time_constant_hours : float, default 2
        First-order smoothing time for steady-state electrical loads in hours;
        ignored when the attached thermal model supplies its own dynamics.
    building_model : Optional['Building5R1C']
        Optional Building5R1C for dynamic heat demand calculation.
        When attached, this model calculates actual H_tr from individual
        component U-values and areas, replacing the u_total_w_per_k estimate.

    Notes
    -----
    Configuration remains inspectable and mutable. ``calculate_load`` resets and
    advances an attached building's thermal state; it does not write result files.
    Construction reads the installed hplib database. Invalid model loading or
    design evaluation raises ValueError with a preserved cause, rather than
    substituting a generic model or estimated capacity.
    """

    indoor_temp_c: float = 21.0
    u_total_w_per_k: float = 200.0  # H_tr [W/K] for sizing; 5R1C overrides for demand
    hp_model: str = "Generic"
    group_id: int = 1  # 1=Air/Water regulated (inverter), 4=Air/Water on-off
    design_temp_c: float = -20.0
    supply_temp_c: float = 45.0
    thermal_capacity_w: Optional[float] = None
    min_cop: float = 1.0

    # Thermal dynamics
    thermal_time_constant_hours: float = 2.0

    # ISO 13790 5R1C building model (optional - for dynamic heat demand)
    building_model: Optional["Building5R1C"] = None

    # Auxiliary heater for extreme cold
    aux_heater_max_w: float = field(default=None, init=False)

    # hplib parameters (set in __post_init__)
    _hp_params: object = field(default=None, init=False, repr=False)
    _p_th_ref: float = field(default=None, init=False)

    def __post_init__(self):
        """Initialize hplib parameters and size auxiliary heater."""
        if not HPLIB_AVAILABLE:
            raise ImportError("hplib required for HeatPumpHPLib. Install with: pip install hplib")

        # Calculate design heat load
        design_heat_load = self.u_total_w_per_k * (self.indoor_temp_c - self.design_temp_c)

        # Use specified capacity or size to match design load
        target_capacity = self.thermal_capacity_w or design_heat_load

        # Get hplib parameters for the model
        # For Generic models, we need to specify the thermal capacity
        try:
            self._hp_params = hpl.get_parameters(
                model=self.hp_model,
                group_id=self.group_id,
                t_in=-7,  # Reference conditions (Keymark A-7)
                t_out=int(self.supply_temp_c),
                p_th=int(target_capacity),
            )
            self._p_th_ref = float(self._hp_params["P_th_h_ref [W]"].values[0])
        except Exception as error:
            raise ValueError(
                f"Cannot load requested hplib model {self.hp_model!r} "
                f"in group {self.group_id}. Check the installed hplib database, "
                "model name, supply temperature and requested capacity. "
                "Select HeatPumpCarnot explicitly if that approximation is intended."
            ) from error
        if not np.isfinite(self._p_th_ref) or self._p_th_ref <= 0:
            raise ValueError(f"hplib model {self.hp_model!r} returned invalid reference capacity")

        # Size auxiliary heat using the requested model at the design temperature.
        try:
            result = hpl.simulate(
                t_in_primary=self.design_temp_c,
                t_in_secondary=self.supply_temp_c - 5,
                parameters=self._hp_params,
                t_amb=self.design_temp_c,
            )
            hp_output_at_design = float(result["P_th"].values[0])
        except Exception as error:
            raise ValueError(
                f"Cannot evaluate hplib model {self.hp_model!r} at design temperature "
                f"{self.design_temp_c:g} C. Check the model operating range and supply temperature."
            ) from error
        if not np.isfinite(hp_output_at_design) or hp_output_at_design < 0:
            raise ValueError(f"hplib model {self.hp_model!r} returned invalid design output")

        # Aux heater covers remaining load with 10% margin
        aux_needed = max(0, design_heat_load - hp_output_at_design)
        self.aux_heater_max_w = max(aux_needed * 1.1, 2000.0)

    def simulate_hplib(
        self,
        outdoor_temp_c: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate heat pump using hplib.


        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Tuple of (COP, P_el [W], P_th [W])
        """
        outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)
        if outdoor_temp_c.ndim != 1 or not np.isfinite(outdoor_temp_c).all():
            raise ValueError("hplib outdoor temperatures must be a finite one-dimensional array")
        n = len(outdoor_temp_c)

        # hplib simulate returns DataFrame with columns COP, P_el, P_th, etc.
        result = hpl.simulate(
            t_in_primary=outdoor_temp_c,
            t_in_secondary=self.supply_temp_c - 5,  # Return temp
            parameters=self._hp_params,
            t_amb=outdoor_temp_c,
        )

        # hplib returns nested arrays when given array input
        # Result is a DataFrame with 1 row, each cell contains an array
        def extract_array(col_name: str) -> np.ndarray:
            val = result[col_name].values[0]
            if isinstance(val, np.ndarray):
                return val.astype(float)
            elif isinstance(val, (list, tuple)):
                return np.array(val, dtype=float)
            else:
                return np.full(n, float(val))

        cop = extract_array("COP")
        p_el = extract_array("P_el")
        p_th = extract_array("P_th")

        for name, values in (("COP", cop), ("P_el", p_el), ("P_th", p_th)):
            if values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError(
                    f"hplib model {self.hp_model!r} returned invalid {name}; "
                    f"expected {n} finite values, received shape {values.shape}"
                )

        # Clamp COP to sensible range
        cop = np.clip(cop, self.min_cop, 10.0)
        p_el = np.clip(p_el, 0, None)
        p_th = np.clip(p_th, 0, None)

        return cop, p_el, p_th

    def heat_demand_w(
        self,
        outdoor_temp_c: np.ndarray,
        solar_gains: Optional[np.ndarray] = None,
        internal_gains: Optional[np.ndarray] = None,
        minutes_per_slot: int = 10,
    ) -> np.ndarray:
        """Compute building heat demand [W].

        If a Building5R1C model is attached, uses ISO 13790 dynamic calculation
        with thermal mass, solar gains, and internal gains. Otherwise falls back
        to simple steady-state Q = U × ΔT.


        Parameters
        ----------
        outdoor_temp_c : numpy.ndarray
            Outdoor temperature series [°C]
        solar_gains : Optional[numpy.ndarray]
            Solar heat gains through windows [W] (optional)
        internal_gains : Optional[numpy.ndarray]
            Internal gains from occupants/appliances [W] (optional)
        minutes_per_slot : int
            Time resolution for 5R1C model


        Returns
        -------
        numpy.ndarray
            Heat demand [W] array
        """
        outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)

        if self.building_model is not None:
            # Use ISO 13790 5R1C dynamic model
            dt_hours = minutes_per_slot / 60.0
            return _compute_5r1c_heat_demand(
                self.building_model, outdoor_temp_c, solar_gains, internal_gains, dt_hours
            )
        else:
            # Fall back to simple steady-state model
            delta = self.indoor_temp_c - outdoor_temp_c
            return np.maximum(0.0, self.u_total_w_per_k * delta)

    def calculate_load(
        self,
        outdoor_temp_c: np.ndarray,
        rng: np.random.Generator | None = None,
        minutes_per_slot: int = 10,
        solar_gains: Optional[np.ndarray] = None,
        internal_gains: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate electrical load using hplib simulation.


        Parameters
        ----------
        outdoor_temp_c : numpy.ndarray
            Outdoor temperature series [°C]
        rng : np.random.Generator | None
            Random generator (unused, for API compatibility)
        minutes_per_slot : int
            Time resolution
        solar_gains : Optional[numpy.ndarray]
            Solar heat gains [W] (for 5R1C model)
        internal_gains : Optional[numpy.ndarray]
            Internal gains [W] (for 5R1C model)


        Returns
        -------
        numpy.ndarray
            Total electrical load [W] (compressor + auxiliary)
        """
        outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)

        # Get heat demand (uses 5R1C if building_model is set)
        q_demand = self.heat_demand_w(
            outdoor_temp_c,
            solar_gains=solar_gains,
            internal_gains=internal_gains,
            minutes_per_slot=minutes_per_slot,
        )

        # Simulate HP performance
        cop, p_el_hp, p_th_hp = self.simulate_hplib(outdoor_temp_c)

        # HP output is capped by its capacity - it delivers what it can
        p_th_hp = np.clip(p_th_hp, 0, None)

        # Deficit must be covered by auxiliary heater (COP=1)
        deficit = np.maximum(0.0, q_demand - p_th_hp)
        p_aux = np.minimum(deficit, self.aux_heater_max_w)

        # Total electrical load
        # Only count HP power when there's demand
        p_el_hp = np.where(q_demand > 0, p_el_hp, 0)
        total = p_el_hp + p_aux

        # Apply thermal mass filter for realistic load dynamics
        # (Skip if using 5R1C model which already has thermal mass dynamics)
        if self.building_model is None and self.thermal_time_constant_hours > 0 and len(total) > 1:
            dt_hours = minutes_per_slot / 60.0
            alpha = dt_hours / (self.thermal_time_constant_hours + dt_hours)

            filtered = np.zeros_like(total)
            filtered[0] = total[0]
            for i in range(1, len(total)):
                filtered[i] = alpha * total[i] + (1 - alpha) * filtered[i - 1]
            total = filtered

        return np.maximum(0.0, total)


# =============================================================================
# Simplified Carnot Model (Fallback when hplib not available)
# =============================================================================


@dataclass(repr=False)
class HeatPumpCarnot(HeatingSystem):
    """Simplified heat pump model using Carnot-based COP approximation.

    This is an explicitly selected theoretical approximation. Adopt its
    efficiency and operating assumptions for the intended experiment; package
    tests do not establish accuracy for a particular installed heat pump.

    COP Model:
        COP = η_carnot × T_sink / (T_sink - T_source)

    Known limitations:
        - No defrost cycle penalty (overestimates COP below -5°C)
        - No part-load efficiency (inverter HPs vary with capacity)
        - Linear approximation vs. empirical non-linear behavior


    Parameters
    ----------
    indoor_temp_c : float
        Target indoor temperature [°C]
    u_total_w_per_k : float
        Building heat loss coefficient H_tr [W/K].
        This is NOT a U-value! H_tr = Σ(U_i × A_i) summed over envelope.
        Used for heat pump SIZING. If building_model attached, 5R1C
        calculates actual demand from individual U-values and areas.
    supply_temp_c : float
        Heating supply water temperature [°C]
    bivalent_temp_c : float
        Temperature where HP reaches max capacity [°C]
    design_temp_c : float
        Outdoor design temperature [°C]
    carnot_efficiency : float
        Fraction of Carnot COP (0.25-0.50)
    hp_coverage_at_bivalent : float
        Fraction of load HP covers at bivalent point
    min_cop : float, default 1
        Lower bound applied to calculated COP, dimensionless.
    thermal_time_constant_hours : float, default 2
        First-order smoothing time for steady-state electrical loads in hours;
        ignored when the attached thermal model supplies its own dynamics.
    building_model : Optional['Building5R1C']
        Optional Building5R1C for dynamic heat demand

    Notes
    -----
    Configuration remains inspectable and mutable. ``calculate_load`` resets and
    advances an attached building's thermal state; it does not write result files.
    """

    indoor_temp_c: float = 21.0
    u_total_w_per_k: float = 200.0  # H_tr [W/K] for sizing; 5R1C overrides for demand
    supply_temp_c: float = 45.0
    bivalent_temp_c: float = -7.0
    design_temp_c: float = -20.0
    carnot_efficiency: float = 0.40
    hp_coverage_at_bivalent: float = 0.85
    min_cop: float = 1.0

    thermostat_deadband_c: float = 0.5
    thermal_time_constant_hours: float = 2.0

    # ISO 13790 5R1C building model (optional)
    building_model: Optional["Building5R1C"] = None

    compressor_power_max_w: float = field(default=None, init=False)
    aux_power_max_w: float = field(default=None, init=False)

    def __post_init__(self):
        """Size the heat pump based on building characteristics."""
        delta_bivalent = self.indoor_temp_c - self.bivalent_temp_c
        q_bivalent = self.u_total_w_per_k * delta_bivalent

        delta_design = self.indoor_temp_c - self.design_temp_c
        q_design = self.u_total_w_per_k * delta_design

        cop_bivalent = self._cop_at_temp(self.bivalent_temp_c)

        q_hp_max = q_bivalent * self.hp_coverage_at_bivalent
        self.compressor_power_max_w = q_hp_max / max(cop_bivalent, 1.0)

        cop_design = self._cop_at_temp(self.design_temp_c)
        q_hp_at_design = self.compressor_power_max_w * cop_design
        aux_needed = q_design - q_hp_at_design
        self.aux_power_max_w = max(aux_needed * 1.1, 2000.0)

    def _cop_at_temp(self, outdoor_temp_c: float) -> float:
        """Calculate COP at a single temperature using Carnot model."""
        T_source = outdoor_temp_c + 273.15
        T_sink = self.supply_temp_c + 273.15

        if T_sink <= T_source:
            return 10.0

        cop_carnot = T_sink / (T_sink - T_source)
        cop = self.carnot_efficiency * cop_carnot
        return max(self.min_cop, cop)

    def cop(self, outdoor_temp_c: np.ndarray) -> np.ndarray:
        """Compute COP array using Carnot-based approximation."""
        outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)
        T_source = outdoor_temp_c + 273.15
        T_sink = self.supply_temp_c + 273.15

        delta = T_sink - T_source
        delta = np.maximum(delta, 1.0)

        cop_carnot = T_sink / delta
        cop = self.carnot_efficiency * cop_carnot
        return np.maximum(self.min_cop, cop)

    def heat_demand_w(
        self,
        outdoor_temp_c: np.ndarray,
        solar_gains: Optional[np.ndarray] = None,
        internal_gains: Optional[np.ndarray] = None,
        minutes_per_slot: int = 10,
    ) -> np.ndarray:
        """Compute heat demand [W] using 5R1C if available."""
        outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)

        if self.building_model is not None:
            # Use ISO 13790 5R1C dynamic model
            dt_hours = minutes_per_slot / 60.0
            return _compute_5r1c_heat_demand(
                self.building_model, outdoor_temp_c, solar_gains, internal_gains, dt_hours
            )
        else:
            delta = self.indoor_temp_c - outdoor_temp_c
            return np.maximum(0.0, self.u_total_w_per_k * delta)

    def electrical_load(
        self,
        outdoor_temp_c: np.ndarray,
        solar_gains: Optional[np.ndarray] = None,
        internal_gains: Optional[np.ndarray] = None,
        minutes_per_slot: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute electrical loads (total, compressor, aux)."""
        outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)
        cop = self.cop(outdoor_temp_c)
        q = self.heat_demand_w(
            outdoor_temp_c,
            solar_gains=solar_gains,
            internal_gains=internal_gains,
            minutes_per_slot=minutes_per_slot,
        )

        p_comp_ideal = np.divide(q, cop, out=np.zeros_like(q), where=cop > 0)
        p_comp = np.minimum(p_comp_ideal, self.compressor_power_max_w)

        q_from_hp = p_comp * cop
        deficit = np.maximum(0.0, q - q_from_hp)
        p_aux = np.minimum(deficit, self.aux_power_max_w)

        return p_comp + p_aux, p_comp, p_aux

    def calculate_load(
        self,
        outdoor_temp_c: np.ndarray,
        rng: np.random.Generator | None = None,
        minutes_per_slot: int = 10,
        solar_gains: Optional[np.ndarray] = None,
        internal_gains: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate electrical load."""
        total, _comp, _aux = self.electrical_load(
            outdoor_temp_c,
            solar_gains=solar_gains,
            internal_gains=internal_gains,
            minutes_per_slot=minutes_per_slot,
        )

        # Skip thermal filter if using 5R1C model (it has its own thermal mass)
        if self.building_model is None and self.thermal_time_constant_hours > 0 and len(total) > 1:
            dt_hours = minutes_per_slot / 60.0
            alpha = dt_hours / (self.thermal_time_constant_hours + dt_hours)

            filtered = np.zeros_like(total)
            filtered[0] = total[0]
            for i in range(1, len(total)):
                filtered[i] = alpha * total[i] + (1 - alpha) * filtered[i - 1]
            total = filtered

        return total


# =============================================================================
# Unified HeatPump Factory (Auto-selects best available model)
# =============================================================================


def HeatPump(
    indoor_temp_c: float = 21.0,
    u_total_w_per_k: float = 200.0,
    supply_temp_c: float = 45.0,
    bivalent_temp_c: float = -7.0,
    design_temp_c: float = -20.0,
    carnot_efficiency: float = 0.40,
    hp_coverage_at_bivalent: float = 0.85,
    hp_model: str = "Generic",
    group_id: int = 1,
    thermal_time_constant_hours: float = 2.0,
    building_model: Optional["Building5R1C"] = None,
) -> HeatingSystem:
    """Create the historically selected heat-pump model (deprecated).

    Automatically selects:
        - HeatPumpHPLib (if hplib installed) - empirically validated
        - HeatPumpCarnot (fallback) - simplified theoretical model


    Parameters
    ----------
    indoor_temp_c : float
        Target indoor temperature [°C]
    u_total_w_per_k : float
        Building heat loss coefficient H_tr [W/K].
        This is NOT a U-value! It's the total heat transfer coefficient:
            H_tr = Σ(U_i × A_i) for walls, roof, floor, windows + ventilation
        Used only for heat pump SIZING (compressor and aux heater capacity).
        If building_model is attached, the 5R1C model calculates actual
        heat demand from individual component U-values and areas.
    supply_temp_c : float
        Heating supply water temperature [°C]
    bivalent_temp_c : float
        Temperature where HP reaches max capacity [°C]
    design_temp_c : float
        Outdoor design temperature [°C]
    carnot_efficiency : float
        Carnot efficiency (only used in fallback model)
    hp_coverage_at_bivalent : float
        Fraction of bivalent load used for sizing
    hp_model : str
        hplib model name ('Generic' or specific manufacturer model)
    group_id : int
        hplib group ID (1=air/water regulated, 4=air/water on-off)
    thermal_time_constant_hours : float
        Building thermal mass time constant
    building_model : Optional['Building5R1C']
        Optional Building5R1C for dynamic heat demand calculation.
        When provided, the 5R1C model calculates heat demand using proper
        physics with individual U-values, thermal mass, solar gains, etc.


    Returns
    -------
    HeatingSystem
        HeatingSystem instance (HeatPumpHPLib or HeatPumpCarnot)
    """
    if HPLIB_AVAILABLE:
        # Calculate thermal capacity from building characteristics
        thermal_capacity = (
            u_total_w_per_k * (indoor_temp_c - bivalent_temp_c) * hp_coverage_at_bivalent
        )

        model = HeatPumpHPLib(
            indoor_temp_c=indoor_temp_c,
            u_total_w_per_k=u_total_w_per_k,
            hp_model=hp_model,
            group_id=group_id,
            design_temp_c=design_temp_c,
            supply_temp_c=supply_temp_c,
            thermal_capacity_w=thermal_capacity,
            thermal_time_constant_hours=thermal_time_constant_hours,
            building_model=building_model,
        )
    else:
        model = HeatPumpCarnot(
            indoor_temp_c=indoor_temp_c,
            u_total_w_per_k=u_total_w_per_k,
            supply_temp_c=supply_temp_c,
            bivalent_temp_c=bivalent_temp_c,
            design_temp_c=design_temp_c,
            carnot_efficiency=carnot_efficiency,
            hp_coverage_at_bivalent=hp_coverage_at_bivalent,
            thermal_time_constant_hours=thermal_time_constant_hours,
            building_model=building_model,
        )
    warnings.warn(
        f"HeatPump() automatically selected {type(model).__name__}. "
        "Automatic model selection is deprecated; instantiate HeatPumpHPLib or "
        "HeatPumpCarnot explicitly. Removal will be no earlier than 0.9.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return model


# =============================================================================
# Other Heating Systems (unchanged)
# =============================================================================


@dataclass(repr=False)
class DistrictHeating(HeatingSystem):
    """District heating (fjärrvärme) - zero electrical load to resident.

    Swedish district heating delivers hot water from CHP plants through pipes.
    From the household electricity meter's perspective, this is invisible:
    - The heat exchanger requires no electricity
    - Circulation pumps are on building common areas or district network
    - The resident sees ZERO heating-related electrical load

    However, we still calculate thermal demand using the 5R1C building model
    to enable analysis of heat demand profiles for district heating networks.


    Parameters
    ----------
    indoor_temp_c : float
        Target indoor temperature [°C]
    building_model : Optional['Building5R1C']
        ISO 13790 5R1C building model for thermal demand
    circulation_pump_w : float
        Small circulation pump if on resident meter [W]
        (typically 0 - pumps are usually on common areas)
    """

    indoor_temp_c: float = 21.0
    building_model: Optional["Building5R1C"] = None
    circulation_pump_w: float = 0.0  # Usually on common areas, not resident meter

    # Track thermal demand for analysis (not electrical)
    _last_thermal_demand_w: np.ndarray = field(default=None, init=False, repr=False)

    def heat_demand_w(
        self,
        outdoor_temp_c: np.ndarray,
        solar_gains: Optional[np.ndarray] = None,
        internal_gains: Optional[np.ndarray] = None,
        minutes_per_slot: int = 10,
    ) -> np.ndarray:
        """Calculate thermal heat demand [W] using 5R1C building model.

        This is the heat that district heating must supply.
        """
        outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)

        if self.building_model is not None:
            dt_hours = minutes_per_slot / 60.0
            self._last_thermal_demand_w = _compute_5r1c_heat_demand(
                self.building_model, outdoor_temp_c, solar_gains, internal_gains, dt_hours
            )
            return self._last_thermal_demand_w
        else:
            # No building model - return zeros (unknown demand)
            self._last_thermal_demand_w = np.zeros(len(outdoor_temp_c))
            return self._last_thermal_demand_w

    def calculate_load(
        self,
        outdoor_temp_c: np.ndarray,
        rng: np.random.Generator | None = None,
        minutes_per_slot: int = 10,
        solar_gains: Optional[np.ndarray] = None,
        internal_gains: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return electrical load [W] - essentially zero for district heating.

        We still compute thermal demand internally (stored in _last_thermal_demand_w)
        for analysis purposes, but electrical load is just the circulation pump.
        """
        # Calculate thermal demand (for analysis, stored internally)
        self.heat_demand_w(
            outdoor_temp_c,
            solar_gains=solar_gains,
            internal_gains=internal_gains,
            minutes_per_slot=minutes_per_slot,
        )

        # Electrical load is just circulation pump (usually 0)
        n = len(outdoor_temp_c)
        if self.circulation_pump_w > 0 and self._last_thermal_demand_w is not None:
            # Pump runs when there's heat demand
            return np.where(self._last_thermal_demand_w > 0, self.circulation_pump_w, 0.0)
        return np.zeros(n, dtype=float)


@dataclass(repr=False)
class DirectElectric(HeatingSystem):
    """Direct resistive electric heating (COP=1).


    Parameters
    ----------
    indoor_temp_c : float
        Target indoor temperature [°C]
    u_total_w_per_k : float
        Building heat loss coefficient H_tr [W/K].
        NOT a U-value! H_tr = Σ(U_i × A_i) over envelope components.
        Used only if building_model is not provided.
    building_model : Optional['Building5R1C']
        ISO 13790 5R1C building model for thermal demand
    """

    indoor_temp_c: float = 21.0
    u_total_w_per_k: float = 200.0  # H_tr [W/K] - fallback if no building_model
    building_model: Optional["Building5R1C"] = None

    def heat_demand_w(
        self,
        outdoor_temp_c: np.ndarray,
        solar_gains: Optional[np.ndarray] = None,
        internal_gains: Optional[np.ndarray] = None,
        minutes_per_slot: int = 10,
    ) -> np.ndarray:
        """Calculate thermal heat demand [W]."""
        outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)

        if self.building_model is not None:
            dt_hours = minutes_per_slot / 60.0
            return _compute_5r1c_heat_demand(
                self.building_model, outdoor_temp_c, solar_gains, internal_gains, dt_hours
            )
        else:
            # Fallback: simple steady-state
            delta = self.indoor_temp_c - outdoor_temp_c
            return np.maximum(0.0, self.u_total_w_per_k * delta)

    def calculate_load(
        self,
        outdoor_temp_c: np.ndarray,
        rng: np.random.Generator | None = None,
        minutes_per_slot: int = 10,
        solar_gains: Optional[np.ndarray] = None,
        internal_gains: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return electrical load [W] = thermal demand (COP=1)."""
        return self.heat_demand_w(
            outdoor_temp_c,
            solar_gains=solar_gains,
            internal_gains=internal_gains,
            minutes_per_slot=minutes_per_slot,
        )


@dataclass(repr=False)
class WoodPelletHybrid(HeatingSystem):
    """Swedish wood/pellet boiler with electric backup.

    Modern pellet boilers or wood boilers are common in rural Sweden.
    Electric backup covers ~5-10% of annual heat demand.


    Parameters
    ----------
    indoor_temp_c : float
        Target indoor temperature [°C]
    u_total_w_per_k : float
        Building heat loss coefficient H_tr [W/K].
        NOT a U-value! H_tr = Σ(U_i × A_i) over envelope components.
        Used only if building_model is not provided.
    wood_coverage_fraction : float
        Fraction of heat demand covered by wood/pellet
    electric_backup_max_w : float
        Maximum electric backup heater power [W]
    building_model : Optional['Building5R1C']
        ISO 13790 5R1C building model for thermal demand
    """

    indoor_temp_c: float = 21.0
    u_total_w_per_k: float = 200.0  # H_tr [W/K] - fallback if no building_model
    wood_coverage_fraction: float = 0.90
    electric_backup_max_w: float = 3000.0
    pellet_circulation_pump_w: float = 50.0
    shoulder_season_temp_c: float = 12.0
    building_model: Optional["Building5R1C"] = None

    def heat_demand_w(
        self,
        outdoor_temp_c: np.ndarray,
        solar_gains: Optional[np.ndarray] = None,
        internal_gains: Optional[np.ndarray] = None,
        minutes_per_slot: int = 10,
    ) -> np.ndarray:
        """Calculate thermal heat demand [W]."""
        outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)

        if self.building_model is not None:
            dt_hours = minutes_per_slot / 60.0
            return _compute_5r1c_heat_demand(
                self.building_model, outdoor_temp_c, solar_gains, internal_gains, dt_hours
            )
        else:
            # Fallback: simple steady-state
            delta = self.indoor_temp_c - outdoor_temp_c
            return np.maximum(0.0, self.u_total_w_per_k * delta)

    def calculate_load(
        self,
        outdoor_temp_c: np.ndarray,
        rng: np.random.Generator | None = None,
        minutes_per_slot: int = 10,
        solar_gains: Optional[np.ndarray] = None,
        internal_gains: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate electric load for wood/pellet + electric backup system."""
        outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)
        n = len(outdoor_temp_c)

        # Get thermal demand from 5R1C model (or fallback)
        q_total = self.heat_demand_w(
            outdoor_temp_c,
            solar_gains=solar_gains,
            internal_gains=internal_gains,
            minutes_per_slot=minutes_per_slot,
        )

        electric_load = np.zeros(n, dtype=float)

        heating_season = outdoor_temp_c < self.shoulder_season_temp_c

        # During heating season, wood covers most load, electric is backup
        electric_fraction = 1.0 - self.wood_coverage_fraction
        electric_load[heating_season] = np.minimum(
            q_total[heating_season] * electric_fraction, self.electric_backup_max_w
        )

        # Circulation pump runs when there's any heat demand
        needs_heat = q_total > 0
        electric_load[needs_heat] += self.pellet_circulation_pump_w

        # Shoulder season: wood stove not lit, electric covers all
        shoulder = ~heating_season & (q_total > 0)
        electric_load[shoulder] = np.minimum(q_total[shoulder], self.electric_backup_max_w)

        return electric_load


# =============================================================================
# Utility Functions
# =============================================================================


def list_available_hp_models(hp_type: Literal["air", "brine", "water"] = "air") -> list:
    """List available heat pump models from hplib database.


    Parameters
    ----------
    hp_type : Literal['air', 'brine', 'water']
        Heat source type ('air', 'brine', or 'water')


    Returns
    -------
    list
        List of available model names (format: 'Manufacturer Model')
    """
    if not HPLIB_AVAILABLE:
        return ["hplib not installed - only Generic models available"]

    db = hpl.load_database()

    type_filter = {
        "air": "Outdoor Air/Water",
        "brine": "Brine/Water",
        "water": "Water/Water",
    }

    filtered = db[db["Type"] == type_filter.get(hp_type, "Outdoor Air/Water")]
    # Combine Manufacturer and Model for a unique identifier
    names = (
        filtered["Manufacturer"].astype(str).str.strip()
        + " "
        + filtered["Model"].astype(str).str.strip()
    )
    return names.unique().tolist()


def get_hp_info(model_name: str) -> dict:
    """Get information about a specific heat pump model.


    Parameters
    ----------
    model_name : str
        Name of the heat pump model (format: 'Manufacturer Model')


    Returns
    -------
    dict
        Dictionary with model specifications
    """
    if not HPLIB_AVAILABLE:
        return {"error": "hplib not installed"}

    db = hpl.load_database()

    # Create combined name column for matching
    db_names = (
        db["Manufacturer"].astype(str).str.strip() + " " + db["Model"].astype(str).str.strip()
    )
    match = db[db_names == model_name]

    if len(match) == 0:
        # Try partial match
        mask = db_names.str.contains(model_name, case=False, na=False)
        match = db[mask]

    if len(match) == 0:
        return {"error": f"Model '{model_name}' not found"}

    row = match.iloc[0]
    return {
        "manufacturer": str(row.get("Manufacturer", "Unknown")).strip(),
        "model": str(row.get("Model", "Unknown")).strip(),
        "type": row.get("Type", "Unknown"),
        "subtype": row.get("Subtype", "Unknown"),
        "group_id": int(row.get("Group", 1)),
        "p_th_ref_w": float(row.get("P_th_h_ref [W]", 0)),
        "p_el_ref_w": float(row.get("P_el_h_ref [W]", 0)),
        "cop_ref": float(row.get("COP_ref", 0)),
        "refrigerant": row.get("Refrigerant", "Unknown"),
        "mape_cop": float(row.get("MAPE_COP", 0)),
        "mape_p_el": float(row.get("MAPE_P_el", 0)),
        "mape_p_th": float(row.get("MAPE_P_th", 0)),
    }
