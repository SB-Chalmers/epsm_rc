"""Heating strategies and heat pump physics.

Implements a Strategy Pattern for heating systems with heat pump models based on
the hplib library (FZJ-IEK3-VSA/hplib) - empirically validated against EHPA
Keymark test data from 5800+ heat pumps.

Heat Pump COP Model:
    Primary: hplib parametric fit (MAPE ~9.8% vs Keymark data)
        COP = p1*T_in + p2*T_out + p3 + p4*T_amb
        P_el = P_el_ref * (p1*T_in + p2*T_out + p3 + p4*T_amb)
    
    Fallback: Simplified Carnot approximation (if hplib not installed)
        COP = η_carnot × T_sink / (T_sink - T_source)

References:
    - hplib: https://github.com/FZJ-IEK3-VSA/hplib (MIT License)
    - Tjaden et al. (2021): DOI 10.5281/zenodo.5521597
    - EN 14825:2022 test conditions used for Keymark certification
    - Schwamberger (1991): VDI Fortschrittsberichte Reihe 6 Nr. 263

All arrays operate at slot resolution defined by the simulator (e.g., 144 slots/day).

# =============================================================================
# NOTE [BUILDING-PHYSICS-MODEL]:
# =============================================================================
# The heat pump models in this module use a simplified steady-state approach:
#
#   Q_demand = U_total × (T_indoor - T_outdoor)
#
# For more accurate building physics simulation including:
#   - Solar gains through windows
#   - Internal gains (occupants, appliances)
#   - Ventilation with heat recovery
#   - Thermal mass dynamics (building time constant)
#
# Use the ISO 13790 5R1C model in src/models/iso13790.py:
#   - Building5R1C: Full 5-resistance, 1-capacitance thermal network
#   - create_swedish_villa(): Pre-configured for Swedish villas by era
#   - create_swedish_apartment(): Pre-configured for Swedish apartments
#
# The 5R1C model is adapted from DIBS (Dynamic ISO Building Simulator)
# by Institut Wohnen und Umwelt (IWU), licensed under MIT.
# Reference: ISO 13790:2008 "Energy performance of buildings"
# =============================================================================

# =============================================================================
# TODO [HEAT-PUMP-MODEL-LIMITATIONS]:
# =============================================================================
# While hplib provides empirically-validated COP curves, the implementation
# still has limitations:
#
# 1. DEFROST CYCLES: At 0-5°C outdoor temp, air-source heat pumps need
#    periodic defrost, consuming ~5-10% of heating energy. hplib captures
#    this implicitly in Keymark data, but we can't model the transient spikes.
#
# 2. PART-LOAD EFFICIENCY: Inverter HPs are MORE efficient at part-load than
#    full-load. The model uses hplib curves which capture some of this, but
#    we don't explicitly model modulation.
#
# 3. SUPPLY TEMPERATURE MODULATION: Real systems use outdoor-reset curves
#    to lower supply temp in mild weather (improving COP). We use fixed
#    supply_temp_c, missing this optimization.
#
# 4. STARTUP/SHUTDOWN DYNAMICS: Compressor startup draws 3-5x normal current
#    for a few seconds. We model average power, not transients.
#
# 5. GROUND-SOURCE LIMITATIONS: For bergvärme (geothermal), the source temp
#    varies seasonally and with extraction rate. We use static assumptions.
# =============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal, TYPE_CHECKING
from abc import ABC, abstractmethod
import warnings

import numpy as np

# Import Building5R1C for type hints (avoid circular import)
if TYPE_CHECKING:
    from .iso13790 import Building5R1C

# Try to import hplib - if not available, fall back to simplified model
try:
    import hplib.hplib as hpl
    HPLIB_AVAILABLE = True
except ImportError:
    HPLIB_AVAILABLE = False
    warnings.warn(
        "hplib not installed. Using simplified Carnot COP model. "
        "For empirically-validated heat pump simulation, install hplib: "
        "pip install hplib",
        UserWarning
    )


class HeatingSystem(ABC):
    """Abstract base class for heating strategies."""

    @abstractmethod
    def calculate_load(self, outdoor_temp_c: np.ndarray) -> np.ndarray:
        """Return electrical load [W] for given outdoor temperature series."""
        raise NotImplementedError


# =============================================================================
# hplib-based Heat Pump Model (Primary - Empirically Validated)
# =============================================================================

@dataclass
class HeatPumpHPLib(HeatingSystem):
    """Heat pump model using hplib (EHPA Keymark validated).
    
    Uses parametric fit equations derived from public Keymark test data:
        COP = p1*T_in + p2*T_out + p3 + p4*T_amb
        P_el = P_el_ref * (p1*T_in + p2*T_out + p3 + p4*T_amb)
    
    Validation metrics (from hplib database):
        - MAPE COP: 9.8% average
        - MAPE P_el: 16.3% average
        - MAPE P_th: 19.7% average
    
    Supports:
        - Generic heat pumps (Group 1-6 averages)
        - Specific manufacturer models (5812 air/water HPs in database)
        - Inverter (regulated) and on-off subtypes
        - ISO 13790 5R1C building model for dynamic heat demand
    
    Args:
        indoor_temp_c: Target indoor temperature [°C]
        u_total_w_per_k: Building heat loss coefficient H_tr [W/K].
            This is NOT a U-value! It's the total heat transfer coefficient:
            H_tr = Σ(U_i × A_i) summed over walls, roof, floor, windows.
            Used only for heat pump SIZING. If building_model is attached,
            the 5R1C model calculates actual heat demand dynamically.
        hp_model: hplib model name or 'Generic' for average performance
        group_id: Heat pump type (1=air/water regulated, 4=air/water on-off, etc.)
        design_temp_c: Outdoor design temperature for aux heater sizing [°C]
        supply_temp_c: Heating supply water temperature [°C] (affects COP)
        thermal_capacity_w: Desired thermal output at design conditions [W]
            If None, auto-sized from building heat loss
        building_model: Optional Building5R1C for dynamic heat demand calculation.
            When attached, this model calculates actual H_tr from individual
            component U-values and areas, replacing the u_total_w_per_k estimate.
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
            raise ImportError(
                "hplib required for HeatPumpHPLib. Install with: pip install hplib"
            )
        
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
            self._p_th_ref = float(self._hp_params['P_th_h_ref [W]'].values[0])
        except Exception as e:
            warnings.warn(f"Failed to load hplib model '{self.hp_model}': {e}. Using default.")
            # Fall back to generic average
            self._hp_params = hpl.get_parameters(
                model="Generic",
                group_id=1,
                t_in=-7,
                t_out=52,
                p_th=int(target_capacity),
            )
            self._p_th_ref = float(self._hp_params['P_th_h_ref [W]'].values[0])
        
        # Size auxiliary heater for deficit at design temperature
        # Simulate at design conditions to find HP output
        try:
            result = hpl.simulate(
                t_in_primary=self.design_temp_c,
                t_in_secondary=self.supply_temp_c - 5,  # Return temp (5K delta)
                parameters=self._hp_params,
                t_amb=self.design_temp_c,
            )
            hp_output_at_design = float(result['P_th'].values[0])
        except:
            # Rough estimate if simulation fails
            hp_output_at_design = self._p_th_ref * 0.5
        
        # Aux heater covers remaining load with 10% margin
        aux_needed = max(0, design_heat_load - hp_output_at_design)
        self.aux_heater_max_w = max(aux_needed * 1.1, 2000.0)
    
    def simulate_hplib(
        self, 
        outdoor_temp_c: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate heat pump using hplib.
        
        Returns:
            Tuple of (COP, P_el [W], P_th [W])
        """
        outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)
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
        
        cop = extract_array('COP')
        p_el = extract_array('P_el')
        p_th = extract_array('P_th')
        
        # Ensure correct length
        if len(cop) != n:
            cop = np.full(n, cop[0] if len(cop) > 0 else 2.5)
        if len(p_el) != n:
            p_el = np.full(n, p_el[0] if len(p_el) > 0 else 1000.0)
        if len(p_th) != n:
            p_th = np.full(n, p_th[0] if len(p_th) > 0 else 3000.0)
        
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
        
        Args:
            outdoor_temp_c: Outdoor temperature series [°C]
            solar_gains: Solar heat gains through windows [W] (optional)
            internal_gains: Internal gains from occupants/appliances [W] (optional)
            minutes_per_slot: Time resolution for 5R1C model
            
        Returns:
            Heat demand [W] array
        """
        outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)
        n = len(outdoor_temp_c)
        
        if self.building_model is not None:
            # Use ISO 13790 5R1C dynamic model
            dt_hours = minutes_per_slot / 60.0
            
            # Default solar/internal gains if not provided
            if solar_gains is None:
                solar_gains = np.zeros(n)
            if internal_gains is None:
                internal_gains = np.full(
                    n, 
                    self.building_model.internal_gains_w_per_m2 * self.building_model.floor_area_m2
                )
            
            # Reset building model to initial state
            self.building_model.reset()
            
            # Calculate heat demand for each timestep
            q_demand = np.zeros(n)
            for i in range(n):
                q_demand[i] = self.building_model.calc_heating_demand(
                    t_out=outdoor_temp_c[i],
                    solar_gains=solar_gains[i] if solar_gains is not None else 0.0,
                    internal_gains=internal_gains[i] if internal_gains is not None else None,
                    dt_hours=dt_hours,
                )
            
            return np.maximum(0.0, q_demand)
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
        
        Args:
            outdoor_temp_c: Outdoor temperature series [°C]
            rng: Random generator (unused, for API compatibility)
            minutes_per_slot: Time resolution
            solar_gains: Solar heat gains [W] (for 5R1C model)
            internal_gains: Internal gains [W] (for 5R1C model)
            
        Returns:
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
                filtered[i] = alpha * total[i] + (1 - alpha) * filtered[i-1]
            total = filtered
        
        return np.maximum(0.0, total)


# =============================================================================
# Simplified Carnot Model (Fallback when hplib not available)
# =============================================================================

@dataclass
class HeatPumpCarnot(HeatingSystem):
    """Simplified heat pump model using Carnot-based COP approximation.
    
    WARNING: This is a theoretical simplification for use when hplib is not
    available. For academic/research use, prefer HeatPumpHPLib.
    
    COP Model:
        COP = η_carnot × T_sink / (T_sink - T_source)
        
    Known limitations:
        - No defrost cycle penalty (overestimates COP below -5°C)
        - No part-load efficiency (inverter HPs vary with capacity)
        - Linear approximation vs. empirical non-linear behavior
        - Estimated MAPE: 15-25% (unvalidated)
    
    Args:
        indoor_temp_c: Target indoor temperature [°C]
        u_total_w_per_k: Building heat loss coefficient H_tr [W/K].
            This is NOT a U-value! H_tr = Σ(U_i × A_i) summed over envelope.
            Used for heat pump SIZING. If building_model attached, 5R1C
            calculates actual demand from individual U-values and areas.
        supply_temp_c: Heating supply water temperature [°C]
        bivalent_temp_c: Temperature where HP reaches max capacity [°C]
        design_temp_c: Outdoor design temperature [°C]
        carnot_efficiency: Fraction of Carnot COP (0.25-0.50)
        hp_coverage_at_bivalent: Fraction of load HP covers at bivalent point
        building_model: Optional Building5R1C for dynamic heat demand
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
        n = len(outdoor_temp_c)
        
        if self.building_model is not None:
            # Use ISO 13790 5R1C dynamic model
            dt_hours = minutes_per_slot / 60.0
            
            if solar_gains is None:
                solar_gains = np.zeros(n)
            if internal_gains is None:
                internal_gains = np.full(
                    n,
                    self.building_model.internal_gains_w_per_m2 * self.building_model.floor_area_m2
                )
            
            self.building_model.reset()
            
            q_demand = np.zeros(n)
            for i in range(n):
                q_demand[i] = self.building_model.calc_heating_demand(
                    t_out=outdoor_temp_c[i],
                    solar_gains=solar_gains[i] if solar_gains is not None else 0.0,
                    internal_gains=internal_gains[i] if internal_gains is not None else None,
                    dt_hours=dt_hours,
                )
            
            return np.maximum(0.0, q_demand)
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
                filtered[i] = alpha * total[i] + (1 - alpha) * filtered[i-1]
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
    **kwargs,
) -> HeatingSystem:
    """Factory function to create the best available heat pump model.
    
    Automatically selects:
        - HeatPumpHPLib (if hplib installed) - empirically validated
        - HeatPumpCarnot (fallback) - simplified theoretical model
    
    Args:
        indoor_temp_c: Target indoor temperature [°C]
        u_total_w_per_k: Building heat loss coefficient H_tr [W/K].
            This is NOT a U-value! It's the total heat transfer coefficient:
                H_tr = Σ(U_i × A_i) for walls, roof, floor, windows + ventilation
            Used only for heat pump SIZING (compressor and aux heater capacity).
            If building_model is attached, the 5R1C model calculates actual
            heat demand from individual component U-values and areas.
        supply_temp_c: Heating supply water temperature [°C]
        bivalent_temp_c: Temperature where HP reaches max capacity [°C]
        design_temp_c: Outdoor design temperature [°C]
        carnot_efficiency: Carnot efficiency (only used in fallback model)
        hp_coverage_at_bivalent: Load coverage at bivalent (only used in fallback)
        hp_model: hplib model name ('Generic' or specific manufacturer model)
        group_id: hplib group ID (1=air/water regulated, 4=air/water on-off)
        thermal_time_constant_hours: Building thermal mass time constant
        building_model: Optional Building5R1C for dynamic heat demand calculation.
            When provided, the 5R1C model calculates heat demand using proper
            physics with individual U-values, thermal mass, solar gains, etc.
        
    Returns:
        HeatingSystem instance (HeatPumpHPLib or HeatPumpCarnot)
    """
    if HPLIB_AVAILABLE:
        # Calculate thermal capacity from building characteristics
        thermal_capacity = u_total_w_per_k * (indoor_temp_c - bivalent_temp_c) * hp_coverage_at_bivalent
        
        return HeatPumpHPLib(
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
        return HeatPumpCarnot(
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


# =============================================================================
# Other Heating Systems (unchanged)
# =============================================================================

@dataclass
class DistrictHeating(HeatingSystem):
    """District heating (fjärrvärme) - zero electrical load to resident.
    
    Swedish district heating delivers hot water from CHP plants through pipes.
    From the household electricity meter's perspective, this is invisible:
    - The heat exchanger requires no electricity
    - Circulation pumps are on building common areas or district network
    - The resident sees ZERO heating-related electrical load
    
    However, we still calculate thermal demand using the 5R1C building model
    to enable analysis of heat demand profiles for district heating networks.
    
    Args:
        indoor_temp_c: Target indoor temperature [°C]
        building_model: ISO 13790 5R1C building model for thermal demand
        circulation_pump_w: Small circulation pump if on resident meter [W]
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
        n = len(outdoor_temp_c)
        
        if self.building_model is not None:
            dt_hours = minutes_per_slot / 60.0
            
            if solar_gains is None:
                solar_gains = np.zeros(n)
            if internal_gains is None:
                internal_gains = np.full(
                    n,
                    self.building_model.internal_gains_w_per_m2 * self.building_model.floor_area_m2
                )
            
            self.building_model.reset()
            
            q_demand = np.zeros(n)
            for i in range(n):
                q_demand[i] = self.building_model.calc_heating_demand(
                    t_out=outdoor_temp_c[i],
                    solar_gains=solar_gains[i] if solar_gains is not None else 0.0,
                    internal_gains=internal_gains[i] if internal_gains is not None else None,
                    dt_hours=dt_hours,
                )
            
            self._last_thermal_demand_w = np.maximum(0.0, q_demand)
            return self._last_thermal_demand_w
        else:
            # No building model - return zeros (unknown demand)
            self._last_thermal_demand_w = np.zeros(n)
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


@dataclass
class DirectElectric(HeatingSystem):
    """Direct resistive electric heating (COP=1).
    
    Args:
        indoor_temp_c: Target indoor temperature [°C]
        u_total_w_per_k: Building heat loss coefficient H_tr [W/K].
            NOT a U-value! H_tr = Σ(U_i × A_i) over envelope components.
            Used only if building_model is not provided.
        building_model: ISO 13790 5R1C building model for thermal demand
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
        n = len(outdoor_temp_c)
        
        if self.building_model is not None:
            dt_hours = minutes_per_slot / 60.0
            
            if solar_gains is None:
                solar_gains = np.zeros(n)
            if internal_gains is None:
                internal_gains = np.full(
                    n,
                    self.building_model.internal_gains_w_per_m2 * self.building_model.floor_area_m2
                )
            
            self.building_model.reset()
            
            q_demand = np.zeros(n)
            for i in range(n):
                q_demand[i] = self.building_model.calc_heating_demand(
                    t_out=outdoor_temp_c[i],
                    solar_gains=solar_gains[i] if solar_gains is not None else 0.0,
                    internal_gains=internal_gains[i] if internal_gains is not None else None,
                    dt_hours=dt_hours,
                )
            
            return np.maximum(0.0, q_demand)
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


@dataclass
class WoodPelletHybrid(HeatingSystem):
    """Swedish wood/pellet boiler with electric backup.
    
    Modern pellet boilers or wood boilers are common in rural Sweden.
    Electric backup covers ~5-10% of annual heat demand.
    
    Args:
        indoor_temp_c: Target indoor temperature [°C]
        u_total_w_per_k: Building heat loss coefficient H_tr [W/K].
            NOT a U-value! H_tr = Σ(U_i × A_i) over envelope components.
            Used only if building_model is not provided.
        wood_coverage_fraction: Fraction of heat demand covered by wood/pellet
        electric_backup_max_w: Maximum electric backup heater power [W]
        building_model: ISO 13790 5R1C building model for thermal demand
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
        n = len(outdoor_temp_c)
        
        if self.building_model is not None:
            dt_hours = minutes_per_slot / 60.0
            
            if solar_gains is None:
                solar_gains = np.zeros(n)
            if internal_gains is None:
                internal_gains = np.full(
                    n,
                    self.building_model.internal_gains_w_per_m2 * self.building_model.floor_area_m2
                )
            
            self.building_model.reset()
            
            q_demand = np.zeros(n)
            for i in range(n):
                q_demand[i] = self.building_model.calc_heating_demand(
                    t_out=outdoor_temp_c[i],
                    solar_gains=solar_gains[i] if solar_gains is not None else 0.0,
                    internal_gains=internal_gains[i] if internal_gains is not None else None,
                    dt_hours=dt_hours,
                )
            
            return np.maximum(0.0, q_demand)
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
            q_total[heating_season] * electric_fraction,
            self.electric_backup_max_w
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
    
    Args:
        hp_type: Heat source type ('air', 'brine', or 'water')
        
    Returns:
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
    
    filtered = db[db['Type'] == type_filter.get(hp_type, "Outdoor Air/Water")]
    # Combine Manufacturer and Model for a unique identifier
    names = (filtered['Manufacturer'].astype(str).str.strip() + ' ' + 
             filtered['Model'].astype(str).str.strip())
    return names.unique().tolist()


def get_hp_info(model_name: str) -> dict:
    """Get information about a specific heat pump model.
    
    Args:
        model_name: Name of the heat pump model (format: 'Manufacturer Model')
        
    Returns:
        Dictionary with model specifications
    """
    if not HPLIB_AVAILABLE:
        return {"error": "hplib not installed"}
    
    db = hpl.load_database()
    
    # Create combined name column for matching
    db_names = (db['Manufacturer'].astype(str).str.strip() + ' ' + 
                db['Model'].astype(str).str.strip())
    match = db[db_names == model_name]
    
    if len(match) == 0:
        # Try partial match
        mask = db_names.str.contains(model_name, case=False, na=False)
        match = db[mask]
    
    if len(match) == 0:
        return {"error": f"Model '{model_name}' not found"}
    
    row = match.iloc[0]
    return {
        "manufacturer": str(row.get('Manufacturer', 'Unknown')).strip(),
        "model": str(row.get('Model', 'Unknown')).strip(),
        "type": row.get('Type', 'Unknown'),
        "subtype": row.get('Subtype', 'Unknown'),
        "group_id": int(row.get('Group', 1)),
        "p_th_ref_w": float(row.get('P_th_h_ref [W]', 0)),
        "p_el_ref_w": float(row.get('P_el_h_ref [W]', 0)),
        "cop_ref": float(row.get('COP_ref', 0)),
        "refrigerant": row.get('Refrigerant', 'Unknown'),
        "mape_cop": float(row.get('MAPE_COP', 0)),
        "mape_p_el": float(row.get('MAPE_P_el', 0)),
        "mape_p_th": float(row.get('MAPE_P_th', 0)),
    }
