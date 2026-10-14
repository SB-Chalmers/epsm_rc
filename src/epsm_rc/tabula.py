"""TABULA seasonal calculation method for building energy demand.

Implements the quasi-steady-state seasonal method from the TABULA/EPISCOPE project
based on EN ISO 13790 monthly/seasonal approach. This method calculates annual
heating energy demand using heating degree days and a gain utilization factor.

Key differences from the hourly 5R1C method:
    - Calculates annual totals, not hourly profiles
    - Uses Gain Utilization Factor (η) to account for thermal mass
    - Includes empirical calibration ("Adaptation Method") option
    - Matches TABULA webtool reference values

The TABULA method is well-suited for:
    - Building stock modeling and energy certificates
    - Policy analysis and building typology comparisons
    - Validation against measured consumption data
    - Quick annual energy estimates

References:
    - TABULA Common Calculation Method (IEE TABULA Project)
    - EN ISO 13790:2008 Seasonal calculation method
    - Swedish TABULA: "Byggnadstypologier Sverige"
    - IEE Project TABULA (2009-2012), EPISCOPE (2013-2016)

License: MIT

Author: SweLoadSim
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
import math

import numpy as np


class ThermalMassClass(Enum):
    """TABULA thermal mass classes for gain utilization calculation.
    
    Determines the building time constant parameter (a0) and affects
    how well the building can utilize internal and solar gains.
    """
    VERY_LIGHT = 1   # Lightweight timber frame
    LIGHT = 2        # Mixed light construction
    MEDIUM = 3       # Standard construction
    HEAVY = 4        # Masonry, concrete
    VERY_HEAVY = 5   # Heavy concrete, thick masonry


# Thermal capacitance per floor area [Wh/(m²·K)] by mass class
# Based on EN ISO 13790 Table 12, converted to Wh
THERMAL_CAPACITANCE_WH = {
    ThermalMassClass.VERY_LIGHT: 22.2,    # 80 kJ/(m²·K) = 22.2 Wh/(m²·K)
    ThermalMassClass.LIGHT: 30.6,         # 110 kJ/(m²·K)
    ThermalMassClass.MEDIUM: 45.8,        # 165 kJ/(m²·K)
    ThermalMassClass.HEAVY: 72.2,         # 260 kJ/(m²·K)
    ThermalMassClass.VERY_HEAVY: 102.8,   # 370 kJ/(m²·K)
}


@dataclass
class ClimateData:
    """Climate data for TABULA seasonal calculation.
    
    TABULA uses simplified climate parameters:
    - Heating days: number of days when heating is needed
    - Average outdoor temperature during heating season
    - Solar irradiation by orientation (annual totals)
    - Degree-days for heat loss calculations
    
    Swedish climate zones from TABULA webtool:
        Zone 1 (North): Luleå area, θe=-1.2°C, 7738 degree-days
        Zone 2 (Central): Östersund area  
        Zone 3 (South): Göteborg area
        
    Note: Parameters are from TABULA webtool API (2025) to ensure
    exact match with published reference values.
    """
    name: str = "SE.Z3"
    heating_days: float = 261.0  # lH [days]
    mean_external_temp: float = 3.5  # θe [°C] during heating season
    
    # Solar irradiation [kWh/(m²·year)] - annual totals
    # By orientation (horizontal, N, E, S, W)
    solar_horizontal: float = 350.0
    solar_north: float = 180.0
    solar_east: float = 280.0
    solar_south: float = 450.0
    solar_west: float = 280.0
    
    @classmethod
    def sweden_zone1(cls) -> "ClimateData":
        """Northern Sweden (Norrland) - coldest zone.
        
        Parameters from TABULA webtool API for Zone 1:
        - θe = -1.2°C (mean external temp during heating season)
        - Heating days = 365
        - Degree-days = (20 - (-1.2)) × 365 = 7738 K·days
        - Solar irradiation from webtool (annual totals)
        """
        return cls(
            name="SE.Z1",
            heating_days=365.0,         # TABULA webtool value
            mean_external_temp=-1.2,    # θe from webtool
            solar_horizontal=711.0,     # I_sol_hor from webtool
            solar_north=309.0,          # I_sol_north
            solar_east=628.0,           # I_sol_east
            solar_south=725.0,          # I_sol_south
            solar_west=628.0,           # I_sol_west
        )
    
    @classmethod
    def sweden_zone2(cls) -> "ClimateData":
        """Central Sweden (Svealand/Jämtland) - moderate zone.
        
        Parameters from TABULA Common Calculation Method document (Table 7):
        - θe = 0.3°C (mean external temp during heating season)
        - Heating days = 286
        - Degree-days = (20 - 0.3) × 286 = 5636 K·days
        - Solar irradiation from TABULA documentation
        
        Source: TABULA_CommonCalculationMethod.pdf, page 35, SE.Zone2
        """
        return cls(
            name="SE.Z2",
            heating_days=286.0,         # TABULA official value
            mean_external_temp=0.3,     # θe from TABULA doc
            solar_horizontal=492.0,     # I_H from TABULA doc
            solar_north=161.0,          # I_N from TABULA doc
            solar_east=372.0,           # I_E from TABULA doc
            solar_south=524.0,          # I_S from TABULA doc
            solar_west=372.0,           # I_W from TABULA doc
        )
    
    @classmethod
    def sweden_zone3(cls) -> "ClimateData":
        """Southern Sweden (Götaland) - mildest zone.
        
        Parameters from TABULA Common Calculation Method document (Table 7):
        - θe = 2.2°C (mean external temp during heating season)
        - Heating days = 261
        - Degree-days = (20 - 2.2) × 261 = 4643 K·days
        - Solar irradiation from TABULA documentation
        
        Source: TABULA_CommonCalculationMethod.pdf, page 35, SE.Zone3
        """
        return cls(
            name="SE.Z3",
            heating_days=261.0,         # TABULA official value
            mean_external_temp=2.2,     # θe from TABULA doc
            solar_horizontal=450.0,     # I_H from TABULA doc
            solar_north=138.0,          # I_N from TABULA doc
            solar_east=328.0,           # I_E from TABULA doc
            solar_south=496.0,          # I_S from TABULA doc
            solar_west=328.0,           # I_W from TABULA doc
        )


@dataclass 
class BuildingTABULA:
    """TABULA seasonal calculation model for building heating demand.
    
    Implements the quasi-steady-state method from EN ISO 13790 as used
    in the TABULA/EPISCOPE building typology project.
    
    The annual heating demand is calculated as:
        Q_H,nd = Q_H,tr + Q_H,ve - η_H × (Q_int + Q_sol)
    
    Where:
        Q_H,tr: Transmission heat losses [kWh/year]
        Q_H,ve: Ventilation heat losses [kWh/year]
        η_H: Gain utilization factor (0-1)
        Q_int: Internal heat gains [kWh/year]
        Q_sol: Solar heat gains [kWh/year]
    
    Args:
        floor_area_m2: Heated floor area [m²]
        thermal_mass_class: Building thermal mass class
        
        # U-values [W/(m²·K)]
        u_wall: Exterior wall U-value
        u_roof: Roof U-value  
        u_floor: Ground floor U-value
        u_window: Window U-value
        u_door: Door U-value
        
        # Areas [m²]
        wall_area_m2: Exterior wall area
        roof_area_m2: Roof area
        floor_area_ground_m2: Ground floor area
        window_area_m2: Window area (total)
        door_area_m2: Door area
        
        # Window distribution (fractions summing to 1.0)
        window_north_fraction: Fraction of windows facing north
        window_east_fraction: Fraction facing east
        window_south_fraction: Fraction facing south
        window_west_fraction: Fraction facing west
        
        # Thermal bridges
        delta_u_thermal_bridge: Thermal bridge surcharge [W/(m²·K)]
        
        # Ventilation
        ach_use: Air changes for use/occupancy [1/h]
        ach_infiltration: Air changes from infiltration [1/h]
        heat_recovery_efficiency: Heat recovery efficiency (0-1)
        
        # Gains
        internal_gains_w_per_m2: Internal heat sources [W/m²]
        g_value: Solar heat gain coefficient of windows
        frame_factor: Glass fraction of window (1 - frame)
        shading_factor: External shading reduction factor
        
        # Setpoints and corrections
        t_set_heating: Indoor setpoint temperature [°C]
        non_uniform_heating_factor: Correction for setbacks, unheated spaces
        temp_adj_ground: Ground temperature adjustment factor (btr)
        
    Example:
        >>> building = BuildingTABULA(
        ...     floor_area_m2=125,
        ...     u_wall=0.60, u_roof=0.40, u_floor=0.40, u_window=2.80,
        ...     wall_area_m2=150, roof_area_m2=100, floor_area_ground_m2=100,
        ...     window_area_m2=25,
        ... )
        >>> climate = ClimateData.sweden_zone3()
        >>> q_annual = building.calc_annual_heating_demand(climate)
        >>> print(f"Annual heating: {q_annual:.1f} kWh/m²/year")
    """
    
    # Building geometry
    floor_area_m2: float = 125.0
    thermal_mass_class: ThermalMassClass = ThermalMassClass.MEDIUM
    
    # U-values [W/(m²·K)]
    u_wall: float = 0.60
    u_roof: float = 0.40
    u_floor: float = 0.40
    u_window: float = 2.80
    u_door: float = 2.00
    
    # Areas [m²]
    wall_area_m2: float = 150.0
    roof_area_m2: float = 100.0
    floor_area_ground_m2: float = 100.0
    window_area_m2: float = 25.0
    door_area_m2: float = 4.0
    
    # Optional: basement wall for MFH
    basement_wall_area_m2: float = 0.0
    u_basement_wall: float = 0.0
    
    # Window orientation distribution (fractions)
    window_north_fraction: float = 0.25
    window_east_fraction: float = 0.25
    window_south_fraction: float = 0.25
    window_west_fraction: float = 0.25
    
    # Building volume
    ceiling_height_m: float = 2.5
    
    # Thermal bridges [W/(m²·K)]
    # TABULA default: 0.15 for SFH, 0.10 for MFH
    delta_u_thermal_bridge: float = 0.15
    
    # Ventilation [1/h]
    ach_use: float = 0.40      # Ventilation for use/occupancy
    ach_infiltration: float = 0.40  # Infiltration through envelope
    heat_recovery_efficiency: float = 0.0
    
    # Internal gains
    internal_gains_w_per_m2: float = 3.0  # TABULA standard: 3.0 W/m²
    
    # Solar properties
    g_value: float = 0.60  # Solar heat gain coefficient
    frame_factor: float = 0.70  # Glass fraction (FF = 0.7 in TABULA)
    shading_factor: float = 0.90  # External shading factor (FS)
    
    # Setpoints and corrections
    t_set_heating: float = 20.0  # θi [°C] - TABULA uses 20°C
    non_uniform_heating_factor: float = 1.0  # fH - correction for setbacks
    temp_adj_ground: float = 0.50  # btr,g for ground floor
    temp_adj_basement: float = 0.50  # btr for basement walls
    
    # Derived properties
    _h_tr: float = field(default=0.0, init=False, repr=False)
    _h_ve: float = field(default=0.0, init=False, repr=False)
    _volume: float = field(default=0.0, init=False, repr=False)
    _total_envelope_area: float = field(default=0.0, init=False, repr=False)
    
    def __post_init__(self):
        """Calculate derived thermal properties."""
        self._calculate_thermal_properties()
    
    def _calculate_thermal_properties(self):
        """Calculate transmission and ventilation heat transfer coefficients."""
        # Building volume [m³]
        self._volume = self.floor_area_m2 * self.ceiling_height_m
        
        # Total envelope area for thermal bridge calculation
        self._total_envelope_area = (
            self.wall_area_m2 + 
            self.roof_area_m2 + 
            self.floor_area_ground_m2 + 
            self.window_area_m2 +
            self.door_area_m2 +
            self.basement_wall_area_m2
        )
        
        # Transmission heat transfer coefficient H_tr [W/K]
        # H_tr = Σ(U × A × b) + ΔU_tb × A_env
        
        # Opaque elements
        h_wall = self.u_wall * self.wall_area_m2  # btr = 1.0 for exterior walls
        h_roof = self.u_roof * self.roof_area_m2  # btr = 1.0 for roof
        h_floor = self.u_floor * self.floor_area_ground_m2 * self.temp_adj_ground
        h_door = self.u_door * self.door_area_m2
        
        # Basement walls (if present)
        h_basement = (
            self.u_basement_wall * self.basement_wall_area_m2 * self.temp_adj_basement
        )
        
        # Windows (btr = 1.0)
        h_window = self.u_window * self.window_area_m2
        
        # Thermal bridges
        h_tb = self.delta_u_thermal_bridge * self._total_envelope_area
        
        self._h_tr = h_wall + h_roof + h_floor + h_door + h_basement + h_window + h_tb
        
        # Ventilation heat transfer coefficient H_ve [W/K]
        # H_ve = ρ·cp·V̇ = 0.34 × n × V (using ρ·cp ≈ 1200 J/(m³·K) → 0.34 Wh/(m³·K))
        ach_total = self.ach_use + self.ach_infiltration
        
        # Heat recovery only applies to mechanical ventilation (ach_use)
        if ach_total > 0:
            ach_recovered = self.ach_use * self.heat_recovery_efficiency
            ach_effective = ach_total - ach_recovered
        else:
            ach_effective = 0
        
        self._h_ve = 0.34 * ach_effective * self._volume
    
    @property
    def h_tr(self) -> float:
        """Transmission heat transfer coefficient [W/K]."""
        return self._h_tr
    
    @property
    def h_ve(self) -> float:
        """Ventilation heat transfer coefficient [W/K]."""
        return self._h_ve
    
    @property
    def h_total(self) -> float:
        """Total heat transfer coefficient [W/K]."""
        return self._h_tr + self._h_ve
    
    @property
    def specific_heat_loss(self) -> float:
        """Specific heat loss per floor area [W/(m²·K)]."""
        return self.h_total / self.floor_area_m2
    
    @property
    def time_constant(self) -> float:
        """Building time constant τ [hours].
        
        τ = C_m / H_total
        
        Used to calculate the gain utilization factor parameter 'a'.
        """
        c_m = THERMAL_CAPACITANCE_WH[self.thermal_mass_class] * self.floor_area_m2
        if self.h_total > 0:
            return c_m / self.h_total
        return 0.0
    
    def calc_transmission_losses(self, climate: ClimateData) -> float:
        """Calculate annual transmission heat losses [kWh/year].
        
        Q_H,tr = H_tr × (θi - θe) × lH × 24 / 1000
        
        Args:
            climate: Climate data with heating days and temperatures
            
        Returns:
            Annual transmission losses [kWh/year]
        """
        delta_t = self.t_set_heating - climate.mean_external_temp
        # Convert W to kWh: W × hours / 1000
        q_tr = self._h_tr * delta_t * climate.heating_days * 24 / 1000
        return max(0, q_tr)
    
    def calc_ventilation_losses(self, climate: ClimateData) -> float:
        """Calculate annual ventilation heat losses [kWh/year].
        
        Q_H,ve = H_ve × (θi - θe) × lH × 24 / 1000
        
        Args:
            climate: Climate data with heating days and temperatures
            
        Returns:
            Annual ventilation losses [kWh/year]
        """
        delta_t = self.t_set_heating - climate.mean_external_temp
        q_ve = self._h_ve * delta_t * climate.heating_days * 24 / 1000
        return max(0, q_ve)
    
    def calc_internal_gains(self, climate: ClimateData) -> float:
        """Calculate annual internal heat gains [kWh/year].
        
        Q_int = φi × A_f × lH × 24 / 1000
        
        Args:
            climate: Climate data with heating days
            
        Returns:
            Annual internal gains [kWh/year]
        """
        # W/m² × m² × days × hours/day / 1000 = kWh
        q_int = (
            self.internal_gains_w_per_m2 * 
            self.floor_area_m2 * 
            climate.heating_days * 24 / 1000
        )
        return q_int
    
    def calc_solar_gains(self, climate: ClimateData) -> float:
        """Calculate annual solar heat gains through windows [kWh/year].
        
        Q_sol = Σ(I_sol,k × A_sol,k) during heating season
        
        Where A_sol = A_window × FF × FS × g
        
        Args:
            climate: Climate data with solar irradiation by orientation
            
        Returns:
            Annual solar gains [kWh/year]
        """
        # Effective solar collecting area per orientation
        a_sol_factor = self.window_area_m2 * self.frame_factor * self.shading_factor * self.g_value
        
        # Solar gains by orientation [kWh/year]
        q_north = climate.solar_north * self.window_north_fraction * a_sol_factor
        q_east = climate.solar_east * self.window_east_fraction * a_sol_factor
        q_south = climate.solar_south * self.window_south_fraction * a_sol_factor
        q_west = climate.solar_west * self.window_west_fraction * a_sol_factor
        
        q_sol = q_north + q_east + q_south + q_west
        return q_sol
    
    def calc_gain_utilization_factor(
        self, 
        total_losses: float, 
        total_gains: float,
    ) -> float:
        """Calculate gain utilization factor η_H.
        
        The gain utilization factor accounts for the fact that not all gains
        can be used to reduce heating demand (some cause overheating).
        
        For heating mode (EN ISO 13790 Eq. 63):
            if γ_H ≠ 1: η_H = (1 - γ_H^a) / (1 - γ_H^(a+1))
            if γ_H = 1: η_H = a / (a + 1)
        
        Where:
            γ_H = Q_gains / Q_losses (gain-to-loss ratio)
            a = a0 + τ/τ0 (time constant parameter)
            a0 = 0.8 (TABULA seasonal method constant)
            τ0 = 30 hours (TABULA seasonal method constant)
        
        Note: Constants a0=0.8 and τ0=30h are from TABULA Common Calculation
        Method document (page 15-16) for the seasonal method. EN ISO 13790
        uses a0=1.0 and τ0=15h for the monthly method.
        
        Args:
            total_losses: Total heat losses [kWh]
            total_gains: Total heat gains [kWh]
            
        Returns:
            Gain utilization factor η_H (0-1)
        """
        if total_losses <= 0:
            return 0.0
        
        # Gain-to-loss ratio
        gamma_h = total_gains / total_losses
        
        # Time constant parameter (TABULA seasonal method constants)
        # a = a0 + τ/τ0, with a0=0.8 and τ0=30h per TABULA doc page 15-16
        a0 = 0.8
        tau0 = 30.0  # hours
        tau = self.time_constant
        a = a0 + tau / tau0
        
        # Calculate utilization factor (EN ISO 13790 Eq. 63)
        if abs(gamma_h - 1.0) < 1e-6:
            # Special case when γ ≈ 1
            eta_h = a / (a + 1)
        elif gamma_h <= 0:
            eta_h = 1.0
        elif gamma_h > 0:
            eta_h = (1 - gamma_h**a) / (1 - gamma_h**(a + 1))
        else:
            eta_h = 1.0
        
        # Clamp to valid range
        return max(0.0, min(1.0, eta_h))
    
    def calc_annual_heating_demand(
        self, 
        climate: ClimateData,
        include_gains: bool = True,
    ) -> float:
        """Calculate annual heating energy demand [kWh/year].
        
        Q_H,nd = (Q_H,tr + Q_H,ve) × fH - η_H × (Q_int + Q_sol)
        
        Args:
            climate: Climate data for location
            include_gains: If True, subtract utilized gains (default)
                          If False, return gross losses only (for comparison)
            
        Returns:
            Annual heating demand [kWh/year]
        """
        # Heat losses
        q_tr = self.calc_transmission_losses(climate)
        q_ve = self.calc_ventilation_losses(climate)
        q_losses = q_tr + q_ve
        
        # Apply non-uniform heating factor
        q_losses_adj = q_losses * self.non_uniform_heating_factor
        
        if not include_gains:
            return q_losses_adj
        
        # Heat gains
        q_int = self.calc_internal_gains(climate)
        q_sol = self.calc_solar_gains(climate)
        q_gains = q_int + q_sol
        
        # Gain utilization factor
        eta_h = self.calc_gain_utilization_factor(q_losses, q_gains)
        
        # Net heating demand
        q_h_nd = q_losses_adj - eta_h * q_gains
        
        return max(0, q_h_nd)
    
    def calc_annual_heating_demand_per_m2(
        self, 
        climate: ClimateData,
        include_gains: bool = True,
    ) -> float:
        """Calculate specific annual heating demand [kWh/(m²·year)].
        
        Args:
            climate: Climate data for location
            include_gains: If True, subtract utilized gains
            
        Returns:
            Specific annual heating demand [kWh/(m²·year)]
        """
        q_total = self.calc_annual_heating_demand(climate, include_gains)
        return q_total / self.floor_area_m2
    
    def calc_energy_breakdown(self, climate: ClimateData) -> dict:
        """Calculate detailed energy balance breakdown.
        
        Returns all components of the heating demand calculation.
        
        Args:
            climate: Climate data for location
            
        Returns:
            Dictionary with:
                - q_tr: Transmission losses [kWh/year]
                - q_ve: Ventilation losses [kWh/year]
                - q_losses: Total losses [kWh/year]
                - q_int: Internal gains [kWh/year]
                - q_sol: Solar gains [kWh/year]
                - q_gains: Total gains [kWh/year]
                - gamma_h: Gain-to-loss ratio
                - eta_h: Gain utilization factor
                - q_utilized: Utilized gains [kWh/year]
                - q_h_nd: Net heating demand [kWh/year]
                - q_h_nd_per_m2: Specific demand [kWh/(m²·year)]
        """
        q_tr = self.calc_transmission_losses(climate)
        q_ve = self.calc_ventilation_losses(climate)
        q_losses = q_tr + q_ve
        
        q_int = self.calc_internal_gains(climate)
        q_sol = self.calc_solar_gains(climate)
        q_gains = q_int + q_sol
        
        gamma_h = q_gains / q_losses if q_losses > 0 else 0
        eta_h = self.calc_gain_utilization_factor(q_losses, q_gains)
        q_utilized = eta_h * q_gains
        
        q_h_nd = max(0, q_losses * self.non_uniform_heating_factor - q_utilized)
        
        return {
            "q_tr": q_tr,
            "q_ve": q_ve,
            "q_losses": q_losses,
            "q_int": q_int,
            "q_sol": q_sol,
            "q_gains": q_gains,
            "gamma_h": gamma_h,
            "eta_h": eta_h,
            "q_utilized": q_utilized,
            "q_h_nd": q_h_nd,
            "q_h_nd_per_m2": q_h_nd / self.floor_area_m2,
            "h_tr": self._h_tr,
            "h_ve": self._h_ve,
            "time_constant": self.time_constant,
        }
    
    def apply_adaptation_factor(
        self, 
        q_calculated: float,
        e_del_measured: float,
        system_efficiency: float = 0.85,
    ) -> tuple[float, float]:
        """Apply TABULA adaptation method to calibrate model.
        
        The adaptation method calculates a correction factor to match
        calculated values to measured consumption data.
        
        f_cal = E_del,measured / (Q_calculated / η_sys)
        
        Args:
            q_calculated: Calculated heating demand [kWh/year]
            e_del_measured: Measured delivered energy [kWh/year]
            system_efficiency: Heating system efficiency (η_sys)
            
        Returns:
            Tuple of (adapted demand, calibration factor)
        """
        if q_calculated <= 0 or system_efficiency <= 0:
            return q_calculated, 1.0
        
        # Calculated delivered energy (before adaptation)
        e_del_calc = q_calculated / system_efficiency
        
        # Calibration factor
        f_cal = e_del_measured / e_del_calc if e_del_calc > 0 else 1.0
        
        # Adapted demand
        q_adapted = q_calculated * f_cal
        
        return q_adapted, f_cal


def create_tabula_sfh(
    floor_area_m2: float = 125.0,
    period: Literal["...1960", "1961-1975", "1976-1985", "1986-1995", "1996-2005"] = "...1960",
    **kwargs,
) -> BuildingTABULA:
    """Create a TABULA single-family house (SFH) archetype.
    
    Uses Swedish TABULA reference U-values and geometry by construction period.
    
    Args:
        floor_area_m2: Heated floor area [m²] (125 or 160 typical)
        period: Construction period
        **kwargs: Override any BuildingTABULA parameter
        
    Returns:
        Configured BuildingTABULA instance
    """
    # U-values by period from Swedish TABULA [W/(m²·K)]
    u_values = {
        "...1960": {"u_wall": 0.60, "u_roof": 0.40, "u_floor": 0.40, "u_window": 2.80, "u_door": 2.00},
        "1961-1975": {"u_wall": 0.35, "u_roof": 0.20, "u_floor": 0.25, "u_window": 2.70, "u_door": 2.00},
        "1976-1985": {"u_wall": 0.20, "u_roof": 0.14, "u_floor": 0.23, "u_window": 2.00, "u_door": 1.20},
        "1986-1995": {"u_wall": 0.15, "u_roof": 0.12, "u_floor": 0.20, "u_window": 1.80, "u_door": 1.20},
        "1996-2005": {"u_wall": 0.15, "u_roof": 0.10, "u_floor": 0.15, "u_window": 1.60, "u_door": 1.20},
    }
    
    # Thermal mass class by period
    mass_classes = {
        "...1960": ThermalMassClass.HEAVY,
        "1961-1975": ThermalMassClass.MEDIUM,
        "1976-1985": ThermalMassClass.MEDIUM,
        "1986-1995": ThermalMassClass.LIGHT,
        "1996-2005": ThermalMassClass.LIGHT,
    }
    
    # Ventilation by period
    ventilation = {
        "...1960": {"ach_use": 0.40, "ach_infiltration": 0.40, "heat_recovery_efficiency": 0.0},
        "1961-1975": {"ach_use": 0.40, "ach_infiltration": 0.35, "heat_recovery_efficiency": 0.0},
        "1976-1985": {"ach_use": 0.40, "ach_infiltration": 0.25, "heat_recovery_efficiency": 0.0},
        "1986-1995": {"ach_use": 0.40, "ach_infiltration": 0.20, "heat_recovery_efficiency": 0.0},
        "1996-2005": {"ach_use": 0.40, "ach_infiltration": 0.15, "heat_recovery_efficiency": 0.5},
    }
    
    # Estimate geometry from floor area
    # SFH: 1.5-story, roughly square footprint
    footprint = floor_area_m2 / 1.5
    side = math.sqrt(footprint)
    height = 2.5 * 1.5  # 1.5 stories
    perimeter = 4 * side
    
    params = {
        "floor_area_m2": floor_area_m2,
        "thermal_mass_class": mass_classes[period],
        "wall_area_m2": perimeter * height * 0.85,  # Minus windows/doors
        "roof_area_m2": footprint * 1.15,  # Pitched roof factor
        "floor_area_ground_m2": footprint,
        "window_area_m2": floor_area_m2 * 0.15,  # 15% window ratio
        "door_area_m2": 4.0,
        "delta_u_thermal_bridge": 0.15,  # SFH thermal bridge surcharge
        "window_south_fraction": 0.40,  # More windows facing south
        "window_north_fraction": 0.15,
        "window_east_fraction": 0.225,
        "window_west_fraction": 0.225,
        **u_values[period],
        **ventilation[period],
    }
    
    params.update(kwargs)
    return BuildingTABULA(**params)


def create_tabula_mfh(
    floor_area_m2: float = 1207.0,
    period: Literal["...1960", "1961-1975", "1976-1985", "1986-1995", "1996-2005"] = "...1960",
    **kwargs,
) -> BuildingTABULA:
    """Create a TABULA multi-family house (MFH) archetype.
    
    Uses Swedish TABULA reference U-values and geometry by construction period.
    Default floor area is 1207 m² (TABULA Swedish MFH reference).
    
    Args:
        floor_area_m2: Heated floor area [m²]
        period: Construction period
        **kwargs: Override any BuildingTABULA parameter
        
    Returns:
        Configured BuildingTABULA instance
    """
    # U-values by period from Swedish TABULA API [W/(m²·K)]
    u_values = {
        "...1960": {"u_wall": 0.60, "u_roof": 0.35, "u_floor": 0.47, "u_window": 2.80, "u_door": 2.00, "u_basement_wall": 1.39},
        "1961-1975": {"u_wall": 0.50, "u_roof": 0.30, "u_floor": 0.46, "u_window": 2.70, "u_door": 2.00, "u_basement_wall": 1.31},
        "1976-1985": {"u_wall": 0.30, "u_roof": 0.17, "u_floor": 0.32, "u_window": 2.00, "u_door": 1.20, "u_basement_wall": 0.57},
        "1986-1995": {"u_wall": 0.25, "u_roof": 0.15, "u_floor": 0.30, "u_window": 1.80, "u_door": 1.20, "u_basement_wall": 0.50},
        "1996-2005": {"u_wall": 0.20, "u_roof": 0.12, "u_floor": 0.25, "u_window": 1.60, "u_door": 1.20, "u_basement_wall": 0.40},
    }
    
    # MFH thermal mass (concrete construction)
    mass_classes = {
        "...1960": ThermalMassClass.VERY_HEAVY,
        "1961-1975": ThermalMassClass.VERY_HEAVY,
        "1976-1985": ThermalMassClass.HEAVY,
        "1986-1995": ThermalMassClass.HEAVY,
        "1996-2005": ThermalMassClass.HEAVY,
    }
    
    # Ventilation by period (MFH typically has exhaust ventilation)
    ventilation = {
        "...1960": {"ach_use": 0.40, "ach_infiltration": 0.20, "heat_recovery_efficiency": 0.0},
        "1961-1975": {"ach_use": 0.40, "ach_infiltration": 0.20, "heat_recovery_efficiency": 0.0},
        "1976-1985": {"ach_use": 0.40, "ach_infiltration": 0.15, "heat_recovery_efficiency": 0.0},
        "1986-1995": {"ach_use": 0.40, "ach_infiltration": 0.15, "heat_recovery_efficiency": 0.0},
        "1996-2005": {"ach_use": 0.40, "ach_infiltration": 0.10, "heat_recovery_efficiency": 0.5},
    }
    
    # MFH geometry: 3-story, rectangular footprint (2:1 aspect)
    n_floors = 3
    footprint = floor_area_m2 / n_floors
    width = math.sqrt(footprint / 2)
    length = 2 * width
    height = 2.5 * n_floors
    perimeter = 2 * (width + length)
    
    params = {
        "floor_area_m2": floor_area_m2,
        "thermal_mass_class": mass_classes[period],
        "wall_area_m2": perimeter * height * 0.80,  # Minus windows/doors
        "roof_area_m2": footprint,
        "floor_area_ground_m2": footprint,
        "window_area_m2": floor_area_m2 * 0.12,  # 12% window ratio for MFH
        "door_area_m2": 8.0,
        "basement_wall_area_m2": perimeter * 2.0,  # 2m basement height
        "delta_u_thermal_bridge": 0.10,  # MFH thermal bridge surcharge
        "temp_adj_ground": 0.50,
        "temp_adj_basement": 0.50,
        "window_south_fraction": 0.30,
        "window_north_fraction": 0.20,
        "window_east_fraction": 0.25,
        "window_west_fraction": 0.25,
        **u_values[period],
        **ventilation[period],
    }
    
    params.update(kwargs)
    return BuildingTABULA(**params)
