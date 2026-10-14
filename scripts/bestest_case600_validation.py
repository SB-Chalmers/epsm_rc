#!/usr/bin/env python3
"""
ASHRAE Standard 140-2020 BESTEST Case 600 Validation
=====================================================

Validates the 5R1C thermal model against ASHRAE Standard 140-2020
Building Thermal Envelope and Fabric Load Test (Section 5.2.1).

Case 600 Specifications (ASHRAE 140-2020 Table 5-1):
- Base Case (low mass)
- Location: Denver, CO (39.83°N, 104.65°W, 1650m elevation)
- Floor: 8m × 6m = 48 m²
- Volume: 129.6 m³ (height 2.7m)  
- Lightweight construction (low thermal mass)
- South-facing windows: 12 m² (two 3m × 2m windows)
- Wall U-value: 0.514 W/m²K (including inside film)
- Roof U-value: 0.318 W/m²K
- Floor: adiabatic (raised floor, not ground-coupled)
- Window U-value: 2.1 W/m²K (from WINDOW7, environmental condition 7)
- SHGC: 0.77 (from WINDOW7 glazing ID 23)
- Infiltration: 0.5 ACH
- Internal gains: 200 W constant (sensible only)
- Heating setpoint: 20°C
- Cooling setpoint: 27°C
- Deadband: thermostat not operating between 20°C and 27°C

Reference Results (ASHRAE 140-2017, Table B8-2):
- Annual heating: 4.296 to 5.709 MWh
- Annual cooling: 6.137 to 8.448 MWh
- Peak heating: 3.437 to 4.354 kW
- Peak cooling: 5.965 to 7.188 kW

Validation Results:
- With --modelica settings (matching LBNL Modelica Buildings Library):
  Heating: 4.424 MWh vs 4.658 MWh Modelica (-5.0%)
  Cooling: 5.596 MWh vs 5.803 MWh Modelica (-3.6%)
- With ISO 13790 defaults:
  Heating: 3.215 MWh (below reference range)
  Cooling: 5.979 MWh (below reference range)

Key Modelica parameters for accurate validation:
- h_is = 2.1 W/m²K (vs ISO default 3.45)
- Wall area = 63.6 m² (geometric sum vs ASHRAE's 55.6 m²)
- b = 1.0 (floor losses vs outdoor temp, not ground temp)
- U_windows = 3.1 W/m²K (vs ASHRAE's 2.1)
- Angle-dependent g-value polynomial: coeFac={1,-0.189,0.644,-0.596}

Author: SweLoadSim Validation Suite
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import pvlib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from epsm_rc.iso13790 import Building5R1C, BuildingClass


# ===========================================================================
# BESTEST Case 600 Building Parameters (ASHRAE 140-2020 Section 5.2.1)
# ===========================================================================

# Geometry
FLOOR_LENGTH = 8.0  # m (East-West)
FLOOR_WIDTH = 6.0   # m (North-South) 
FLOOR_AREA = FLOOR_LENGTH * FLOOR_WIDTH  # 48 m²
CEILING_HEIGHT = 2.7  # m
VOLUME = FLOOR_AREA * CEILING_HEIGHT  # 129.6 m³

# Window areas (all south-facing)
WINDOW_HEIGHT = 2.0  # m
WINDOW_WIDTH = 3.0   # m
N_WINDOWS = 2
WINDOW_AREA = N_WINDOWS * WINDOW_HEIGHT * WINDOW_WIDTH  # 12 m²

# Wall areas (excluding windows) - from ASHRAE 140 Table 5-2
# Note: BESTEST specifies 55.6 m² opaque wall area, which differs from
# the calculated gross area (63.6 m²). The difference may be due to
# thermal bridging corrections or area interpretation differences.
# Using the specified value for consistency with reference programs.
WALL_AREA_TOTAL = 55.6  # From ASHRAE 140 Table 5-2 "Sun-Zone Opaque Surface Area"

# Alternative calculation (not used):
# WALL_AREA_NORTH = FLOOR_LENGTH * CEILING_HEIGHT  # 21.6 m²
# WALL_AREA_SOUTH = FLOOR_LENGTH * CEILING_HEIGHT - WINDOW_AREA  # 9.6 m²
# WALL_AREA_EAST = FLOOR_WIDTH * CEILING_HEIGHT   # 16.2 m²
# WALL_AREA_WEST = FLOOR_WIDTH * CEILING_HEIGHT   # 16.2 m²
# WALL_AREA_TOTAL = WALL_AREA_NORTH + WALL_AREA_SOUTH + WALL_AREA_EAST + WALL_AREA_WEST  # 63.6 m²

ROOF_AREA = FLOOR_AREA  # 48 m²

# U-values (ASHRAE 140-2020 Table 5-3)
# These are assembly U-values including interior film coefficient
U_WALLS = 0.514   # W/m²K (exterior walls)
U_ROOF = 0.318    # W/m²K (ceiling)
U_FLOOR = 0.039   # W/m²K (insulated raised floor - effectively adiabatic)

# Window properties (ASHRAE 140-2020 Section 5.2.1.7, WINDOW7 ID 23)
# From Case600-W7-DblPaneClr-ID23.txt:
U_WINDOWS = 2.10  # W/m²K (center-of-glass, environmental condition 7)
SHGC = 0.769      # Solar heat gain coefficient at normal incidence
# Note: Standard 140 uses center-of-glass values, no frame

# Ventilation (ASHRAE 140-2020 Section 5.2.1.8)
# Note: Standard specifies 0.5 ACH at sea level.
# Modelica adjusts to 0.41 ACH for Denver altitude (1650m) due to lower air density.
# Using spec value here; the difference is ~18% of ventilation losses.
ACH_INFILTRATION = 0.5  # Air changes per hour (per ASHRAE 140 spec)

# Ground temperature
# Per Modelica Buildings Library: floor connected to fixed 10°C ground temp
# Note: With U_floor=0.039 W/m²K, floor losses are only ~2% of total HLC, 
# so ground boundary condition has minimal impact on results.
GROUND_TEMP_C = 10.0  # °C (283.15 K)

# Internal gains (ASHRAE 140-2020 Section 5.2.1.6)
INTERNAL_GAINS_W = 200.0  # Watts, constant 24/7

# Setpoints (ASHRAE 140-2020 Section 5.2.1.5)
T_SET_HEATING = 20.0  # °C
T_SET_COOLING = 27.0  # °C

# Modelica angle-dependent g-value polynomial (from GlazedElements.mo)
# coeFac = {1, -0.189, 0.644, -0.596} for double-pane clear glass (WINDOW7 ID 23)
# Models reduced transmittance at grazing angles due to Fresnel reflections.
# Applied to direct beam only; diffuse uses normal-incidence g-value.
MODELICA_GVALUE_ANGLE_COEF = (1.0, -0.189, 0.644, -0.596)

# Location
LATITUDE = 39.833   # °N
LONGITUDE = -104.65  # °W
ALTITUDE = 1650     # m
TIMEZONE = 'Etc/GMT+7'  # Denver standard time (no DST in TMY)


# ===========================================================================
# Reference Results (ASHRAE 140-2017 Table B8-2, participating programs)
# ===========================================================================

REFERENCE_RESULTS = {
    'annual_heating_mwh': {'min': 4.296, 'max': 5.709},
    'annual_cooling_mwh': {'min': 6.137, 'max': 8.448},
    'peak_heating_kw': {'min': 3.437, 'max': 4.354},
    'peak_cooling_kw': {'min': 5.965, 'max': 7.188},
}

# EnergyPlus Reference Results (from ASHRAE 140-2020 validation)
# These are actual E+ results, which are within the BESTEST range
ENERGYPLUS_RESULTS = {
    'annual_heating_mwh': 4.673,  # -7.4% from BESTEST avg (5.046)
    'annual_cooling_mwh': 6.792,  # -3.7% from BESTEST avg (7.053)
    'peak_heating_kw': 3.838,     # -2.9% from BESTEST avg (3.952)
    'peak_cooling_kw': 6.664,     # +2.0% from BESTEST avg (6.535)
}


def load_denver_tmy(filepath: Path) -> pd.DataFrame:
    """Load Denver TMY weather data from ASHRAE 140 accompanying files.
    
    File format: TMY3 CSV (725650TY.csv)
    - Line 1: Station metadata (ID, name, state, tz, lat, lon, elev)
    - Line 2: Column headers
    - Lines 3+: Hourly data
    
    Returns:
        DataFrame with columns: datetime, t_outdoor, dni, dhi, ghi, wind_speed
    """
    # Read metadata line
    with open(filepath, 'r') as f:
        meta_line = f.readline().strip()
    
    # Parse metadata
    parts = meta_line.split(',')
    station_id = parts[0]
    station_name = parts[1].strip('"')
    timezone_offset = float(parts[3])
    lat = float(parts[4])
    lon = float(parts[5])
    elev = float(parts[6])
    
    print(f"Loading weather data: {station_name} ({station_id})")
    print(f"  Location: {lat}°N, {lon}°W, {elev}m elevation")
    print(f"  Timezone: UTC{timezone_offset:+.0f}")
    
    # Read data (skip metadata line)
    df = pd.read_csv(filepath, skiprows=1)
    
    # Parse datetime - handle hour 24:00 by converting to next day 00:00
    datetimes = []
    for date_str, time_str in zip(df['Date (MM/DD/YYYY)'], df['Time (HH:MM)']):
        base_date = pd.to_datetime(date_str, format='%m/%d/%Y')
        if time_str == '24:00':
            # Hour 24 = midnight of next day
            dt = base_date + pd.Timedelta(hours=24)
        else:
            hour, minute = map(int, time_str.split(':'))
            dt = base_date + pd.Timedelta(hours=hour, minutes=minute)
        datetimes.append(dt)
    
    df['datetime'] = datetimes
    
    # Extract relevant columns
    result = pd.DataFrame({
        'datetime': df['datetime'],
        't_outdoor': df['Dry-bulb (C)'].astype(float),
        'dni': df['DNI (W/m^2)'].astype(float),
        'dhi': df['DHI (W/m^2)'].astype(float),
        'ghi': df['GHI (W/m^2)'].astype(float),
        'wind_speed': df['Wspd (m/s)'].astype(float),
    })
    
    # Set datetime index
    result.set_index('datetime', inplace=True)
    
    # Handle missing values (coded as -9900)
    result = result.replace(-9900, np.nan)
    result = result.ffill().bfill()
    
    print(f"  Loaded {len(result)} hours of data")
    print(f"  Temperature range: {result['t_outdoor'].min():.1f}°C to {result['t_outdoor'].max():.1f}°C")
    print(f"  Mean temperature: {result['t_outdoor'].mean():.1f}°C")
    
    return result


def calculate_south_window_irradiance(
    weather: pd.DataFrame,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    altitude: float = ALTITUDE,
) -> np.ndarray:
    """Calculate total irradiance on south-facing vertical windows.
    
    Uses pvlib to compute plane-of-array irradiance for:
    - Surface tilt: 90° (vertical)
    - Surface azimuth: 180° (south-facing)
    
    Follows ASHRAE 140-2020 Section 5.2.1.12 for solar distribution
    (not relevant here since we only need incident irradiance).
    
    Args:
        weather: DataFrame with dni, dhi, ghi columns
        latitude: Site latitude (default: Denver)
        longitude: Site longitude (default: Denver)
        altitude: Site altitude in meters (default: Denver)
    
    Returns:
        Array of total POA irradiance [W/m²] for each hour
    """
    # Create location
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        tz=TIMEZONE
    )
    
    # Get solar position for each hour
    # Note: TMY data is local standard time (no DST), times represent end of hour
    # Need to localize the index to proper timezone
    times = weather.index.tz_localize(TIMEZONE)
    solar_position = location.get_solarposition(times)
    
    # South-facing vertical window
    surface_tilt = 90  # degrees from horizontal
    surface_azimuth = 180  # degrees clockwise from north (south-facing)
    
    # Get DNI, DHI, GHI
    dni = weather['dni'].values.astype(float)
    dhi = weather['dhi'].values.astype(float)
    ghi = weather['ghi'].values.astype(float)
    
    # Clip unrealistic values (sometimes TMY data has errors)
    dni = np.clip(dni, 0, 1400)  # Max DNI ~1400 W/m² at high altitude
    dhi = np.clip(dhi, 0, 700)
    ghi = np.clip(ghi, 0, 1400)
    
    # Calculate POA irradiance using isotropic sky model (simpler, more stable)
    # For vertical south-facing surface
    poa_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        model='isotropic',  # More stable than Perez for edge cases
    )
    
    # Get total POA (beam + diffuse + ground reflected)
    poa_global = poa_irradiance['poa_global'].fillna(0).values
    
    # Clip any unrealistic results (should not exceed ~1200 W/m² on vertical)
    poa_global = np.clip(poa_global, 0, 1200)
    
    # Statistics
    daily_max = poa_global.reshape(-1, 24).max(axis=1)
    print(f"\nSouth window irradiance:")
    print(f"  Annual total: {np.sum(poa_global)/1000:.0f} kWh/m²")
    print(f"  Peak hourly: {np.max(poa_global):.0f} W/m²")
    print(f"  Mean daily peak: {np.mean(daily_max):.0f} W/m²")
    
    return poa_global


def create_case_600_building(
    convective_fraction: float = 0.4,
    use_modelica_settings: bool = False,
) -> Building5R1C:
    """Create BESTEST Case 600 building.
    
    Follows ASHRAE 140-2020 Section 5.2.1 specifications with corrections
    from LBNL Modelica Buildings Library implementation.
    
    Key modeling notes:
    1. Floor is connected to constant ground temperature (10°C) per Modelica.
       The 2m soil layer provides thermal inertia with the floor insulation.
    
    2. Windows properties from WINDOW7 (ID 23, environmental condition 7):
       - U-value: 2.10 W/m²K (center-of-glass)
       - SHGC: 0.769 at normal incidence
    
    3. Thermal mass: ASHRAE 140 Case 600 is "low mass" (lightweight construction).
       This maps to BuildingClass.VERY_LIGHT in ISO 13790.
    
    4. Internal gains: 200 W constant, with 60% radiative / 40% convective split
       per EnergyPlus/Modelica: qRadGai=120W, qConGai=80W
       This differs from ISO 13790's default 50/50 split.
    
    5. Infiltration: 0.41 ACH adjusted for Denver altitude (0.5 at sea level)
       per Modelica implementation comment.
       
    Args:
        convective_fraction: Internal gains convective fraction. Default 0.4 (40%)
                            to match EnergyPlus/Modelica BESTEST setup.
        use_modelica_settings: If True, use Modelica Buildings Library specific
                              parameters (h_is=2.1, actual surface areas, etc.)
                              to exactly replicate Modelica ISO 13790 results.
    """
    # Modelica-specific parameters from Buildings.ThermalZones.ISO13790:
    # - hInt = 2.1 (BESTEST override, vs ISO default 3.45)
    # - ratSur = (sum(AWal) + sum(AWin) + ARoo + AFlo) / AFlo = actual surfaces
    # - heaC = 42167 J/(m²K) from Case600Mass record
    # - facMas = 2.95 (mass area factor)
    # - U-values from Modelica differ from ASHRAE 140 Table 5-3!
    # - b = 1.0 (ground temp adjustment) - floor losses vs outdoor temp!
    if use_modelica_settings:
        # Modelica wall areas: AWal = {21.6, 16.2, 9.6, 16.2}
        # This is OPAQUE wall area (south is gross minus 12m² windows = 9.6m²)
        wall_area_modelica = 21.6 + 16.2 + 9.6 + 16.2  # 63.6 m² (vs ASHRAE's 55.6 m²)
        # A_tot per Modelica: sum(AWal) + sum(AWin) + ARoo + AFlo
        a_tot_modelica = wall_area_modelica + WINDOW_AREA + ROOF_AREA + FLOOR_AREA  # 171.6 m²
        
        # Modelica thermal mass (after subtracting air volume contribution)
        # C = heaC * AFlo - VRoo * 1.2 * 1014
        # = 42167 * 48 - 129.6 * 1.2 * 1014 = 1866399 J/K
        # Per floor area: 1866399 / 48 = 38883 J/(m²K)
        thermal_cap_modelica = 42167  # J/(m²K) - direct from Modelica Case600Mass
        
        h_is = 2.1  # Modelica BESTEST override (convective only)
        mass_area_factor = 2.95  # facMas from Case600Mass
        ach_infl = 0.414  # Modelica altitude-adjusted ACH
        
        # Modelica U-values (differ from ASHRAE 140-2020 Table 5-3!)
        u_walls = 0.53    # Modelica uses 0.53, ASHRAE says 0.514
        u_roof = 0.33     # Modelica uses 0.33, ASHRAE says 0.318
        u_floor = 0.038   # Modelica uses 0.038, ASHRAE says 0.039
        u_windows = 3.1   # KEY DIFFERENCE: Modelica uses 3.1, ASHRAE says 2.10
        
        # CRITICAL: Modelica BESTEST uses b=1, meaning floor losses calculated
        # against outdoor temp, not ground temp. This is a major difference!
        use_ground_temps = None  # Don't use ground temp array
        temp_adj_ground = 1.0    # b=1 from Modelica Zone.mo
    else:
        wall_area_modelica = None  # Use ASHRAE spec value
        a_tot_modelica = None  # Use ISO default 4.5 * A_f
        thermal_cap_modelica = None  # Use building class
        h_is = 3.45  # ISO 13790 default
        mass_area_factor = None  # Derive from building class
        ach_infl = ACH_INFILTRATION  # Use spec value
        u_walls = U_WALLS
        u_roof = U_ROOF
        u_floor = U_FLOOR
        u_windows = U_WINDOWS
        # Use constant 10°C ground temp per original BESTEST interpretation
        use_ground_temps = [GROUND_TEMP_C] * 12
        temp_adj_ground = 0.5  # Not used when ground_temps_monthly is set
    
    # Wall area: use Modelica's geometric calculation or ASHRAE spec
    wall_area = wall_area_modelica if wall_area_modelica else WALL_AREA_TOTAL
    
    building = Building5R1C(
        # Geometry
        floor_area_m2=FLOOR_AREA,
        ceiling_height_m=CEILING_HEIGHT,
        
        # Thermal mass class - Case 600 is lightweight
        building_class=BuildingClass.VERY_LIGHT,
        thermal_capacitance_j_m2k=thermal_cap_modelica,
        
        # U-values (Modelica values differ from ASHRAE 140-2020 Table 5-3!)
        u_walls=u_walls,
        u_roof=u_roof,
        u_floor=u_floor,
        u_windows=u_windows,
        
        # Areas
        wall_area_m2=wall_area,  # Use Modelica (63.6) or ASHRAE (55.6) m²
        roof_area_m2=ROOF_AREA,
        floor_area_ground_m2=FLOOR_AREA,  
        window_area_m2=WINDOW_AREA,  # South-facing only
        
        # Ventilation: infiltration only (altitude-corrected), no mechanical ventilation
        ach_infiltration=ach_infl,
        ach_ventilation=0.0,
        heat_recovery_efficiency=0.0,
        
        # Solar gains
        g_value=SHGC,
        frame_factor=1.0,  # Pure glazing, no frame in BESTEST
        # Use angle-dependent g-value when replicating Modelica
        g_value_angle_coefficients=MODELICA_GVALUE_ANGLE_COEF if use_modelica_settings else None,
        
        # Internal gains (with EnergyPlus/Modelica-compatible convective fraction)
        internal_gains_w_per_m2=INTERNAL_GAINS_W / FLOOR_AREA,
        convective_fraction_internal=convective_fraction,
        
        # Setpoints
        t_set_heating=T_SET_HEATING,
        t_set_cooling=T_SET_COOLING,
        
        # Floor boundary: 
        # - ASHRAE mode: constant 10°C ground temperature
        # - Modelica mode: b=1.0 (floor losses vs outdoor temp, no ground temp array)
        ground_temps_monthly=use_ground_temps,
        temp_adj_ground=temp_adj_ground,
        
        # Thermal bridges: none in BESTEST
        delta_u_thermal_bridge=0.0,
        
        # Sky radiation: BESTEST doesn't explicitly include sky radiation loss
        # through opaque elements (only window solar gain), so disable.
        delta_theta_er=0.0,
        
        # Modelica-specific overrides (only active if use_modelica_settings=True)
        h_is=h_is,
        a_tot_surfaces_m2=a_tot_modelica,
        mass_area_factor_override=mass_area_factor,
    )
    
    return building


def run_annual_simulation(
    building: Building5R1C,
    weather: pd.DataFrame,
    solar_irradiance: np.ndarray,
) -> dict:
    """Run annual hourly simulation for BESTEST Case 600.
    
    Args:
        building: Building5R1C instance
        weather: DataFrame with t_outdoor column
        solar_irradiance: Array of total POA irradiance on south windows [W/m²]
    
    Returns:
        Dictionary with simulation results
    """
    # Get outdoor temperature
    t_outdoor = weather['t_outdoor'].values
    
    # Calculate solar gains through windows
    # Solar gain = POA irradiance × window area × SHGC × frame_factor
    solar_gains = solar_irradiance * building.window_area_m2 * building.g_value * building.frame_factor
    
    # Internal gains (constant)
    internal_gains = np.full(len(t_outdoor), INTERNAL_GAINS_W)
    
    # Initialize building
    building.reset(t_initial=T_SET_HEATING)
    
    # Run simulation with unlimited heating AND cooling capacity
    # BESTEST assumes ideal HVAC (infinite capacity)
    print(f"\nRunning annual simulation ({len(t_outdoor)} hours)...")
    results = building.simulate(
        t_outdoor=t_outdoor,
        solar_gains=solar_gains,
        internal_gains=internal_gains,
        dt_hours=1.0,
        max_heating_power=np.inf,  # Unlimited heating
        max_cooling_power=np.inf,  # Unlimited cooling
    )
    
    return results


def analyze_results(results: dict, weather: pd.DataFrame) -> dict:
    """Analyze simulation results and compare to ASHRAE 140 reference.
    
    Args:
        results: Dictionary from simulate() method
        weather: Weather DataFrame for timestamps
    
    Returns:
        Dictionary with computed metrics
    """
    # Extract heating/cooling demand
    # Positive = heating, Negative = cooling (convention in 5R1C model)
    demand = results['heating_demand']
    
    heating_demand = np.maximum(demand, 0)
    cooling_demand = np.maximum(-demand, 0)  # Make positive for cooling
    
    # Annual totals
    annual_heating_kwh = np.sum(heating_demand) / 1000
    annual_cooling_kwh = np.sum(cooling_demand) / 1000
    annual_heating_mwh = annual_heating_kwh / 1000
    annual_cooling_mwh = annual_cooling_kwh / 1000
    
    # Peak loads
    peak_heating_w = np.max(heating_demand)
    peak_cooling_w = np.max(cooling_demand)
    peak_heating_kw = peak_heating_w / 1000
    peak_cooling_kw = peak_cooling_w / 1000
    
    # Find peak hours
    idx_peak_heating = np.argmax(heating_demand)
    idx_peak_cooling = np.argmax(cooling_demand)
    
    # Temperature statistics
    t_air = results['t_air']
    
    metrics = {
        'annual_heating_kwh': annual_heating_kwh,
        'annual_cooling_kwh': annual_cooling_kwh,
        'annual_heating_mwh': annual_heating_mwh,
        'annual_cooling_mwh': annual_cooling_mwh,
        'peak_heating_kw': peak_heating_kw,
        'peak_cooling_kw': peak_cooling_kw,
        'peak_heating_hour': idx_peak_heating,
        'peak_cooling_hour': idx_peak_cooling,
        'mean_indoor_temp': np.mean(t_air),
        'min_indoor_temp': np.min(t_air),
        'max_indoor_temp': np.max(t_air),
        'hours_heating': np.sum(heating_demand > 0),
        'hours_cooling': np.sum(cooling_demand > 0),
    }
    
    return metrics


def print_comparison(metrics: dict):
    """Print results comparison against ASHRAE 140 reference ranges."""
    
    print("\n" + "=" * 70)
    print("BESTEST CASE 600 VALIDATION RESULTS")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    ASHRAE Standard 140 Comparison                   │")
    print("├───────────────────┬──────────────┬───────────────┬─────────────────┤")
    print("│ Metric            │ This Model   │ Reference     │ Status          │")
    print("├───────────────────┼──────────────┼───────────────┼─────────────────┤")
    
    results_summary = []
    
    # Annual Heating
    ref = REFERENCE_RESULTS['annual_heating_mwh']
    value = metrics['annual_heating_mwh']
    ref_mid = (ref['min'] + ref['max']) / 2
    pct_diff = (value - ref_mid) / ref_mid * 100
    status = "✓ PASS" if ref['min'] <= value <= ref['max'] else "✗ FAIL"
    results_summary.append(('annual_heating_mwh', status == "✓ PASS", pct_diff))
    print(f"│ Annual Heating    │ {value:>6.3f} MWh  │ {ref['min']:.3f}-{ref['max']:.3f}   │ {status:^15} │")
    
    # Annual Cooling
    ref = REFERENCE_RESULTS['annual_cooling_mwh']
    value = metrics['annual_cooling_mwh']
    ref_mid = (ref['min'] + ref['max']) / 2
    pct_diff = (value - ref_mid) / ref_mid * 100
    status = "✓ PASS" if ref['min'] <= value <= ref['max'] else "✗ FAIL"
    results_summary.append(('annual_cooling_mwh', status == "✓ PASS", pct_diff))
    print(f"│ Annual Cooling    │ {value:>6.3f} MWh  │ {ref['min']:.3f}-{ref['max']:.3f}   │ {status:^15} │")
    
    # Peak Heating
    ref = REFERENCE_RESULTS['peak_heating_kw']
    value = metrics['peak_heating_kw']
    ref_mid = (ref['min'] + ref['max']) / 2
    pct_diff = (value - ref_mid) / ref_mid * 100
    status = "✓ PASS" if ref['min'] <= value <= ref['max'] else "✗ FAIL"
    results_summary.append(('peak_heating_kw', status == "✓ PASS", pct_diff))
    print(f"│ Peak Heating      │ {value:>6.3f} kW   │ {ref['min']:.3f}-{ref['max']:.3f}   │ {status:^15} │")
    
    # Peak Cooling
    ref = REFERENCE_RESULTS['peak_cooling_kw']
    value = metrics['peak_cooling_kw']
    ref_mid = (ref['min'] + ref['max']) / 2
    pct_diff = (value - ref_mid) / ref_mid * 100
    status = "✓ PASS" if ref['min'] <= value <= ref['max'] else "✗ FAIL"
    results_summary.append(('peak_cooling_kw', status == "✓ PASS", pct_diff))
    print(f"│ Peak Cooling      │ {value:>6.3f} kW   │ {ref['min']:.3f}-{ref['max']:.3f}   │ {status:^15} │")
    
    print("└───────────────────┴──────────────┴───────────────┴─────────────────┘")
    
    # Deviation analysis
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                         Deviation Analysis                          │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    for metric, passed, pct in results_summary:
        label = metric.replace('_', ' ').replace('mwh', '').replace('kw', '').title().strip()
        sign = "+" if pct > 0 else ""
        print(f"│ {label:<20}: {sign}{pct:>6.1f}% from reference midpoint           │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                         Additional Statistics                       │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Mean indoor temperature:     {metrics['mean_indoor_temp']:>6.2f}°C                           │")
    print(f"│ Indoor temp range:           {metrics['min_indoor_temp']:.1f}°C to {metrics['max_indoor_temp']:.1f}°C                       │")
    print(f"│ Hours with heating:          {metrics['hours_heating']:>5} ({metrics['hours_heating']/87.6:.1f}%)                       │")
    print(f"│ Hours with cooling:          {metrics['hours_cooling']:>5} ({metrics['hours_cooling']/87.6:.1f}%)                       │")
    print(f"│ Peak heating at hour:        {metrics['peak_heating_hour']:>5}                              │")
    print(f"│ Peak cooling at hour:        {metrics['peak_cooling_hour']:>5}                              │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    # Summary
    n_pass = sum(1 for _, passed, _ in results_summary if passed)
    print(f"\nOverall: {n_pass}/4 metrics within reference range")
    
    # Comparison against EnergyPlus specifically
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    Comparison vs EnergyPlus                        │")
    print("├───────────────────┬──────────────┬──────────────┬─────────────────┤")
    print("│ Metric            │ This Model   │ EnergyPlus   │ Difference      │")
    print("├───────────────────┼──────────────┼──────────────┼─────────────────┤")
    
    eplus_comparison = [
        ('Annual Heating', metrics['annual_heating_mwh'], ENERGYPLUS_RESULTS['annual_heating_mwh'], 'MWh'),
        ('Annual Cooling', metrics['annual_cooling_mwh'], ENERGYPLUS_RESULTS['annual_cooling_mwh'], 'MWh'),
        ('Peak Heating', metrics['peak_heating_kw'], ENERGYPLUS_RESULTS['peak_heating_kw'], 'kW'),
        ('Peak Cooling', metrics['peak_cooling_kw'], ENERGYPLUS_RESULTS['peak_cooling_kw'], 'kW'),
    ]
    
    for label, model_val, eplus_val, unit in eplus_comparison:
        diff_pct = (model_val - eplus_val) / eplus_val * 100
        sign = "+" if diff_pct > 0 else ""
        print(f"│ {label:<17} │ {model_val:>6.3f} {unit:<3}  │ {eplus_val:>6.3f} {unit:<3}  │ {sign}{diff_pct:>6.1f}%        │")
    
    print("└───────────────────┴──────────────┴──────────────┴─────────────────┘")
    
    # Analysis of deviations
    print("\n" + "=" * 70)
    print("ANALYSIS OF DEVIATIONS")
    print("=" * 70)
    print("""
The ISO 13790 5R1C simplified model can achieve close agreement with
detailed simulations when parameters are carefully matched:

WITH MODELICA SETTINGS (--modelica flag):
  - Heating: ~5% below Modelica Buildings Library ISO 13790
  - Cooling: ~4% below Modelica
  - Uses LBNL's parameter choices: h_is=2.1, actual surface areas,
    angle-dependent g-value, b=1.0 for ground coupling

WITH ISO 13790 DEFAULTS:
  - Heating: ~25-35% below BESTEST reference range
  - The ISO 13790 default h_is=3.45 W/m²K and A_tot=4.5*A_f lead to
    different thermal dynamics than detailed programs

KEY PARAMETER DIFFERENCES FROM DETAILED PROGRAMS:
1. Surface heat transfer: h_is = 2.1-3.45 W/m²K vs ~8.3 combined (ASHRAE)
2. Thermal mass coupling: Lumped mass vs distributed thermal gradients
3. Solar angle dependence: Polynomial SHGC modifier vs ray-tracing
4. Ground coupling: b factor vs detailed soil modeling

REMAINING ~5% GAP SOURCES:
- Numerical solver differences (Dassl vs explicit Euler)
- Solar geometry calculation precision (pvlib vs Modelica.BoundaryConditions)
- Heat distribution to nodes (Modelica's GainSurface/GainMass blocks)

The 5R1C model captures essential physics and is suitable for early-stage
design, parametric studies, and district-scale modeling where computational
efficiency is important.

Reference: ISO 13790:2008 Annex C, LBNL Modelica Buildings Library v11
""")
    
    return n_pass


def print_building_parameters(building: Building5R1C):
    """Print building heat loss parameters."""
    
    print("\n" + "=" * 70)
    print("CASE 600 BUILDING PARAMETERS")
    print("=" * 70)
    
    print(f"\nGeometry:")
    print(f"  Floor area:    {FLOOR_AREA:.1f} m²")
    print(f"  Volume:        {VOLUME:.1f} m³")
    print(f"  Wall area:     {building.wall_area_m2:.1f} m² (opaque)")
    print(f"  Window area:   {WINDOW_AREA:.1f} m² (south)")
    print(f"  Roof area:     {ROOF_AREA:.1f} m²")
    
    print(f"\nU-values:")
    print(f"  Walls:         {building.u_walls:.3f} W/m²K")
    print(f"  Roof:          {building.u_roof:.3f} W/m²K")
    print(f"  Floor:         {building.u_floor:.3f} W/m²K")
    print(f"  Windows:       {building.u_windows:.2f} W/m²K")
    print(f"  SHGC:          {building.g_value:.3f}")
    
    print(f"\nVentilation:")
    print(f"  Infiltration:  {building.ach_infiltration:.3f} ACH")
    
    print(f"\nCalculated conductances (ISO 13790):")
    print(f"  H_tr_op:       {building._h_tr_op:.2f} W/K (opaque transmission)")
    print(f"  H_tr_w:        {building._h_tr_w:.2f} W/K (window transmission)")
    print(f"  H_ve:          {building._h_ve:.2f} W/K (ventilation)")
    print(f"  Total HLC:     {building.heat_loss_coefficient:.2f} W/K")
    
    print(f"\nISO 13790 network parameters:")
    atot_note = "(Modelica actual surfaces)" if building.a_tot_surfaces_m2 else "(4.5 × floor area)"
    print(f"  h_is:          {building.h_is:.2f} W/m²K {'(Modelica BESTEST)' if building.h_is != 3.45 else '(ISO default)'}")
    print(f"  A_tot:         {building._a_tot:.1f} m² {atot_note}")
    print(f"  A_m:           {building._mass_area:.1f} m² (thermal mass area)")
    print(f"  C_m:           {building._c_m/1e6:.2f} MJ/K (thermal capacitance)")
    print(f"  H_tr_is:       {building._h_tr_is:.2f} W/K (surface-air: h_is × A_tot)")
    print(f"  H_tr_ms:       {building._h_tr_ms:.2f} W/K (mass-surface)")
    print(f"  H_tr_em:       {building._h_tr_em:.2f} W/K (exterior-mass)")


def main():
    """Run BESTEST Case 600 validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BESTEST Case 600 Validation')
    parser.add_argument('--modelica', action='store_true',
                       help='Use Modelica Buildings Library settings (h_is=2.1, actual surfaces)')
    parser.add_argument('--compare', action='store_true',
                       help='Run both ISO and Modelica settings and compare')
    args = parser.parse_args()
    
    print("=" * 70)
    print("ASHRAE Standard 140-2020 BESTEST Case 600 Validation")
    print("5R1C ISO 13790 Simplified Hourly Method")
    print("=" * 70)
    
    # Path to Denver weather file
    weather_file = ROOT / 'validation' / 'iso13790_comparison' / '140-2020-AccompanyingFiles' / \
                   'Sec5-2AFiles' / 'Normative Materials' / '725650TY.csv'
    
    if not weather_file.exists():
        print(f"\nERROR: Weather file not found: {weather_file}")
        print("Please ensure the ASHRAE 140-2020 accompanying files are in place.")
        return
    
    # Load weather data
    print("\n" + "-" * 70)
    print("WEATHER DATA")
    print("-" * 70)
    weather = load_denver_tmy(weather_file)
    
    # Calculate south-facing window irradiance
    solar_irradiance = calculate_south_window_irradiance(weather)
    
    if args.compare:
        # Run both configurations
        print("\n" + "=" * 70)
        print("COMPARISON: ISO 13790 Defaults vs Modelica Settings")
        print("=" * 70)
        
        results_list = []
        for use_modelica, label in [(False, "ISO 13790 Defaults"), (True, "Modelica Settings")]:
            print(f"\n{'='*70}")
            print(f"Configuration: {label}")
            print("="*70)
            
            building = create_case_600_building(use_modelica_settings=use_modelica)
            print_building_parameters(building)
            results = run_annual_simulation(building, weather, solar_irradiance)
            metrics = analyze_results(results, weather)
            results_list.append((label, metrics))
        
        # Print side-by-side comparison
        print("\n" + "=" * 70)
        print("SIDE-BY-SIDE COMPARISON")
        print("=" * 70)
        iso_metrics, mod_metrics = results_list[0][1], results_list[1][1]
        
        print(f"\n{'Metric':<25} {'ISO Default':>15} {'Modelica':>15} {'Diff':>12}")
        print("-" * 70)
        print(f"{'Annual Heating (MWh)':<25} {iso_metrics['annual_heating_mwh']:>15.3f} {mod_metrics['annual_heating_mwh']:>15.3f} {(mod_metrics['annual_heating_mwh']/iso_metrics['annual_heating_mwh']-1)*100:>+11.1f}%")
        print(f"{'Annual Cooling (MWh)':<25} {iso_metrics['annual_cooling_mwh']:>15.3f} {mod_metrics['annual_cooling_mwh']:>15.3f} {(mod_metrics['annual_cooling_mwh']/iso_metrics['annual_cooling_mwh']-1)*100:>+11.1f}%")
        print(f"{'Peak Heating (kW)':<25} {iso_metrics['peak_heating_kw']:>15.3f} {mod_metrics['peak_heating_kw']:>15.3f} {(mod_metrics['peak_heating_kw']/iso_metrics['peak_heating_kw']-1)*100:>+11.1f}%")
        print(f"{'Peak Cooling (kW)':<25} {iso_metrics['peak_cooling_kw']:>15.3f} {mod_metrics['peak_cooling_kw']:>15.3f} {(mod_metrics['peak_cooling_kw']/iso_metrics['peak_cooling_kw']-1)*100:>+11.1f}%")
        
        print("\nBESTEST Reference Ranges:")
        print(f"  Heating: 4.296 - 5.709 MWh | Cooling: 6.137 - 8.448 MWh")
        print("\nModelica Actual Results (from Docker run):")
        print(f"  Heating: 4.658 MWh | Cooling: 5.803 MWh")
        
        return results_list
    
    # Single run (default or Modelica based on flag)
    use_modelica = args.modelica
    config_name = "Modelica Settings" if use_modelica else "ISO 13790 Defaults"
    
    # Create building
    print("\n" + "-" * 70)
    print(f"BUILDING MODEL ({config_name})")
    print("-" * 70)
    building = create_case_600_building(use_modelica_settings=use_modelica)
    print_building_parameters(building)
    
    # Run simulation
    print("\n" + "-" * 70)
    print("SIMULATION")
    print("-" * 70)
    results = run_annual_simulation(building, weather, solar_irradiance)
    
    # Analyze results
    metrics = analyze_results(results, weather)
    
    # Print comparison
    print_comparison(metrics)
    
    return metrics


if __name__ == '__main__':
    metrics = main()
