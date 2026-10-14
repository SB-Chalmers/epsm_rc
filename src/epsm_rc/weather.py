"""Weather data from EPW files for Swedish load simulation.

Uses EnergyPlus Weather (EPW) format - Typical Meteorological Year data.
Default: Göteborg (bundled). Other Swedish locations available from:
- climate.onebuilding.org (free)
- EnergyPlus website (IWEC data)

EPW Field Reference (0-indexed from data rows):
  0: Year, 1: Month, 2: Day, 3: Hour, 4: Minute
  5: Data Source, 6: Dry Bulb Temp [°C], 7: Dew Point Temp [°C]
  8: Relative Humidity [%], 9: Atmospheric Pressure [Pa]
  10: Extraterrestrial Horizontal Radiation [Wh/m²]
  11: Extraterrestrial Direct Normal Radiation [Wh/m²]
  12: Horizontal Infrared Radiation from Sky [Wh/m²]
  13: Global Horizontal Radiation [Wh/m²]
  14: Direct Normal Radiation [Wh/m²]
  15: Diffuse Horizontal Radiation [Wh/m²]
  ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
import numpy as np


# Bundled default: Göteborg TMY
DEFAULT_EPW = Path(__file__).parent / "data" / "epw" / "goteborg.epw"

# Cache loaded EPW files
_cache: dict[Path, "EPWData"] = {}


@dataclass
class EPWData:
    """Parsed EPW weather data (8760 hourly values)."""
    
    location: str
    latitude: float
    longitude: float
    elevation: float
    timezone: float  # UTC offset
    
    # Hourly arrays (8760 values)
    temp_c: np.ndarray           # Dry-bulb temperature [°C]
    ghi_wh_m2: np.ndarray        # Global Horizontal Irradiance [Wh/m²]
    dni_wh_m2: np.ndarray        # Direct Normal Irradiance [Wh/m²]  
    dhi_wh_m2: np.ndarray        # Diffuse Horizontal Irradiance [Wh/m²]
    
    # Monthly aggregates (computed on init)
    monthly_mean_temp: np.ndarray = field(default_factory=lambda: np.zeros(12))
    monthly_mean_ghi: np.ndarray = field(default_factory=lambda: np.zeros(12))
    monthly_total_ghi: np.ndarray = field(default_factory=lambda: np.zeros(12))
    
    def __post_init__(self):
        """Compute monthly statistics."""
        self._compute_monthly_stats()
    
    def _compute_monthly_stats(self):
        """Calculate monthly mean temperature and irradiance values."""
        # Hours per month (approximate, using standard month lengths)
        month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        hour_idx = 0
        
        for m in range(12):
            n_hours = month_hours[m]
            month_temps = self.temp_c[hour_idx:hour_idx + n_hours]
            month_ghi = self.ghi_wh_m2[hour_idx:hour_idx + n_hours]
            
            self.monthly_mean_temp[m] = np.mean(month_temps)
            self.monthly_total_ghi[m] = np.sum(month_ghi) / 1000  # kWh/m²
            
            # Mean GHI during daylight hours (when GHI > 0)
            daylight_ghi = month_ghi[month_ghi > 0]
            if len(daylight_ghi) > 0:
                self.monthly_mean_ghi[m] = np.mean(daylight_ghi)
            else:
                self.monthly_mean_ghi[m] = 0
            
            hour_idx += n_hours
    
    @property
    def annual_mean_temp(self) -> float:
        """Annual mean temperature [°C]."""
        return float(np.mean(self.temp_c))
    
    @property
    def annual_ghi_kwh_m2(self) -> float:
        """Annual total global horizontal irradiance [kWh/m²]."""
        return float(np.sum(self.ghi_wh_m2) / 1000)
    
    @property
    def heating_degree_days_18(self) -> float:
        """Heating degree days base 18°C."""
        # Daily mean temperatures
        daily_temps = self.temp_c.reshape(-1, 24).mean(axis=1)
        return float(np.sum(np.maximum(18 - daily_temps, 0)))
    
    def get_temperature(
        self,
        start_date: date,
        end_date: date,
        resolution_minutes: int = 10,
        noise_std: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Get temperature for date range, interpolated to resolution.
        
        Args:
            start_date: Start date
            end_date: End date (exclusive)
            resolution_minutes: Output resolution (default 10 min)
            noise_std: Microclimate noise std [°C] for household diversity
            rng: Random generator for noise
            
        Returns:
            Temperature array at specified resolution
        """
        n_days = (end_date - start_date).days
        slots_per_hour = 60 // resolution_minutes
        n_slots = n_days * 24 * slots_per_hour
        
        # Collect hourly temps (wrap around year)
        hourly = []
        for day in range(n_days):
            d = start_date + timedelta(days=day)
            hour_of_year = (d.timetuple().tm_yday - 1) * 24
            for h in range(24):
                idx = (hour_of_year + h) % 8760
                hourly.append(self.temp_c[idx])
        
        hourly = np.array(hourly)
        
        # Interpolate to finer resolution
        if slots_per_hour > 1:
            x = np.arange(len(hourly))
            x_fine = np.linspace(0, len(hourly) - 1, n_slots)
            temps = np.interp(x_fine, x, hourly)
        else:
            temps = hourly
        
        # Add microclimate noise for household diversity
        if noise_std > 0 and rng is not None:
            temps = temps + rng.normal(0, noise_std, len(temps))
        
        return temps
    
    def get_irradiance(
        self,
        start_date: date,
        end_date: date,
        component: str = "ghi",
        resolution_minutes: int = 60,
    ) -> np.ndarray:
        """Get solar irradiance for date range.
        
        Args:
            start_date: Start date
            end_date: End date (exclusive)
            component: "ghi" (global), "dni" (direct), or "dhi" (diffuse)
            resolution_minutes: Output resolution (default 60 min)
            
        Returns:
            Irradiance array [W/m²] at specified resolution
        """
        if component == "ghi":
            source = self.ghi_wh_m2
        elif component == "dni":
            source = self.dni_wh_m2
        elif component == "dhi":
            source = self.dhi_wh_m2
        else:
            raise ValueError(f"Unknown component: {component}")
        
        n_days = (end_date - start_date).days
        slots_per_hour = 60 // resolution_minutes
        n_slots = n_days * 24 * slots_per_hour
        
        # Collect hourly values (wrap around year)
        hourly = []
        for day in range(n_days):
            d = start_date + timedelta(days=day)
            hour_of_year = (d.timetuple().tm_yday - 1) * 24
            for h in range(24):
                idx = (hour_of_year + h) % 8760
                hourly.append(source[idx])
        
        hourly = np.array(hourly)
        
        # Interpolate if needed
        if slots_per_hour > 1:
            x = np.arange(len(hourly))
            x_fine = np.linspace(0, len(hourly) - 1, n_slots)
            irradiance = np.interp(x_fine, x, hourly)
        else:
            irradiance = hourly
        
        return irradiance
    
    def get_monthly_irradiance_profile(self) -> dict:
        """Get monthly irradiance statistics for building energy calculations.
        
        Returns:
            Dictionary with monthly GHI statistics useful for solar gain calculations
        """
        return {
            "monthly_mean_ghi_w_m2": self.monthly_mean_ghi.tolist(),
            "monthly_total_ghi_kwh_m2": self.monthly_total_ghi.tolist(),
            "annual_ghi_kwh_m2": self.annual_ghi_kwh_m2,
            "latitude": self.latitude,
            "location": self.location,
        }
    
    def summary(self) -> str:
        """Return a summary string of the weather data."""
        return (
            f"EPW Weather: {self.location}\n"
            f"  Latitude: {self.latitude:.2f}°, Longitude: {self.longitude:.2f}°\n"
            f"  Elevation: {self.elevation:.0f} m\n"
            f"  Annual mean temp: {self.annual_mean_temp:.1f}°C\n"
            f"  HDD18: {self.heating_degree_days_18:.0f}\n"
            f"  Annual GHI: {self.annual_ghi_kwh_m2:.0f} kWh/m²\n"
            f"  Monthly mean GHI [W/m²]: {', '.join(f'{v:.0f}' for v in self.monthly_mean_ghi)}"
        )


def load_epw(path: Path | str | None = None) -> EPWData:
    """Load EPW weather file. Uses bundled Göteborg data if no path given.
    
    Args:
        path: Path to .epw file (default: bundled Göteborg)
        
    Returns:
        EPWData with hourly temperatures and irradiance
    """
    path = Path(path) if path else DEFAULT_EPW
    
    if path in _cache:
        return _cache[path]
    
    if not path.exists():
        raise FileNotFoundError(f"EPW file not found: {path}")
    
    with open(path) as f:
        lines = f.readlines()
    
    # Header line 1: LOCATION,city,state,country,source,WMO,lat,lon,tz,elev
    h = lines[0].strip().split(',')
    location = h[1]
    latitude = float(h[6])
    longitude = float(h[7])
    timezone = float(h[8])
    elevation = float(h[9])
    
    # Data starts at line 9 (0-indexed: line 8)
    temps = []
    ghi = []
    dni = []
    dhi = []
    
    for line in lines[8:]:
        if not line.strip():
            continue
        fields = line.split(',')
        temps.append(float(fields[6]))   # Dry bulb temp
        ghi.append(float(fields[13]))    # Global Horizontal Radiation
        dni.append(float(fields[14]))    # Direct Normal Radiation
        dhi.append(float(fields[15]))    # Diffuse Horizontal Radiation
    
    # Ensure exactly 8760 hours (pad or truncate if needed)
    def ensure_8760(arr):
        arr = np.array(arr[:8760])
        if len(arr) < 8760:
            arr = np.tile(arr, 8760 // len(arr) + 1)[:8760]
        return arr
    
    temps = ensure_8760(temps)
    ghi = ensure_8760(ghi)
    dni = ensure_8760(dni)
    dhi = ensure_8760(dhi)
    
    epw = EPWData(
        location=location,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        timezone=timezone,
        temp_c=temps,
        ghi_wh_m2=ghi,
        dni_wh_m2=dni,
        dhi_wh_m2=dhi,
    )
    _cache[path] = epw
    return epw


def get_temperature(
    start_date: date,
    end_date: date,
    resolution_minutes: int = 10,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    epw_path: Path | str | None = None,
) -> np.ndarray:
    """Convenience function: get temperature from EPW data.
    
    Args:
        start_date: Start date
        end_date: End date (exclusive)
        resolution_minutes: Output resolution (default 10 min)
        noise_std: Microclimate noise for diversity [°C]
        rng: Random generator
        epw_path: Custom EPW file (default: Göteborg)
        
    Returns:
        Temperature array
    """
    epw = load_epw(epw_path)
    return epw.get_temperature(
        start_date=start_date,
        end_date=end_date,
        resolution_minutes=resolution_minutes,
        noise_std=noise_std,
        rng=rng,
    )


def get_irradiance(
    start_date: date,
    end_date: date,
    component: str = "ghi",
    resolution_minutes: int = 60,
    epw_path: Path | str | None = None,
) -> np.ndarray:
    """Convenience function: get solar irradiance from EPW data.
    
    Args:
        start_date: Start date
        end_date: End date (exclusive)
        component: "ghi" (global), "dni" (direct), or "dhi" (diffuse)
        resolution_minutes: Output resolution (default 60 min)
        epw_path: Custom EPW file (default: Göteborg)
        
    Returns:
        Irradiance array [W/m²]
    """
    epw = load_epw(epw_path)
    return epw.get_irradiance(
        start_date=start_date,
        end_date=end_date,
        component=component,
        resolution_minutes=resolution_minutes,
    )


def get_monthly_irradiance(epw_path: Path | str | None = None) -> dict:
    """Convenience function: get monthly irradiance statistics.
    
    Args:
        epw_path: Custom EPW file (default: Göteborg)
        
    Returns:
        Dictionary with monthly GHI values and statistics
    """
    epw = load_epw(epw_path)
    return epw.get_monthly_irradiance_profile()


def parse_ground_temperatures(epw_path: Path | str) -> dict[str, list[float]]:
    """Parse ground temperatures from EPW file header.
    
    EPW files contain monthly ground temperatures at multiple depths, calculated
    from the weather data. Format:
    GROUND TEMPERATURES,<num_depths>,<depth1>,<cond>,<density>,<cp>,<12 temps>,<depth2>,...
    
    Per ISO 13370, the 0.5m depth is most relevant for floor heat transfer as it
    represents the temperature at shallow foundation depth.
    
    Args:
        epw_path: Path to EPW weather file
        
    Returns:
        Dictionary with:
            - 'depths_m': List of depth values [m]
            - 'monthly_temps_by_depth': Dict mapping depth to 12 monthly temps [°C]
            - 'ground_temps_0_5m': Convenient accessor for 0.5m depth (most common)
            - 'annual_mean': Annual mean ground temperature at shallowest depth
            
    Raises:
        ValueError: If GROUND TEMPERATURES header not found in EPW file
        
    Example:
        >>> temps = parse_ground_temperatures("goteborg.epw")
        >>> temps['ground_temps_0_5m']  # 12 monthly values at 0.5m depth
        [4.14, 1.32, 0.55, 1.21, 5.03, 9.30, 13.32, 16.23, 17.08, 15.71, 12.42, 8.27]
        >>> temps['annual_mean']
        8.04
    """
    path = Path(epw_path)
    
    if not path.exists():
        raise FileNotFoundError(f"EPW file not found: {path}")
    
    with open(path) as f:
        for line in f:
            if line.startswith('GROUND TEMPERATURES'):
                break
        else:
            raise ValueError(f"GROUND TEMPERATURES header not found in EPW file: {path}")
    
    # Parse the ground temperature line
    # Format: GROUND TEMPERATURES,<num_depths>,<depth>,<cond>,<density>,<cp>,<12 temps>,...
    # Note: soil properties (cond, density, cp) may be empty
    parts = line.strip().split(',')
    
    num_depths = int(parts[1])
    depths = []
    monthly_temps_by_depth = {}
    
    # Each depth block has 16 fields:
    # depth, soil_conductivity, soil_density, soil_specific_heat, 12 monthly temps
    idx = 2
    for _ in range(num_depths):
        # Parse depth (may be like ".5" for 0.5m)
        depth_str = parts[idx].strip()
        if depth_str.startswith('.'):
            depth_str = '0' + depth_str
        depth = float(depth_str)
        depths.append(depth)
        
        # Skip 3 soil property fields (may be empty)
        idx += 4
        
        # Read 12 monthly temperatures
        monthly_temps = []
        for m in range(12):
            temp_str = parts[idx + m].strip()
            monthly_temps.append(float(temp_str))
        monthly_temps_by_depth[depth] = monthly_temps
        idx += 12
    
    # Find the shallowest depth (typically 0.5m)
    shallowest_depth = min(depths)
    ground_temps_shallow = monthly_temps_by_depth[shallowest_depth]
    annual_mean = sum(ground_temps_shallow) / 12
    
    return {
        'depths_m': depths,
        'monthly_temps_by_depth': monthly_temps_by_depth,
        'ground_temps_0_5m': monthly_temps_by_depth.get(0.5, ground_temps_shallow),
        'annual_mean': annual_mean,
        'shallowest_depth_m': shallowest_depth,
    }


# Pre-load default weather on import
default_weather = load_epw()
