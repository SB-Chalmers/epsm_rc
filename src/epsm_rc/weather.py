# File: weather.py
# Purpose: Parse explicitly selected EPW inputs after shared checksum/calendar validation.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# Author: Sanjay Somanath
# Research attribution: Existing numerical model credits remain in iso13790.py.
# Date created: Unknown; retained weather module predates this refactor.
# Date modified: 2026-10-20
# Changelog:
#   2026-10-14 - Require explicit selection and shared validation; remove annual padding/truncation.
"""Weather data from EPW files for Swedish load simulation.

Uses EnergyPlus Weather (EPW) format - Typical Meteorological Year data.
Weather selection is explicit; Göteborg is an optional named bundled preset.
Strict loading requires epsm_rc[platform] and delegates calendar/checksum checks
to Core. Complete ordinary years are retained; leap calendars are unsupported.
Other Swedish locations available from:
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

from calendar import isleap
from copy import copy
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from epsm_core import Weather

# Compatibility path constant; it is never selected implicitly.
DEFAULT_EPW = Path(__file__).parent / "data" / "epw" / "goteborg.epw"

# Cache loaded EPW files
_cache: dict[tuple[Path, str, int], EPWData] = {}


@dataclass
class EPWData:
    """Parsed ordinary-year weather with detached metadata and read-only hourly arrays.

    Parameters
    ----------
    location : str
        EPW location label.
    latitude, longitude : float
        Geographic coordinates in degrees.
    elevation : float
        Site elevation in metres above sea level.
    timezone : float
        Local standard-time offset from UTC in hours.
    temp_c : ndarray
        Exactly 8760 finite dry-bulb temperatures in degrees Celsius.
    ghi_wh_m2, dni_wh_m2, dhi_wh_m2 : ndarray
        Exactly 8760 hourly global, direct-normal and diffuse-horizontal values
        in Wh/m². For one-hour intervals these have the same numeric values as W/m².
    path, sha256 : optional
        Original file path and content fingerprint when loaded from an EPW.
    calendar_year : int
        Ordinary Gregorian year. Leap years are rejected.

    Notes
    -----
    Construction performs no I/O. Arrays are copied into immutable byte storage.
    Monthly statistics are computed at construction. ``load_epw`` returns separate
    wrappers, so replacing public attributes cannot change cached weather.

    Examples
    --------
    Construct a synthetic constant climate without Core or file access.

    >>> import numpy as np
    >>> from epsm_rc import EPWData
    >>> zeros = np.zeros(8760)
    >>> weather = EPWData("synthetic", 57, 12, 20, 1, zeros, zeros, zeros, zeros)
    >>> weather.annual_mean_temp
    0.0
    """

    location: str
    latitude: float
    longitude: float
    elevation: float
    timezone: float  # UTC offset

    # Hourly arrays (8760 values)
    temp_c: np.ndarray  # Dry-bulb temperature [°C]
    ghi_wh_m2: np.ndarray  # Global Horizontal Irradiance [Wh/m²]
    dni_wh_m2: np.ndarray  # Direct Normal Irradiance [Wh/m²]
    dhi_wh_m2: np.ndarray  # Diffuse Horizontal Irradiance [Wh/m²]

    path: Path | None = None
    sha256: str | None = None
    calendar_year: int = 2023
    _captured_weather: Weather | None = field(default=None, repr=False)

    # Monthly aggregates (computed on init)
    monthly_mean_temp: np.ndarray = field(default_factory=lambda: np.zeros(12))
    monthly_mean_ghi: np.ndarray = field(default_factory=lambda: np.zeros(12))
    monthly_total_ghi: np.ndarray = field(default_factory=lambda: np.zeros(12))

    def __post_init__(self):
        """Compute monthly statistics."""
        if isleap(self.calendar_year):
            raise ValueError(
                "Leap calendar is unsupported by standalone RC weather; no records are truncated"
            )
        for name in ("temp_c", "ghi_wh_m2", "dni_wh_m2", "dhi_wh_m2"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (8760,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {name}: require 8760 finite hourly values")
            values = np.frombuffer(values.tobytes(), dtype=float)
            setattr(self, name, values)
        self._compute_monthly_stats()

    def __repr__(self):
        pieces = []
        used = 0
        for character in self.location[:32]:
            escaped = repr(character)[1:-1].replace("'", "\\'")
            if used + len(escaped) > 30:
                pieces.append("…")
                break
            pieces.append(escaped)
            used += len(escaped)
        else:
            if len(self.location) > 32:
                pieces.append("…")
        location = "'" + "".join(pieces) + "'"
        return f"EPWData(location={location}, year={self.calendar_year}, hours=8760)"

    __str__ = __repr__

    def _compute_monthly_stats(self):
        """Calculate monthly mean temperature and irradiance values."""
        # Gregorian month lengths for the declared validated calendar.
        from calendar import monthrange

        month_hours = [monthrange(self.calendar_year, month)[1] * 24 for month in range(1, 13)]
        hour_idx = 0

        for m in range(12):
            n_hours = month_hours[m]
            month_temps = self.temp_c[hour_idx : hour_idx + n_hours]
            month_ghi = self.ghi_wh_m2[hour_idx : hour_idx + n_hours]

            self.monthly_mean_temp[m] = np.mean(month_temps)
            self.monthly_total_ghi[m] = np.sum(month_ghi) / 1000  # kWh/m²

            # Mean GHI during daylight hours (when GHI > 0)
            daylight_ghi = month_ghi[month_ghi > 0]
            if len(daylight_ghi) > 0:
                self.monthly_mean_ghi[m] = np.mean(daylight_ghi)
            else:
                self.monthly_mean_ghi[m] = 0

            hour_idx += n_hours

    def validate(self):
        """Recheck captured file identity/calendar before reusing parsed input."""
        if self._captured_weather is not None:
            self._captured_weather.validate()
        return self

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
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Get temperature for date range, interpolated to resolution.

        Parameters
        ----------
        start_date : date
            Start date
        end_date : date
            End date (exclusive)
        resolution_minutes : int
            Output resolution (default 10 min)
        noise_std : float
            Microclimate noise std [°C] for household diversity
        rng : np.random.Generator | None
            Random generator for noise

        Returns
        -------
        np.ndarray
            Temperature array at specified resolution
        """
        self.validate()
        n_days = (end_date - start_date).days
        slots_per_hour = 60 // resolution_minutes
        n_slots = n_days * 24 * slots_per_hour

        # Collect hourly temps (wrap around year)
        hourly = []
        for day in range(n_days):
            d = start_date + timedelta(days=day)
            try:
                calendar_day = date(self.calendar_year, d.month, d.day)
            except ValueError as error:
                raise ValueError(
                    "February 29 is unsupported by this ordinary-year calendar"
                ) from error
            hour_of_year = (calendar_day.timetuple().tm_yday - 1) * 24
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

        Parameters
        ----------
        start_date : date
            Start date
        end_date : date
            End date (exclusive)
        component : str
            "ghi" (global), "dni" (direct), or "dhi" (diffuse)
        resolution_minutes : int
            Output resolution (default 60 min)

        Returns
        -------
        np.ndarray
            Irradiance array [W/m²] at specified resolution
        """
        self.validate()
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
            try:
                calendar_day = date(self.calendar_year, d.month, d.day)
            except ValueError as error:
                raise ValueError(
                    "February 29 is unsupported by this ordinary-year calendar"
                ) from error
            hour_of_year = (calendar_day.timetuple().tm_yday - 1) * 24
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

        Returns
        -------
        dict
            Dictionary with monthly GHI statistics useful for solar gain calculations
        """
        self.validate()
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


def _detached(epw):
    """Keep public attribute edits separate from canonical cached weather."""
    result = copy(epw)
    for name in ("monthly_mean_temp", "monthly_mean_ghi", "monthly_total_ghi"):
        setattr(result, name, getattr(epw, name).copy())
    return result


def load_epw(
    path: Path | str | Weather | None = None,
    *,
    preset: str | None = None,
) -> EPWData:
    """Load explicit EPW input, validating Core checksum/calendar before parsing.

    Supply one local path, captured Core Weather or ``preset="goteborg"``.
    Requires coordinated ``epsm_rc[platform]`` (Python 3.11+); imports remain lazy.
    Changed captured Weather fails. Cache keys include resolved path, checksum
    and calendar; every call revalidates file identity and coverage. Required
    solar arrays are finite and cannot contain EPW missing-value sentinels.
    No records are repeated, padded or truncated. Leap calendars fail explicitly.
    Returned objects are detached. Hourly arrays share immutable byte-backed
    storage; replacing an attribute cannot alter another caller or the cache.
    """
    if path is None and preset is None:
        raise ValueError("Supply an explicit EPW path/Core Weather or select preset='goteborg'")
    if path is not None and preset is not None:
        raise ValueError("Supply either one weather input or a preset, not both")
    if preset is not None:
        if preset != "goteborg":
            raise ValueError(f"Unknown weather preset: {preset!r}; available preset: 'goteborg'")
        path = DEFAULT_EPW
    try:
        from epsm_core import Weather
    except ImportError as error:
        raise ImportError(
            "Strict EPW loading requires epsm_rc[platform] with coordinated Core/Data (Python 3.11+)"
        ) from error
    weather = path if isinstance(path, Weather) else Weather.from_epw(path)
    weather.validate()
    if isleap(weather.calendar_year):
        raise ValueError(
            "Leap calendar is unsupported by standalone RC weather; no records are truncated"
        )
    key = (weather.path.resolve(), weather.sha256, weather.calendar_year)
    if key in _cache:
        return _detached(_cache[key])
    from pvlib.iotools import read_epw

    data, metadata = read_epw(weather.path, coerce_year=weather.calendar_year)
    arrays = {}
    for column, name in (
        ("temp_air", "temp_c"),
        ("ghi", "ghi_wh_m2"),
        ("dni", "dni_wh_m2"),
        ("dhi", "dhi_wh_m2"),
    ):
        values = data[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (
            column != "temp_air" and ((values < 0) | (values >= 9999)).any()
        ):
            raise ValueError(f"Missing or invalid EPW {column}: require finite physical values")
        arrays[name] = values
    weather.validate()  # Bind parsed values to the same captured bytes.
    epw = EPWData(
        location=metadata["city"],
        latitude=metadata["latitude"],
        longitude=metadata["longitude"],
        elevation=metadata["altitude"],
        timezone=metadata["TZ"],
        **arrays,
        path=weather.path.resolve(),
        sha256=weather.sha256,
        calendar_year=weather.calendar_year,
        _captured_weather=weather,
    )
    # Keep one current parsed entry per file rather than retaining obsolete versions.
    for old_key in list(_cache):
        if old_key[0] == key[0]:
            del _cache[old_key]
    _cache[key] = epw
    return _detached(epw)


def get_temperature(
    start_date: date,
    end_date: date,
    resolution_minutes: int = 10,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    epw_path: Path | str | Weather | None = None,
    *,
    preset: str | None = None,
) -> np.ndarray:
    """Convenience function: get temperature from EPW data.

    Parameters
    ----------
    start_date : date
        Start date
    end_date : date
        End date (exclusive)
    resolution_minutes : int
        Output resolution (default 10 min)
    noise_std : float
        Microclimate noise for diversity [°C]
    rng : np.random.Generator | None
        Random generator
    epw_path : Path | str | Weather | None
        Explicit local EPW or captured Core Weather; or select preset.

    Returns
    -------
    np.ndarray
        Temperature array
    """
    epw = load_epw(epw_path, preset=preset)
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
    epw_path: Path | str | Weather | None = None,
    *,
    preset: str | None = None,
) -> np.ndarray:
    """Convenience function: get solar irradiance from EPW data.

    Parameters
    ----------
    start_date : date
        Start date
    end_date : date
        End date (exclusive)
    component : str
        "ghi" (global), "dni" (direct), or "dhi" (diffuse)
    resolution_minutes : int
        Output resolution (default 60 min)
    epw_path : Path | str | Weather | None
        Explicit local EPW or captured Core Weather; or select preset.

    Returns
    -------
    np.ndarray
        Irradiance array [W/m²]
    """
    epw = load_epw(epw_path, preset=preset)
    return epw.get_irradiance(
        start_date=start_date,
        end_date=end_date,
        component=component,
        resolution_minutes=resolution_minutes,
    )


def get_monthly_irradiance(
    epw_path: Path | str | Weather | None = None,
    *,
    preset: str | None = None,
) -> dict:
    """Convenience function: get monthly irradiance statistics.

    Parameters
    ----------
    epw_path : Path | str | Weather | None
        Explicit local EPW or captured Core Weather; or select preset.

    Returns
    -------
    dict
        Dictionary with monthly GHI values and statistics
    """
    epw = load_epw(epw_path, preset=preset)
    return epw.get_monthly_irradiance_profile()


def parse_ground_temperatures(epw_path: Path | str) -> dict[str, list[float]]:
    """Parse ground temperatures from EPW file header.

    EPW files contain monthly ground temperatures at multiple depths, calculated
    from the weather data. Format:
    GROUND TEMPERATURES,<num_depths>,<depth1>,<cond>,<density>,<cp>,<12 temps>,<depth2>,...

    Per ISO 13370, the 0.5m depth is most relevant for floor heat transfer as it
    represents the temperature at shallow foundation depth.

    Parameters
    ----------
    epw_path : Path | str
        Path to EPW weather file

    Returns
    -------
    Dictionary with

            - 'depths_m': List of depth values [m]
            - 'monthly_temps_by_depth': Dict mapping depth to 12 monthly temps [°C]
            - 'ground_temps_0_5m': Convenient accessor for 0.5m depth (most common)
            - 'annual_mean': Annual mean ground temperature at shallowest depth


    Raises
    ------
    ValueError
        If GROUND TEMPERATURES header not found in EPW file

    Examples
    --------
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
            if line.startswith("GROUND TEMPERATURES"):
                break
        else:
            raise ValueError(f"GROUND TEMPERATURES header not found in EPW file: {path}")

    # Parse the ground temperature line
    # Format: GROUND TEMPERATURES,<num_depths>,<depth>,<cond>,<density>,<cp>,<12 temps>,...
    # Note: soil properties (cond, density, cp) may be empty
    parts = line.strip().split(",")

    num_depths = int(parts[1])
    depths = []
    monthly_temps_by_depth = {}

    # Each depth block has 16 fields:
    # depth, soil_conductivity, soil_density, soil_specific_heat, 12 monthly temps
    idx = 2
    for _ in range(num_depths):
        # Parse depth (may be like ".5" for 0.5m)
        depth_str = parts[idx].strip()
        if depth_str.startswith("."):
            depth_str = "0" + depth_str
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
        "depths_m": depths,
        "monthly_temps_by_depth": monthly_temps_by_depth,
        "ground_temps_0_5m": monthly_temps_by_depth.get(0.5, ground_temps_shallow),
        "annual_mean": annual_mean,
        "shallowest_depth_m": shallowest_depth,
    }


def __getattr__(name):
    # Compatibility: explicit access loads weather; imports perform no weather I/O.
    if name == "default_weather":
        raise ValueError(
            'default_weather is retired; select load_epw(preset="goteborg") explicitly'
        )
    raise AttributeError(name)
