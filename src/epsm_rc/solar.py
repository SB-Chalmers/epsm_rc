"""Solar gains calculation for ISO 13790 5R1C model.

Uses pvlib for accurate solar position and irradiance calculations.

Provides two methods:
1. Simplified: GHI × total_window_area × g_value × frame × 0.5
2. Detailed: Proper solar geometry with orientation-specific windows using pvlib

Features:
- Can load location and irradiance data directly from EPW files
- Uses Perez model for diffuse sky irradiance (more accurate than isotropic)
- Falls back to Erbs decomposition if DNI/DHI not available

References:
    - ISO 13790:2008 Annex E (simplified solar)
    - pvlib: https://pvlib-python.readthedocs.io/
    - Perez et al. (1990) for diffuse sky model
    - BEDOT Thesis (Chalmers 2018) for validation approach
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# pvlib for solar calculations
import pvlib
from pvlib import solarposition, irradiance, atmosphere
from pvlib.iotools import read_epw

from .radiation import plane_irradiance


def calc_g_value_angle_modifier(
    aoi: np.ndarray,
    coefficients: Tuple[float, ...],
) -> np.ndarray:
    """Calculate angle-dependent g-value modifier using polynomial.
    
    Models how solar heat gain coefficient varies with angle of incidence.
    Applied to direct beam component only (diffuse uses normal g-value).
    
    The polynomial form is: modifier = sum(coef[i] * theta^i)
    where theta is incidence angle in radians.
    
    Args:
        aoi: Angle of incidence in degrees (0=normal, 90=grazing)
        coefficients: Polynomial coefficients (a0, a1, a2, ...)
            Example for double-pane clear: (1.0, -0.189, 0.644, -0.596)
            At normal incidence (theta=0): modifier = 1.0
            At grazing angles: modifier decreases
    
    Returns:
        G-value modifier (0.0 to 1.0), same shape as aoi
        
    References:
        - Modelica Buildings Library (coeFac in GlazedElements.mo)
        - EnergyPlus Engineering Reference, Tables 2-3 for angular properties
    """
    # Convert degrees to radians
    theta = np.radians(np.clip(aoi, 0, 90))
    
    # Evaluate polynomial: sum(coef[i] * theta^i)
    modifier = np.zeros_like(theta, dtype=float)
    for i, coef in enumerate(coefficients):
        modifier += coef * (theta ** i)
    
    # Clamp to valid range (0 to 1)
    return np.clip(modifier, 0.0, 1.0)


@dataclass
class SolarGainsCalculator:
    """Calculate solar gains using pvlib for accurate solar geometry.
    
    This implements a detailed solar calculation that accounts for:
    - Solar position throughout the year (via pvlib)
    - Window orientation effects
    - Direct beam vs diffuse radiation
    - Perez sky model for accurate diffuse irradiance on tilted surfaces
    - Ground reflections
    
    Args:
        latitude: Site latitude in degrees (default: 57.7 for Gothenburg)
        longitude: Site longitude in degrees (default: 12.0 for Gothenburg)
        altitude: Site altitude in meters (default: 0)
        timezone: Timezone string (default: 'UTC')
        ground_reflectance: Albedo of ground (default: 0.2)
        
    Example:
        >>> # From coordinates
        >>> calc = SolarGainsCalculator(latitude=57.7, longitude=12.0)
        >>> window_areas = {'south': 15.0, 'north': 5.0, 'east': 8.0, 'west': 8.0}
        >>> gains = calc.calculate_hourly_solar_gains(ghi, window_areas, g_value=0.5)
        
        >>> # From EPW file (recommended - includes DNI/DHI)
        >>> calc = SolarGainsCalculator.from_epw("weather.epw")
        >>> gains = calc.calculate_from_epw(window_areas, g_value=0.5)
    """
    
    latitude: float = 57.7   # Gothenburg, Sweden
    longitude: float = 12.0  # Gothenburg, Sweden  
    altitude: float = 0.0    # Sea level
    timezone: str = 'UTC'
    ground_reflectance: float = 0.2  # Typical grass/soil albedo
    
    # EPW data (populated when loading from file)
    _epw_data: Optional[pd.DataFrame] = field(default=None, repr=False)
    _epw_metadata: Optional[dict] = field(default=None, repr=False)
    _validated_weather: object = field(default=None, repr=False)
    
    @classmethod
    def from_epw(cls, epw_path=None, ground_reflectance: float = 0.2, *, preset=None) -> 'SolarGainsCalculator':
        """Create calculator from EPW weather file.
        
        This extracts location (lat, lon, altitude, timezone) from the EPW
        and stores the weather data for use in calculations.
        
        Args:
            epw_path: Explicit EPW path or captured Core Weather; alternatively select preset
            ground_reflectance: Ground albedo (default: 0.2)
            
        Returns:
            SolarGainsCalculator initialized with EPW location and data
            
        Example:
            >>> calc = SolarGainsCalculator.from_epw("SWE_VG_Goteborg.epw")
            >>> print(f"Location: {calc.latitude:.2f}°N, {calc.longitude:.2f}°E")
        """
        from .weather import load_epw

        weather = load_epw(epw_path, preset=preset)
        data, metadata = read_epw(weather.path, coerce_year=weather.calendar_year)
        # Recheck that the parser consumed the same validated bytes.
        weather.validate()
        
        # Extract location from metadata
        latitude = metadata['latitude']
        longitude = metadata['longitude']
        altitude = metadata.get('altitude', 0.0)
        
        # Timezone offset to string (EPW has numeric offset)
        tz_offset = metadata.get('TZ', 0)
        if tz_offset == 0:
            timezone = 'UTC'
        else:
            # pvlib expects timezone string, use Etc/GMT convention
            # Note: Etc/GMT sign is inverted (Etc/GMT-1 = UTC+1)
            timezone = f'Etc/GMT{-int(tz_offset):+d}'
        
        calc = cls(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            timezone=timezone,
            ground_reflectance=ground_reflectance,
        )
        calc._epw_data = data
        calc._epw_metadata = metadata
        calc._validated_weather = weather
        
        return calc
    
    def update_location(self, latitude: float, longitude: float, 
                       altitude: float = 0.0, timezone: str = 'UTC') -> None:
        """Update the calculator's location.
        
        Args:
            latitude: Site latitude in degrees
            longitude: Site longitude in degrees  
            altitude: Site altitude in meters
            timezone: Timezone string
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone = timezone
    
    def _create_time_index(self, year: int = 2019) -> pd.DatetimeIndex:
        """Create hourly time index for a year."""
        return pd.date_range(
            start=f'{year}-01-01 00:30',  # Mid-hour for hourly data
            periods=8760,
            freq='h',
            tz=self.timezone
        )
    
    def _get_solar_position(self, times: pd.DatetimeIndex) -> pd.DataFrame:
        """Get solar position using pvlib."""
        return solarposition.get_solarposition(
            times, 
            self.latitude, 
            self.longitude,
            self.altitude,
            method='nrel_numpy'  # Fast and accurate
        )
    
    def _decompose_ghi(
        self, 
        ghi: np.ndarray, 
        solar_zenith: np.ndarray,
        times: pd.DatetimeIndex,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompose GHI into DNI and DHI using Erbs model.
        
        Only used as fallback when DNI/DHI not provided.
        
        Returns:
            Tuple of (dni, dhi) arrays in W/m²
        """
        # Create DataFrame for pvlib
        ghi_series = pd.Series(ghi, index=times)
        zenith_series = pd.Series(solar_zenith, index=times)
        
        # Use Erbs model to decompose GHI
        out = irradiance.erbs(ghi_series, zenith_series, times)
        
        return out['dni'].values, out['dhi'].values
    
    def _get_poa_irradiance_perez(
        self,
        surface_tilt: float,
        surface_azimuth: float,
        solar_zenith: np.ndarray,
        solar_azimuth: np.ndarray,
        dni: np.ndarray,
        dhi: np.ndarray,
        ghi: np.ndarray,
        dni_extra: np.ndarray,
        airmass: np.ndarray,
    ) -> np.ndarray:
        """Calculate plane-of-array irradiance using Perez sky model.
        
        The Perez model provides more accurate diffuse irradiance on tilted
        surfaces than the isotropic model by accounting for circumsolar
        and horizon brightening effects.
        
        Args:
            surface_tilt: Tilt from horizontal (90 = vertical wall)
            surface_azimuth: Azimuth (180 = south in pvlib convention)
            solar_zenith: Solar zenith angle array
            solar_azimuth: Solar azimuth array
            dni: Direct normal irradiance array
            dhi: Diffuse horizontal irradiance array
            ghi: Global horizontal irradiance array
            dni_extra: Extraterrestrial DNI array
            airmass: Relative airmass array
            
        Returns:
            Total POA irradiance array [W/m²]
        """
        return plane_irradiance(surface_tilt, surface_azimuth, solar_zenith,
            solar_azimuth, dni, ghi, dhi, dni_extra=dni_extra, airmass=airmass,
            albedo=self.ground_reflectance, legacy_standalone=True)["poa_global"]

    def _get_poa_irradiance_isotropic(
        self,
        surface_tilt: float,
        surface_azimuth: float,
        solar_zenith: np.ndarray,
        solar_azimuth: np.ndarray,
        dni: np.ndarray,
        dhi: np.ndarray,
        ghi: np.ndarray,
    ) -> np.ndarray:
        """Calculate POA irradiance using isotropic sky model (simpler fallback).
        
        Args:
            surface_tilt: Tilt from horizontal (90 = vertical wall)
            surface_azimuth: Azimuth (180 = south in pvlib convention)
            solar_zenith: Solar zenith angle array
            solar_azimuth: Solar azimuth array
            dni: Direct normal irradiance array
            dhi: Diffuse horizontal irradiance array
            ghi: Global horizontal irradiance array
            
        Returns:
            Total POA irradiance array [W/m²]
        """
        return plane_irradiance(surface_tilt, surface_azimuth, solar_zenith,
            solar_azimuth, dni, ghi, dhi, albedo=self.ground_reflectance,
            model="isotropic", legacy_standalone=True)["poa_global"]

    def calculate_hourly_solar_gains(
        self,
        ghi: np.ndarray,
        window_areas: dict[str, float],
        g_value: float,
        frame_factor: float = 0.8,
        year: int = 2019,
        dni: Optional[np.ndarray] = None,
        dhi: Optional[np.ndarray] = None,
        use_perez: bool = True,
        g_value_angle_coefficients: Optional[Tuple[float, ...]] = None,
    ) -> np.ndarray:
        """Calculate hourly solar gains with proper geometry using pvlib.
        
        Args:
            ghi: Global horizontal irradiance [W/m²] for 8760 hours
            window_areas: Dict of {orientation: area_m2}
                          orientations: 'south', 'north', 'east', 'west'
            g_value: Solar heat gain coefficient (SHGC/g-value) at normal incidence
            frame_factor: Glazed fraction of window (1 - frame ratio), typically 0.7-0.85
            year: Year for solar position calculation (affects leap year handling)
            dni: Direct normal irradiance [W/m²]. If None, decomposed from GHI using Erbs.
            dhi: Diffuse horizontal irradiance [W/m²]. If None, decomposed from GHI using Erbs.
            use_perez: Use Perez sky model for diffuse (recommended). Falls back to isotropic
                      if dni/dhi not provided.
            g_value_angle_coefficients: Optional polynomial coefficients for angle-dependent
                g-value. If provided, applies angle correction to beam component.
                Example: (1.0, -0.189, 0.644, -0.596) for double-pane clear glass.
            
        Returns:
            Hourly solar gains [W] for 8760 hours
        """
        n_hours = len(ghi)
        if n_hours != 8760:
            raise ValueError(f"Expected 8760 hours, got {n_hours}")
        
        # Create time index
        times = self._create_time_index(year)
        
        # Get solar position
        solpos = self._get_solar_position(times)
        solar_zenith = solpos['apparent_zenith'].values
        solar_azimuth = solpos['azimuth'].values  # pvlib: N=0, E=90, S=180, W=270
        
        # Get or decompose DNI/DHI
        if dni is None or dhi is None:
            dni, dhi = self._decompose_ghi(ghi, solar_zenith, times)
            # Erbs decomposition doesn't provide dni_extra, so use isotropic model
            use_perez = False
        
        # Surface azimuths in pvlib convention (N=0, E=90, S=180, W=270)
        azimuths = {
            'south': 180,
            'north': 0,
            'east': 90,
            'west': 270,
        }
        
        # For Perez model, we need extraterrestrial DNI and airmass
        if use_perez:
            # Extraterrestrial irradiance
            dni_extra = irradiance.get_extra_radiation(times)
            
            # Airmass (relative, not pressure-corrected)
            airmass = atmosphere.get_relative_airmass(solar_zenith, model='kastenyoung1989')
            # Handle very high zenith angles
            airmass = np.where(solar_zenith < 90, airmass, 0)
        
        # Calculate solar gains for each orientation
        solar_gains = np.zeros(n_hours)
        
        for orientation, area in window_areas.items():
            if area <= 0:
                continue
            
            surface_azimuth = azimuths.get(orientation, 180)
            surface_tilt = 90  # Vertical wall
            
            # Calculate beam vs diffuse separately if using angle-dependent g-value
            if g_value_angle_coefficients is not None:
                # Get angle of incidence for this surface
                aoi = irradiance.aoi(surface_tilt, surface_azimuth, 
                                     solar_zenith, solar_azimuth)
                aoi = np.clip(aoi, 0, 90)  # Clamp to valid range
                
                # Calculate angle-corrected g-value for beam
                g_modifier = calc_g_value_angle_modifier(aoi, g_value_angle_coefficients)
                g_value_beam = g_value * g_modifier
                g_value_diffuse = g_value  # Use normal g-value for diffuse
                
                channels = plane_irradiance(surface_tilt, surface_azimuth,
                    solar_zenith, solar_azimuth, dni, ghi, dhi,
                    albedo=self.ground_reflectance,
                    dni_extra=dni_extra if use_perez else None,
                    airmass=airmass if use_perez else None,
                    model="perez" if use_perez else "isotropic", legacy_standalone=True)
                beam = np.maximum(0, channels["poa_direct"])
                diffuse = np.maximum(0, channels["poa_sky_diffuse"]
                                     + channels["poa_ground_diffuse"])

                # Apply different g-values to beam and diffuse
                solar_gains += (beam * g_value_beam + diffuse * g_value_diffuse) * area * frame_factor
            else:
                # Standard calculation - uniform g-value
                if use_perez:
                    poa = self._get_poa_irradiance_perez(
                        surface_tilt, surface_azimuth,
                        solar_zenith, solar_azimuth,
                        dni, dhi, ghi,
                        dni_extra, airmass
                    )
                else:
                    poa = self._get_poa_irradiance_isotropic(
                        surface_tilt, surface_azimuth,
                        solar_zenith, solar_azimuth,
                        dni, dhi, ghi
                    )
                solar_gains += poa * area * g_value * frame_factor
        
        return solar_gains
    
    def calculate_from_epw(
        self,
        window_areas: dict[str, float],
        g_value: float,
        frame_factor: float = 0.8,
        use_perez: bool = True,
        g_value_angle_coefficients: Optional[Tuple[float, ...]] = None,
    ) -> np.ndarray:
        """Calculate solar gains using stored EPW data.
        
        This is the recommended method when using from_epw() constructor,
        as it uses the EPW's DNI and DHI directly (more accurate than
        decomposition from GHI).
        
        Args:
            window_areas: Dict of {orientation: area_m2}
            g_value: Solar heat gain coefficient
            frame_factor: Glazed fraction of window
            use_perez: Use Perez sky model for diffuse
            g_value_angle_coefficients: Optional polynomial coefficients for
                angle-dependent g-value (ISO 13790 §11.4.2)
            
        Returns:
            Hourly solar gains [W] for 8760 hours
            
        Raises:
            ValueError: If no EPW data loaded (use from_epw constructor)
        """
        if self._epw_data is None:
            raise ValueError("No EPW data loaded. Use SolarGainsCalculator.from_epw() to load.")
        
        if self._validated_weather is not None:
            self._validated_weather.validate()
        data = self._epw_data
        
        # EPW has ghi, dni, dhi columns
        ghi = data['ghi'].values
        dni = data['dni'].values
        dhi = data['dhi'].values
        
        # Get year from EPW data
        year = data.index[0].year
        
        return self.calculate_hourly_solar_gains(
            ghi=ghi,
            window_areas=window_areas,
            g_value=g_value,
            frame_factor=frame_factor,
            year=year,
            dni=dni,
            dhi=dhi,
            use_perez=use_perez,
            g_value_angle_coefficients=g_value_angle_coefficients,
        )
    
    def calculate_opaque_solar_gains(
        self,
        wall_area: float,
        roof_area: float,
        u_walls: float,
        u_roof: float,
        wall_absorptivity: float = 0.6,
        roof_absorptivity: float = 0.6,
        r_se_wall: float = 0.04,
        r_se_roof: float = 0.04,
    ) -> np.ndarray:
        """Calculate solar gains through opaque surfaces (ISO 13790 §11.3.4 Eq. 46 + §11.3.2 Eq. 44).
        
        Solar radiation absorbed by external wall/roof surfaces creates a heat
        flow inward proportional to α_S × R_se × U × A × I_sol. This is
        analogous to a sol-air temperature correction.
        
        Wall irradiance is computed as the average POA irradiance on 4 cardinal
        vertical surfaces (N/E/S/W) using the Perez sky model. Roof irradiance
        uses GHI (horizontal global).
        
        Args:
            wall_area: Total net wall area [m²] (all orientations combined)
            roof_area: Roof area [m²]
            u_walls: Wall U-value [W/m²K]
            u_roof: Roof U-value [W/m²K]
            wall_absorptivity: Solar absorptivity of walls (0-1)
            roof_absorptivity: Solar absorptivity of roof (0-1)
            r_se_wall: Exterior surface resistance for walls [m²K/W]
            r_se_roof: Exterior surface resistance for roof [m²K/W]
            
        Returns:
            Hourly opaque solar gains [W] for 8760 hours
        """
        if self._epw_data is None:
            raise ValueError("No EPW data loaded. Use SolarGainsCalculator.from_epw() to load.")
        
        data = self._epw_data
        ghi = data['ghi'].values
        dni = data['dni'].values
        dhi = data['dhi'].values
        n_hours = len(ghi)
        
        # Roof irradiance = GHI (horizontal surface)
        i_sol_roof = np.asarray(ghi, dtype=float)
        
        # Average wall irradiance: POA on vertical surfaces for 4 cardinal dirs
        year = data.index[0].year
        times = self._create_time_index(year)
        solpos = self._get_solar_position(times)
        solar_zenith = solpos['apparent_zenith'].values
        solar_azimuth = solpos['azimuth'].values
        
        # Extraterrestrial radiation and airmass for Perez model
        from pvlib import irradiance as pv_irr, atmosphere as pv_atm
        dni_extra = pv_irr.get_extra_radiation(times)
        airmass = pv_atm.get_relative_airmass(solar_zenith, model='kastenyoung1989')
        airmass = np.where(solar_zenith < 90, airmass, 0)
        
        i_sol_wall_avg = np.zeros(n_hours)
        for surface_azimuth in [0, 90, 180, 270]:  # N, E, S, W
            poa = self._get_poa_irradiance_perez(
                surface_tilt=90, surface_azimuth=surface_azimuth,
                solar_zenith=solar_zenith, solar_azimuth=solar_azimuth,
                dni=dni, dhi=dhi, ghi=ghi,
                dni_extra=dni_extra, airmass=airmass,
            )
            i_sol_wall_avg += np.maximum(0, poa)
        i_sol_wall_avg /= 4.0  # Average over 4 orientations
        
        # Effective solar collecting areas (ISO 13790 §11.3.4 Eq. 46)
        a_sol_wall = wall_absorptivity * r_se_wall * u_walls * wall_area
        a_sol_roof = roof_absorptivity * r_se_roof * u_roof * roof_area
        
        # Total opaque solar gains [W]
        return a_sol_wall * i_sol_wall_avg + a_sol_roof * i_sol_roof
    
    def get_epw_ghi(self) -> np.ndarray:
        """Get GHI array from loaded EPW data.
        
        Returns:
            GHI array [W/m²] for 8760 hours
            
        Raises:
            ValueError: If no EPW data loaded
        """
        if self._epw_data is None:
            raise ValueError("No EPW data loaded. Use SolarGainsCalculator.from_epw() to load.")
        return self._epw_data['ghi'].values
    
    def get_epw_irradiance(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get GHI, DNI, DHI arrays from loaded EPW data.
        
        Returns:
            Tuple of (ghi, dni, dhi) arrays [W/m²] for 8760 hours
            
        Raises:
            ValueError: If no EPW data loaded
        """
        if self._epw_data is None:
            raise ValueError("No EPW data loaded. Use SolarGainsCalculator.from_epw() to load.")
        data = self._epw_data
        return data['ghi'].values, data['dni'].values, data['dhi'].values
    
    def calculate_single_hour(
        self,
        ghi: float,
        hour_of_year: int,
        window_areas: dict[str, float],
        g_value: float,
        frame_factor: float = 0.8,
        year: int = 2019,
        g_value_angle_coefficients: Optional[Tuple[float, ...]] = None,
    ) -> float:
        """Calculate solar gains for a single hour.
        
        Convenience method for real-time simulation.
        Note: Less efficient than batch calculation due to overhead.
        Uses isotropic sky model for simplicity.
        
        Args:
            ghi: Global horizontal irradiance [W/m²]
            hour_of_year: Hour index (0-8759)
            window_areas: Dict of {orientation: area_m2}
            g_value: Solar heat gain coefficient
            frame_factor: Glazed fraction of window
            year: Year for solar position
            g_value_angle_coefficients: Optional polynomial for angle-dependent g-value
            
        Returns:
            Solar gains [W]
        """
        if ghi <= 0:
            return 0.0
        
        # Create single-hour time index
        times = pd.DatetimeIndex([
            pd.Timestamp(f'{year}-01-01 00:30', tz=self.timezone) + pd.Timedelta(hours=hour_of_year)
        ])
        
        # Get solar position
        solpos = solarposition.get_solarposition(
            times, self.latitude, self.longitude, self.altitude
        )
        solar_zenith = solpos['apparent_zenith'].values[0]
        solar_azimuth = solpos['azimuth'].values[0]
        
        # Skip if sun is below horizon
        if solar_zenith >= 90:
            return 0.0
        
        # Decompose GHI (simplified for single hour)
        ghi_series = pd.Series([ghi], index=times)
        zenith_series = pd.Series([solar_zenith], index=times)
        out = irradiance.erbs(ghi_series, zenith_series, times)
        dni = out['dni'].values[0]
        dhi = out['dhi'].values[0]
        
        # Surface azimuths in pvlib convention
        azimuths = {'south': 180, 'north': 0, 'east': 90, 'west': 270}
        surface_tilt = 90  # Vertical wall
        
        total_gains = 0.0
        for orientation, area in window_areas.items():
            if area <= 0:
                continue
            
            surface_azimuth = azimuths.get(orientation, 180)
            
            if g_value_angle_coefficients is not None:
                # Angle-dependent g-value: separate beam and diffuse
                aoi = irradiance.aoi(surface_tilt, surface_azimuth, 
                                     solar_zenith, solar_azimuth)
                aoi = np.clip(aoi, 0, 90)
                
                g_modifier = calc_g_value_angle_modifier(
                    np.array([aoi]), g_value_angle_coefficients
                )[0]
                g_value_beam = g_value * g_modifier
                
                # Beam component
                beam = irradiance.beam_component(
                    surface_tilt, surface_azimuth,
                    solar_zenith, solar_azimuth, dni
                )
                beam = max(0, beam)
                
                # Diffuse components
                sky_diffuse = irradiance.isotropic(surface_tilt, dhi)
                ground_diffuse = irradiance.get_ground_diffuse(
                    surface_tilt, ghi, albedo=self.ground_reflectance
                )
                diffuse = max(0, sky_diffuse + ground_diffuse)
                
                total_gains += (beam * g_value_beam + diffuse * g_value) * area * frame_factor
            else:
                # Standard calculation
                poa = self._get_poa_irradiance_isotropic(
                    surface_tilt, surface_azimuth,
                    np.array([solar_zenith]), np.array([solar_azimuth]),
                    np.array([dni]), np.array([dhi]), np.array([ghi])
                )[0]
                total_gains += poa * area * g_value * frame_factor
        
        return total_gains


def simplified_solar_gains(
    ghi: np.ndarray,
    total_window_area: float,
    g_value: float,
    frame_factor: float = 0.8,
    average_factor: float = 0.5,
) -> np.ndarray:
    """Simplified solar gains calculation (current default).
    
    This is the ISO 13790 simplified approach that uses a single
    factor to account for average orientation and incidence effects.
    
    Args:
        ghi: Global horizontal irradiance [W/m²]
        total_window_area: Total window area [m²]
        g_value: Solar heat gain coefficient
        frame_factor: Glazed fraction (typically 0.7-0.85)
        average_factor: Orientation/incidence factor (default 0.5)
        
    Returns:
        Solar gains [W]
    """
    solar_factor = total_window_area * g_value * frame_factor * average_factor
    return ghi * solar_factor


def estimate_window_orientation_split(
    total_window_area: float,
    aspect_ratio: float = 1.0,
    south_bias: float = 1.5,
) -> dict[str, float]:
    """Estimate window areas by orientation from total.
    
    For buildings without orientation data, this provides a reasonable
    estimate assuming slightly more glazing on south façade.
    
    Args:
        total_window_area: Total window area [m²]
        aspect_ratio: Building length/width ratio (affects E/W vs N/S split)
        south_bias: Ratio of south glazing to north (default 1.5)
        
    Returns:
        Dict of {'south': area, 'north': area, 'east': area, 'west': area}
    """
    # Total = S + N + E + W
    # S = south_bias × N
    # Assume E+W = N+S (equal perimeter on long vs short sides)
    
    ns_total = total_window_area / 2
    ew_total = total_window_area / 2
    
    north = ns_total / (1 + south_bias)
    south = ns_total - north
    east = ew_total / 2
    west = ew_total / 2
    
    return {
        'south': south,
        'north': north,
        'east': east,
        'west': west,
    }
