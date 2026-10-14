"""Tests for weather module - EPW loading and temperature retrieval."""

from datetime import date
from pathlib import Path
import tempfile

import numpy as np
import pytest

from epsm_rc.weather import (
    load_epw,
    get_temperature,
    get_irradiance,
    get_monthly_irradiance,
    EPWData,
    DEFAULT_EPW,
)


class TestLoadEPW:
    """Tests for EPW file loading."""
    
    def test_loads_default_goteborg(self):
        """Default load uses bundled Göteborg EPW."""
        epw = load_epw()
        
        assert "goteborg" in epw.location.lower()  # Göteborg in some form
        assert 57.0 < epw.latitude < 58.0  # Göteborg latitude
        assert len(epw.temp_c) == 8760  # Full year hourly data
    
    def test_loads_custom_path(self, tmp_path):
        """Can load custom EPW file."""
        # Create minimal EPW
        epw_content = self._create_minimal_epw()
        epw_path = tmp_path / "test.epw"
        epw_path.write_text(epw_content)
        
        epw = load_epw(epw_path)
        
        assert epw.location == "TEST_CITY"
        assert epw.latitude == 59.0
        assert len(epw.temp_c) == 8760
    
    def test_caches_loaded_data(self):
        """Same file should return cached instance."""
        epw1 = load_epw()
        epw2 = load_epw()
        
        assert epw1 is epw2  # Same object from cache
    
    def test_missing_file_raises(self):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_epw("/nonexistent/path.epw")
    
    def _create_minimal_epw(self) -> str:
        """Create minimal valid EPW file content."""
        # Header lines (8 total)
        lines = [
            "LOCATION,TEST_CITY,STATE,COUNTRY,SOURCE,123456,59.0,18.0,1,50",
            "DESIGN CONDITIONS,0",
            "TYPICAL/EXTREME PERIODS,0",
            "GROUND TEMPERATURES,0",
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
            "COMMENTS 1,Test EPW",
            "COMMENTS 2,Minimal",
            "DATA PERIODS,1,1,Data,Sunday,1/1,12/31",
        ]
        
        # Generate 8760 hours of data (full year)
        for month in range(1, 13):
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
            for day in range(1, days_in_month + 1):
                for hour in range(1, 25):  # EPW uses 1-24
                    # Simple sinusoidal temperature pattern
                    temp = 5.0 + 10.0 * np.sin((hour - 6) * np.pi / 12)
                    # GHI peaks at midday in summer
                    ghi = max(0, 500 * np.sin((hour - 6) * np.pi / 12) * (0.5 + 0.5 * np.sin((month - 3) * np.pi / 6)))
                    dni = ghi * 0.8
                    dhi = ghi * 0.2
                    # EPW data format with irradiance fields
                    data = f"2024,{month},{day},{hour},60,?,{temp:.1f},0,0,0,0,0,0,{ghi:.0f},{dni:.0f},{dhi:.0f},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
                    lines.append(data)
        
        return "\n".join(lines)


class TestGetTemperature:
    """Tests for temperature retrieval."""
    
    def test_returns_correct_length(self):
        """Temperature array should match expected length."""
        temps = get_temperature(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 16),  # 1 day
            resolution_minutes=10,
        )
        
        assert len(temps) == 144  # 24 hours * 6 slots/hour
    
    def test_multi_day(self):
        """Multiple days should work."""
        temps = get_temperature(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 8),  # 7 days
            resolution_minutes=60,
        )
        
        assert len(temps) == 7 * 24  # 7 days * 24 hours
    
    def test_winter_is_cold(self):
        """January temperatures should be cold in Sweden."""
        temps = get_temperature(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 16),
        )
        
        assert np.mean(temps) < 5.0  # Should be cold in January
    
    def test_summer_is_warm(self):
        """July temperatures should be warmer."""
        temps = get_temperature(
            start_date=date(2024, 7, 15),
            end_date=date(2024, 7, 16),
        )
        
        assert np.mean(temps) > 10.0  # Should be warmer in July
    
    def test_noise_adds_variation(self):
        """Noise should add variation between calls."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        
        temps1 = get_temperature(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 16),
            noise_std=1.0,
            rng=rng1,
        )
        temps2 = get_temperature(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 16),
            noise_std=1.0,
            rng=rng2,
        )
        
        # Same base data, different noise
        assert not np.allclose(temps1, temps2)
    
    def test_no_noise_is_deterministic(self):
        """Without noise, same dates should give same result."""
        temps1 = get_temperature(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 16),
            noise_std=0.0,
        )
        temps2 = get_temperature(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 16),
            noise_std=0.0,
        )
        
        np.testing.assert_array_equal(temps1, temps2)


class TestEPWData:
    """Tests for EPWData class methods."""
    
    def test_get_temperature_wraps_year(self):
        """Temperature lookup should wrap around year boundary."""
        epw = load_epw()
        
        # Get temps spanning year end (should work without error)
        temps = epw.get_temperature(
            start_date=date(2024, 12, 30),
            end_date=date(2025, 1, 2),  # Crosses year boundary
            resolution_minutes=60,
        )
        
        assert len(temps) == 3 * 24  # 3 days
        assert not np.any(np.isnan(temps))


class TestDefaultWeather:
    """Test module-level default weather."""
    
    def test_default_epw_exists(self):
        """Bundled EPW file should exist."""
        assert DEFAULT_EPW.exists()
    
    def test_default_weather_is_goteborg(self):
        """Default weather should be Göteborg."""
        from epsm_rc.weather import default_weather
        
        assert "GOTEBORG" in default_weather.location.upper()


class TestIrradianceData:
    """Tests for solar irradiance extraction from EPW."""
    
    def test_ghi_array_length(self):
        """GHI array should have 8760 hourly values."""
        epw = load_epw()
        assert len(epw.ghi_wh_m2) == 8760
    
    def test_dni_array_length(self):
        """DNI array should have 8760 hourly values."""
        epw = load_epw()
        assert len(epw.dni_wh_m2) == 8760
    
    def test_dhi_array_length(self):
        """DHI array should have 8760 hourly values."""
        epw = load_epw()
        assert len(epw.dhi_wh_m2) == 8760
    
    def test_ghi_non_negative(self):
        """GHI should never be negative."""
        epw = load_epw()
        assert np.all(epw.ghi_wh_m2 >= 0)
    
    def test_ghi_zero_at_night(self):
        """GHI should be zero during night hours."""
        epw = load_epw()
        # Check midnight hours (around hour 0) in winter
        # January has short days in Sweden
        jan_midnight = epw.ghi_wh_m2[0]  # Hour 0 of Jan 1
        assert jan_midnight == 0
    
    def test_ghi_higher_in_summer(self):
        """Summer GHI should be higher than winter."""
        epw = load_epw()
        
        # January mean (first 744 hours)
        jan_mean = np.mean(epw.ghi_wh_m2[:744])
        
        # July mean (hours 4344-5088, approximately)
        july_start = sum([31, 28, 31, 30, 31, 30]) * 24  # Start of July
        july_mean = np.mean(epw.ghi_wh_m2[july_start:july_start + 744])
        
        assert july_mean > jan_mean * 2  # At least 2x higher in summer
    
    def test_annual_ghi_reasonable_range(self):
        """Annual GHI should be in reasonable range for Sweden."""
        epw = load_epw()
        
        # Swedish annual GHI typically 850-1100 kWh/m²
        assert 800 < epw.annual_ghi_kwh_m2 < 1200


class TestMonthlyStatistics:
    """Tests for monthly aggregation methods."""
    
    def test_monthly_mean_temp_length(self):
        """Should have 12 monthly mean temperatures."""
        epw = load_epw()
        assert len(epw.monthly_mean_temp) == 12
    
    def test_monthly_total_ghi_length(self):
        """Should have 12 monthly GHI totals."""
        epw = load_epw()
        assert len(epw.monthly_total_ghi) == 12
    
    def test_monthly_mean_ghi_length(self):
        """Should have 12 monthly mean GHI values."""
        epw = load_epw()
        assert len(epw.monthly_mean_ghi) == 12
    
    def test_winter_temp_lower_than_summer(self):
        """Winter months should be colder than summer."""
        epw = load_epw()
        
        jan_temp = epw.monthly_mean_temp[0]
        jul_temp = epw.monthly_mean_temp[6]
        
        assert jul_temp > jan_temp + 10  # At least 10°C warmer
    
    def test_monthly_ghi_sum_equals_annual(self):
        """Sum of monthly GHI should approximately equal annual."""
        epw = load_epw()
        
        monthly_sum = np.sum(epw.monthly_total_ghi)
        annual = epw.annual_ghi_kwh_m2
        
        # Should be within 1% (rounding differences)
        assert abs(monthly_sum - annual) / annual < 0.01


class TestHeatingDegreeDays:
    """Tests for HDD calculation."""
    
    def test_hdd18_positive(self):
        """HDD18 should be positive in Sweden."""
        epw = load_epw()
        assert epw.heating_degree_days_18 > 0
    
    def test_hdd18_reasonable_range(self):
        """HDD18 should be in reasonable range for Göteborg."""
        epw = load_epw()
        
        # Göteborg typically 3200-3600 HDD18
        assert 3000 < epw.heating_degree_days_18 < 4000


class TestGetIrradiance:
    """Tests for irradiance retrieval function."""
    
    def test_get_ghi(self):
        """Can retrieve GHI for date range."""
        irr = get_irradiance(
            start_date=date(2024, 6, 21),
            end_date=date(2024, 6, 22),
            component="ghi",
        )
        
        assert len(irr) == 24
        assert np.max(irr) > 0  # Some irradiance on midsummer
    
    def test_get_dni(self):
        """Can retrieve DNI for date range."""
        irr = get_irradiance(
            start_date=date(2024, 6, 21),
            end_date=date(2024, 6, 22),
            component="dni",
        )
        
        assert len(irr) == 24
    
    def test_get_dhi(self):
        """Can retrieve DHI for date range."""
        irr = get_irradiance(
            start_date=date(2024, 6, 21),
            end_date=date(2024, 6, 22),
            component="dhi",
        )
        
        assert len(irr) == 24
    
    def test_invalid_component_raises(self):
        """Invalid component should raise ValueError."""
        with pytest.raises(ValueError):
            get_irradiance(
                start_date=date(2024, 6, 21),
                end_date=date(2024, 6, 22),
                component="invalid",
            )


class TestGetMonthlyIrradiance:
    """Tests for monthly irradiance retrieval."""
    
    def test_returns_dict(self):
        """Should return dictionary with expected keys."""
        result = get_monthly_irradiance()
        
        assert isinstance(result, dict)
        assert "monthly_mean_ghi_w_m2" in result
        assert "monthly_total_ghi_kwh_m2" in result
        assert "annual_ghi_kwh_m2" in result
        assert "latitude" in result
        assert "location" in result
    
    def test_monthly_lists_length(self):
        """Monthly lists should have 12 values."""
        result = get_monthly_irradiance()
        
        assert len(result["monthly_mean_ghi_w_m2"]) == 12
        assert len(result["monthly_total_ghi_kwh_m2"]) == 12


class TestMultipleEPWFiles:
    """Tests for loading different EPW files."""
    
    def test_load_stockholm_if_exists(self):
        """Stockholm EPW should load if available."""
        epw_path = DEFAULT_EPW.parent / "stockholm.epw"
        if epw_path.exists():
            epw = load_epw(epw_path)
            assert "STOCKHOLM" in epw.location.upper()
            assert 59.0 < epw.latitude < 60.0
            assert len(epw.temp_c) == 8760
            assert len(epw.ghi_wh_m2) == 8760
    
    def test_load_lulea_if_exists(self):
        """Luleå EPW should load if available."""
        epw_path = DEFAULT_EPW.parent / "lulea.epw"
        if epw_path.exists():
            epw = load_epw(epw_path)
            assert "LULEA" in epw.location.upper()
            assert 65.0 < epw.latitude < 66.0
            assert len(epw.temp_c) == 8760
            assert len(epw.ghi_wh_m2) == 8760
    
    def test_lulea_colder_than_goteborg(self):
        """Luleå should be significantly colder than Göteborg."""
        lulea_path = DEFAULT_EPW.parent / "lulea.epw"
        if lulea_path.exists():
            goteborg = load_epw()
            lulea = load_epw(lulea_path)
            
            assert lulea.annual_mean_temp < goteborg.annual_mean_temp - 3
            assert lulea.heating_degree_days_18 > goteborg.heating_degree_days_18 + 1000


class TestEPWSummary:
    """Tests for EPW summary method."""
    
    def test_summary_returns_string(self):
        """Summary should return a string."""
        epw = load_epw()
        summary = epw.summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 50
    
    def test_summary_contains_location(self):
        """Summary should contain location name."""
        epw = load_epw()
        summary = epw.summary()
        
        assert epw.location in summary
    
    def test_summary_contains_key_values(self):
        """Summary should contain key weather statistics."""
        epw = load_epw()
        summary = epw.summary()
        
        assert "HDD18" in summary
        assert "GHI" in summary
        assert "°C" in summary
