"""Standalone weather uses explicit selection and shared identity/calendar checks."""

from datetime import date

import numpy as np
import pytest

from epsm_rc import Building5R1C, ParametricBuilding, SolarGainsCalculator
from epsm_rc.weather import (
    DEFAULT_EPW,
    get_irradiance,
    get_monthly_irradiance,
    get_temperature,
    load_epw,
)


@pytest.fixture
def epw_path(tmp_path):
    path = tmp_path / "local.epw"
    path.write_bytes(DEFAULT_EPW.read_bytes())
    return path


@pytest.mark.parametrize(
    "operation",
    [
        lambda: load_epw(),
        lambda: get_temperature(date(2023, 1, 1), date(2023, 1, 2)),
        lambda: get_irradiance(date(2023, 1, 1), date(2023, 1, 2)),
        lambda: get_monthly_irradiance(),
    ],
)
def test_missing_weather_selection_rejected(operation):
    with pytest.raises(ValueError, match="explicit|preset"):
        operation()


@pytest.mark.parametrize("alteration", ["truncate", "duplicate", "invalid_ghi"])
def test_invalid_weather_is_not_repaired(epw_path, alteration):
    lines = epw_path.read_text().splitlines()
    if alteration == "truncate":
        lines.pop()
    elif alteration == "duplicate":
        lines[9] = lines[8]
    else:
        row = lines[8].split(",")
        row[13] = "nan"
        lines[8] = ",".join(row)
    epw_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="calendar|invalid|finite"):
        load_epw(epw_path)


def test_valid_records_retained_and_monthly_statistics_exact(epw_path):
    epw = load_epw(epw_path)
    rows = [line.split(",") for line in epw_path.read_text().splitlines()[8:]]
    np.testing.assert_array_equal(epw.temp_c, [float(row[6]) for row in rows])
    np.testing.assert_array_equal(epw.ghi_wh_m2, [float(row[13]) for row in rows])
    for month in range(1, 13):
        selected = [row for row in rows if int(row[1]) == month]
        assert epw.monthly_mean_temp[month - 1] == pytest.approx(
            np.mean([float(row[6]) for row in selected])
        )
    assert len(epw.temp_c) == 8760
    assert len(epw.sha256) == 64
    assert epw.calendar_year == 2023


def test_cache_rechecks_checksum_and_stale_weather_rejected(epw_path):
    core = pytest.importorskip("epsm_core")
    captured = core.Weather.from_epw(epw_path)
    original = load_epw(epw_path)
    again = load_epw(epw_path)
    assert original is not again
    np.testing.assert_array_equal(original.temp_c, again.temp_c)
    lines = epw_path.read_text().splitlines()
    row = lines[8].split(",")
    row[6] = "12.3"
    lines[8] = ",".join(row)
    epw_path.write_text("\n".join(lines) + "\n")
    updated = load_epw(epw_path)
    assert updated.sha256 != original.sha256
    assert updated.temp_c[0] == 12.3
    with pytest.raises(ValueError, match="changed|fingerprint"):
        load_epw(captured)


def test_explicit_preset_and_core_weather_supported(epw_path):
    core = pytest.importorskip("epsm_core")
    assert "GOTEBORG" in load_epw(preset="goteborg").location.upper()
    assert load_epw(core.Weather.from_epw(epw_path)).temp_c.shape == (8760,)
    assert get_temperature(date(2023, 1, 1), date(2023, 1, 2), preset="goteborg").shape == (144,)
    assert get_irradiance(date(2023, 1, 1), date(2023, 1, 2), preset="goteborg").shape == (24,)
    assert "annual_ghi_kwh_m2" in get_monthly_irradiance(preset="goteborg")
    with pytest.raises(ValueError, match="one|both|either"):
        load_epw(epw_path, preset="goteborg")


def test_complete_leap_year_rejected_with_explicit_calendar_error(epw_path):
    lines = epw_path.read_text().splitlines()
    leap_day = []
    for row in lines[8:]:
        fields = row.split(",")
        if fields[1:3] == ["2", "28"]:
            fields[2] = "29"
            leap_day.append(",".join(fields))
    march = next(i for i, row in enumerate(lines[8:], 8) if row.split(",")[1:3] == ["3", "1"])
    lines[7] = "DATA PERIODS,1,1,Data,Monday,1/1/2024,12/31/2024"
    lines[march:march] = leap_day
    epw_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="[Ll]eap.*calendar|calendar.*[Ll]eap"):
        load_epw(epw_path)


def test_solar_and_simulation_use_strict_loader(epw_path):
    epw_path.write_text("\n".join(epw_path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="calendar"):
        SolarGainsCalculator.from_epw(epw_path)
    building = Building5R1C()
    with pytest.raises(ValueError, match="calendar"):
        building.simulate_with_epw(epw_path)
    assert building.ground_temps_monthly is None  # Validation precedes solver mutation.


def test_solar_parser_cannot_change_captured_weather(epw_path, monkeypatch):
    from epsm_rc import solar

    original_parser = solar.read_epw

    def changed_parser(*args, **kwargs):
        parsed = original_parser(*args, **kwargs)
        epw_path.write_text(epw_path.read_text().replace("COMMENTS 1", "COMMENTS 1 changed", 1))
        return parsed

    monkeypatch.setattr(solar, "read_epw", changed_parser)
    with pytest.raises(ValueError, match="changed|fingerprint"):
        SolarGainsCalculator.from_epw(epw_path)


def test_legacy_parameters_have_no_selected_weather():
    assert ParametricBuilding().epw_path is None


def test_removed_default_weather_requires_explicit_preset():
    from epsm_rc import weather

    with pytest.raises(ValueError, match="preset"):
        _ = weather.default_weather


def test_seasonal_lookup_uses_month_day_without_leap_shift():
    epw = load_epw(preset="goteborg")
    ordinary = epw.get_temperature(date(2023, 7, 1), date(2023, 7, 2), 60)
    same_day = epw.get_temperature(date(2024, 7, 1), date(2024, 7, 2), 60)
    np.testing.assert_array_equal(ordinary, same_day)
    with pytest.raises(ValueError, match="calendar|February 29"):
        epw.get_temperature(date(2024, 2, 29), date(2024, 3, 1))
