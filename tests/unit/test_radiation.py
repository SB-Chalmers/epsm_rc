"""Radiation channels, calendars and explicit policy changes at the Core handoff."""

from datetime import datetime, timedelta
from importlib.resources import files

import numpy as np
import pytest
from epsm_core import Weather

from epsm_rc import PreparationCache, RCSettings, simulate_building
from epsm_rc.radiation import plane_irradiance


def test_legacy_standalone_keeps_direct_beam_projection():
    from pvlib import irradiance

    zenith, azimuth = np.array([30.1234, 72.1123]), np.array([173.4, 241.7])
    dni, ghi, dhi = np.array([700.0, 510.0]), np.array([600.0, 300.0]), np.array([100.0, 80.0])
    channels = plane_irradiance(
        90,
        180,
        zenith,
        azimuth,
        dni,
        ghi,
        dhi,
        albedo=0.2,
        dni_extra=np.full(2, 1367.0),
        airmass=np.array([1.2, 3.1]),
        legacy_standalone=True,
    )
    expected = irradiance.beam_component(90, 180, zenith, azimuth, dni)
    np.testing.assert_array_equal(channels["poa_direct"], expected)
    np.testing.assert_array_equal(
        channels["poa_global"],
        np.maximum(0, expected + channels["poa_sky_diffuse"] + channels["poa_ground_diffuse"]),
    )


@pytest.mark.parametrize("dhi", [np.array([100.0, 40.0, 0.0]), np.zeros(3)])
def test_plane_components_are_finite_conserved_and_detached(dhi):
    zenith = np.array([30.0, 89.9, 105.0])
    sun_azimuth = np.array([180.0, 260.0, 0.0])
    dni, ghi = np.array([700.0, 200.0, 0.0]), np.array([600.0, 30.0, 0.0])
    originals = [array.copy() for array in (zenith, sun_azimuth, dni, ghi, dhi)]
    for array in (zenith, sun_azimuth, dni, ghi, dhi):
        array.flags.writeable = False
    channels = plane_irradiance(
        90,
        180,
        zenith,
        sun_azimuth,
        dni,
        ghi,
        dhi,
        albedo=0.2,
        dni_extra=np.full(3, 1367.0),
        airmass=np.array([1.15, 36.0, np.nan]),
    )
    for values in channels.values():
        assert np.isfinite(values).all()
        assert values[-1] == 0
    np.testing.assert_allclose(
        channels["poa_global"],
        channels["poa_direct"] + channels["poa_sky_diffuse"] + channels["poa_ground_diffuse"],
        atol=1e-12,
    )
    for actual, original in zip((zenith, sun_azimuth, dni, ghi, dhi), originals):
        np.testing.assert_array_equal(actual, original)


def policy(name="epsm_rc_5r1c_v2", **overrides):
    return RCSettings.from_policy(
        name,
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
        **overrides,
    )


def test_policy_samples_midpoint_but_preserves_output_labels(
    reference_building, synthetic_weather, monkeypatch
):
    from epsm_rc import platform

    weather = Weather.from_epw(synthetic_weather)
    original = platform.solarposition.get_solarposition
    sampled = []

    def capture(times, *args, **kwargs):
        sampled.append(times)
        return original(times, *args, **kwargs)

    monkeypatch.setattr(platform.solarposition, "get_solarposition", capture)
    old = simulate_building(reference_building, weather=weather, settings=policy("epsm_rc_5r1c_v1"))
    new = simulate_building(reference_building, weather=weather, settings=policy())
    assert sampled[0][0].minute == 0
    assert sampled[1][0].minute == 30
    np.testing.assert_array_equal(old.time_axis, new.time_axis)
    # With explicitly zero solar, timing/optics cannot change thermal results.
    for name, values in old.hourly.items():
        np.testing.assert_array_equal(values, new.hourly[name])
    assert new.run_id != old.run_id
    assert new.diagnostics["radiation"]["window_optics_method"] == "simple_glazing_angular_v1"


def test_angular_channels_account_for_inward_absorption_and_cache(reference_building):
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    cache = PreparationCache()
    first = simulate_building(
        reference_building, weather=weather, settings=policy(), preparation_cache=cache
    )
    second = simulate_building(
        reference_building, weather=weather, settings=policy(), preparation_cache=cache
    )
    assert first.run_id == second.run_id
    assert first.diagnostics == second.diagnostics
    for name in first.hourly:
        np.testing.assert_array_equal(first.hourly[name], second.hourly[name])
    hourly = first.hourly
    assert hourly["window_inward_absorbed_solar_w"].sum() > 0
    np.testing.assert_allclose(
        hourly["window_solar_heat_gain_w"],
        hourly["window_transmitted_solar_w"] + hourly["window_inward_absorbed_solar_w"],
        rtol=2e-15,
        atol=2e-11,
    )
    for name, energy in first.diagnostics["radiation"]["annual_kwh"].items():
        assert energy == float(hourly[name].sum()) / 1000
    legacy = simulate_building(
        reference_building,
        weather=weather,
        settings=policy("epsm_rc_5r1c_v1"),
        preparation_cache=cache,
    )
    assert first.run_id != legacy.run_id
    assert first.scalars["space_heating_demand_kwh"] != legacy.scalars["space_heating_demand_kwh"]


def test_leap_year_and_fractional_timezone_keep_all_intervals(
    reference_building, synthetic_weather, tmp_path
):
    rows = synthetic_weather.read_text().splitlines()
    headers = rows[:8]
    headers[0] = headers[0].replace(",57,12,1,20", ",57,12,5.5,20")
    sample = rows[8].split(",")
    target = tmp_path / "leap.epw"
    with target.open("w") as stream:
        stream.write("\n".join(headers) + "\n")
        for hour in range(8784):
            stamp = datetime(2024, 1, 1) + timedelta(hours=hour)
            row = sample.copy()
            row[:4] = ["2024", str(stamp.month), str(stamp.day), str(stamp.hour + 1)]
            stream.write(",".join(row) + "\n")
    weather = Weather.from_epw(target, calendar_year=2024)
    assert weather.climate.time_zone_hours == 5.5
    result = simulate_building(reference_building, weather=weather, settings=policy())
    assert len(result.time_axis) == 8784
    assert result.time_axis[-1] == np.datetime64("2024-12-31T23")
    assert result.diagnostics["accepted"]
    assert all(len(values) == 8784 for values in result.hourly.values())


def test_unsupported_optics_fail_before_thermal_simulation(
    reference_building, synthetic_weather, monkeypatch
):
    from dataclasses import replace

    from epsm_rc import platform

    elements = dict(reference_building.construction_set.elements)
    window = next(role for role in elements if role.value == "WINDOW")
    elements[window] = replace(elements[window], u_value=8.0)
    reference_building.construction_set = replace(
        reference_building.construction_set, elements=elements
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported optics must fail before constructing a thermal solver")

    monkeypatch.setattr(platform, "Building5R1C", forbidden)
    with pytest.raises(ValueError, match="0 < U <= 7"):
        simulate_building(
            reference_building, weather=Weather.from_epw(synthetic_weather), settings=policy()
        )
