"""Run ownership and one building preparation across independent thermal domains."""

from copy import deepcopy
from importlib.resources import files

import numpy as np
import pytest
from epsm_core import Building, Footprint, GeometrySpec, Weather

from epsm_rc import RCSettings, load_epw, simulate_building


@pytest.mark.parametrize("basement", [False, True])
def test_each_run_prepares_one_snapshot_graph_and_weather(basement, monkeypatch):
    from epsm_core import pipeline
    from epsm_core import resolved as core_resolved

    from epsm_rc import platform

    geometry = GeometrySpec.from_footprint(
        Footprint.rectangle(10, 8),
        storeys=2,
        storey_height_m=3,
        basement_depth_m=2.5 if basement else 0,
        basement_conditioned=basement,
    )
    building = Building(preset="illustrative-reference", geometry=geometry)
    weather = Weather.from_epw(files("epsm_rc") / "data/epw/goteborg.epw")
    calls = {"snapshot_inputs": 0, "geometry": 0, "weather": 0}

    def count(owner, name, key):
        original = getattr(owner, name)

        def tracked(*args, **kwargs):
            calls[key] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, tracked)

    count(core_resolved, "resolve_snapshot_inputs", "snapshot_inputs")
    count(pipeline, "resolve_geometry", "geometry")
    count(platform, "read_epw", "weather")
    before = building.snapshot().sha256
    settings = RCSettings(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    first = simulate_building(building, weather=weather, settings=settings)
    assert calls == {"snapshot_inputs": 1, "geometry": 1, "weather": 1}
    assert building.snapshot().sha256 == before
    expected = deepcopy(first.hourly)
    first.hourly["space_heating_demand_kwh"][:] = -1
    second = simulate_building(building, weather=weather, settings=settings)
    for name, values in expected.items():
        np.testing.assert_array_equal(second.hourly[name], values)


def test_cached_weather_has_no_mutable_cross_caller_aliases():
    first = load_epw(preset="goteborg")
    expected = first.temp_c.copy()
    first.temp_c = np.full(8760, 100.0)
    first.monthly_mean_temp[:] = -100
    second = load_epw(preset="goteborg")
    np.testing.assert_array_equal(second.temp_c, expected)
    assert np.all(second.monthly_mean_temp > -100)
    with pytest.raises(ValueError):
        second.temp_c.setflags(write=True)


def test_unsupported_airflow_fails_before_weather_or_state_creation(monkeypatch, tmp_path):
    from epsm_rc import platform

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported inputs reached preparation")

    monkeypatch.setattr(platform, "read_epw", forbidden)
    monkeypatch.setattr(platform, "Building5R1C", forbidden)
    destination = tmp_path / "result"
    with pytest.raises(ValueError, match="airflow"):
        simulate_building(
            Building(preset="illustrative-reference"),
            weather="absent.epw",
            settings=RCSettings(),
            directory=destination,
        )
    assert not destination.exists()


def test_changed_weather_after_geometry_resolution_fails_before_state(monkeypatch, tmp_path):
    from epsm_core import resolved as core_resolved

    from epsm_rc import platform

    path = tmp_path / "weather.epw"
    path.write_bytes(files("epsm_rc").joinpath("data/epw/goteborg.epw").read_bytes())
    weather = Weather.from_epw(path)
    original = core_resolved.resolve_snapshot_inputs

    def change_weather_after_resolution(*args, **kwargs):
        inputs = original(*args, **kwargs)
        lines = path.read_text().splitlines()
        record = lines[8].split(",")
        record[6] = str(float(record[6]) + 1)
        lines[8] = ",".join(record)
        path.write_text("\n".join(lines) + "\n")
        return inputs

    def forbidden(*args, **kwargs):
        pytest.fail("Changed weather reached thermal state creation")

    monkeypatch.setattr(core_resolved, "resolve_snapshot_inputs", change_weather_after_resolution)
    monkeypatch.setattr(platform, "Building5R1C", forbidden)
    settings = RCSettings(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    with pytest.raises(ValueError, match="changed"):
        simulate_building(
            Building(preset="illustrative-reference"),
            weather=weather,
            settings=settings,
            directory=tmp_path / "result",
        )
    assert not (tmp_path / "result").exists()


def test_empty_building_readiness_precedes_unavailable_weather(monkeypatch, tmp_path):
    from epsm_rc import platform

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete building reached weather inspection or thermal state creation")

    monkeypatch.setattr(Weather, "from_epw", forbidden)
    monkeypatch.setattr(platform, "Building5R1C", forbidden)
    settings = RCSettings(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    with pytest.raises(ValueError, match="[Mm]issing|[Nn]ot ready|geometry"):
        simulate_building(
            Building(),
            weather=tmp_path / "missing.epw",
            settings=settings,
            directory=tmp_path / "result",
        )
    assert not (tmp_path / "result").exists()


def test_weather_representation_is_bounded_and_printable():
    weather = load_epw(preset="goteborg")
    weather.location = "\x00\n\x1b" * 100
    text = repr(weather)
    assert len(text) <= 160
    assert all(character.isprintable() for character in text)
