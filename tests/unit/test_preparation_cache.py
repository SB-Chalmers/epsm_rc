"""Preparation reuse must never reuse a thermal state or conceal input edits."""

from contextlib import nullcontext
from importlib.resources import files

import numpy as np
import pytest
from epsm_core import ScheduleSpec, Weather

from epsm_rc import PreparationCache, RCSettings, simulate_building


def settings():
    return RCSettings.from_policy(
        "epsm_rc_5r1c_v1",
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
    )


def assert_same(a, b):
    assert a.run_id == b.run_id
    assert a.scalars == b.scalars
    assert a.diagnostics == b.diagnostics
    for key in a.hourly:
        np.testing.assert_array_equal(a.hourly[key], b.hourly[key])


def test_preparation_reuses_work_but_not_thermal_state(reference_building):
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    cache = PreparationCache(max_entries=32)
    expected = simulate_building(reference_building, weather=weather, settings=settings())
    for _ in range(2):
        actual = simulate_building(
            reference_building, weather=weather, settings=settings(), preparation_cache=cache
        )
        assert_same(expected, actual)
    assert cache.hits > 0
    assert len(cache) <= cache.max_entries
    reference_building.operation = reference_building.operation.updated(
        heating_setpoint=ScheduleSpec.constant(24)
    )
    edited = simulate_building(
        reference_building, weather=weather, settings=settings(), preparation_cache=cache
    )
    assert_same(edited, simulate_building(reference_building, weather=weather, settings=settings()))
    assert edited.run_id != expected.run_id
    cache.clear()
    assert len(cache) == cache.hits == cache.misses == 0


def test_preparation_cache_rejects_changed_weather(reference_building, synthetic_weather):
    weather = Weather.from_epw(synthetic_weather)
    cache = PreparationCache()
    simulate_building(
        reference_building, weather=weather, settings=settings(), preparation_cache=cache
    )
    synthetic_weather.write_text(synthetic_weather.read_text().replace("Controlled", "Altered"))
    with pytest.raises(ValueError, match="contents changed"):
        simulate_building(
            reference_building, weather=weather, settings=settings(), preparation_cache=cache
        )


def test_preparation_cache_is_bounded_and_copies_arrays():
    cache = PreparationCache(max_entries=1)
    first = cache.get_or_compute("example", {"input": 1}, lambda: np.array([1.0]))
    first[0] = 99
    assert cache.get_or_compute("example", {"input": 1}, lambda: None)[0] == 1
    cache.get_or_compute("example", {"input": 2}, lambda: np.array([2.0]))
    assert len(cache) == 1
    assert cache.misses == 2
    for size in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            PreparationCache(max_entries=size)


@pytest.mark.parametrize("batch", [False, True])
def test_source_identity_changes_invalidate_prepared_weather(
    reference_building, synthetic_weather, monkeypatch, batch
):
    from epsm_rc import platform

    weather = Weather.from_epw(synthetic_weather)
    cache = PreparationCache()
    with cache.batch() if batch else nullcontext():
        first = simulate_building(
            reference_building, weather=weather, settings=settings(), preparation_cache=cache
        )
        misses = cache.misses
        original = platform._implementation_identity()
        monkeypatch.setattr(
            platform, "_implementation_identity", lambda *args: original | {"test_source": "changed"}
        )
        second = simulate_building(
            reference_building, weather=weather, settings=settings(), preparation_cache=cache
        )
    assert second.run_id != first.run_id
    assert cache.misses > misses
