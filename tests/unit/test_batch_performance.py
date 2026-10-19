"""Result-preserving reductions of deterministic trial and preparation work."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from epsm_rc import Building5R1C, PreparationCache


@pytest.mark.parametrize("temperature,heating,cooling", [(-20, 0, 0), (45, np.inf, 0)])
def test_unavailable_conditioning_keeps_exact_free_state_with_one_step(
    temperature, heating, cooling, monkeypatch
):
    model = Building5R1C(floor_area_m2=100, ventilation_behaviour=0)
    model._prepare_deterministic_hourly()
    if temperature > 30:
        model.t_m = model.t_air = model.t_s = 35.0
    expected = deepcopy(model)
    expected.step(temperature, internal_gains=100)
    calls = []
    original = model.step

    def tracked(*args, **kwargs):
        calls.append(kwargs.get("energy_demand", 0.0))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "step", tracked)
    power = model.calc_heating_demand(
        temperature, internal_gains=100, max_heating_power=heating, max_cooling_power=cooling
    )
    assert power == 0
    assert (model.t_air, model.t_s, model.t_m) == (expected.t_air, expected.t_s, expected.t_m)
    assert calls == [0.0]


def test_readonly_preparation_shares_arrays_without_exposing_mutable_storage():
    cache = PreparationCache(max_entries=1)
    source = np.arange(12.0).reshape(3, 4)
    first = cache._get_or_compute_readonly("arrays", {}, lambda: {"channels": (source,)})
    source[:] = 99
    second = cache._get_or_compute_readonly("arrays", {}, lambda: pytest.fail("cache miss"))
    array = second["channels"][0]
    assert first is second
    np.testing.assert_array_equal(array, np.arange(12.0).reshape(3, 4))
    with pytest.raises(TypeError):
        second["channels"] = ()
    with pytest.raises(ValueError):
        array[0] = 2
    with pytest.raises(ValueError):
        array.setflags(write=True)
    with pytest.raises(ValueError):
        array.base.setflags(write=True)
    cache.get_or_compute("mutable", {}, lambda: [1])
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == cache.hits == cache.misses == 0


def test_readonly_preparation_rejects_mutable_object_arrays():
    cache = PreparationCache()
    with pytest.raises(TypeError, match="object"):
        cache._get_or_compute_readonly("objects", {}, lambda: np.array([{}], dtype=object))
    assert len(cache) == 0


def test_mutable_and_readonly_cache_entries_cannot_alias():
    cache = PreparationCache()
    ordinary = cache.get_or_compute("array", {}, lambda: np.array([1.0]))
    shared = cache._get_or_compute_readonly("array", {}, lambda: np.array([2.0]))
    ordinary[0] = 9
    assert shared[0] == 2
    assert cache.get_or_compute("array", {}, lambda: None)[0] == 1


def test_batch_source_identity_checks_edits_additions_and_deletions(tmp_path, monkeypatch):
    import hashlib
    import os
    from types import SimpleNamespace

    from epsm_rc import _identity

    root = tmp_path / "package"
    root.mkdir()
    source = root / "__init__.py"
    source.write_text("value = 1\n")
    monkeypatch.setattr(_identity, "import_module", lambda name: SimpleNamespace(__file__=source))
    cache = _identity.SourceFingerprintCache()
    expected = _identity.implementation_identity()
    digest = hashlib.sha256(b"__init__.py\0value = 1\n\0").hexdigest()
    assert set(expected["source_sha256"].values()) == {digest}
    first = _identity.implementation_identity(cache)
    assert first == expected
    reads = []
    original = Path.read_bytes

    def tracked(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    assert _identity.implementation_identity(cache) == first
    assert reads == []
    stamp = source.stat()
    source.write_text("value = 2\n")
    # A same-length edit with restored mtime must still be noticed through ctime.
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    edited = _identity.implementation_identity(cache)
    assert edited != first
    nested = root / "nested"
    nested.mkdir()
    extra = nested / "new.py"
    extra.write_text("value = 3\n")
    added = _identity.implementation_identity(cache)
    assert added != edited
    extra.unlink()
    assert _identity.implementation_identity(cache) == edited
    nested.rmdir()
    assert _identity.implementation_identity(cache) == edited
    original_version = _identity.metadata.version
    monkeypatch.setattr(
        _identity.metadata, "version",
        lambda name: "different" if name == "numpy" else original_version(name),
    )
    assert _identity.implementation_identity(cache) != edited


def test_batch_context_is_scoped_nested_and_released_after_failure():
    cache = PreparationCache()
    assert cache._source_cache is None
    with pytest.raises(RuntimeError), cache.batch() as active:
        assert active is cache
        first = cache._source_cache
        assert first is not None
        with cache.batch():
            assert cache._source_cache is first
        assert cache._source_cache is first
        raise RuntimeError("interrupted batch")
    assert cache._source_cache is None
    with cache.batch():
        assert cache._source_cache is not first


@pytest.mark.parametrize("prepared,windows,steps", [(False, 0, 3), (True, 1, 2)])
def test_trial_shortcut_does_not_change_standalone_or_window_opening(
    prepared, windows, steps, monkeypatch
):
    model = Building5R1C(floor_area_m2=100, ventilation_behaviour=windows)
    if prepared:
        model._prepare_deterministic_hourly()
    original = model.step
    calls = []

    def tracked(*args, **kwargs):
        calls.append(kwargs.get("energy_demand", 0.0))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "step", tracked)
    assert model.calc_heating_demand(-20, max_heating_power=0) == 0
    assert len(calls) == steps


@pytest.mark.parametrize("policy", ["epsm_rc_5r1c_v1", "epsm_rc_5r1c_v2"])
def test_batch_retains_exact_results_after_building_and_result_edits(reference_building, policy):
    from importlib.resources import files

    from epsm_core import ScheduleSpec, Weather

    from epsm_rc import RCSettings, simulate_building

    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    settings = RCSettings.from_policy(
        policy, airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    cache = PreparationCache()
    with cache.batch():
        for setpoint in (20, 24, 20):
            reference_building.operation = reference_building.operation.updated(
                heating_setpoint=ScheduleSpec.constant(setpoint)
            )
            actual = simulate_building(
                reference_building, weather=weather, settings=settings, preparation_cache=cache
            )
            expected = simulate_building(reference_building, weather=weather, settings=settings)
            assert actual.run_id == expected.run_id
            assert actual.diagnostics == expected.diagnostics
            assert actual.scalars == expected.scalars
            for key in actual.hourly:
                np.testing.assert_array_equal(actual.hourly[key], expected.hourly[key])
                actual.hourly[key][:] = 0
        cache.clear()
        assert len(cache) == cache.hits == cache.misses == 0


def test_batch_still_rejects_changed_weather(reference_building, tmp_path, synthetic_weather):
    from epsm_core import Weather

    from epsm_rc import RCSettings, simulate_building

    path = tmp_path / "weather.epw"
    path.write_bytes(synthetic_weather.read_bytes())
    weather = Weather.from_epw(path)
    settings = RCSettings(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    with PreparationCache().batch() as cache:
        simulate_building(
            reference_building, weather=weather, settings=settings, preparation_cache=cache
        )
        path.write_text(path.read_text().replace("Controlled", "Altered"))
        with pytest.raises(ValueError, match="contents changed"):
            simulate_building(
                reference_building, weather=weather, settings=settings, preparation_cache=cache
            )
