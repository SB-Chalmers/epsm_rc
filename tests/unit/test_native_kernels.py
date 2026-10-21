"""Native boundary, dispatch and numerical regressions against Python trial steps."""

import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pytest

from epsm_rc import Building5R1C, _core_rs
from epsm_rc.coupled import step_domains
from epsm_rc.domains import HourlyInputs, _immutable_columns, _immutable_series
from epsm_rc.heating import DirectElectric


def domain_inputs(count=24):
    hours = np.arange(count, dtype=float)
    return {
        "temperatures": 5 + 20 * np.sin(hours / 4),
        "gains": np.full(count, 400.0),
        "convective": np.full(count, 200.0),
        "heating_setpoints": np.full(count, 20.0),
        "cooling_setpoints": np.full(count, 26.0),
        "wind": np.full(count, 2.0),
        "infiltration": np.full(count, 0.3),
        "natural": np.zeros(count),
        "ventilation": np.full(count, 0.5),
        "solar": np.maximum(0.0, 2000 * np.sin(hours / 6)),
        "opaque": np.zeros(count),
        "month_hours": np.full(count, 744, dtype=np.int64),
        "coefficients": (1.0, 0.01, 0.02, 0.0),
        "aggregate_split": False,
        "heating_limits_w": np.where(hours % 8 < 2, 0.0, np.inf),
        "cooling_limits_w": np.full(count, -1000.0),
    }


def model():
    result = Building5R1C(ground_temps_monthly=list(range(12)))
    result._prepare_deterministic_hourly()
    result.reset(18)
    return result


def hourly_inputs(building, inputs):
    return HourlyInputs(
        model=building,
        **{key: value for key, value in inputs.items() if not key.endswith("limits_w")},
    )


@pytest.mark.parametrize(
    "channel", [key for key, value in domain_inputs().items() if isinstance(value, np.ndarray)]
)
def test_single_domain_rejects_mismatched_lengths_without_changing_state(channel):
    inputs = domain_inputs()
    inputs[channel] = inputs[channel][:-1]
    building = model()
    before = building.t_air, building.t_s, building.t_m
    with pytest.raises(ValueError, match="length"):
        _core_rs.run_single_domain_hourly(building, **inputs)
    assert (building.t_air, building.t_s, building.t_m) == before


@pytest.mark.parametrize("domain", [0, 1])
@pytest.mark.parametrize("channel", ["temperatures", "solar", "month_hours", "cooling_limits_w"])
def test_coupled_domains_validate_both_input_lengths(domain, channel):
    inputs = [domain_inputs(), domain_inputs()]
    inputs[domain][channel] = inputs[domain][channel][:-1]
    buildings = [model(), model()]
    arguments = {
        f"{key}{index}": value for index, item in enumerate(inputs) for key, value in item.items()
    }
    with pytest.raises(ValueError, match="length"):
        _core_rs.run_coupled_domains_hourly(*buildings, **arguments, conductance_w_k=50.0)
    assert all((item.t_air, item.t_s, item.t_m) == (18, 18, 18) for item in buildings)


@pytest.mark.parametrize(
    "channel", ["solar_gains", "internal_gains", "solar_gains_opaque", "occupancy_schedule"]
)
def test_standalone_native_boundary_rejects_short_series(channel):
    building = Building5R1C()
    building.reset(18)
    arguments = {
        "t_outdoor": np.full(4, -5.0),
        "solar_gains": np.zeros(4),
        "internal_gains": np.zeros(4),
        "solar_gains_opaque": np.zeros(4),
        "occupancy_schedule": np.ones(4, dtype=bool),
        "dt_hours": 1.0,
        "max_heating_power": np.inf,
        "max_cooling_power": 0.0,
        "include_sky_radiation": True,
    }
    arguments[channel] = arguments[channel][:-1]
    with pytest.raises(ValueError, match="length|cover"):
        _core_rs.simulate_5r1c_hourly(building, **arguments)
    assert (building.t_air, building.t_s, building.t_m) == (18, 18, 18)


@pytest.mark.parametrize(
    "gains", [[0.0], [0.0, 0.0, 0.0], [[0.0, 0.0]], [np.nan, 0.0], [-1.0, 0.0]]
)
def test_heating_rejects_bad_series_before_reset(gains):
    building = Building5R1C()
    building.reset(18)
    with pytest.raises(ValueError):
        DirectElectric(building_model=building).heat_demand_w([-5.0, -5.0], solar_gains=gains)
    assert (building.t_air, building.t_s, building.t_m) == (18, 18, 18)


def test_subclass_demand_override_is_used_by_simulation_and_heating():
    class CustomBuilding(Building5R1C):
        def calc_heating_demand(self, *args, **kwargs):
            return 123.0

    building = CustomBuilding()
    np.testing.assert_array_equal(building.simulate([-5.0, -5.0])["heating_demand"], [123, 123])
    np.testing.assert_array_equal(
        DirectElectric(building_model=building).heat_demand_w([-5.0, -5.0]), [123, 123]
    )


@pytest.mark.parametrize("entry", ["simulate", "heating"])
def test_custom_reset_is_preserved(entry):
    class CustomBuilding(Building5R1C):
        def reset(self, t_initial=None):
            self.reset_calls = getattr(self, "reset_calls", 0) + 1
            super().reset(12)

    building = CustomBuilding()
    reference = Building5R1C()
    reference.reset(12)
    if entry == "simulate":
        expected = reference.calc_heating_demand(-5, max_heating_power=0)
        result = building.simulate([-5], max_heating_power=0)
        assert result["heating_demand"][0] == expected
        assert result["t_air"][0] == pytest.approx(reference.t_air)
    else:
        expected = reference.calc_heating_demand(-5)
        result = DirectElectric(building_model=building).heat_demand_w([-5], minutes_per_slot=60)
        assert result[0] == pytest.approx(expected)
    assert building.reset_calls == 1


def test_custom_active_setpoints_are_preserved():
    class CustomBuilding(Building5R1C):
        def get_active_setpoints(self, hour_of_year=None):
            return 23.0, 29.0

    result = CustomBuilding().simulate([-5.0, -5.0])
    np.testing.assert_allclose(result["t_air"], 23.0, atol=1e-10)


@pytest.mark.parametrize("entry", ["scalar", "simulate", "heating"])
@pytest.mark.parametrize("patch", ["subclass", "class", "instance"])
def test_step_overrides_are_preserved(monkeypatch, entry, patch):
    calls = []
    original = Building5R1C.step

    def tracked(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    if patch == "subclass":

        class CustomBuilding(Building5R1C):
            step = tracked

        building = CustomBuilding()
    else:
        building = Building5R1C()
        if patch == "class":
            monkeypatch.setattr(Building5R1C, "step", tracked)
        else:
            monkeypatch.setattr(building, "step", tracked.__get__(building))
    if entry == "scalar":
        building.calc_heating_demand(-5.0)
    elif entry == "simulate":
        building.simulate([-5.0])
    else:
        DirectElectric(building_model=building).heat_demand_w([-5.0])
    assert calls


def test_class_demand_patch_is_preserved(monkeypatch):
    monkeypatch.setattr(Building5R1C, "calc_heating_demand", lambda *args, **kwargs: 123.0)
    np.testing.assert_array_equal(Building5R1C().simulate([-5.0])["heating_demand"], [123])
    np.testing.assert_array_equal(
        DirectElectric(building_model=Building5R1C()).heat_demand_w([-5.0]), [123]
    )


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("windows", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("dt", [0.25, 1.0, 2.0])
def test_native_simulation_matches_python_trials(prepared, windows, dt):
    rng = np.random.default_rng(123)
    native = Building5R1C(
        ground_temps_monthly=rng.uniform(0, 15, 12).tolist(),
        occupancy_schedule=rng.integers(0, 2, 96).astype(bool),
        t_set_heating_setback=16,
        t_set_cooling_setback=29,
        ventilation_behaviour=windows,
        use_operative_temperature=prepared,
        ach_infiltration=0 if dt == 0.25 else 0.3,
        ach_ventilation=0 if dt == 0.25 else 0.5,
    )
    if prepared:
        native._prepare_deterministic_hourly()
    reference = deepcopy(native)
    # This forces the retained Python trial/reset algorithm and Python step.
    reference.step = reference.step
    forcing = dict(
        t_outdoor=rng.uniform(-20, 40, 48),
        solar_gains=rng.uniform(0, 8000, 48),
        internal_gains=rng.uniform(0, 1000, 48),
        solar_gains_opaque=rng.uniform(0, 200, 48),
        dt_hours=dt,
        max_heating_power=1000 if prepared else np.inf,
        max_cooling_power=-1000,
        include_sky_radiation=not prepared,
    )
    state = random.getstate()
    try:
        random.seed(123)
        expected = reference.simulate(**forcing)
        random.seed(123)
        actual = native.simulate(**forcing)
    finally:
        random.setstate(state)
    for key in expected:
        if expected[key].dtype == bool:
            np.testing.assert_array_equal(actual[key], expected[key])
        else:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(
        [native.t_air, native.t_s, native.t_m],
        [reference.t_air, reference.t_s, reference.t_m],
        rtol=1e-10,
        atol=1e-8,
    )


@pytest.mark.parametrize("immutable", [False, True])
def test_single_domain_matches_python_hourly_trial_runner(immutable):
    inputs = domain_inputs()
    if immutable:
        inputs = {
            key: _immutable_series(value, value.dtype) if isinstance(value, np.ndarray) else value
            for key, value in inputs.items()
        }
    native, reference = model(), model()
    reference.step = reference.step
    hourly = hourly_inputs(reference, inputs)
    powers, temperatures = [], []
    for hour in range(len(inputs["temperatures"])):
        powers.append(
            reference.calc_heating_demand(
                **hourly.at(hour),
                max_heating_power=inputs["heating_limits_w"][hour],
                max_cooling_power=inputs["cooling_limits_w"][hour],
            )
        )
        temperatures.append(reference.t_air)
    actual = _core_rs.run_single_domain_hourly(native, **inputs)
    np.testing.assert_allclose(actual, [powers, temperatures], rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("immutable", [False, True])
def test_coupled_domain_batch_matches_python_trial_runner(immutable):
    inputs = domain_inputs()
    if immutable:
        inputs = {
            key: _immutable_series(value, value.dtype) if isinstance(value, np.ndarray) else value
            for key, value in inputs.items()
        }
    native, reference = [model(), model()], [model(), model()]
    hourly = [hourly_inputs(item, inputs) for item in reference]
    solutions = []
    for hour in range(len(inputs["temperatures"])):
        solutions.append(
            step_domains(
                reference,
                [item.at(hour) for item in hourly],
                50.0,
                heating_limit_w=(inputs["heating_limits_w"][hour],) * 2,
                cooling_limit_w=(inputs["cooling_limits_w"][hour],) * 2,
            )
        )
    arguments = {f"{key}{index}": value for index in range(2) for key, value in inputs.items()}
    powers, temperatures, exchanges, residuals = _core_rs.run_coupled_domains_hourly(
        *native, **arguments, conductance_w_k=50.0
    )
    np.testing.assert_allclose(
        powers, np.array([item.power_w for item in solutions]).T, rtol=1e-10, atol=1e-8
    )
    np.testing.assert_allclose(
        temperatures, np.array([item.temperature_c for item in solutions]).T, rtol=1e-10, atol=1e-8
    )
    np.testing.assert_allclose(
        exchanges, np.array([item.exchange_w for item in solutions]).T, rtol=1e-10, atol=1e-8
    )
    np.testing.assert_array_equal(exchanges[0], -exchanges[1])
    assert np.max(residuals) < 1e-7


def test_parallel_simulations_keep_independent_states_and_match_serial_results():
    def simulate(temperature):
        return Building5R1C().simulate(np.full(8760, temperature))

    temperatures = [-10.0, 0.0, 10.0, 30.0]
    expected = [simulate(value) for value in temperatures]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(simulate, temperatures))
    for first, second in zip(actual, expected):
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


def test_prepared_storage_cannot_be_changed_through_an_alias():
    original = np.arange(24.0)
    frozen = _immutable_series(original)
    original[:] = -1
    np.testing.assert_array_equal(frozen, np.arange(24.0))
    with pytest.raises(ValueError):
        frozen.setflags(write=True)


def test_packed_columns_preserve_values_without_exposing_mutable_storage():
    originals = [np.arange(4.0), np.full(4, 2.0)]
    first, second = _immutable_columns(*originals)
    originals[0][:] = -1
    np.testing.assert_array_equal(first, np.arange(4.0))
    np.testing.assert_array_equal(second, np.full(4, 2.0))
    assert first.base is second.base
    with pytest.raises(ValueError):
        first.setflags(write=True)
    with pytest.raises(ValueError):
        first.base.setflags(write=True)


@pytest.mark.parametrize("alias", ["writable", "readonly", "subclass"])
def test_native_worker_snapshots_writable_aliases_and_releases_gil(alias):
    inputs = domain_inputs(200_000)
    original = inputs["temperatures"]
    for key, value in inputs.items():
        if isinstance(value, np.ndarray) and key != "temperatures":
            inputs[key] = _immutable_series(value, value.dtype)
    if alias == "readonly":
        inputs["temperatures"] = original.view()
        inputs["temperatures"].setflags(write=False)
    elif alias == "subclass":

        class MisleadingArray(np.ndarray):
            @property
            def base(self):
                return b"not the storage owner"

        inputs["temperatures"] = original.view(MisleadingArray)
    expected = _core_rs.run_single_domain_hourly(model(), **inputs)
    ready, begin, changed = threading.Event(), threading.Event(), threading.Event()

    def mutate():
        ready.set()
        begin.wait()
        original[:] = 45.0
        changed.set()

    thread = threading.Thread(target=mutate)
    thread.start()
    ready.wait()
    building = model()
    interval = sys.getswitchinterval()
    try:
        # The writer can acquire the GIL only when the native call detaches,
        # after its snapshot is complete. Keep Python's periodic handoff out
        # of that short interval so the test does not depend on a race.
        sys.setswitchinterval(1.0)
        begin.set()
        actual = _core_rs.run_single_domain_hourly(building, **inputs)
        progressed_during_call = changed.is_set()
    finally:
        sys.setswitchinterval(interval)
        begin.set()
        thread.join()
    assert progressed_during_call
    for first, second in zip(actual, expected):
        np.testing.assert_array_equal(first, second)
