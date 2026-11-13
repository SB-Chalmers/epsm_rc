"""ISO 52016-1 zone network: balances, limits and the shared-Building selection."""

from importlib.resources import files

import numpy as np
import pytest

from epsm_rc import RCSettings
from epsm_rc.iso52016 import (
    H_CI,
    H_RI,
    Network52016,
    ThermalZone,
    Zone52016,
    ZoneElement,
    classify_mass,
    mass_distribution,
    solve_thermostats,
)


def _elements():
    return (
        ZoneElement(
            "wall",
            100,
            "opaque",
            "exterior",
            "horizontal",
            2.0,
            150_000,
            "D",
            solar_absorptance=0.6,
            sky_view_factor=0.5,
        ),
        ZoneElement(
            "window", 20, "window", "exterior", "horizontal", 0.5, 0, "D", sky_view_factor=0.5
        ),
        ZoneElement(
            "roof",
            50,
            "opaque",
            "exterior",
            "upwards",
            4.0,
            300_000,
            "I",
            solar_absorptance=0.6,
            sky_view_factor=1.0,
        ),
        ZoneElement(
            "floor",
            50,
            "opaque",
            "ground",
            "downwards",
            1.0,
            400_000,
            "IE",
            boundary_resistance_m2_k_w=1.5,
        ),
        ZoneElement("internal", 100, "opaque", "adiabatic", "horizontal", 0.5, 100_000, "D"),
    )


def test_mass_classes_conserve_capacity():
    for name in ("I", "E", "IE", "D", "M"):
        assert mass_distribution(name, 120_000).sum() == pytest.approx(120_000)


def test_mass_class_follows_insulation_position():
    concrete, insulation = (0.1, 400_000.0), (3.0, 2_000.0)
    assert classify_mass([insulation, concrete]) == "I"  # external insulation
    assert classify_mass([concrete, insulation]) == "E"  # internal insulation
    assert classify_mass([concrete, insulation, concrete]) == "IE"
    assert classify_mass([(0.5, 100_000.0), (0.5, 100_000.0)]) == "D"


@pytest.mark.parametrize("control", ["air", "operative"])
def test_hourly_energy_balance_closes(control):
    zone = Zone52016(
        _elements(), floor_area_m2=50, air_volume_m3=125, thermal_bridge_w_k=10, control=control
    )
    built = zone._built
    state = zone.initial_state(20)
    rng = np.random.default_rng(1)
    for hour in range(500):
        theta_e = 10 * np.sin(hour / 24 * 2 * np.pi) + rng.normal()
        sun = max(0.0, 500 * np.sin(hour / 24 * 2 * np.pi))
        absorbed = np.full(5, sun)
        h_ve = 30 + 10 * rng.random()
        forcing = zone.forcing_vector(theta_e, 5, 1000, 1.6 * sun, absorbed)
        new, power, _, _ = zone.step(state, forcing, h_ve, 20, 24, np.inf, -np.inf)
        inflow = (10 + h_ve) * (theta_e - new[0]) + 1000 + 1.6 * sun + power
        for element, start in zip(zone.elements, built["outer"]):
            boundary = theta_e if element.boundary == "exterior" else 5
            inflow += element.area_m2 * element.outer_conductance() * (boundary - new[start])
            if element.boundary == "exterior":
                inflow -= element.area_m2 * element.sky_view_factor * 4.14 * 11
                if element.kind == "opaque":
                    inflow += element.area_m2 * element.solar_absorptance * sun
        stored = (built["c_dt"] * (new - state)).sum()
        assert inflow == pytest.approx(stored, abs=1e-6)
        state = new


def test_thermostat_holds_control_temperature_and_respects_limits():
    zone = Zone52016(_elements(), floor_area_m2=50, air_volume_m3=125, control="operative")
    state = zone.initial_state(20)
    forcing = zone.forcing_vector(-10, 5, 0, 0, np.zeros(5))
    state, power, _, operative = zone.step(state, forcing, 30, 20, 26, np.inf, -np.inf)
    assert power > 0 and operative == pytest.approx(20)
    _, limited, _, operative = zone.step(state, forcing, 30, 20, 26, 100.0, -np.inf)
    assert limited == 100.0 and operative < 20
    _, off, _, _ = zone.step(state, forcing, 30, 20, 26, 0.0, -np.inf)
    assert off == 0.0


def test_single_wall_steady_loss_matches_series_resistance():
    wall = ZoneElement("wall", 10, "opaque", "exterior", "horizontal", 2.0, 50_000, "D")
    zone = Zone52016((wall,), floor_area_m2=10, air_volume_m3=25, control="air", delta_theta_er_k=0)
    state = zone.initial_state(20)
    forcing = zone.forcing_vector(0, 0, 0, 0, np.zeros(1))
    for _ in range(24 * 30):
        state, power, _, _ = zone.step(state, forcing, 0.0, 20, 26, np.inf, -np.inf)
    # One surface: its radiative exchange is with itself, so only convection remains.
    u = 1 / (1 / H_CI["horizontal"] + 2.0 + 1 / (20.0 + 4.14))
    assert power == pytest.approx(10 * u * 20, rel=1e-9)
    assert H_RI > 0


def test_invalid_elements_are_rejected():
    with pytest.raises(ValueError, match="boundary resistance"):
        ZoneElement("g", 1, "opaque", "ground", "downwards", 1.0, 0, "D")
    with pytest.raises(ValueError, match="windows"):
        ZoneElement("w", 1, "window", "adiabatic", "horizontal", 1.0, 0, "D")


def test_settings_default_to_the_choices_closest_to_energyplus():
    from epsm_rc.options import ControlTemperature, HeatRecovery, ThermalModel

    settings = RCSettings()
    assert settings.thermal_model is ThermalModel.ISO52016_1
    assert settings.control_temperature is ControlTemperature.AIR
    assert settings.heat_recovery is HeatRecovery.WHEN_BENEFICIAL
    assert RCSettings(thermal_model=ThermalModel.ISO13790_5R1C).thermal_model == "iso13790_5r1c"
    configuration = settings.effective_configuration()
    assert configuration["methods"]["thermal_model"] == "iso52016_1"
    assert "mass_area_factors" in configuration["inactive_coefficients"]
    with pytest.warns(DeprecationWarning):
        assert RCSettings(thermal_model="iso52016_1_hourly") == settings
    with pytest.warns(DeprecationWarning):
        legacy = RCSettings(heat_recovery_control="continuous")
    assert legacy.heat_recovery is HeatRecovery.ALWAYS
    with pytest.raises(ValueError, match="choose one of"):
        RCSettings(thermal_model="iso99999")
    with pytest.raises(ValueError):
        RCSettings(control_temperature="radiant")
    with pytest.raises(TypeError):
        RCSettings(heat_recovery="always", heat_recovery_control="continuous")


def test_shared_building_runs_both_models(reference_building):
    core = pytest.importorskip("epsm_core")
    from epsm_rc import simulate_building

    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    common = dict(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    old = simulate_building(
        reference_building,
        weather=weather,
        settings=RCSettings(thermal_model="iso13790_5r1c", **common),
    )
    new = simulate_building(reference_building, weather=weather, settings=RCSettings(**common))
    assert old.run_id != new.run_id
    assert set(old.hourly) == set(new.hourly)
    heating = new.scalars["space_heating_demand_kwh"]
    assert np.isfinite(heating) and heating > 0
    # Same building and forcing: the networks differ, annual heating stays comparable.
    assert heating == pytest.approx(old.scalars["space_heating_demand_kwh"], rel=0.25)
    assert new.hourly["indoor_temperature_c"].min() > 15


def test_recovery_bypass_only_removes_recovery(reference_building):
    core = pytest.importorskip("epsm_core")
    from epsm_rc import simulate_building

    with pytest.raises(ValueError):
        RCSettings(heat_recovery="sometimes")
    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    common = dict(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    assert reference_building.hvac.airflow.sensible_recovery_effectiveness > 0
    for model in ("iso13790_5r1c", "iso52016_1"):
        continuous = simulate_building(
            reference_building,
            weather=weather,
            settings=RCSettings(thermal_model=model, heat_recovery="always", **common),
        )
        bypass = simulate_building(
            reference_building,
            weather=weather,
            settings=RCSettings(thermal_model=model, **common),
        )
        old = continuous.scalars["space_heating_demand_kwh"]
        new = bypass.scalars["space_heating_demand_kwh"]
        # Bypassed hours lose recovery, so heating cannot fall; the heating
        # season keeps recovery, so the change stays small.
        assert old <= new <= 1.1 * old


def test_native_kernel_matches_python_recurrence(reference_building, monkeypatch):
    core = pytest.importorskip("epsm_core")
    from epsm_rc import _iso52016_platform as adapter
    from epsm_rc import simulate_building

    if adapter._native_kernel() is None:
        pytest.skip("native extension without the ISO 52016-1 kernel")
    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    settings = RCSettings(
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
    )
    original = adapter.run_network
    results = {}
    for native in (False, True):
        monkeypatch.setattr(
            adapter, "run_network", lambda *a, _n=native, **k: original(*a, native=_n, **k)
        )
        results[native] = simulate_building(reference_building, weather=weather, settings=settings)
    for name in ("space_heating_demand_kwh", "cooling_demand_kwh", "indoor_temperature_c"):
        np.testing.assert_allclose(
            results[True].hourly[name], results[False].hourly[name], rtol=0, atol=1e-9
        )


def test_native_5r1c_bypass_matches_python_recurrence(reference_building, monkeypatch):
    core = pytest.importorskip("epsm_core")
    from epsm_rc import iso13790, simulate_building

    pytest.importorskip("epsm_rc._core_rs")
    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    settings = RCSettings(
        thermal_model="iso13790_5r1c",
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
    )
    calls = []
    original = iso13790._can_use_native
    monkeypatch.setattr(
        iso13790, "_can_use_native", lambda *a, **k: calls.append(original(*a, **k)) or calls[-1]
    )
    native = simulate_building(reference_building, weather=weather, settings=settings)
    assert any(calls)  # the native 5R1C recurrence ran
    monkeypatch.setattr(iso13790, "_can_use_native", lambda *a, **k: False)
    python = simulate_building(reference_building, weather=weather, settings=settings)
    for name in ("space_heating_demand_kwh", "cooling_demand_kwh", "indoor_temperature_c"):
        np.testing.assert_allclose(native.hourly[name], python.hourly[name], rtol=0, atol=1e-9)


def _two_zone_network():
    floor = ZoneElement(
        "floor",
        50,
        "opaque",
        "zone",
        "downwards",
        1.0,
        200_000,
        "D",
        zone=0,
        adjacent_zone=1,
        adjacent_direction="downwards",
    )
    roof = ZoneElement(
        "roof",
        50,
        "opaque",
        "exterior",
        "upwards",
        4.0,
        100_000,
        "I",
        solar_absorptance=0.6,
        sky_view_factor=1.0,
    )
    wall = ZoneElement("wall", 80, "opaque", "exterior", "horizontal", 2.0, 150_000, "D")
    window = ZoneElement("window", 10, "window", "exterior", "horizontal", 0.5, 0, "D")
    lower_wall = ZoneElement(
        "lower",
        60,
        "opaque",
        "ground",
        "horizontal",
        1.5,
        300_000,
        "IE",
        boundary_resistance_m2_k_w=1.0,
        zone=1,
    )
    lower_floor = ZoneElement(
        "slab",
        50,
        "opaque",
        "ground",
        "downwards",
        1.0,
        400_000,
        "D",
        boundary_resistance_m2_k_w=2.0,
        zone=1,
    )
    zones = (
        ThermalZone("main", 50, 125, thermal_bridge_w_k=5, control="air"),
        ThermalZone("lower", 50, 100, conditioned=False, internal_capacity_j_m2_k=0),
    )
    return Network52016(
        (floor, roof, wall, window, lower_wall, lower_floor), zones, zone_bridges=((0, 1, 3.0),)
    )


def test_two_zone_energy_balance_closes_per_network():
    network = _two_zone_network()
    built = network._built
    state = network.initial_state(15)
    rng = np.random.default_rng(2)
    for hour in range(300):
        theta_e = 5 * np.sin(hour / 24 * 2 * np.pi) + rng.normal()
        sun = max(0.0, 400 * np.sin(hour / 24 * 2 * np.pi))
        absorbed = np.full(len(network.elements), sun)
        h_ve = np.array([30 + 5 * rng.random(), 10 * rng.random()])
        forcing = network.forcing_vector(theta_e, [8, 6], [800, 200], [1.2 * sun, 0], absorbed)
        new, power, air, _ = network.step(
            state, forcing, h_ve, [20, 20], [24, 24], [np.inf, 0], [-np.inf, 0]
        )
        assert power[1] == 0 and air[0] >= 20 - 1e-9
        inflow = h_ve @ (theta_e - new[:2]) + 5 * (theta_e - new[0]) + 1000 + 1.2 * sun
        inflow += power.sum()
        for element, start in zip(network.elements, built["outer"]):
            if element.boundary == "exterior":
                inflow += element.area_m2 * element.outer_conductance() * (theta_e - new[start])
                inflow -= element.area_m2 * element.sky_view_factor * 4.14 * 11
                if element.kind == "opaque":
                    inflow += element.area_m2 * element.solar_absorptance * sun
            elif element.boundary == "ground":
                boundary = 8 if element.zone == 0 else 6
                inflow += element.area_m2 * element.outer_conductance() * (boundary - new[start])
        stored = (built["c_dt"] * (new - state)).sum()
        assert inflow == pytest.approx(stored, abs=1e-6)
        state = new


def test_single_zone_network_matches_zone_wrapper():
    zone = Zone52016(_elements(), floor_area_m2=50, air_volume_m3=125, control="air")
    forcing = zone.forcing_vector(-5, 5, 300, 100, np.full(5, 50.0))
    state = zone.initial_state(20)
    new, power, air, operative = zone.step(state, forcing, 40, 21, 25, np.inf, -np.inf)
    assert air == pytest.approx(21) and power > 0
    assert zone.network.zone_count == 1 and np.isfinite(operative)


def test_coupled_thermostats_hold_both_setpoints_and_limits():
    gain = np.array([[2e-3, 4e-4], [5e-4, 3e-3]])
    power, temperature = solve_thermostats(
        [15.0, 14.0], gain, [20.0, 18.0], [26.0, 26.0], [np.inf, np.inf], [-np.inf, -np.inf]
    )
    assert temperature == pytest.approx([20.0, 18.0]) and (power > 0).all()
    power, temperature = solve_thermostats(
        [15.0, 14.0], gain, [20.0, 18.0], [26.0, 26.0], [1000.0, np.inf], [-np.inf, -np.inf]
    )
    assert power[0] == 1000.0 and temperature[0] < 20 and temperature[1] == pytest.approx(18)
    # A zone inside its band floats while its neighbour cools.
    power, temperature = solve_thermostats(
        [30.0, 22.0], gain, [20.0, 18.0], [26.0, 26.0], [np.inf, np.inf], [-np.inf, -np.inf]
    )
    assert power[1] == 0 and temperature[0] == pytest.approx(26) and 18 <= temperature[1] <= 26


def test_zone_boundary_requires_an_adjacent_zone():
    with pytest.raises(ValueError, match="adjacent zone"):
        ZoneElement("f", 1, "opaque", "zone", "downwards", 1.0, 0, "D")
    with pytest.raises(ValueError, match="only zone boundaries"):
        ZoneElement("f", 1, "opaque", "adiabatic", "downwards", 1.0, 0, "D", adjacent_zone=1)


def _zoned_building(basement=True, attic=True):
    from epsm_core import (
        Building,
        Footprint,
        GeometrySpec,
        RoofSpec,
        ScheduleSpec,
        UnconditionedExchangeSpec,
        ZoneUseSpec,
    )

    geometry = dict(storeys=2, storey_height_m=3)
    if attic:
        geometry["roof"] = RoofSpec(shape="gable", pitch_deg=30)
    if basement:
        geometry.update(basement_depth_m=2.5, basement_conditioned=True)
    building = Building(
        preset="illustrative-reference",
        geometry=GeometrySpec.from_footprint(Footprint.rectangle(10, 8), **geometry),
        unconditioned_exchange=(
            (UnconditionedExchangeSpec(zone_kind="attic", outdoor_air_ach=0.5, source="test"),)
            if attic
            else ()
        ),
    )
    if basement:
        building.zone_uses = (
            ZoneUseSpec(
                zone_kind="basement",
                use=building.operation.updated(heating_setpoint=ScheduleSpec.constant(18)),
                source="test",
            ),
        )
    return building


def test_basement_and_attic_run_as_zones_of_one_network(monkeypatch):
    core = pytest.importorskip("epsm_core")
    from epsm_rc import _iso52016_platform as adapter
    from epsm_rc import simulate_building

    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    common = dict(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    building = _zoned_building()
    with pytest.raises(ValueError, match="unconditioned_zone"):
        simulate_building(
            building,
            weather=weather,
            settings=RCSettings(cold_attic_policy="quasi_steady_buffer", **common),
        )
    with pytest.raises(ValueError, match="quasi_steady_buffer"):
        simulate_building(
            building, weather=weather, settings=RCSettings(thermal_model="iso13790_5r1c", **common)
        )
    original = adapter.run_network
    results = {}
    for native in (False, True):
        if native and adapter._native_kernel() is None:
            break
        monkeypatch.setattr(
            adapter, "run_network", lambda *a, _n=native, **k: original(*a, native=_n, **k)
        )
        results[native] = simulate_building(
            building, weather=weather, settings=RCSettings(**common)
        )
    result = results[False]
    methods = result.diagnostics["effective_configuration"]["methods"]
    assert methods["thermal_domain"] == "iso52016_1_multizone_main_and_conditioned_basement"
    attic = result.hourly["attic_air_temperature_c"]
    basement = result.hourly["basement_indoor_temperature_c"]
    # The attic sits between outdoor air (Göteborg mean about 8 °C) and the
    # heated rooms; the basement holds 18 °C.
    assert 8 < attic.mean() < result.hourly["main_indoor_temperature_c"].mean()
    assert basement.min() > 18 - 1e-6
    exchange = result.hourly["interzone_heat_to_basement_w"]
    np.testing.assert_allclose(exchange, -result.hourly["interzone_heat_to_main_w"])
    reference = simulate_building(
        building,
        weather=weather,
        settings=RCSettings(
            thermal_model="iso13790_5r1c", cold_attic_policy="quasi_steady_buffer", **common
        ),
    )
    assert result.scalars["space_heating_demand_kwh"] == pytest.approx(
        reference.scalars["space_heating_demand_kwh"], rel=0.15
    )
    if True in results:
        for name in result.hourly:
            np.testing.assert_allclose(
                results[True].hourly[name], result.hourly[name], rtol=0, atol=1e-8
            )


def test_attic_without_basement_reports_attic_temperature():
    core = pytest.importorskip("epsm_core")
    from epsm_rc import simulate_building

    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    result = simulate_building(
        _zoned_building(basement=False),
        weather=weather,
        settings=RCSettings(
            airflow_policy="sensible_only_unlimited_supply",
            ground_policy="conductance_multiplier",
        ),
    )
    assert result.units["attic_air_temperature_c"] == "C"
    assert result.diagnostics["attic_reduction"]["effective_method"] == "unconditioned_zone"
