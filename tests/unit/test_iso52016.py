"""ISO 52016-1 zone network: balances, limits and the shared-Building selection."""

from importlib.resources import files

import numpy as np
import pytest

from epsm_rc import RCSettings
from epsm_rc.iso52016 import (
    H_CI,
    H_RI,
    Zone52016,
    ZoneElement,
    classify_mass,
    mass_distribution,
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


def test_settings_select_model_by_policy_and_explicit_field():
    assert RCSettings().thermal_model == "iso13790_5r1c"
    iso = RCSettings.from_policy("epsm_rc_iso52016_v1")
    assert iso.thermal_model == "iso52016_1_hourly"
    assert iso.window_optics_method == "simple_glazing_angular_v1"
    explicit = RCSettings(thermal_model="iso52016_1_hourly")
    assert explicit.window_optics_method == "constant_shgc"
    configuration = explicit.effective_configuration()
    assert configuration["methods"]["thermal_model"] == "iso52016_1_hourly"
    assert "mass_area_factors" in configuration["inactive_coefficients"]
    with pytest.raises(ValueError):
        RCSettings(thermal_model="iso99999")
    with pytest.raises(ValueError):
        RCSettings(iso52016_control_temperature="radiant")


def test_shared_building_runs_both_models(reference_building):
    core = pytest.importorskip("epsm_core")
    from epsm_rc import simulate_building

    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    common = dict(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    old = simulate_building(reference_building, weather=weather, settings=RCSettings(**common))
    new = simulate_building(
        reference_building,
        weather=weather,
        settings=RCSettings(thermal_model="iso52016_1_hourly", **common),
    )
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
        RCSettings(heat_recovery_control="sometimes")
    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    common = dict(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    assert reference_building.hvac.airflow.sensible_recovery_effectiveness > 0
    for model in ("iso13790_5r1c", "iso52016_1_hourly"):
        continuous = simulate_building(
            reference_building,
            weather=weather,
            settings=RCSettings(thermal_model=model, **common),
        )
        bypass = simulate_building(
            reference_building,
            weather=weather,
            settings=RCSettings(
                thermal_model=model, heat_recovery_control="load_dependent_bypass", **common
            ),
        )
        old = continuous.scalars["space_heating_demand_kwh"]
        new = bypass.scalars["space_heating_demand_kwh"]
        # Bypassed hours lose recovery, so heating cannot fall; the heating
        # season keeps recovery, so the change stays small.
        assert old <= new <= 1.1 * old
