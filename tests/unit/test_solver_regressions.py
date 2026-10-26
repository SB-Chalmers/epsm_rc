"""Regressions for solver edge cases: zero ventilation, window trials and gain inputs."""

from importlib.resources import files

import numpy as np
import pytest

from epsm_rc import Building5R1C, estimate_ground_temps_from_air


def standalone(native, **parameters):
    building = Building5R1C(**parameters)
    if not native:
        # An instance attribute disables native dispatch, as for user overrides.
        building.step = building.step
    return building


@pytest.mark.parametrize("native", [True, False], ids=["native", "python"])
def test_zero_ventilation_is_the_continuous_limit_of_small_airflow(native):
    results = {}
    for ach in (0.0, 1e-9):
        building = standalone(
            native,
            ach_infiltration=ach,
            ach_ventilation=0.0,
            internal_gains_w_per_m2=0.0,
            delta_theta_er=0.0,
        )
        results[ach] = building.simulate(np.zeros(3000))
    zero, small = results[0.0], results[1e-9]
    np.testing.assert_allclose(zero["heating_demand"], small["heating_demand"], rtol=1e-6)
    # Heat supplied to the air node must still warm the surface and mass nodes.
    assert zero["t_s"][-1] == pytest.approx(small["t_s"][-1], abs=1e-6)
    assert zero["t_m"][-1] > 15


@pytest.mark.parametrize("native", [True, False], ids=["native", "python"])
def test_zero_ventilation_surface_and_mass_receive_air_node_heat(native):
    building = standalone(native, ach_infiltration=0.0, ach_ventilation=0.0)
    building.reset(20)
    phi_ia, phi_st, phi_m = 1000.0, 200.0, 300.0
    h_ms, h_w = building._h_tr_ms, building._h_tr_w
    expected_t_s = (h_ms * 20 + phi_st + h_w * 0 + phi_ia) / (h_ms + h_w)
    assert building.calc_t_s(20, 0, phi_ia, phi_st) == pytest.approx(expected_t_s)
    expected_phi_m_tot = (
        phi_m + building._h_tr_em * 0 + h_ms * (phi_st + h_w * 0 + phi_ia) / (h_ms + h_w)
    )
    assert building.calc_phi_m_tot(0, phi_ia, phi_st, phi_m) == pytest.approx(expected_phi_m_tot)


def test_platform_zero_infiltration_without_supply_matches_small_infiltration():
    core = pytest.importorskip("epsm_core")
    from epsm_rc import RCSettings, simulate_building

    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    settings = RCSettings.from_policy(
        "epsm_rc_5r1c_v2",
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
    )
    annual = []
    for ach in (0.0, 1e-6):
        building = core.Building(preset="illustrative-reference")
        building.hvac = building.hvac.with_changes(
            airflow=core.AirflowSpec(
                topology="S",
                envelope_infiltration_ach=ach,
                mechanical_vent_l_s_m2=0.0,
                infiltration_coefficients=(1, 0, 0, 0),
            )
        )
        result = simulate_building(building, weather=weather, settings=settings)
        annual.append(result.scalars["space_heating_demand_kwh"])
    assert annual[0] == pytest.approx(annual[1], rel=1e-4)


@pytest.mark.parametrize("native", [True, False], ids=["native", "python"])
def test_window_decision_is_held_across_thermostat_trials(native):
    building = standalone(
        native, ventilation_behaviour=1.0, ach_window_open=3.0, t_set_heating=20, t_set_cooling=24
    )
    building.reset(26.0)
    cooled = 0
    for hour in range(48):
        daytime = 8 < hour % 24 < 18
        demand = building.calc_heating_demand(
            22.0 + 4 * (hour % 24 > 8),
            solar_gains=6000.0 if daytime else 0.0,
            max_cooling_power=float("inf"),
            hour_of_year=hour,
        )
        if demand < 0:
            cooled += 1
            assert building.t_air == pytest.approx(24.0, abs=1e-6)
    assert cooled


def test_simulate_with_epw_applies_detailed_shading_factors():
    pytest.importorskip("epsm_core", reason="Strict EPW loading requires platform extra")
    weather = files("epsm_rc").joinpath("data/epw/goteborg.epw")
    open_sky = Building5R1C(window_area_south=20).simulate_with_epw(weather)
    overhang = Building5R1C(window_area_south=20, f_shading_overhang=0.5).simulate_with_epw(weather)
    np.testing.assert_allclose(overhang["solar_gains"], 0.5 * open_sky["solar_gains"])


def test_strict_iso_does_not_halve_solar_gains():
    building = Building5R1C(delta_theta_er=0.0, convective_fraction_internal=0.5)
    strict = building.calc_heat_flows(0.0, internal_gains=0.0, solar_gains=1000.0, strict_iso=True)
    flexible = building.calc_heat_flows(
        0.0, internal_gains=0.0, solar_gains=1000.0, strict_iso=False
    )
    np.testing.assert_allclose(strict, flexible)


def test_ground_estimate_uses_phase_shift_argument():
    default = estimate_ground_temps_from_air(7.0)
    assert default == estimate_ground_temps_from_air(7.0, phase_shift_months=0.98)
    shifted = estimate_ground_temps_from_air(7.0, phase_shift_months=1.98)
    # A one-month larger offset moves the whole profile one month earlier.
    assert shifted == pytest.approx(default[1:] + default[:1], abs=0.11)
    with pytest.raises(ValueError, match="phase_shift_months"):
        estimate_ground_temps_from_air(7.0, phase_shift_months=float("nan"))
