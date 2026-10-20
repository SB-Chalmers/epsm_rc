"""Public standalone calculations validate once and preserve scientific controls."""

import numpy as np
import pytest

from epsm_rc import Building5R1C, create_swedish_apartment, create_swedish_villa
from epsm_rc.parametric import ParametricBuilding, get_5r1c_params
from epsm_rc.tabula import BuildingTABULA, ClimateData


def test_sky_switch_reaches_all_thermostat_trials():
    weather = np.full(24, -5.0)
    disabled = Building5R1C().simulate(weather, include_sky_radiation=False)
    reference = Building5R1C(delta_theta_er=0).simulate(weather)
    enabled = Building5R1C().simulate(weather)
    np.testing.assert_array_equal(disabled["heating_demand"], reference["heating_demand"])
    assert enabled["heating_demand"].sum() > disabled["heating_demand"].sum()


@pytest.mark.parametrize("dt_hours", [1.0, 0.5])
def test_setback_clock_is_independent_of_ground_and_uses_elapsed_hours(dt_hours):
    count = int(4 / dt_hours)
    building = Building5R1C(
        occupancy_schedule=np.array([False, True, False, True]),
        t_set_heating_setback=16,
        delta_theta_er=0,
    )
    actual = building.simulate(np.full(count, -5.0), dt_hours=dt_hours)
    reference = Building5R1C(
        occupancy_schedule=building.occupancy_schedule,
        t_set_heating_setback=16,
        delta_theta_er=0,
    )
    reference.reset()
    powers = [
        reference.calc_heating_demand(-5, dt_hours=dt_hours, hour_of_year=int(i * dt_hours))
        for i in range(count)
    ]
    np.testing.assert_allclose(actual["heating_demand"], powers, rtol=0, atol=1e-10)
    assert actual["t_air"][0] < 21


def test_supplied_radiation_channels_are_forwarded():
    building = Building5R1C(
        solar_method="detailed", shading_factor=1, non_perpendicular_incidence_factor=1
    )
    ghi = np.full(8760, 300.0)
    dni, dhi = np.zeros(8760), ghi.copy()
    actual = building.calc_solar_gains_annual(ghi, dni=dni, dhi=dhi)
    calculator = building._get_solar_calculator()
    expected = calculator.calculate_hourly_solar_gains(
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        window_areas=building._get_window_areas_by_orientation(),
        g_value=building.g_value,
        frame_factor=building.frame_factor,
    )
    np.testing.assert_array_equal(actual, expected)


def test_epw_default_ground_matches_explicit_boundary_on_first_and_repeated_runs():
    from importlib.resources import files

    from epsm_rc.iso13790 import SWEDISH_DEFAULT_GROUND_TEMPS

    pytest.importorskip("epsm_core", reason="Strict EPW loading requires platform extra")
    weather = files("epsm_rc").joinpath("data/epw/goteborg.epw")
    automatic = Building5R1C()
    first = automatic.simulate_with_epw(weather)
    repeated = automatic.simulate_with_epw(weather)
    explicit = Building5R1C(
        ground_temps_monthly=SWEDISH_DEFAULT_GROUND_TEMPS.copy()
    ).simulate_with_epw(weather)

    # The selected boundary values and all other physical inputs are identical;
    # neither how the defaults were supplied nor a prior run may change demand.
    for actual in (first, repeated):
        for name in ("heating_demand", "t_air", "t_m", "ground_temps_monthly"):
            np.testing.assert_array_equal(actual[name], explicit[name])
        assert actual["heating_demand_kwh"] == explicit["heating_demand_kwh"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"floor_area_m2": 0},
        {"u_walls": -0.1},
        {"ach_infiltration": -0.1},
        {"h_is": 0},
        {"heat_recovery_efficiency": 1.1},
        {"shading_factor": np.nan},
        {"t_set_heating": 27, "t_set_cooling": 26},
        {"ground_temps_monthly": [np.nan] * 12},
    ],
)
def test_invalid_model_values_fail_at_construction(kwargs):
    with pytest.raises(ValueError):
        Building5R1C(**kwargs)


@pytest.mark.parametrize(
    "weather,kwargs",
    [
        ([-5, np.nan], {}),
        ([[-5, -5]], {}),
        ([], {}),
        ([-5, -5], {"solar_gains": [0]}),
        ([-5, -5], {"internal_gains": [0, np.inf]}),
        ([-5, -5], {"dt_hours": 0}),
        ([-5, -5], {"dt_hours": np.inf}),
        ([-5, -5], {"max_heating_power": -1}),
        ([-5, -5], {"max_cooling_power": np.nan}),
    ],
)
def test_invalid_series_leave_temperatures_unchanged(weather, kwargs):
    building = Building5R1C()
    building.reset(18)
    before = building.t_air, building.t_s, building.t_m
    with pytest.raises(ValueError):
        building.simulate(weather, **kwargs)
    assert (building.t_air, building.t_s, building.t_m) == before


def test_public_edits_refresh_physics_and_invalid_edits_fail_before_advance():
    building = Building5R1C()
    building.u_walls = 1.0
    actual = building.simulate(np.full(12, -5.0))
    reference = Building5R1C(u_walls=1.0).simulate(np.full(12, -5.0))
    np.testing.assert_array_equal(actual["heating_demand"], reference["heating_demand"])
    before = building.t_air, building.t_s, building.t_m
    building.ach_ventilation = -1
    with pytest.raises(ValueError):
        building.calc_heating_demand(-5)
    assert (building.t_air, building.t_s, building.t_m) == before


def test_zero_envelope_and_unlimited_capacity_remain_supported():
    result = Building5R1C(
        wall_area_m2=0,
        roof_area_m2=0,
        floor_area_ground_m2=0,
        window_area_m2=0,
        u_walls=0,
        u_roof=0,
        u_floor=0,
        u_windows=0,
    ).simulate([-5], max_heating_power=np.inf, max_cooling_power=-np.inf)
    assert np.isfinite(result["heating_demand"]).all()


def test_validation_runs_once_and_prepared_kernels_do_not_revalidate(monkeypatch):
    building = Building5R1C()
    calls = []
    original = building._validate_inputs

    def tracked():
        calls.append(1)
        original()

    monkeypatch.setattr(building, "_validate_inputs", tracked)
    building.simulate(np.full(24, -5.0))
    assert len(calls) == 1
    building._prepare_deterministic_hourly()
    calls.clear()
    building.calc_heating_demand(-5)
    assert calls == []


@pytest.mark.parametrize("factory", [create_swedish_villa, create_swedish_apartment])
def test_supplied_noop_factory_flag_warns(factory):
    with pytest.warns(DeprecationWarning, match="use_config.*no effect"):
        factory(use_config=False)


def test_noop_mass_class_warns_but_explicit_capacity_remains_available():
    with pytest.warns(DeprecationWarning, match="thermal_mass_class.*no effect"):
        building = ParametricBuilding(thermal_mass_class="HEAVY", thermal_capacitance_j_m2k=200000)
    assert get_5r1c_params(building)["thermal_capacitance_j_m2k"] == 200000


def test_invalid_radiation_channels_fail_before_solar_initialization():
    building = Building5R1C(solar_method="detailed")
    with pytest.raises(ValueError):
        building.calc_solar_gains_annual(np.ones(8760), dni=np.ones(10), dhi=np.ones(8760))
    assert building._solar_calc is None


def test_cooling_capacity_magnitude_retains_signed_compatibility():
    arguments = {"t_out": 40, "solar_gains": 3000}
    first, second = Building5R1C(), Building5R1C()
    first.reset(35)
    second.reset(35)
    positive = first.calc_heating_demand(**arguments, max_cooling_power=1000)
    signed = second.calc_heating_demand(**arguments, max_cooling_power=-1000)
    assert positive == signed == -1000


def test_parametric_cooling_setpoint_and_rotated_rectangle():
    ordinary = ParametricBuilding(length_m=20, width_m=10, t_cooling=29)
    assert get_5r1c_params(ordinary)["t_set_cooling"] == 29
    pytest.importorskip("epsm_core", reason="Rotated geometry requires platform extra")
    rotated = ParametricBuilding(length_m=20, width_m=10, orientation_deg=90)
    assert rotated.wall_area_north == pytest.approx(ordinary.wall_area_east)
    assert rotated.wall_area_east == pytest.approx(ordinary.wall_area_north)


def test_tabula_refreshes_after_public_edit():
    building = BuildingTABULA()
    building.u_wall = 1.0
    climate = ClimateData.sweden_zone3()
    assert building.calc_annual_heating_demand(climate) == pytest.approx(
        BuildingTABULA(u_wall=1.0).calc_annual_heating_demand(climate)
    )


@pytest.mark.parametrize("model", [Building5R1C(), BuildingTABULA(), ParametricBuilding()])
def test_public_model_inspection_is_bounded(model):
    assert len(repr(model)) <= 160
    assert "\n" not in repr(model)
    assert model.summary()
