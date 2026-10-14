"""Physical balance and public Core-to-RC handoff regressions."""

from importlib.resources import files

import numpy as np
import pytest

from epsm_rc import RCSettings, simulate_building
from epsm_rc.platform import _Platform5R1C


def test_sensible_recovery_reduces_mechanical_losses_once():
    # 300 m3, 0.2 ACH infiltration and 0.5 ACH mechanical flow. With 80%
    # recovery and a 20 K temperature difference, losses are 400 W + 200 W.
    model = _Platform5R1C(
        floor_area_m2=120,
        ceiling_height_m=2.5,
        ach_infiltration=0.2,
        ach_ventilation=0.5,
        heat_recovery_efficiency=0.8,
    )
    model.t_air = 20
    loss = model._h_ve * (model.t_air - model.calc_supply_air_temperature(0))
    assert loss == pytest.approx(600)


def test_recovery_does_not_reduce_infiltration_without_mechanical_flow():
    model = _Platform5R1C(
        floor_area_m2=120,
        ceiling_height_m=2.5,
        ach_infiltration=0.2,
        ach_ventilation=0,
        heat_recovery_efficiency=0.8,
    )
    model.t_air = 20
    loss = model._h_ve * (model.t_air - model.calc_supply_air_temperature(0))
    assert loss == pytest.approx(400)


def test_actual_building_calls_start_fresh_and_preserve_common_delivery(tmp_path):
    core = pytest.importorskip("epsm_core")
    building = core.Building()
    before = building.snapshot().sha256
    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    first = simulate_building(building, weather=weather, directory=tmp_path / "first")
    second = simulate_building(building, weather=weather)
    assert building.snapshot().sha256 == before
    assert len(first.time_axis) == 8760
    assert first.run_id == second.run_id
    for name, values in first.hourly.items():
        np.testing.assert_array_equal(values, second.hourly[name])
        assert np.isfinite(values).all()
        if name.endswith("kwh"):
            assert first.units[name] == "kWh"
            assert first.scalars[name] == pytest.approx(values.sum())
    heating = first.scalars["space_heating_demand_kwh"]
    assert heating > 0
    delivery = building.hvac.delivery.heating
    # This named reference has district heat, no COP curve and no losses.
    assert delivery.heat_pump_cop is None
    assert delivery.net_distribution_loss_kwh_m2_year == 0
    assert first.scalars["space_heating_carrier_input_kwh"] == pytest.approx(
        heating / delivery.efficiency
    )
    airflow = building.hvac.airflow
    fan_kwh = (
        building.geometry.conditioned_floor_area_m2
        * airflow.mechanical_vent_l_s_m2
        * sum(airflow.mechanical_vent_fraction.hourly(weather.calendar_year))
        * delivery.specific_fan_power_w_per_l_s
        / 1000
    )
    assert first.scalars["ventilation_fan_electricity_kwh"] == pytest.approx(fan_kwh)
    with np.load(first.artifacts["hourly"]) as saved:
        np.testing.assert_array_equal(saved["time_axis"], first.time_axis)
        np.testing.assert_array_equal(
            saved["space_heating_demand_kwh"], first.hourly["space_heating_demand_kwh"]
        )


def test_reduction_settings_validate_fractions_and_depth():
    for parameters in (
        {"effective_mass_depth_m": 0},
        {"effective_mass_depth_m": float("nan")},
        {"ground_coupling_factor": 1.1},
        {"ground_reflectance": -0.1},
    ):
        with pytest.raises(ValueError):
            RCSettings(**parameters)
