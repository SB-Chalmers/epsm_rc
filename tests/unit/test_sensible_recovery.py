"""Sensible recovery applies once to mechanical flow, with natural flows unrecovered."""

from importlib.resources import files

import numpy as np
import pytest
from shapely.geometry import Polygon

from epsm_rc import Building5R1C
from epsm_rc.platform import _Platform5R1C


def model(model_class=Building5R1C, mechanical=0.5, recovery=0.8):
    building = model_class(
        floor_area_m2=120, ceiling_height_m=2.5,
        ach_infiltration=0.2, ach_ventilation=mechanical,
        heat_recovery_efficiency=recovery,
    )
    building.t_air = 20.0
    return building


@pytest.mark.parametrize('model_class', [Building5R1C, _Platform5R1C])
def test_300_m3_recovery_loss_is_600_w(model_class):
    building = model(model_class)
    assert building._building_vol == 300
    assert building._h_ve * (20 - building.calc_supply_air_temperature(0)) == pytest.approx(600)


@pytest.mark.parametrize('model_class', [Building5R1C, _Platform5R1C])
def test_infiltration_and_actual_heating_unchanged_without_mechanical_flow(model_class):
    results = []
    for recovery in (0.0, 0.8, 1.0):
        building = model(model_class, mechanical=0.0, recovery=recovery)
        assert building._h_ve * (20 - building.calc_supply_air_temperature(0)) == pytest.approx(400)
        results.append(building.simulate(
            t_outdoor=np.zeros(48), solar_gains=np.zeros(48),
            internal_gains=np.zeros(48), include_sky_radiation=False,
        )['heating_demand'])
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])


def test_window_flow_is_unrecovered_and_mechanical_recovery_continues():
    building = model()
    building.ventilation_behaviour = 1.0
    building.ach_window_open = 0.3
    building.t_air = 30.0
    building.t_set_cooling = 26.0
    building._update_ventilation_with_window_opening(10.0)
    assert building._windows_open
    # 100 W/K per ACH, with 0.2 infiltration + 0.1 recovered mechanical + 0.3 window.
    assert building._h_ve == pytest.approx(60.0)
    assert building._h_ve * (30 - building.calc_supply_air_temperature(10)) == pytest.approx(1200)
    building.t_air = 20.0
    building._update_ventilation_with_window_opening(10.0)
    assert not building._windows_open
    assert building._h_ve == pytest.approx(30.0)


def test_total_flow_override_retains_known_mechanical_share():
    building = model()
    building._update_ventilation_coefficient(ach_total=0.9)
    assert building._h_ve == pytest.approx(50.0)
    assert building._h_ve * (20 - building.calc_supply_air_temperature(0)) == pytest.approx(1000)


@pytest.mark.parametrize('total', [0.1, -0.2, float('nan'), float('inf')])
def test_total_flow_override_cannot_omit_known_mechanical_flow(total):
    building = model()
    with pytest.raises(ValueError, match='mechanical|finite|nonnegative'):
        building._update_ventilation_coefficient(ach_total=total)


@pytest.mark.parametrize('topology,mechanical,recovery,natural,expected_loss', [
    ('FTX', 0.5, 0.8, 0.0, 600.0), ('S', 0.0, 0.0, 0.3, 1000.0),
])
def test_public_core_adapter_uses_same_recovery_with_unrecovered_natural_flow(
    topology, mechanical, recovery, natural, expected_loss, monkeypatch,
):
    core = pytest.importorskip('epsm_core')
    from epsm_rc import RCSettings, platform
    building = core.Building(preset='illustrative-reference')
    building.geometry = core.GeometrySpec.from_footprint(
        Polygon([(0, 0), (12, 0), (12, 10), (0, 10)]), storeys=1, storey_height_m=2.5,
    )
    airflow = core.AirflowSpec(
        topology=topology, envelope_infiltration_ach=0.2, natural_exchange_ach=natural,
        mechanical_vent_l_s_m2=mechanical * 300 / 3.6 / 120,
        sensible_recovery_effectiveness=recovery, infiltration_coefficients=(1, 0, 0, 0),
    )
    building.hvac = building.hvac.with_changes(airflow=airflow)
    captured = []

    def actual_solver(**kwargs):
        instance = Building5R1C(**kwargs)
        captured.append(instance)
        return instance

    monkeypatch.setattr(platform, 'Building5R1C', actual_solver)
    result = platform.simulate_building(
        building, weather=core.Weather.from_epw(files('epsm_rc').joinpath('data/epw/goteborg.epw')),
        settings=RCSettings.from_policy('epsm_rc_5r1c_v1',
            airflow_policy='sensible_only_unlimited_supply', ground_policy='conductance_multiplier'),
    )
    assert len(captured) == 1
    actual = captured[0]
    assert actual._building_vol == pytest.approx(300)
    assert actual._h_ve * (20 - actual.calc_supply_air_temperature(0)) == pytest.approx(expected_loss)
    assert result.scalars['space_heating_demand_kwh'] > 0
