"""Shared explicit gain channels reach the real RC calculation."""
from importlib.resources import files

import numpy as np
import pytest

from epsm_core import GainPartitionSpec, ScheduleSpec, Weather
from epsm_rc import RCSettings, simulate_building


def partitions(**changes):
    values = dict(policy="explicit", people_fraction_radiant=.25,
        people_sensible_heat_fraction=.6, equipment_fraction_radiant=.2,
        equipment_fraction_latent=.1, lighting_fraction_radiant=.3,
        lighting_fraction_visible=.2, lighting_return_air_fraction=0)
    return GainPartitionSpec(**{**values, **changes})


def calculate(building, **changes):
    building.operation = building.operation.updated(gain_partitions=partitions(**changes))
    return simulate_building(building,
        weather=Weather.from_epw(files("epsm_rc") / "data/epw/goteborg.epw"),
        settings=RCSettings.from_policy("epsm_rc_5r1c_v1",
            airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"))


def test_people_sensible_fraction_changes_forcing_and_annual_heat(reference_building):
    low = calculate(reference_building, people_sensible_heat_fraction=0)
    high = calculate(reference_building, people_sensible_heat_fraction=1)
    assert low.scalars["space_heating_demand_kwh"] > high.scalars["space_heating_demand_kwh"]
    assert np.all(high.hourly["internal_gains_convective_w"] >= low.hourly["internal_gains_convective_w"])
    assert np.any(high.hourly["internal_gains_convective_w"] > low.hourly["internal_gains_convective_w"])
    assert low.run_id != high.run_id
    assert high.diagnostics["gain_reduction"]["aggregate_split_applied"] is False


def test_scheduled_heat_and_reference_area_reach_solver(reference_building):
    reference_building.load_reference_area_m2 = 100
    reference_building.operation = reference_building.operation.updated(
        occupant_density_per_m2=0, equipment_gain_w_m2=4,
        equipment_fraction=ScheduleSpec.constant(.5), lighting_gain_w_m2=2,
        equipment_heat_fraction_schedule=ScheduleSpec(values=(.5, .8) * 12))
    result = calculate(reference_building)
    assert result.hourly["internal_gains_convective_w"][:2] == pytest.approx((140, 200))
    assert result.hourly["internal_gains_radiant_w"][:2] == pytest.approx((140, 140))
    assert result.units["internal_gains_convective_w"] == "W"


@pytest.mark.parametrize("changes,match", [
    ({"people_sensible_heat_fraction": None}, "people"),
    ({"lighting_return_air_fraction": .1}, "return.air"),
])
def test_unsupported_gains_fail_before_solver(reference_building, changes, match, monkeypatch):
    from epsm_rc import platform

    reference_building.operation = reference_building.operation.updated(lighting_gain_w_m2=2)
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported gains reached solver")
    monkeypatch.setattr(platform, "Building5R1C", forbidden)
    with pytest.raises(ValueError, match=match):
        calculate(reference_building, **changes)


def test_inactive_engine_only_channels_are_accepted(reference_building):
    reference_building.operation = reference_building.operation.updated(
        occupant_density_per_m2=0, lighting_gain_w_m2=0)
    result = calculate(reference_building, people_sensible_heat_fraction=None,
                       lighting_return_air_fraction=.1)
    assert result.diagnostics["accepted"]


def test_explicit_channels_override_aggregate_coefficient(reference_building):
    from epsm_rc.platform import _gain_inputs

    reference_building.operation = reference_building.operation.updated(gain_partitions=partitions())
    results = [_gain_inputs(reference_building.operation, 100, 2023,
        RCSettings(convective_internal_gains_fraction=fraction)) for fraction in (0, 1)]
    for index in (0, 1, 2):
        np.testing.assert_array_equal(results[0][index], results[1][index])
    assert not results[0][3]["aggregate_split_applied"]
