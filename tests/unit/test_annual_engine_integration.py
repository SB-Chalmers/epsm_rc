"""Release gate: real annual engines consume one explicit physical snapshot."""
import os
from pathlib import Path

import numpy as np
import pytest

from epsm_core import (
    Building, Footprint, FoundationSpec, GainPartitionSpec, GeometrySpec,
    GroundTemperatureSpec, ScheduleSpec, Weather, run_building,
)
from epsm_rc import RCSettings, simulate_building


@pytest.mark.energyplus
@pytest.mark.parametrize("sensible", [0.0, 1.0])
def test_annual_shared_explicit_inputs(sensible, tmp_path, synthetic_weather):
    engine = os.environ.get("ENERGYPLUS_EXE")
    if not engine or not Path(engine).is_file():
        if os.environ.get("EPSM_REQUIRE_ENERGYPLUS") == "1":
            pytest.fail("Release integration requires EnergyPlus 24.2")
        pytest.skip("EnergyPlus 24.2 is required for annual integration")
    building = Building(preset="illustrative-reference", geometry=GeometrySpec.from_footprint(
        Footprint.rectangle(8, 6), storeys=1, storey_height_m=3))
    building.foundation = FoundationSpec(method="prescribed_temperature")
    building.ground_temperature = GroundTemperatureSpec(
        monthly_temperatures_c=(18.0,) * 12, source="explicit controlled integration boundary")
    building.load_reference_area_m2 = 40
    building.operation = building.operation.updated(
        gain_partitions=GainPartitionSpec(policy="explicit",
            people_fraction_radiant=.25, people_sensible_heat_fraction=sensible,
            equipment_fraction_radiant=.2, equipment_fraction_latent=.1,
            lighting_fraction_radiant=.3, lighting_fraction_visible=.2,
            lighting_return_air_fraction=0),
        equipment_fraction=ScheduleSpec.constant(.5),
        equipment_heat_fraction_schedule=ScheduleSpec(values=(.5, .8) * 12),
        lighting_gain_w_m2=1)
    weather = Weather.from_epw(synthetic_weather)
    snapshot = building.snapshot()
    physical = building.to_spec(weather).physical_hash()
    detailed = run_building(snapshot, weather=weather, directory=tmp_path / "energyplus", energyplus=engine)
    reduced = simulate_building(snapshot.to_building(), weather=weather, directory=tmp_path / "rc",
        settings=RCSettings.from_policy("epsm_rc_5r1c_v1", airflow_policy="sensible_only_unlimited_supply"))
    assert building.snapshot().sha256 == snapshot.sha256
    assert snapshot.to_spec(weather).physical_hash() == physical
    assert reduced.diagnostics["physical_inputs_sha256"] == physical
    assert detailed.diagnostics["building_snapshot_id"] == snapshot.sha256
    assert reduced.diagnostics["ground_reduction"]["monthly_temperatures_c"] == (18.0,) * 12
    np.testing.assert_array_equal(detailed.time_axis, reduced.time_axis)
    for result in (detailed, reduced):
        assert result.diagnostics["accepted"]
        assert len(result.time_axis) == 8760
        assert result.scalars["space_heating_demand_kwh"] > 0
        for name, values in result.hourly.items():
            assert np.isfinite(values).all()
            if name.endswith("kwh"):
                assert result.scalars[name] == pytest.approx(float(values.sum()), rel=.001, abs=.001)
