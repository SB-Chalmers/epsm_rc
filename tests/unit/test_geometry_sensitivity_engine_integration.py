"""Independent physical trends for new geometry in both real annual engines."""

import os
from pathlib import Path

import pytest
from epsm_core import (
    Building,
    Footprint,
    FoundationSpec,
    GeometrySpec,
    GroundTemperatureSpec,
    RoofSpec,
    ScheduleSpec,
    UnconditionedExchangeSpec,
    Weather,
    ZoneUseSpec,
    run_building,
)

from epsm_rc import RCSettings, simulate_building


@pytest.mark.energyplus
def test_warmer_basement_and_greater_attic_exchange_increase_heating(synthetic_weather, tmp_path):
    engine = os.environ.get("ENERGYPLUS_EXE")
    if not engine or not Path(engine).is_file():
        if os.environ.get("EPSM_REQUIRE_ENERGYPLUS") == "1":
            pytest.fail("Geometry release requires EnergyPlus 24.2")
        pytest.skip("EnergyPlus 24.2 is required for the geometry release comparison")
    weather = Weather.from_epw(synthetic_weather)
    settings = RCSettings.from_policy(
        "epsm_rc_5r1c_v2",
        airflow_policy="sensible_only_unlimited_supply",
        cold_attic_policy="quasi_steady_buffer",
    )
    outputs = {}
    for label, temperature, attic_ach in (
        ("reference", 18, 0.5),
        ("warmer_basement", 20, 0.5),
        ("more_attic_exchange", 18, 5),
    ):
        building = Building(
            preset="illustrative-reference",
            geometry=GeometrySpec.from_footprint(
                Footprint.rectangle(10, 8),
                storeys=3,
                storey_height_m=3,
                roof=RoofSpec(shape="gable", pitch_deg=30),
                basement_depth_m=2.5,
                basement_conditioned=True,
            ),
            unconditioned_exchange=(
                UnconditionedExchangeSpec(
                    zone_kind="attic",
                    outdoor_air_ach=attic_ach,
                    source="explicit synthetic attic sensitivity",
                ),
            ),
        )
        building.facade = building.facade.updated(default_wwr=0)
        building.foundation = FoundationSpec(method="prescribed_temperature")
        building.ground_temperature = GroundTemperatureSpec(
            monthly_temperatures_c=(15.0,) * 12, source="controlled synthetic ground"
        )
        building.operation = building.operation.updated(
            heating_setpoint=ScheduleSpec.constant(20),
            cooling_enabled=False,
            occupant_density_per_m2=0,
            equipment_gain_w_m2=0,
            lighting_gain_w_m2=0,
            dhw_kwh_m2_year=0,
        )
        building.zone_uses = (
            ZoneUseSpec(
                zone_kind="basement",
                use=building.operation.updated(heating_setpoint=ScheduleSpec.constant(temperature)),
                source="explicit synthetic basement thermostat",
            ),
        )
        snapshot = building.snapshot()
        detailed = run_building(
            snapshot,
            weather=weather,
            energyplus=engine,
            directory=tmp_path / label / "energyplus",
        )
        reduced = simulate_building(
            snapshot.to_building(),
            weather=weather,
            settings=settings,
            directory=tmp_path / label / "rc",
        )
        assert detailed.diagnostics["accepted"] and reduced.diagnostics["accepted"]
        assert building.snapshot().sha256 == snapshot.sha256
        outputs[label] = (detailed, reduced)
    for index in (0, 1):
        baseline = outputs["reference"][index].scalars["space_heating_demand_kwh"]
        assert baseline > 0
        assert outputs["warmer_basement"][index].scalars["space_heating_demand_kwh"] > baseline
        assert outputs["more_attic_exchange"][index].scalars["space_heating_demand_kwh"] > baseline
