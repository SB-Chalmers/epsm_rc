"""Release gate: real annual engines consume one explicit physical snapshot."""

import os
from pathlib import Path

import numpy as np
import pytest
from epsm_core import (
    Building,
    Footprint,
    FoundationSpec,
    GainPartitionSpec,
    GeometrySpec,
    GroundTemperatureSpec,
    PartyWall,
    ScheduleSpec,
    Weather,
    run_building,
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
    building = Building(
        preset="illustrative-reference",
        geometry=GeometrySpec.from_footprint(
            Footprint.rectangle(8, 6), storeys=1, storey_height_m=3
        ),
    )
    building.foundation = FoundationSpec(method="prescribed_temperature")
    building.ground_temperature = GroundTemperatureSpec(
        monthly_temperatures_c=(18.0,) * 12, source="explicit controlled integration boundary"
    )
    building.load_reference_area_m2 = 40
    building.operation = building.operation.updated(
        gain_partitions=GainPartitionSpec(
            policy="explicit",
            people_fraction_radiant=0.25,
            people_sensible_heat_fraction=sensible,
            equipment_fraction_radiant=0.2,
            equipment_fraction_latent=0.1,
            lighting_fraction_radiant=0.3,
            lighting_fraction_visible=0.2,
            lighting_return_air_fraction=0,
        ),
        equipment_fraction=ScheduleSpec.constant(0.5),
        equipment_heat_fraction_schedule=ScheduleSpec(values=(0.5, 0.8) * 12),
        lighting_gain_w_m2=1,
    )
    weather = Weather.from_epw(synthetic_weather)
    snapshot = building.snapshot()
    physical = building.to_spec(weather).physical_hash()
    detailed = run_building(
        snapshot, weather=weather, directory=tmp_path / "energyplus", energyplus=engine
    )
    reduced = simulate_building(
        snapshot.to_building(),
        weather=weather,
        directory=tmp_path / "rc",
        settings=RCSettings.from_policy(
            "epsm_rc_5r1c_v1", airflow_policy="sensible_only_unlimited_supply"
        ),
    )
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
                assert result.scalars[name] == pytest.approx(
                    float(values.sum()), rel=0.001, abs=0.001
                )


@pytest.mark.energyplus
def test_partial_shared_wall_lowers_heating_in_both_real_engines(tmp_path, synthetic_weather):
    engine = os.environ.get("ENERGYPLUS_EXE")
    if not engine or not Path(engine).is_file():
        if os.environ.get("EPSM_REQUIRE_ENERGYPLUS") == "1":
            pytest.fail("Shared-wall release integration requires EnergyPlus 24.2")
        pytest.skip("EnergyPlus 24.2 is required for shared-wall annual integration")
    building = Building(
        preset="illustrative-reference",
        geometry=GeometrySpec.from_footprint(
            Footprint.rectangle(10, 8), storeys=2, storey_height_m=3
        ),
    )
    building.facade = building.facade.updated(default_wwr=0)
    building.foundation = FoundationSpec(method="prescribed_temperature")
    building.ground_temperature = GroundTemperatureSpec(
        monthly_temperatures_c=(18.0,) * 12, source="controlled test boundary"
    )
    building.operation = building.operation.updated(
        occupant_density_per_m2=0,
        equipment_gain_w_m2=0,
        lighting_gain_w_m2=0,
        heating_setpoint=ScheduleSpec.constant(20),
        cooling_enabled=False,
    )
    weather = Weather.from_epw(synthetic_weather)
    results = {}
    for label, contacts in (
        ("detached", ()),
        (
            "shared",
            (
                PartyWall(
                    facade_id="main.r0.e0",
                    neighbor_id="next",
                    neighbor_conditioning="unknown",
                    contact_confirmed=True,
                    assumption_source="explicit test adiabatic contact",
                    start_m=2,
                    end_m=7,
                    bottom_m=1,
                    top_m=4,
                ),
            ),
        ),
    ):
        building.geometry = building.geometry.updated(party_walls=contacts)
        snapshot = building.snapshot()
        detailed = run_building(
            snapshot, weather=weather, directory=tmp_path / label / "energyplus", energyplus=engine
        )
        reduced = simulate_building(
            snapshot.to_building(),
            weather=weather,
            directory=tmp_path / label / "rc",
            settings=RCSettings.from_policy(
                "epsm_rc_5r1c_v2", airflow_policy="sensible_only_unlimited_supply"
            ),
        )
        assert building.snapshot().sha256 == snapshot.sha256
        assert detailed.diagnostics["building_snapshot_id"] == snapshot.sha256
        assert (
            reduced.diagnostics["physical_inputs_sha256"]
            == snapshot.to_spec(weather).physical_hash()
        )
        np.testing.assert_array_equal(detailed.time_axis, reduced.time_axis)
        for result in (detailed, reduced):
            assert result.diagnostics["accepted"]
            assert len(result.time_axis) == 8760
            assert all(np.isfinite(values).all() for values in result.hourly.values())
        results[label] = (detailed, reduced)
    for index in (0, 1):
        assert (
            0
            < results["shared"][index].scalars["space_heating_demand_kwh"]
            < results["detached"][index].scalars["space_heating_demand_kwh"]
        )
    assert results["shared"][1].diagnostics["boundary_reduction"][
        "adiabatic_opaque_area_m2"
    ] == pytest.approx(15)


@pytest.mark.energyplus
@pytest.mark.parametrize("shape", ["flat", "gable", "hip", "mono_pitch"])
def test_conditioned_basement_and_finite_attic_share_physical_controls_in_real_engines(
    shape,
    tmp_path,
    synthetic_weather,
):
    from epsm_core import RoofSpec, UnconditionedExchangeSpec, ZoneUseSpec

    engine = os.environ.get("ENERGYPLUS_EXE")
    if not engine or not Path(engine).is_file():
        if os.environ.get("EPSM_REQUIRE_ENERGYPLUS") == "1":
            pytest.fail("Geometry release integration requires EnergyPlus 24.2")
        pytest.skip("EnergyPlus 24.2 is required for geometry annual integration")
    building = Building(
        preset="illustrative-reference",
        geometry=GeometrySpec.from_footprint(
            Footprint.rectangle(10, 8),
            storeys=3,
            storey_height_m=3,
            basement_depth_m=2.5,
            basement_conditioned=True,
            roof=RoofSpec(shape=shape),
        ),
        unconditioned_exchange=()
        if shape == "flat"
        else (
            UnconditionedExchangeSpec(
                zone_kind="attic", outdoor_air_ach=0.5, source="controlled attic exchange"
            ),
        ),
    )
    building.facade = building.facade.updated(default_wwr=0)
    building.foundation = FoundationSpec(method="prescribed_temperature")
    building.ground_temperature = GroundTemperatureSpec(
        monthly_temperatures_c=(18.0,) * 12, source="controlled synthetic boundary"
    )
    building.operation = building.operation.updated(
        occupant_density_per_m2=0,
        equipment_gain_w_m2=0,
        lighting_gain_w_m2=0,
        dhw_kwh_m2_year=0,
        heating_setpoint=ScheduleSpec.constant(20),
        cooling_enabled=False,
    )
    building.zone_uses = (
        ZoneUseSpec(
            zone_kind="basement",
            use=building.operation.updated(heating_setpoint=ScheduleSpec.constant(18)),
            source="controlled basement 18 C",
        ),
    )
    weather = Weather.from_epw(synthetic_weather)
    snapshot = building.snapshot()
    detailed = run_building(
        snapshot, weather=weather, directory=tmp_path / "energyplus", energyplus=engine
    )
    reduced = simulate_building(
        snapshot.to_building(),
        weather=weather,
        directory=tmp_path / "rc",
        settings=RCSettings.from_policy(
            "epsm_rc_5r1c_v2",
            airflow_policy="sensible_only_unlimited_supply",
            cold_attic_policy="quasi_steady_buffer",
        ),
    )
    assert building.snapshot().sha256 == snapshot.sha256
    assert (
        reduced.diagnostics["physical_inputs_sha256"] == snapshot.to_spec(weather).physical_hash()
    )
    assert detailed.diagnostics["building_snapshot_id"] == snapshot.sha256
    np.testing.assert_array_equal(detailed.time_axis, reduced.time_axis)
    np.testing.assert_allclose(reduced.hourly["main_indoor_temperature_c"], 20, atol=1e-7)
    np.testing.assert_allclose(reduced.hourly["basement_indoor_temperature_c"], 18, atol=1e-7)
    for result in (detailed, reduced):
        assert result.diagnostics["accepted"]
        assert len(result.time_axis) == 8760
        assert result.scalars["space_heating_demand_kwh"] > 0
        assert all(np.isfinite(values).all() for values in result.hourly.values())
        assert result.scalars["space_heating_carrier_input_kwh"] == pytest.approx(
            result.scalars["space_heating_demand_kwh"] / building.hvac.delivery.heating.efficiency
        )
    assert reduced.diagnostics["domain_reduction"]["maximum_interzone_exchange_sum_w"] == 0
