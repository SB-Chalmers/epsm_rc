"""Conditioned basement transmission follows Core's physical ground boundary."""

from importlib.resources import files

import numpy as np
import pytest
from epsm_core import Building, BuildingPart, Footprint, GeometrySpec, Weather

from epsm_rc import RCSettings, simulate_building


def basement_building(depth=2.5, *, conditioned=True):
    building = Building(
        preset="illustrative-reference",
        geometry=GeometrySpec.from_footprint(
            Footprint.rectangle(10, 8),
            storeys=3,
            storey_height_m=3,
            basement_depth_m=depth,
            basement_conditioned=conditioned,
        ),
    )
    building.facade = building.facade.updated(default_wwr=0)
    return building


def evaluate(building, factor=0.5, **kwargs):
    return simulate_building(
        building,
        weather=Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw")),
        settings=RCSettings(
            airflow_policy="sensible_only_unlimited_supply",
            ground_policy="conductance_multiplier",
            ground_coupling_factor=factor,
            self_shadowing_policy="ignore",
        ),
        **kwargs,
    )


@pytest.mark.parametrize("factor", [0.2, 0.8])
def test_heated_basement_keeps_floor_area_volume_and_ground_wall_conductance(factor):
    building = basement_building()
    before = building.snapshot().sha256
    result = evaluate(building, factor)
    assert building.snapshot().sha256 == before
    assert building.geometry.conditioned_floor_area_m2 == pytest.approx(320)
    assert building.geometry.conditioned_volume_m3 == pytest.approx(920)
    elements = building.elements
    walls = [e for e in elements if e.role.value == "BASEMENT_WALL" and e.boundary == "Ground"]
    slabs = [e for e in elements if e.boundary == "Ground" and e not in walls]
    assert sum(e.area_m2 for e in walls) == pytest.approx(90)
    assert sum(e.area_m2 for e in slabs) == pytest.approx(80)
    expected_ua = sum(e.area_m2 * e.construction.u_value for e in walls + slabs)
    assert result.diagnostics["reduced_properties"]["effective_ground_conductance_w_k"] == (
        pytest.approx(expected_ua * factor * (1 + building.thermal_bridge_uplift_fraction))
    )
    assert result.diagnostics["boundary_reduction"]["ground_wall_area_m2"] == pytest.approx(90)
    assert result.diagnostics["boundary_reduction"]["ground_floor_area_m2"] == pytest.approx(80)
    assert result.diagnostics["boundary_reduction"][
        "conditioned_basement_floor_area_m2"
    ] == pytest.approx(80)
    assert result.diagnostics["accepted"]
    assert np.isfinite(result.hourly["space_heating_demand_kwh"]).all()
    assert result.scalars["space_heating_demand_kwh"] > 0


def test_unheated_basement_stays_unsupported_before_solver(monkeypatch):
    from epsm_rc import platform

    def forbidden(*args, **kwargs):
        pytest.fail("Unconditioned basement reached the conditioned-domain solver")

    monkeypatch.setattr(platform, "Building5R1C", forbidden)
    with pytest.raises(ValueError, match="unconditioned basement"):
        evaluate(basement_building(conditioned=False))


def test_internal_basement_ceiling_does_not_add_ground_transmission():
    building = basement_building()
    initial = evaluate(building)
    ceiling = next(e for e in building.elements if e.identifier == "main.ground_floor")
    assert ceiling.boundary == "Internal"
    top = next(e.construction for e in building.elements if e.role.value == "TOP_FLOOR")
    altered = top.with_changes(
        application=ceiling.role,
        components=tuple(c.with_changes(application=ceiling.role) for c in top.components),
    )
    building.assign_construction(altered, element_ids=(ceiling.identifier,))
    changed = evaluate(building)
    assert altered.u_value != ceiling.construction.u_value
    assert (
        changed.diagnostics["reduced_properties"]["effective_ground_conductance_w_k"]
        == (initial.diagnostics["reduced_properties"]["effective_ground_conductance_w_k"])
    )


def test_adjacent_conditioned_basements_count_shared_material_once():
    building = basement_building()
    wing = BuildingPart(
        identifier="wing",
        footprint=Footprint(exterior=((10, 0), (14, 0), (14, 8), (10, 8))),
        storey_elevations_m=(0, 3, 6, 9),
    )
    building.geometry = building.geometry.updated(parts=(*building.geometry.parts, wing))
    result = evaluate(building)
    reduction = result.diagnostics["boundary_reduction"]
    assert reduction["ground_wall_area_m2"] == pytest.approx(44 * 2.5)
    assert reduction["ground_floor_area_m2"] == pytest.approx(112)
    internal = [
        e for e in building.elements if e.role.value == "BASEMENT_WALL" and e.boundary == "Internal"
    ]
    assert sum(e.area_m2 for e in internal) == pytest.approx(8 * 2.5)


def test_distinct_basement_controls_survive_domain_preparation_and_cached_resumption(
    synthetic_weather,
):
    from epsm_core import ScheduleSpec, ZoneUseSpec

    from epsm_rc import PreparationCache

    building = basement_building()
    building.operation = building.operation.updated(
        occupant_density_per_m2=0,
        equipment_gain_w_m2=0,
        lighting_gain_w_m2=0,
        heating_setpoint=ScheduleSpec.constant(20),
        cooling_enabled=False,
    )
    building.zone_uses = (
        ZoneUseSpec(
            zone_kind="basement",
            use=building.operation.updated(heating_setpoint=ScheduleSpec.constant(18)),
            source="declared basement 18 C",
        ),
    )
    cache = PreparationCache()
    settings = RCSettings(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    before = building.snapshot().sha256
    results = [
        simulate_building(
            building,
            weather=Weather.from_epw(synthetic_weather),
            settings=settings,
            preparation_cache=cache,
        )
        for _ in range(2)
    ]
    assert cache.hits > 0
    assert building.snapshot().sha256 == before
    assert results[0].run_id == results[1].run_id
    for key in results[0].hourly:
        np.testing.assert_array_equal(results[0].hourly[key], results[1].hourly[key])
    result = results[0]
    np.testing.assert_allclose(result.hourly["main_indoor_temperature_c"], 20, atol=1e-7)
    np.testing.assert_allclose(result.hourly["basement_indoor_temperature_c"], 18, atol=1e-7)
    np.testing.assert_array_equal(
        result.hourly["interzone_heat_to_main_w"], -result.hourly["interzone_heat_to_basement_w"]
    )
    np.testing.assert_allclose(
        result.hourly["space_heating_demand_kwh"],
        result.hourly["main_space_heating_demand_kwh"]
        + result.hourly["basement_space_heating_demand_kwh"],
    )
    assert result.diagnostics["domain_reduction"]["maximum_control_balance_residual_w"] < 1e-6
    assert (
        sum(
            domain["reference_area_m2"]
            for domain in result.diagnostics["domain_reduction"]["domains"]
        )
        == 320
    )


def test_coupled_basement_preserves_leap_day_and_zone_reference_areas(tmp_path, synthetic_weather):
    from datetime import datetime, timedelta

    from epsm_core import ScheduleSpec, ZoneUseSpec

    lines = synthetic_weather.read_text().splitlines()
    header, template = lines[:8], lines[8].split(",")
    leap = tmp_path / "leap.epw"
    rows = []
    for hour in range(8784):
        stamp = datetime(2016, 1, 1) + timedelta(hours=hour)
        row = template.copy()
        row[:5] = ["2016", str(stamp.month), str(stamp.day), str(stamp.hour + 1), "60"]
        rows.append(",".join(row))
    leap.write_text("\n".join(header + rows) + "\n")
    building = basement_building()
    building.load_reference_area_m2 = 280
    building.zone_uses = (
        ZoneUseSpec(
            zone_kind="basement",
            use=building.operation.updated(
                heating_setpoint=ScheduleSpec.constant(18),
                occupant_density_per_m2=0,
                equipment_gain_w_m2=0,
                lighting_gain_w_m2=0,
                dhw_kwh_m2_year=0,
            ),
            source="explicit basement programme",
        ),
    )
    result = simulate_building(
        building,
        weather=Weather.from_epw(leap, calendar_year=2016),
        settings=RCSettings(
            airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
        ),
    )
    assert len(result.time_axis) == 8784
    assert sum("02-29" in str(stamp) for stamp in result.time_axis) == 24
    domain = {item["name"]: item for item in result.diagnostics["domain_reduction"]["domains"]}
    assert domain["main"]["reference_area_m2"] == 210
    assert domain["basement"]["reference_area_m2"] == 70
    assert result.hourly["internal_gains_convective_w"].sum() > 0
    assert result.diagnostics["domains"]["basement"]["gain_reduction"]["aggregate_split_applied"]
