"""Shared walls retain installed mass without outdoor heat or radiation exchange."""

from importlib.resources import files

import numpy as np
import pytest
from epsm_core import Building, BuildingPart, Footprint, GeometrySpec, PartyWall, Weather

from epsm_rc import PreparationCache, RCSettings, simulate_building


def building_with_contacts(*, contacts=(), glazing=0.0):
    geometry = GeometrySpec.from_footprint(
        Footprint.rectangle(10, 8), storeys=2, storey_height_m=3
    ).updated(party_walls=contacts)
    building = Building(preset="illustrative-reference", geometry=geometry)
    building.facade = building.facade.updated(default_wwr=glazing)
    return building


def contact(facade_id="main.r0.e0", **extent):
    return PartyWall(
        facade_id=facade_id,
        neighbor_id="adjacent-building",
        neighbor_conditioning="unknown",
        contact_confirmed=True,
        assumption_source="Shared contact is explicitly treated as adiabatic",
        **extent,
    )


def evaluate(building, **kwargs):
    settings = RCSettings.from_policy(
        "epsm_rc_5r1c_v2",
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
    )
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    return simulate_building(building, weather=weather, settings=settings, **kwargs)


def test_shared_wall_removes_transmission_and_sky_but_keeps_material_capacity():
    detached = building_with_contacts()
    attached = building_with_contacts(contacts=(contact(),))
    baseline, shared = evaluate(detached), evaluate(attached)
    a, b = [result.diagnostics["reduced_properties"] for result in (baseline, shared)]
    wall = next(
        element
        for element in detached.elements
        if element.facade_id == "main.r0.e0" and element.storey == 0
    )
    shared_area = 10 * 6
    uplift = 1 + detached.thermal_bridge_uplift_fraction
    removed_ua = shared_area * wall.construction.u_value * uplift
    assert a["opaque_transmission_conductance_w_k"] - b[
        "opaque_transmission_conductance_w_k"
    ] == pytest.approx(removed_ua)
    assert a["sky_radiation_loss_w"] - b["sky_radiation_loss_w"] == pytest.approx(
        removed_ua * 0.04 * 5 * 11 * 0.5
    )
    assert a["thermal_capacitance_j_k"] == pytest.approx(b["thermal_capacitance_j_k"])
    assert shared.scalars["space_heating_demand_kwh"] < baseline.scalars["space_heating_demand_kwh"]
    boundary = shared.diagnostics["boundary_reduction"]
    assert boundary["adiabatic_opaque_area_m2"] == pytest.approx(shared_area)
    assert boundary["neighbor_temperature_modelled"] is False


def test_partial_contact_uses_only_actual_shared_area():
    detached = evaluate(building_with_contacts())
    building = building_with_contacts(contacts=(contact(start_m=2, end_m=7, bottom_m=1, top_m=4),))
    partial = evaluate(building)
    wall = next(
        element
        for element in building.elements
        if element.facade_id == "main.r0.e0" and element.boundary == "Adiabatic"
    )
    removed_ua = 5 * 3 * wall.construction.u_value * (1 + building.thermal_bridge_uplift_fraction)
    assert detached.diagnostics["reduced_properties"][
        "opaque_transmission_conductance_w_k"
    ] - partial.diagnostics["reduced_properties"][
        "opaque_transmission_conductance_w_k"
    ] == pytest.approx(removed_ua)
    assert partial.diagnostics["boundary_reduction"]["adiabatic_opaque_area_m2"] == pytest.approx(
        15
    )


def test_fully_shared_walls_receive_no_opaque_solar_gain_or_sky_loss():
    geometry = GeometrySpec.from_footprint(Footprint.rectangle(10, 8), storeys=2, storey_height_m=3)
    attached = building_with_contacts(
        contacts=tuple(contact(facade.facade_id) for facade in geometry.facades())
    )
    # Remove the roof's outside film contribution to isolate the four walls.
    settings = RCSettings.from_policy(
        "epsm_rc_5r1c_v2",
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
        outside_roof_resistance_m2_k_w=0,
    )
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    result = simulate_building(attached, weather=weather, settings=settings)
    np.testing.assert_array_equal(result.hourly["opaque_solar_heat_gain_w"], 0)
    np.testing.assert_array_equal(result.hourly["sky_radiation_loss_w"], 0)
    assert result.diagnostics["reduced_properties"]["thermal_capacitance_j_k"] > 0


def test_partial_contact_windows_are_outdoors_and_cached_runs_remain_independent():
    building = building_with_contacts(
        contacts=(contact(start_m=0, end_m=5, bottom_m=0, top_m=6),), glazing=0.2
    )
    assert all(
        element.boundary == "Outdoors"
        for element in building.elements
        if element.role.value == "WINDOW"
    )
    cache = PreparationCache()
    ordinary = evaluate(building)
    prepared = evaluate(building, preparation_cache=cache)
    resumed = evaluate(Building.from_dict(building.to_dict()), preparation_cache=cache)
    assert cache.hits > 0
    assert ordinary.run_id == prepared.run_id == resumed.run_id
    for key in ordinary.hourly:
        np.testing.assert_array_equal(ordinary.hourly[key], prepared.hourly[key])
        np.testing.assert_array_equal(ordinary.hourly[key], resumed.hourly[key])
    assert ordinary.diagnostics["boundary_reduction"]["outdoor_window_area_m2"] == pytest.approx(
        sum(element.area_m2 for element in building.elements if element.role.value == "WINDOW")
    )


def test_same_domain_attached_parts_count_shared_material_once_and_remove_both_exposures(
    monkeypatch,
):
    from epsm_rc import platform

    models = []
    original = platform.Building5R1C

    def capture(*args, **kwargs):
        model = original(*args, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(platform, "Building5R1C", capture)
    results = []
    for start in (11, 10):
        building = building_with_contacts()
        wing = BuildingPart(
            identifier="wing",
            footprint=Footprint(exterior=((start, 0), (start + 4, 0), (start + 4, 8), (start, 8))),
            storey_elevations_m=(0, 3, 6),
        )
        building.geometry = building.geometry.updated(parts=(*building.geometry.parts, wing))
        weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
        results.append(
            simulate_building(
                building,
                weather=weather,
                settings=RCSettings.from_policy(
                    "epsm_rc_5r1c_v2",
                    airflow_policy="sensible_only_unlimited_supply",
                    ground_policy="conductance_multiplier",
                    self_shadowing_policy="ignore",
                ),
            )
        )
    separated, attached = results
    # Two 8 m x 6 m outdoor sides become one 48 m² installed internal wall.
    assert models[0].wall_area_m2 - models[1].wall_area_m2 == pytest.approx(96)
    assert attached.diagnostics["boundary_reduction"]["internal_opaque_area_m2"] == pytest.approx(
        48 + 80 + 32  # shared wall plus one intermediate floor in each part
    )
    wall = next(
        construction
        for construction in models[0].constructions
        if construction.name.endswith(".wall")
    )
    assert (
        separated.diagnostics["reduced_properties"]["thermal_capacitance_j_k"]
        - attached.diagnostics["reduced_properties"]["thermal_capacitance_j_k"]
    ) == pytest.approx(48 * wall.thermal_mass_per_area)
