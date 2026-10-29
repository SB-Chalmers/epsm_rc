"""Core's absolute thermal-bridge surcharge reaches the 5R1C conductance once."""

from importlib.resources import files

import pytest
from epsm_core import Building, Weather
from epsm_core.bridges import BRIDGED_ROLES

from epsm_rc import RCSettings, simulate_building


def evaluate(building):
    settings = RCSettings.from_policy(
        "epsm_rc_5r1c_v2",
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
    )
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    return simulate_building(building, weather=weather, settings=settings)


def test_surcharge_adds_delta_u_times_envelope_area_to_opaque_conductance():
    from epsm_core.quantities import quantity_takeoff

    plain = Building(preset="illustrative-reference", thermal_bridge_uplift_fraction=0.0)
    bridged = Building(preset="illustrative-reference")
    bridged.assign_inputs(thermal_bridge_delta_u_w_m2k=0.1)
    a, b = (evaluate(item).diagnostics["reduced_properties"] for item in (plain, bridged))
    spec = bridged.to_spec(Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw")))
    areas = {}
    for item in quantity_takeoff(spec):
        envelope_face = item.role in BRIDGED_ROLES and not item.element_id.endswith(
            ".internal_wall"
        )
        if item.area_m2 and (
            item.kind == "glazing" or (item.kind == "opaque_layer" and envelope_face)
        ):
            areas[item.element_id, item.role] = item.area_m2
    added = b["opaque_transmission_conductance_w_k"] - a["opaque_transmission_conductance_w_k"]
    assert added == pytest.approx(0.1 * sum(areas.values()), rel=1e-6)
