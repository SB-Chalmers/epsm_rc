"""Finite attic-node balances and physically useful limiting cases."""

import pytest

from epsm_rc.attic import eliminate_attic_node


def test_finite_attic_conserves_ceiling_shell_and_ventilation_balance():
    hc, hs, hv = 40.0, 100.0, 60.0
    inside, outside, shell_gain = 20.0, 0.0, 300.0
    attic = (hc * inside + (hs + hv) * outside + shell_gain) / (hc + hs + hv)
    equivalent, transmitted = eliminate_attic_node(hc, hs, hv)
    assert hc * (inside - attic) == pytest.approx(
        equivalent * (inside - outside) - transmitted * shell_gain
    )
    assert hc * (inside - attic) + shell_gain == pytest.approx((hs + hv) * (attic - outside))
    assert 0 < attic < inside


def test_sealed_attic_is_series_resistance_and_infinite_ventilation_is_ceiling_limit():
    assert eliminate_attic_node(40, 100, 0) == pytest.approx((40 * 100 / 140, 40 / 140))
    conductance, fraction = eliminate_attic_node(40, 100, 1e12)
    assert conductance == pytest.approx(40)
    assert fraction < 1e-9
    assert eliminate_attic_node(0, 100, 60) == (0, 0)


@pytest.mark.parametrize("values", [(0, 0, 0), (-1, 2, 3), (1, float("nan"), 0)])
def test_invalid_attic_conductance_rejected(values):
    with pytest.raises(ValueError):
        eliminate_attic_node(*values)


@pytest.mark.parametrize("shape", ["flat", "gable", "mono_pitch", "hip"])
def test_public_cold_attic_uses_finite_buffer_and_retains_explicit_exchange(
    shape, synthetic_weather
):
    if shape == "hip":
        pytest.importorskip("py_straight_skeleton", reason="hip roofs need epsm_core[geometry]")
    from epsm_core import (
        Building,
        Footprint,
        GeometrySpec,
        RoofSpec,
        UnconditionedExchangeSpec,
        Weather,
    )

    from epsm_rc import RCSettings, simulate_building

    building = Building(
        preset="illustrative-reference",
        geometry=GeometrySpec.from_footprint(
            Footprint.rectangle(10, 8),
            storeys=3,
            storey_height_m=3,
            roof=RoofSpec(shape=shape, topology="cold_attic"),
        ),
        unconditioned_exchange=(
            UnconditionedExchangeSpec(
                zone_kind="attic", outdoor_air_ach=0.5, source="explicit test ventilation"
            ),
        ),
    )
    before = building.snapshot().sha256
    result = simulate_building(
        building,
        weather=Weather.from_epw(synthetic_weather),
        settings=RCSettings.from_policy(
            "epsm_rc_5r1c_v2",
            airflow_policy="sensible_only_unlimited_supply",
            ground_policy="conductance_multiplier",
            cold_attic_policy="quasi_steady_buffer",
        ),
    )
    attic = result.diagnostics["attic_reduction"]
    assert attic["effective_outdoor_air_ach"] == 0.5
    assert attic["shell_storage"] == "omitted"
    assert len(attic["attics"]) == 1
    node = attic["attics"][0]
    assert 0 < node["equivalent_conductance_w_k"] < node["ceiling_conductance_w_k"]
    assert node["ceiling_area_m2"] == pytest.approx(80)
    assert result.diagnostics["accepted"]
    assert building.snapshot().sha256 == before


def test_cold_attic_cannot_infer_exchange_or_bypass_policy(synthetic_weather):
    from epsm_core import Building, RoofSpec, UnconditionedExchangeSpec, Weather

    from epsm_rc import RCSettings, simulate_building

    building = Building(preset="illustrative-reference")
    building.geometry = building.geometry.updated(roof=RoofSpec(shape="gable"))
    settings = {
        "airflow_policy": "sensible_only_unlimited_supply",
        "ground_policy": "conductance_multiplier",
    }
    weather = Weather.from_epw(synthetic_weather)
    with pytest.raises(ValueError, match="cold_attic_policy"):
        simulate_building(building, weather=weather, settings=RCSettings(**settings))
    with pytest.raises(ValueError, match="explicit attic outdoor exchange"):
        simulate_building(
            building,
            weather=weather,
            settings=RCSettings(**settings, cold_attic_policy="quasi_steady_buffer"),
        )
    building.unconditioned_exchange = (
        UnconditionedExchangeSpec(
            zone_kind="attic", outdoor_air_ach=0, source="explicit sealed attic"
        ),
    )
    result = simulate_building(
        building,
        weather=weather,
        settings=RCSettings(**settings, cold_attic_policy="quasi_steady_buffer"),
    )
    assert result.diagnostics["attic_reduction"]["effective_outdoor_air_ach"] == 0


def test_vertical_attic_interface_to_taller_main_space_remains_explicitly_unsupported(
    synthetic_weather,
):
    from epsm_core import (
        Building,
        BuildingPart,
        Footprint,
        GeometrySpec,
        RoofSpec,
        UnconditionedExchangeSpec,
        Weather,
    )

    from epsm_rc import RCSettings, simulate_building

    building = Building(
        preset="illustrative-reference",
        geometry=GeometrySpec.from_footprint(
            Footprint.rectangle(10, 8),
            storeys=2,
            storey_height_m=3,
            roof=RoofSpec(shape="flat", topology="cold_attic"),
        ),
        unconditioned_exchange=(
            UnconditionedExchangeSpec(
                zone_kind="attic", outdoor_air_ach=0.5, source="declared test exchange"
            ),
        ),
    )
    wing = BuildingPart(
        identifier="wing",
        footprint=Footprint(exterior=((10, 0), (14, 0), (14, 8), (10, 8))),
        storey_elevations_m=(0, 3, 6, 9),
        roof=RoofSpec(shape="flat"),
    )
    building.geometry = building.geometry.updated(parts=(*building.geometry.parts, wing))
    with pytest.raises(ValueError, match="attic.*interfaces"):
        simulate_building(
            building,
            weather=Weather.from_epw(synthetic_weather),
            settings=RCSettings(
                airflow_policy="sensible_only_unlimited_supply",
                ground_policy="conductance_multiplier",
                self_shadowing_policy="ignore",
                cold_attic_policy="quasi_steady_buffer",
            ),
        )


def test_internal_attic_wall_storage_is_omitted_when_attics_share_one_zone(synthetic_weather):
    import numpy as np
    from epsm_core import (
        Building,
        BuildingPart,
        Footprint,
        GeometrySpec,
        RoofSpec,
        UnconditionedExchangeSpec,
        Weather,
        ZoningSpec,
    )

    from epsm_rc import RCSettings, simulate_building

    building = Building(
        preset="illustrative-reference",
        geometry=GeometrySpec.from_footprint(
            Footprint.rectangle(10, 8),
            storeys=2,
            storey_height_m=3,
            roof=RoofSpec(shape="flat", topology="cold_attic"),
            zoning=ZoningSpec(mode="whole_building"),
        ),
        unconditioned_exchange=(
            UnconditionedExchangeSpec(
                zone_kind="attic", outdoor_air_ach=0.5, source="explicit shared attic exchange"
            ),
        ),
    )
    building.geometry = building.geometry.updated(
        parts=(
            *building.geometry.parts,
            BuildingPart(
                identifier="wing",
                footprint=Footprint(exterior=((10, 0), (14, 0), (14, 8), (10, 8))),
                storey_elevations_m=(0, 3, 6),
            ),
        )
    )
    settings = RCSettings(
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
        self_shadowing_policy="ignore",
        cold_attic_policy="quasi_steady_buffer",
    )
    weather = Weather.from_epw(synthetic_weather)
    baseline = simulate_building(building, weather=weather, settings=settings)
    wall = next(
        element
        for element in building.elements
        if element.identifier.endswith(".attic_internal_wall")
    )
    original = wall.construction
    changed = original.with_changes(
        components=tuple(
            component.with_changes(
                layers=tuple(
                    slot.with_changes(
                        layer=slot.layer.with_changes(
                            materials=tuple(
                                share.with_changes(
                                    material=share.material.with_changes(
                                        specific_heat=2 * share.material.specific_heat
                                    )
                                )
                                for share in slot.layer.materials
                            )
                        )
                    )
                    for slot in component.layers
                )
            )
            for component in original.components
        )
    )
    building.assign_construction(changed, element_ids=(wall.identifier,))
    altered = simulate_building(building, weather=weather, settings=settings)
    assert (
        baseline.diagnostics["reduced_properties"]["thermal_capacitance_j_k"]
        == (altered.diagnostics["reduced_properties"]["thermal_capacitance_j_k"])
    )
    for key in baseline.hourly:
        np.testing.assert_array_equal(baseline.hourly[key], altered.hourly[key])
