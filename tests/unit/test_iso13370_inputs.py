# File: test_iso13370_inputs.py
# Purpose: Independent physical geometry and boundary-contract checks for ISO ground inputs.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-10-29

"""Independent physical geometry and boundary-contract checks for ISO ground inputs."""

from types import SimpleNamespace

import pytest

from epsm_rc.ground_inputs import resolve_ground_inputs


def settings(**changes):
    return SimpleNamespace(
        **{
            "iso13370_wall_thickness_m": 0.3,
            "iso13370_wall_thickness_source": "Explicit test assumption for ISO wall width",
            "monthly_ground_temperatures_c": None,
            **changes,
        }
    )


def capture(synthetic_weather, *, basement=0, conditioned=True, rotation=0, building=None):
    from epsm_core import Building, Footprint, GeometrySpec, UnconditionedExchangeSpec, Weather
    from epsm_core.resolved import resolve_snapshot_inputs

    if building is None:
        geometry = GeometrySpec.from_footprint(
            Footprint.rectangle(10, 8),
            storeys=3,
            storey_height_m=2.5,
            basement_depth_m=basement,
            basement_conditioned=bool(basement and conditioned),
            orientation_deg=rotation,
        )
        building = Building(
            preset="illustrative-reference",
            geometry=geometry,
            unconditioned_exchange=(
                UnconditionedExchangeSpec(
                    zone_kind="basement",
                    outdoor_air_ach=0,
                    source="Explicit closed test basement",
                ),
            )
            if basement and not conditioned
            else (),
        )
    inputs = resolve_snapshot_inputs(building.snapshot(), Weather.from_epw(synthetic_weather))
    return inputs.building, inputs.spec, inputs.resolved


@pytest.mark.parametrize("rotation", [0, 37, 90])
def test_rectangle_uses_physical_area_exposed_perimeter_and_soil(synthetic_weather, rotation):
    building, spec, resolved = capture(synthetic_weather, rotation=rotation)
    result = resolve_ground_inputs(building, spec, resolved, settings())
    assert result.kind == "slab_on_ground"
    assert result.area_m2 == pytest.approx(80)
    assert result.exposed_perimeter_m == pytest.approx(36)
    assert result.report["characteristic_dimension_m"] == pytest.approx(80 / 18)
    assert result.soil_volumetric_heat_capacity_j_m3k == pytest.approx(1842 * 419)
    assert result.basement_wall_resistance_m2k_w is None
    assert result.coefficient_inputs()["basement_depth_m"] == 0


def test_explicit_exposure_fraction_scales_outer_perimeter(synthetic_weather):
    building, _, _ = capture(synthetic_weather)
    building.foundation = building.foundation.updated(exposed_perimeter_fraction=0.5)
    building, spec, resolved = capture(synthetic_weather, building=building)
    result = resolve_ground_inputs(building, spec, resolved, settings())
    assert result.area_m2 == pytest.approx(80)
    assert result.exposed_perimeter_m == pytest.approx(18)


def test_heated_basement_has_one_soil_slab_full_walls_and_no_main_ground(synthetic_weather):
    building, spec, resolved = capture(synthetic_weather, basement=2.5)
    result = resolve_ground_inputs(building, spec, resolved, settings())
    assert result.kind == "heated_basement"
    assert result.area_m2 == pytest.approx(80)
    assert result.exposed_perimeter_m == pytest.approx(36)
    assert result.basement_depth_m == pytest.approx(2.5)
    assert result.report["buried_wall_area_m2"] == pytest.approx(90)
    assert result.basement_wall_resistance_m2k_w > 0
    main = SimpleNamespace(
        name="main",
        elements=tuple(element for element in building.elements if element.boundary != "Ground"),
    )
    assert resolve_ground_inputs(building, spec, resolved, settings(), domain=main) is None


def test_bridges_modify_material_resistance_once_without_changing_inventory(synthetic_weather):
    building, _, _ = capture(synthetic_weather)
    building.thermal_bridge_uplift_fraction = 0.1
    building, spec, resolved = capture(synthetic_weather, building=building)
    before = building.snapshot().sha256
    result = resolve_ground_inputs(building, spec, resolved, settings())
    # Reference slab:150mm insulation lambda.04 plus150mm concrete lambda1.4.
    nominal_r = 0.15 / 0.04 + 0.15 / 1.4
    expected_r = (nominal_r + 0.17) / 1.1 - 0.17
    assert result.floor_resistance_m2k_w == pytest.approx(expected_r)
    assert result.report["floor_material_resistances"][0][
        "nominal_layer_resistance_m2k_w"
    ] == pytest.approx(nominal_r)
    assert building.snapshot().sha256 == before


@pytest.mark.parametrize(
    "change",
    [
        {"iso13370_wall_thickness_m": None},
        {"iso13370_wall_thickness_m": 0},
        {"iso13370_wall_thickness_m": float("nan")},
        {"iso13370_wall_thickness_source": " "},
    ],
)
def test_width_and_source_are_required(synthetic_weather, change):
    building, spec, resolved = capture(synthetic_weather)
    with pytest.raises(ValueError, match="wall_thickness"):
        resolve_ground_inputs(building, spec, resolved, settings(**change))


def test_footing_block_width_is_not_used_as_wall_width(synthetic_weather):
    building, _, _ = capture(synthetic_weather)
    building.foundation = building.foundation.updated(footing_width_m=0.9)
    building, spec, resolved = capture(synthetic_weather, building=building)
    result = resolve_ground_inputs(
        building, spec, resolved, settings(iso13370_wall_thickness_m=0.24)
    )
    assert result.wall_thickness_m == 0.24


@pytest.mark.parametrize(
    "change, message",
    [
        ({"wall_height_above_grade_m": 0.2}, "plinth"),
        ({"wall_depth_below_slab_m": 0.1}, "buried wall"),
        ({"deep_ground_boundary": "GroundWater"}, "groundwater"),
        ({"exposed_perimeter_fraction": 0}, "positive exposed"),
    ],
)
def test_unsupported_boundaries_are_explicit(synthetic_weather, change, message):
    building, _, _ = capture(synthetic_weather)
    building.foundation = building.foundation.updated(**change)
    building, spec, resolved = capture(synthetic_weather, building=building)
    with pytest.raises(ValueError, match=message):
        resolve_ground_inputs(building, spec, resolved, settings())


def test_explicit_footing_material_is_not_silently_ignored(synthetic_weather):
    building, spec, _ = capture(synthetic_weather)
    building.foundation = building.foundation.updated(
        footing_material=spec.envelope.ground.layers[0].material
    )
    building, spec, resolved = capture(synthetic_weather, building=building)
    with pytest.raises(ValueError, match="footing"):
        resolve_ground_inputs(building, spec, resolved, settings())


def test_unconditioned_basement_and_second_soil_boundary_are_rejected(synthetic_weather):
    building, spec, resolved = capture(synthetic_weather, basement=2.5, conditioned=False)
    with pytest.raises(ValueError, match="unconditioned"):
        resolve_ground_inputs(building, spec, resolved, settings())
    building, spec, resolved = capture(synthetic_weather)
    with pytest.raises(ValueError, match="second monthly"):
        resolve_ground_inputs(
            building, spec, resolved, settings(monthly_ground_temperatures_c=(8,) * 12)
        )


def multipart_house(synthetic_weather, footprints, *, basement=0):
    """Reference house whose ground floor is the given footprints, in local metres."""
    from epsm_core import BuildingPart, Footprint

    building, _, _ = capture(synthetic_weather, basement=basement)
    first = building.geometry.parts[0]
    parts = tuple(
        BuildingPart(
            identifier=f"part{index}",
            footprint=Footprint(exterior=exterior),
            storey_elevations_m=elevations,
        )
        for index, (exterior, elevations) in enumerate(footprints)
    )
    if basement:
        parts = (first, *parts[1:])
    building.geometry = building.geometry.updated(parts=parts)
    return capture(synthetic_weather, building=building)


def rectangle(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


MAIN = (rectangle(0, 0, 10, 8), (0, 2.5))

# House 28235f44 (single-family, SE.06), one of 16 houses RC refused in the
# calibration of #102: two parts side by side for 5.9 m, apart by 8.9e-16 m.
# Runner's local metres at full precision; rounding would hide or create the gap.
HOUSE_28235F44 = (
    (
        (
            (2.900000000023283, 17.229000000283122),
            (2.9039999592352186, 13.60603695738741),
            (8.758999252347767, 13.613541763211796),
            (8.754000000015367, 17.235999999567866),
        ),
        (0.0, 3.085),
    ),
    (
        (
            (0.0, 9.773000000044703),
            (0.008000000030733645, 3.843999999575317),
            (2.01400000002468, 3.8459999999031425),
            (2.0190000000293367, 0.0),
            (11.474000000045635, 0.012000000104308128),
            (11.457000000053085, 13.61699999962002),
            (2.095000000030268, 13.604999999515712),
            (2.1000000000349246, 9.775000000372529),
        ),
        (0.0, 2.317436294271008, 4.634872588542016),
    ),
)


@pytest.mark.parametrize("gap", [0.0, 1e-9, 0.0005, 0.013])
def test_parts_side_by_side_form_one_slab_without_their_shared_edge(synthetic_weather, gap):
    wing = (rectangle(10 + gap, 0, 16, 5), (0, 2.5))
    building, spec, resolved = multipart_house(synthetic_weather, (MAIN, wing))
    inputs = resolve_ground_inputs(building, spec, resolved, settings())
    # 10 x 8 main part plus a 6 x 5 wing against its east wall: an L-shaped slab.
    assert inputs.kind == "slab_on_ground"
    assert inputs.report["building_parts"] == 2
    assert len(inputs.slabs) == 1
    assert inputs.area_m2 == pytest.approx(80 + (6 - gap) * 5)
    assert inputs.exposed_perimeter_m == pytest.approx(16 + 5 + 6 + 3 + 10 + 8, abs=1e-6)


def test_house_28235f44_from_the_calibration_is_one_slab(synthetic_weather):
    building, spec, resolved = multipart_house(synthetic_weather, HOUSE_28235F44)
    inputs = resolve_ground_inputs(building, spec, resolved, settings())
    assert len(inputs.slabs) == 1
    assert inputs.area_m2 == pytest.approx(161.3843, abs=1e-3)
    # The parts' perimeters (18.954 + 50.133 m) less the 5.895 m they share, twice.
    assert inputs.exposed_perimeter_m == pytest.approx(57.377, abs=0.01)


def test_parts_apart_or_meeting_at_a_corner_are_separate_slabs(synthetic_weather):
    from epsm_rc.ground import ground_coefficients

    for wing in (rectangle(12, 0, 18, 5), rectangle(10, 8, 16, 13)):
        building, spec, resolved = multipart_house(synthetic_weather, (MAIN, (wing, (0, 2.5))))
        inputs = resolve_ground_inputs(building, spec, resolved, settings())
        assert [slab for slab in inputs.slabs] == [pytest.approx(s) for s in ((80, 36), (30, 22))]
        assert inputs.exposed_perimeter_m == pytest.approx(58)
        combined = inputs.coefficients()
        alone = [
            ground_coefficients(
                **{**inputs.coefficient_inputs(), "area_m2": a, "exposed_perimeter_m": p}
            )
            for a, p in inputs.slabs
        ]
        for name in ("steady_conductance_w_k", "periodic_internal_w_k", "periodic_external_w_k"):
            assert getattr(combined, name) == pytest.approx(sum(getattr(c, name) for c in alone))


def test_three_parts_chain_into_one_slab(synthetic_weather):
    parts = (MAIN, (rectangle(10, 0, 16, 5), (0, 2.5)), (rectangle(16.0005, 0, 20, 5), (0, 2.5)))
    building, spec, resolved = multipart_house(synthetic_weather, parts)
    inputs = resolve_ground_inputs(building, spec, resolved, settings())
    assert len(inputs.slabs) == 1
    assert inputs.exposed_perimeter_m == pytest.approx(20 + 5 + 10 + 3 + 10 + 8, abs=0.01)


def test_multipart_basements_are_rejected(synthetic_weather):
    wing = (rectangle(10, 0, 16, 5), (0, 2.5))
    building, spec, resolved = multipart_house(synthetic_weather, (MAIN, wing), basement=2.5)
    with pytest.raises(ValueError, match="one building part"):
        resolve_ground_inputs(building, spec, resolved, settings())
