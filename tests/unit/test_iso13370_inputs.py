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
            Footprint.rectangle(10, 8), storeys=3, storey_height_m=2.5,
            basement_depth_m=basement, basement_conditioned=bool(basement and conditioned),
            orientation_deg=rotation,
        )
        building = Building(
            preset="illustrative-reference", geometry=geometry,
            unconditioned_exchange=(UnconditionedExchangeSpec(
                zone_kind="basement", outdoor_air_ach=0, source="Explicit closed test basement",
            ),) if basement and not conditioned else (),
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
    main = SimpleNamespace(name="main", elements=tuple(
        element for element in building.elements if element.boundary != "Ground"
    ))
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


@pytest.mark.parametrize("change", [
    {"iso13370_wall_thickness_m": None},
    {"iso13370_wall_thickness_m": 0},
    {"iso13370_wall_thickness_m": float("nan")},
    {"iso13370_wall_thickness_source": " "},
])
def test_width_and_source_are_required(synthetic_weather, change):
    building, spec, resolved = capture(synthetic_weather)
    with pytest.raises(ValueError, match="wall_thickness"):
        resolve_ground_inputs(building, spec, resolved, settings(**change))


def test_footing_block_width_is_not_used_as_wall_width(synthetic_weather):
    building, _, _ = capture(synthetic_weather)
    building.foundation = building.foundation.updated(footing_width_m=0.9)
    building, spec, resolved = capture(synthetic_weather, building=building)
    result = resolve_ground_inputs(building, spec, resolved, settings(iso13370_wall_thickness_m=0.24))
    assert result.wall_thickness_m == 0.24


@pytest.mark.parametrize("change, message", [
    ({"wall_height_above_grade_m": 0.2}, "plinth"),
    ({"wall_depth_below_slab_m": 0.1}, "buried wall"),
    ({"deep_ground_boundary": "GroundWater"}, "groundwater"),
    ({"exposed_perimeter_fraction": 0}, "positive exposed"),
])
def test_unsupported_boundaries_are_explicit(synthetic_weather, change, message):
    building, _, _ = capture(synthetic_weather)
    building.foundation = building.foundation.updated(**change)
    building, spec, resolved = capture(synthetic_weather, building=building)
    with pytest.raises(ValueError, match=message):
        resolve_ground_inputs(building, spec, resolved, settings())


def test_explicit_footing_material_is_not_silently_ignored(synthetic_weather):
    building, spec, _ = capture(synthetic_weather)
    building.foundation = building.foundation.updated(footing_material=spec.envelope.ground.layers[0].material)
    building, spec, resolved = capture(synthetic_weather, building=building)
    with pytest.raises(ValueError, match="footing"):
        resolve_ground_inputs(building, spec, resolved, settings())


def test_unconditioned_basement_and_second_soil_boundary_are_rejected(synthetic_weather):
    building, spec, resolved = capture(synthetic_weather, basement=2.5, conditioned=False)
    with pytest.raises(ValueError, match="unconditioned"):
        resolve_ground_inputs(building, spec, resolved, settings())
    building, spec, resolved = capture(synthetic_weather)
    with pytest.raises(ValueError, match="second monthly"):
        resolve_ground_inputs(building, spec, resolved, settings(monthly_ground_temperatures_c=(8,) * 12))


def test_multiple_foundation_parts_are_rejected(synthetic_weather):
    from epsm_core import BuildingPart, Footprint

    building, _, _ = capture(synthetic_weather)
    wing = BuildingPart(
        identifier="wing", footprint=Footprint(exterior=((12, 0), (18, 0), (18, 5), (12, 5))),
        storey_elevations_m=(0, 2.5),
    )
    building.geometry = building.geometry.updated(parts=(*building.geometry.parts, wing))
    building, spec, resolved = capture(synthetic_weather, building=building)
    with pytest.raises(ValueError, match="one physical foundation"):
        resolve_ground_inputs(building, spec, resolved, settings())
