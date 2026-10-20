"""Legacy authoring delegates geometry and windows without supplying physical defaults."""

import pytest
from shapely.affinity import rotate
from shapely.geometry import Polygon

from epsm_rc.building_geometry import BuildingGeometry


@pytest.mark.parametrize("case", ["rectangle", "rotated_concave"])
def test_legacy_areas_facades_and_windows_match_core(case, monkeypatch):
    core = pytest.importorskip("epsm_core")
    from epsm_core import facades
    from epsm_core import geometry as core_geometry

    polygon = (
        Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])
        if case == "rectangle"
        else rotate(Polygon([(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)]), 33)
    )
    native = core.GeometrySpec.from_footprint(polygon, storeys=2, storey_height_m=3)
    catalog_calls, window_calls = [], []
    original_catalog, original_windows = core_geometry.facade_catalog, facades.window_rectangles

    def catalog_spy(*args, **kwargs):
        catalog_calls.append(args)
        return original_catalog(*args, **kwargs)

    def windows_spy(*args, **kwargs):
        window_calls.append(args)
        return original_windows(*args, **kwargs)

    monkeypatch.setattr(core_geometry, "facade_catalog", catalog_spy)
    monkeypatch.setattr(facades, "window_rectangles", windows_spy)
    wrapper = BuildingGeometry.from_shapely(polygon, num_floors=2, floor_height=3)
    assert wrapper.core_geometry == native
    assert wrapper.floor_area == native.conditioned_floor_area_m2
    assert wrapper.volume == native.conditioned_volume_m3
    assert catalog_calls
    native_faces = original_catalog(native)
    assert [segment.normal_angle for segment in wrapper.wall_segments] == [
        face.azimuth_deg for face in native_faces
    ]
    selected = {
        segment.segment_id: 0.1 + segment.segment_id * 0.02 for segment in wrapper.wall_segments
    }
    wrapper.set_wwr_per_segment(selected)
    params = wrapper.get_5r1c_params()
    assert window_calls
    glazing = core.GlazingSpec(
        u_value_w_m2k=1.2,
        solar_heat_gain_coefficient=0.4,
        visible_transmittance=0.5,
    )
    native_building = core.Building.empty(geometry=native, facade=wrapper.to_core_facade(glazing))
    assert params["total_window_area"] == pytest.approx(
        sum(
            element.area_m2
            for element in native_building.elements
            if element.role.value == "WINDOW"
        )
    )
    assert params["total_wall_area_net"] == pytest.approx(
        sum(
            element.area_m2
            for element in native_building.elements
            if element.role.value == "EXT_WALL"
        )
    )


def test_omitted_and_partial_wwr_remain_missing():
    geometry = BuildingGeometry.from_rectangle(20, 10, num_floors=2, floor_height=3)
    assert geometry.wwr_per_segment == {}
    geometry.set_wwr_cardinal({"N": 0.2})
    assert len(geometry.wwr_per_segment) == 1
    with pytest.raises(ValueError, match="Missing WWR|missing.*WWR"):
        geometry.get_5r1c_params()


def test_core_conversion_does_not_inject_fabric_or_operation():
    geometry = BuildingGeometry.from_rectangle(20, 10, num_floors=2, floor_height=3)
    building = geometry.to_core_building()
    assert building.preset is None
    assert building.geometry == geometry.core_geometry
    assert building.construction_set is None
    assert building.facade is None
    assert building.hvac is None
    assert building.operation is None
    assert all(element.construction is None for element in building.elements)
    geometry.set_wwr_uniform(0.2)
    with pytest.raises(ValueError, match="glazing"):
        geometry.to_core_building()


def test_legacy_export_fails_with_compiler_migration_message():
    geometry = BuildingGeometry.from_rectangle(20, 10, num_floors=2, floor_height=3)
    geometry.set_wwr_uniform(0.2)
    with pytest.raises(ValueError, match="EnergyPlusCompiler"):
        geometry.get_eplus_surfaces()


def test_missing_height_is_not_filled():
    with pytest.raises(ValueError, match="num_floors|height"):
        BuildingGeometry.from_rectangle(20, 10)


def test_edge_connected_multipart_is_rejected_by_core():
    from shapely.geometry import MultiPolygon

    polygon = MultiPolygon(
        [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),
        ]
    )
    assert not polygon.is_valid
    with pytest.raises(ValueError, match="valid|intersect"):
        BuildingGeometry.from_shapely(polygon, num_floors=2, floor_height=3)


def test_missing_core_has_actionable_migration_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "epsm_core", None)
    with pytest.raises(ImportError, match=r"epsm_rc\[platform\]"):
        BuildingGeometry.from_rectangle(20, 10, num_floors=2, floor_height=3)


def test_3d_gis_input_cannot_lose_elevation_during_projection():
    from epsm_rc.gis_footprint import FootprintImporter

    feature = {
        "type": "Polygon",
        "coordinates": [
            [
                [11.97, 57.70, 3],
                [11.98, 57.70, 3],
                [11.98, 57.71, 3],
                [11.97, 57.71, 3],
                [11.97, 57.70, 3],
            ]
        ],
    }
    with pytest.raises(ValueError, match="2.D|two.dimensional"):
        FootprintImporter.from_geojson(feature, num_floors=2, floor_height=3)
