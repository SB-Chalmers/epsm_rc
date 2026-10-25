"""Core-backed footprint imports preserve courtyard rings."""

import json

import pytest
from shapely.geometry import Polygon, mapping

from epsm_rc.building_geometry import BuildingGeometry
from epsm_rc.gis_footprint import FootprintImporter


@pytest.fixture
def courtyard_polygon():
    """A metric 400 m² exterior surrounding a 100 m² courtyard."""
    return Polygon(
        [(0, 0), (20, 0), (20, 20), (0, 20)],
        holes=[[(5, 5), (15, 5), (15, 15), (5, 15)]],
    )


def test_from_shapely_preserves_courtyard(courtyard_polygon):
    assert courtyard_polygon.area == 300.0
    geometry = BuildingGeometry.from_shapely(courtyard_polygon, num_floors=2, floor_height=3)
    assert geometry.footprint_area == 300.0
    assert len(geometry.shapely_polygon.interiors) == 1


@pytest.mark.parametrize("source_kind", ["dict", "file"])
def test_from_geojson_preserves_courtyard(courtyard_polygon, source_kind, tmp_path):
    pytest.importorskip("pyproj")
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": mapping(courtyard_polygon),
    }
    source = feature
    if source_kind == "file":
        source = tmp_path / "courtyard.geojson"
        source.write_text(json.dumps(feature))

    geometry = FootprintImporter.from_geojson(
        source,
        crs="EPSG:3006",
        num_floors=2,
        floor_height=3,
    )
    assert geometry.footprint_area == 300.0
    assert geometry.shapely_polygon.interiors[0].bounds == (-5.0, -5.0, 5.0, 5.0)


def test_from_geojson_multi_preserves_courtyard(courtyard_polygon):
    pytest.importorskip("pyproj")
    features = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(courtyard_polygon)}],
    }
    geometries = FootprintImporter.from_geojson_multi(
        features,
        crs="EPSG:3006",
        num_floors=2,
        floor_height=3,
    )
    assert len(geometries) == 1
    assert geometries[0].footprint_area == 300.0


def test_simple_polygon_imports_preserve_metric_area():
    pytest.importorskip("pyproj")
    polygon = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    assert (
        BuildingGeometry.from_shapely(polygon, num_floors=2, floor_height=3).footprint_area == 400.0
    )
    feature = {"type": "Feature", "properties": {}, "geometry": mapping(polygon)}
    assert (
        FootprintImporter.from_geojson(
            feature,
            crs="EPSG:3006",
            num_floors=2,
            floor_height=2.5,
        ).footprint_area
        == 400.0
    )


@pytest.mark.parametrize("source_kind", ["relation_id", "url"])
def test_osm_relation_rejects_courtyard_members(monkeypatch, source_kind):
    requests = pytest.importorskip("requests")
    response_data = {
        "elements": [
            {
                "type": "relation",
                "id": 42,
                "members": [
                    {"type": "way", "ref": 100, "role": "outer"},
                    {"type": "way", "ref": 101, "role": "inner"},
                ],
            },
            {"type": "way", "id": 100, "nodes": [1, 2, 3, 4, 1]},
            {"type": "node", "id": 1, "lon": 11.97, "lat": 57.70},
            {"type": "node", "id": 2, "lon": 11.98, "lat": 57.70},
            {"type": "node", "id": 3, "lon": 11.98, "lat": 57.71},
            {"type": "node", "id": 4, "lon": 11.97, "lat": 57.71},
        ],
    }

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return response_data

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    kwargs = (
        {"relation_id": 42}
        if source_kind == "relation_id"
        else {"url": "https://www.openstreetmap.org/relation/42"}
    )
    with pytest.raises(ValueError, match="courtyard|interior ring"):
        FootprintImporter.from_osm(**kwargs)
