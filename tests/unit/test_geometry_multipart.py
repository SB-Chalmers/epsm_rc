"""Disconnected footprints preserve all parts in a common coordinate frame."""

import pytest
from shapely.geometry import MultiPolygon, Polygon, mapping

from epsm_rc.building_geometry import BuildingGeometry
from epsm_rc.gis_footprint import FootprintImporter


@pytest.fixture
def multipart_polygon():
    return MultiPolygon([
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(30, 0), (40, 0), (40, 5), (30, 5)]),
    ])


def assert_parts_preserved(geometry):
    assert geometry.footprint_area == 150.0
    parts = sorted(geometry.core_geometry.parts, key=lambda part: part.footprint.polygon.bounds)
    assert len(parts) == 2
    assert parts[1].footprint.polygon.bounds[0] - parts[0].footprint.polygon.bounds[0] == 30.0
    assert parts[1].footprint.polygon.centroid.y - parts[0].footprint.polygon.centroid.y == -2.5


def test_direct_shapely_preserves_multipart(multipart_polygon):
    assert multipart_polygon.area == 150.0
    assert_parts_preserved(BuildingGeometry.from_shapely(
        multipart_polygon, num_floors=2, floor_height=3,
    ))


@pytest.mark.parametrize("batch", [False, True])
def test_geojson_preserves_multipart(multipart_polygon, batch):
    feature = {"type": "Feature", "properties": {}, "geometry": mapping(multipart_polygon)}
    if batch:
        features = {"type": "FeatureCollection", "features": [feature]}
        geometry = FootprintImporter.from_geojson_multi(
            features, crs="EPSG:3006", num_floors=2, floor_height=3,
        )[0]
    else:
        geometry = FootprintImporter.from_geojson(
            feature, crs="EPSG:3006", num_floors=2, floor_height=3,
        )
    assert_parts_preserved(geometry)


@pytest.mark.parametrize("batch", [False, True])
def test_shapefile_preserves_multipart(multipart_polygon, batch, tmp_path):
    fiona = pytest.importorskip("fiona")
    path = tmp_path / "parts.shp"
    with fiona.open(
        path, "w", driver="ESRI Shapefile", crs="EPSG:3006",
        schema={"geometry": "Polygon", "properties": {}},
    ) as destination:
        destination.write({"geometry": mapping(multipart_polygon), "properties": {}})
    if batch:
        geometry = FootprintImporter.from_shapefile(path, num_floors=2, floor_height=3)[0]
    else:
        geometry = FootprintImporter.from_shapefile(path, fid=0, num_floors=2, floor_height=3)
    assert_parts_preserved(geometry)


def test_shapefile_single_polygon_retains_metric_area(tmp_path):
    fiona = pytest.importorskip("fiona")
    path = tmp_path / "single.shp"
    with fiona.open(
        path, "w", driver="ESRI Shapefile", crs="EPSG:3006",
        schema={"geometry": "Polygon", "properties": {}},
    ) as destination:
        destination.write({
            "geometry": mapping(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])),
            "properties": {},
        })
    assert FootprintImporter.from_shapefile(
        path, fid=0, num_floors=2, floor_height=2.5,
    ).footprint_area == 100.0
    assert FootprintImporter.from_shapefile(
        path, num_floors=2, floor_height=2.5,
    )[0].footprint_area == 100.0


def mock_osm(monkeypatch, members, ways):
    requests = pytest.importorskip("requests")
    data = {
        "elements": [
            {"type": "relation", "id": 42, "members": members},
            *ways,
            {"type": "node", "id": 1, "lon": 11.97, "lat": 57.70},
            {"type": "node", "id": 2, "lon": 11.98, "lat": 57.70},
            {"type": "node", "id": 3, "lon": 11.98, "lat": 57.71},
            {"type": "node", "id": 4, "lon": 11.97, "lat": 57.71},
        ]
    }

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return data

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())


@pytest.mark.parametrize("case", ["multiple", "ambiguous", "open", "empty_members"])
def test_osm_rejects_unsupported_relation(monkeypatch, case):
    members = [{"type": "way", "ref": 100, "role": "outer"}]
    ways = [{"type": "way", "id": 100, "nodes": [1, 2, 3, 4, 1]}]
    if case == "multiple":
        members.append({"type": "way", "ref": 101, "role": "outer"})
        ways.append({"type": "way", "id": 101, "nodes": [1, 2, 3, 4, 1]})
    elif case == "ambiguous":
        members[0]["role"] = ""
    elif case == "open":
        ways[0]["nodes"] = [1, 2, 3, 4]
    else:
        members.clear()
    mock_osm(monkeypatch, members, ways)
    with pytest.raises(ValueError, match="relation|outer|closed"):
        FootprintImporter.from_osm(relation_id=42)


def test_osm_uses_declared_single_closed_outer(monkeypatch):
    pytest.importorskip("pyproj")
    mock_osm(
        monkeypatch,
        [{"type": "way", "ref": 100, "role": "outer"}],
        # Unrelated way comes first: relation roles determine the selected way.
        [
            {"type": "way", "id": 99, "nodes": [1, 2, 3, 1]},
            {"type": "way", "id": 100, "nodes": [1, 2, 3, 4, 1]},
        ],
    )
    assert len(FootprintImporter.from_osm(
        relation_id=42, num_floors=2, floor_height=2.5,
    ).vertices) == 4
