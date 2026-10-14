"""Disconnected footprints must not be reduced to their largest part."""

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


def test_direct_shapely_rejects_multipart(multipart_polygon):
    assert multipart_polygon.area == 150.0
    with pytest.raises(ValueError, match="MultiPolygon|multipart"):
        BuildingGeometry.from_shapely(multipart_polygon)


@pytest.mark.parametrize("batch", [False, True])
def test_geojson_rejects_multipart(multipart_polygon, batch, caplog):
    feature = {"type": "Feature", "properties": {}, "geometry": mapping(multipart_polygon)}
    if batch:
        features = {"type": "FeatureCollection", "features": [feature]}
        assert FootprintImporter.from_geojson_multi(features, crs="EPSG:3006") == []
        assert "MultiPolygon" in caplog.text
    else:
        with pytest.raises(ValueError, match="MultiPolygon|multipart"):
            FootprintImporter.from_geojson(feature, crs="EPSG:3006")


@pytest.mark.parametrize("batch", [False, True])
def test_shapefile_rejects_multipart(multipart_polygon, batch, caplog, tmp_path):
    fiona = pytest.importorskip("fiona")
    path = tmp_path / "parts.shp"
    with fiona.open(
        path, "w", driver="ESRI Shapefile", crs="EPSG:3006",
        schema={"geometry": "Polygon", "properties": {}},
    ) as destination:
        destination.write({"geometry": mapping(multipart_polygon), "properties": {}})
    if batch:
        assert FootprintImporter.from_shapefile(path) == []
        assert "MultiPolygon" in caplog.text
    else:
        with pytest.raises(ValueError, match="MultiPolygon|multipart"):
            FootprintImporter.from_shapefile(path, fid=0)


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
