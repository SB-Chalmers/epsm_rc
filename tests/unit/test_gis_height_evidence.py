"""Height imports distinguish evidence, omitted inputs and caller overrides."""

import pytest
from shapely.geometry import Polygon, mapping

from epsm_rc.gis_footprint import FootprintImporter


@pytest.fixture
def feature():
    return {
        "type": "Feature",
        "geometry": mapping(Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])),
        "properties": {"building:levels": 3, "height": 9.0},
    }


def test_explicit_two_floors_overrides_source_three(feature):
    geometry = FootprintImporter.from_geojson(feature, crs="EPSG:3006", num_floors=2)
    assert geometry.num_floors == 2
    assert geometry.floor_height == 4.5
    assert geometry.total_height == 9.0
    assert geometry.height_evidence["num_floors_source"] == "caller"
    assert geometry.height_evidence["source_properties"]["building:levels"] == 3


def test_source_levels_and_total_height_are_retained(feature):
    geometry = FootprintImporter.from_geojson(feature, crs="EPSG:3006")
    assert geometry.num_floors == 3
    assert geometry.floor_height == 3.0
    assert geometry.total_height == 9.0


def test_explicit_floor_height_overrides_total_height(feature):
    geometry = FootprintImporter.from_geojson(feature, crs="EPSG:3006", floor_height=2.8)
    assert geometry.total_height == pytest.approx(8.4)
    assert geometry.height_evidence["floor_height_source"] == "caller"
    assert geometry.height_evidence["source_properties"]["height"] == 9.0


@pytest.mark.parametrize("properties,match", [({}, "num_floors"), ({"levels": 3}, "floor_height")])
def test_missing_evidence_requires_explicit_assumption(feature, properties, match):
    feature["properties"] = properties
    with pytest.raises(ValueError, match=match):
        FootprintImporter.from_geojson(feature, crs="EPSG:3006")


@pytest.mark.parametrize(
    "properties",
    [
        {"levels": 2.5, "height": 9},
        {"levels": 0, "height": 9},
        {"levels": 3, "height": "unknown"},
        {"levels": 3, "height": float("nan")},
    ],
)
def test_invalid_evidence_is_rejected(feature, properties):
    feature["properties"] = properties
    with pytest.raises(ValueError, match="height|levels|floors"):
        FootprintImporter.from_geojson(feature, crs="EPSG:3006")


def test_explicit_assumptions_allow_missing_source(feature):
    feature["properties"] = {}
    geometry = FootprintImporter.from_geojson(
        feature,
        crs="EPSG:3006",
        num_floors=2,
        floor_height=2.5,
    )
    assert geometry.total_height == 5.0


def test_batch_reports_missing_height(feature, caplog):
    feature["properties"] = {"levels": 3}
    assert (
        FootprintImporter.from_geojson_multi(
            {"type": "FeatureCollection", "features": [feature]},
            crs="EPSG:3006",
        )
        == []
    )
    assert "floor_height" in caplog.text


@pytest.mark.parametrize("batch", [False, True])
def test_shapefile_uses_height_evidence(feature, tmp_path, batch):
    fiona = pytest.importorskip("fiona")
    path = tmp_path / "height.shp"
    with fiona.open(
        path,
        "w",
        driver="ESRI Shapefile",
        crs="EPSG:3006",
        schema={"geometry": "Polygon", "properties": {"levels": "int", "height": "float"}},
    ) as destination:
        destination.write(
            {"geometry": feature["geometry"], "properties": {"levels": 3, "height": 9}}
        )
    geometry = (
        FootprintImporter.from_shapefile(path, num_floors=2)[0]
        if batch
        else FootprintImporter.from_shapefile(path, fid=0, num_floors=2)
    )
    assert geometry.num_floors == 2
    assert geometry.floor_height == 4.5


def test_osm_explicit_floors_preserve_source_height(monkeypatch):
    requests = pytest.importorskip("requests")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "elements": [
                    {
                        "type": "way",
                        "id": 100,
                        "nodes": [1, 2, 3, 4, 1],
                        "tags": {"building:levels": "3", "height": "9 m"},
                    },
                    {"type": "node", "id": 1, "lon": 11.97, "lat": 57.70},
                    {"type": "node", "id": 2, "lon": 11.98, "lat": 57.70},
                    {"type": "node", "id": 3, "lon": 11.98, "lat": 57.71},
                    {"type": "node", "id": 4, "lon": 11.97, "lat": 57.71},
                ]
            }

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    geometry = FootprintImporter.from_osm(way_id=100, num_floors=2)
    assert geometry.num_floors == 2
    assert geometry.total_height == 9.0
