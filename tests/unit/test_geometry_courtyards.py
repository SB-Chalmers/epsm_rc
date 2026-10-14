"""Legacy footprint imports must not fill courtyard holes silently."""

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


def test_from_shapely_rejects_courtyard(courtyard_polygon):
    assert courtyard_polygon.area == 300.0
    with pytest.raises(ValueError, match="courtyard|interior ring"):
        BuildingGeometry.from_shapely(courtyard_polygon)


@pytest.mark.parametrize("source_kind", ["dict", "file"])
def test_from_geojson_rejects_courtyard(courtyard_polygon, source_kind, tmp_path):
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": mapping(courtyard_polygon),
    }
    source = feature
    if source_kind == "file":
        source = tmp_path / "courtyard.geojson"
        source.write_text(json.dumps(feature))

    with pytest.raises(ValueError, match="courtyard|interior ring"):
        FootprintImporter.from_geojson(source, crs="EPSG:3006")


def test_from_geojson_multi_reports_unsupported_courtyard(courtyard_polygon, caplog):
    features = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": mapping(courtyard_polygon)}
        ],
    }
    assert FootprintImporter.from_geojson_multi(features, crs="EPSG:3006") == []
    assert "Skipping feature 0" in caplog.text
    assert "interior ring" in caplog.text


def test_simple_polygon_imports_preserve_metric_area():
    pytest.importorskip("pyproj")
    polygon = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    assert BuildingGeometry.from_shapely(polygon).footprint_area == 400.0
    feature = {"type": "Feature", "properties": {}, "geometry": mapping(polygon)}
    assert FootprintImporter.from_geojson(feature, crs="EPSG:3006").footprint_area == 400.0
