"""Core-backed geometry metrics and deterministic GIS import controls."""

import math

import pytest

from epsm_rc.building_geometry import BuildingGeometry
from epsm_rc.gis_footprint import FootprintImporter


class TestBuildingGeometry:
    def test_rectangular_geometry_metrics_and_balance(self):
        geometry = BuildingGeometry.from_polygon(
            [(0, 0), (20, 0), (20, 10), (0, 10)], num_floors=2, floor_height=3.0,
        )
        assert math.isclose(geometry.footprint_area, 200.0)
        assert math.isclose(geometry.floor_area, 400.0)
        assert math.isclose(geometry.total_height, 6.0)
        geometry.set_wwr_cardinal({"N": 0.1, "S": 0.3, "E": 0.2, "W": 0.2})
        parameters = geometry.get_5r1c_params()
        assert parameters["total_wall_area_gross"] == pytest.approx(
            parameters["total_wall_area_net"] + parameters["total_window_area"],
        )


class TestFootprintImporter:
    """Importer tests with mocked network and deterministic geometry."""

    def test_from_osm_url_parsing_and_floor_tag(self, monkeypatch):
        # Mock response for https://www.openstreetmap.org/way/90851303
        way_id = 90851303
        mocked_json = {
            "elements": [
                {
                    "type": "way",
                    "id": way_id,
                    "nodes": [1, 2, 3, 4, 1],
                    "tags": {"building": "yes", "building:levels": "4"},
                },
                {"type": "node", "id": 1, "lon": 11.9710, "lat": 57.7080},
                {"type": "node", "id": 2, "lon": 11.9714, "lat": 57.7080},
                {"type": "node", "id": 3, "lon": 11.9714, "lat": 57.7082},
                {"type": "node", "id": 4, "lon": 11.9710, "lat": 57.7082},
            ]
        }

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return mocked_json

        def _fake_get(*args, **kwargs):
            return _Resp()

        import requests
        monkeypatch.setattr(requests, 'get', _fake_get)

        geom = FootprintImporter.from_osm(
            url="https://www.openstreetmap.org/way/90851303", floor_height=2.5,
        )

        assert geom.num_floors == 4  # inferred from building:levels tag
        assert geom.footprint_area > 1.0
        assert len(geom.vertices) == 4

    def test_from_geojson_feature_collection(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"levels": 3},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [11.9710, 57.7080],
                            [11.9713, 57.7080],
                            [11.9713, 57.7082],
                            [11.9710, 57.7082],
                            [11.9710, 57.7080],
                        ]],
                    },
                }
            ],
        }

        geom = FootprintImporter.from_geojson(geojson, floor_height=2.5)
        assert geom.num_floors == 3
        assert geom.footprint_area > 1.0

    def test_from_osm_invalid_url_raises(self):
        try:
            FootprintImporter.from_osm(url="https://www.openstreetmap.org/node/123")
            assert False, "Expected ValueError for non-way/non-relation OSM URL"
        except ValueError as e:
            assert "Cannot parse OSM URL" in str(e)
