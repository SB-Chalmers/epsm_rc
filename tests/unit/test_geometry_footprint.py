"""Tests for polygon geometry and GIS footprint importers."""

from __future__ import annotations

import math
import importlib.util
import re
from pathlib import Path

import pytest
from epsm_rc.building_geometry import BuildingGeometry

try:
    from epsm_rc.gis_footprint import FootprintImporter
    _HAS_FOOTPRINT_IMPORTER = True
except ImportError:
    _HAS_FOOTPRINT_IMPORTER = False


def _write_polygon_idf_for_test(geom: BuildingGeometry, output_path: Path) -> Path:
    """Write a minimal IDF with polygon walls/roof/floor/windows for geometry checks."""
    surfaces = geom.get_eplus_surfaces()
    h = geom.total_height

    idf = [
        "Version, 24.2;",
        "GlobalGeometryRules, UpperLeftCorner, Counterclockwise, World;",
        f"Zone, Zone1, 0, 0, 0, 0, 1, 1, {h}, {geom.volume};",
    ]

    for surf in surfaces:
        if surf['type'] != 'wall':
            continue
        v = surf['vertices']
        idf.append(
            "\n".join([
                "BuildingSurface:Detailed,",
                f"    {surf['name']},",
                "    Wall,",
                "    WallConstruction,",
                "    Zone1, ,",
                "    Outdoors, ,",
                "    SunExposed, WindExposed,",
                "    autocalculate,",
                "    4,",
                f"    {v[0][0]:.4f}, {v[0][1]:.4f}, {v[0][2]:.4f},",
                f"    {v[1][0]:.4f}, {v[1][1]:.4f}, {v[1][2]:.4f},",
                f"    {v[2][0]:.4f}, {v[2][1]:.4f}, {v[2][2]:.4f},",
                f"    {v[3][0]:.4f}, {v[3][1]:.4f}, {v[3][2]:.4f};",
            ])
        )

    roof_verts = [(v[0], v[1], h) for v in geom.vertices]
    idf.append(
        "\n".join([
            "BuildingSurface:Detailed,",
            "    Roof, Roof, RoofConstruction,",
            "    Zone1, , Outdoors, , SunExposed, WindExposed, autocalculate,",
            f"    {len(roof_verts)},",
            ",\n".join(f"    {v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}" for v in roof_verts) + ";",
        ])
    )

    floor_verts = [(v[0], v[1], 0.0) for v in reversed(geom.vertices)]
    idf.append(
        "\n".join([
            "BuildingSurface:Detailed,",
            "    Floor, Floor, FloorConstruction,",
            "    Zone1, , Ground, , NoSun, NoWind, autocalculate,",
            f"    {len(floor_verts)},",
            ",\n".join(f"    {v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}" for v in floor_verts) + ";",
        ])
    )

    for surf in surfaces:
        if surf['type'] != 'window':
            continue
        v = surf['vertices']
        idf.append(
            "\n".join([
                "FenestrationSurface:Detailed,",
                f"    {surf['name']},",
                "    Window,",
                "    WindowConstruction,",
                f"    {surf['parent']}, ,",
                "    autocalculate, , 1,",
                "    4,",
                f"    {v[0][0]:.4f}, {v[0][1]:.4f}, {v[0][2]:.4f},",
                f"    {v[1][0]:.4f}, {v[1][1]:.4f}, {v[1][2]:.4f},",
                f"    {v[2][0]:.4f}, {v[2][1]:.4f}, {v[2][2]:.4f},",
                f"    {v[3][0]:.4f}, {v[3][1]:.4f}, {v[3][2]:.4f};",
            ])
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(idf) + "\n")
    return output_path


def _vertical_quad_area(vertices):
    """Area of vertical quad with E+ vertex ordering (top-left, bottom-left, bottom-right, top-right)."""
    # Width from bottom-left to bottom-right (x/y plane)
    width = math.hypot(vertices[2][0] - vertices[1][0], vertices[2][1] - vertices[1][1])
    # Height from bottom-left to top-left (z axis)
    height = abs(vertices[0][2] - vertices[1][2])
    return width * height


def _extract_from_idf(path):
    """Parse key geometry info from IDF text (aligned with _compare_idfs.py approach)."""
    with open(path) as f:
        content = f.read()

    content_nc = re.sub(r"![^\n]*", "", content)
    objects = re.split(r";", content_nc)

    result = {"walls": {}, "windows": {}, "floor": None, "roof": None}

    for obj in objects:
        obj = obj.strip()
        if not obj:
            continue
        fields = [fld.strip() for fld in obj.split(",")]

        if fields[0] == "BuildingSurface:Detailed":
            name = fields[1]
            stype = fields[2]

            coords = []
            for i in range(12, len(fields), 3):
                if i + 2 < len(fields):
                    try:
                        coords.append((float(fields[i]), float(fields[i + 1]), float(fields[i + 2])))
                    except ValueError:
                        break

            if stype == "Wall" and len(coords) >= 4:
                w = math.sqrt((coords[2][0] - coords[1][0]) ** 2 + (coords[2][1] - coords[1][1]) ** 2)
                h = abs(coords[0][2] - coords[1][2])
                result["walls"][name] = {"area": w * h}

            elif stype in {"Roof", "Floor"} and len(coords) >= 3:
                area = 0.0
                for j in range(len(coords)):
                    x1, y1 = coords[j][0], coords[j][1]
                    x2, y2 = coords[(j + 1) % len(coords)][0], coords[(j + 1) % len(coords)][1]
                    area += x1 * y2 - x2 * y1
                result["roof" if stype == "Roof" else "floor"] = abs(area) / 2

        if fields[0] == "FenestrationSurface:Detailed":
            name = fields[1]
            coords = []
            for i in range(10, len(fields), 3):
                if i + 2 < len(fields):
                    try:
                        coords.append((float(fields[i]), float(fields[i + 1]), float(fields[i + 2])))
                    except ValueError:
                        break
            if len(coords) >= 4:
                w = math.sqrt((coords[2][0] - coords[1][0]) ** 2 + (coords[2][1] - coords[1][1]) ** 2)
                h = abs(coords[0][2] - coords[1][2])
                result["windows"][name] = {"area": w * h}

    return result


class TestBuildingGeometry:
    """Core geometry and envelope aggregation tests."""

    def test_rectangular_geometry_metrics_and_balance(self):
        geom = BuildingGeometry.from_polygon(
            [(0, 0), (20, 0), (20, 10), (0, 10)],
            num_floors=2,
            floor_height=3.0,
        )

        assert math.isclose(geom.footprint_area, 200.0, rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(geom.floor_area, 400.0, rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(geom.total_height, 6.0, rel_tol=0.0, abs_tol=1e-9)

        geom.set_wwr_cardinal({'N': 0.10, 'S': 0.30, 'E': 0.20, 'W': 0.20})
        p = geom.get_5r1c_params()

        # Envelope consistency: gross walls = net walls + windows
        assert math.isclose(
            p['total_wall_area_gross'],
            p['total_wall_area_net'] + p['total_window_area'],
            rel_tol=0.0,
            abs_tol=1e-6,
        )

    def test_eplus_surface_generation_includes_windows(self):
        geom = BuildingGeometry.from_rectangle(length=20, width=10, num_floors=1, floor_height=3.0)
        geom.set_wwr_uniform(0.25)

        surfaces = geom.get_eplus_surfaces()
        walls = [s for s in surfaces if s['type'] == 'wall']
        windows = [s for s in surfaces if s['type'] == 'window']

        assert len(walls) == len(geom.wall_segments)
        assert len(windows) == len(geom.wall_segments)
        assert all(len(s['vertices']) == 4 for s in walls)
        assert all(len(s['vertices']) == 4 for s in windows)

    def test_compare_idf_style_area_balance(self):
        """Mirror _compare_idfs.py style checks using geometry-generated E+ surfaces."""
        geom = BuildingGeometry.from_polygon(
            [(0, 0), (30, 0), (30, 12), (18, 12), (18, 20), (0, 20)],
            num_floors=3,
            floor_height=2.8,
        )
        geom.set_wwr_cardinal({'N': 0.10, 'S': 0.30, 'E': 0.20, 'W': 0.15})

        p = geom.get_5r1c_params()
        surfaces = geom.get_eplus_surfaces()
        walls = [s for s in surfaces if s['type'] == 'wall']
        windows = [s for s in surfaces if s['type'] == 'window']

        wall_area_from_surfaces = sum(_vertical_quad_area(s['vertices']) for s in walls)
        window_area_from_surfaces = sum(_vertical_quad_area(s['vertices']) for s in windows)
        net_wall_from_surfaces = wall_area_from_surfaces - window_area_from_surfaces
        wwr_from_surfaces = window_area_from_surfaces / wall_area_from_surfaces

        # Equivalent of compare_idfs summary invariants
        assert math.isclose(wall_area_from_surfaces, p['total_wall_area_gross'], rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(window_area_from_surfaces, p['total_window_area'], rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(net_wall_from_surfaces, p['total_wall_area_net'], rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(wwr_from_surfaces, p['total_window_area'] / p['total_wall_area_gross'], rel_tol=0.0, abs_tol=1e-9)

        # Floor/roof terms used by 5R1C params
        assert math.isclose(p['roof_area_m2'], geom.footprint_area, rel_tol=0.0, abs_tol=1e-9)
        assert math.isclose(p['floor_area_m2'], geom.floor_area, rel_tol=0.0, abs_tol=1e-9)

    def test_cardinal_wwr_preserved_for_complex_shape(self):
        """Per-cardinal WWR targets should be preserved after surface generation."""
        geom = BuildingGeometry.from_polygon(
            [(0, 0), (28, 0), (28, 10), (18, 10), (18, 22), (0, 22)],
            num_floors=2,
            floor_height=2.9,
        )
        expected = {'N': 0.12, 'S': 0.29, 'E': 0.21, 'W': 0.17}
        geom.set_wwr_cardinal(expected)

        surfaces = geom.get_eplus_surfaces()
        walls = [s for s in surfaces if s['type'] == 'wall']
        windows = [s for s in surfaces if s['type'] == 'window']

        wall_area = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        for w in walls:
            wall_area[w['cardinal']] += _vertical_quad_area(w['vertices'])

        segment_to_cardinal = {seg.segment_id: seg.cardinal_direction for seg in geom.wall_segments}
        window_area = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        for win in windows:
            cardinal = segment_to_cardinal[win['segment_id']]
            window_area[cardinal] += _vertical_quad_area(win['vertices'])

        for cardinal in ['N', 'S', 'E', 'W']:
            if wall_area[cardinal] > 0:
                realized = window_area[cardinal] / wall_area[cardinal]
                assert math.isclose(realized, expected[cardinal], rel_tol=0.0, abs_tol=1e-6)


@pytest.mark.skipif(
    not _HAS_FOOTPRINT_IMPORTER or importlib.util.find_spec("pyproj") is None,
    reason="FootprintImporter (swe_load_sim) or pyproj not available",
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

        geom = FootprintImporter.from_osm(url="https://www.openstreetmap.org/way/90851303")

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

        geom = FootprintImporter.from_geojson(geojson)
        assert geom.num_floors == 3
        assert geom.footprint_area > 1.0

    def test_from_osm_invalid_url_raises(self):
        try:
            FootprintImporter.from_osm(url="https://www.openstreetmap.org/node/123")
            assert False, "Expected ValueError for non-way/non-relation OSM URL"
        except ValueError as e:
            assert "Cannot parse OSM URL" in str(e)


class TestPolygonIdfIntegration:
    """End-to-end text IDF checks similar to _compare_idfs.py."""

    def test_polygon_idf_geometry_matches_5r1c_params(self, tmp_path):
        geom = BuildingGeometry.from_polygon(
            [(0, 0), (24, 0), (24, 10), (14, 10), (14, 18), (0, 18)],
            num_floors=2,
            floor_height=2.8,
        )
        geom.set_wwr_cardinal({'N': 0.12, 'S': 0.28, 'E': 0.20, 'W': 0.16})

        idf_path = tmp_path / "poly_test.idf"
        _write_polygon_idf_for_test(geom, idf_path)

        extracted = _extract_from_idf(idf_path)
        p = geom.get_5r1c_params()

        wall_area_idf = sum(v["area"] for v in extracted["walls"].values())
        window_area_idf = sum(v["area"] for v in extracted["windows"].values())

        assert extracted["floor"] is not None
        assert extracted["roof"] is not None

        assert math.isclose(extracted["floor"], p["roof_area_m2"], rel_tol=0.0, abs_tol=1e-3)
        assert math.isclose(extracted["roof"], p["roof_area_m2"], rel_tol=0.0, abs_tol=1e-3)
        assert math.isclose(wall_area_idf, p["total_wall_area_gross"], rel_tol=0.0, abs_tol=1e-3)
        assert math.isclose(window_area_idf, p["total_window_area"], rel_tol=0.0, abs_tol=1e-3)

        net_wall_idf = wall_area_idf - window_area_idf
        assert math.isclose(net_wall_idf, p["total_wall_area_net"], rel_tol=0.0, abs_tol=1e-3)
