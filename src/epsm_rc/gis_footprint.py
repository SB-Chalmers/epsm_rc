"""
Import building footprints from GIS sources → BuildingGeometry.

Three entry-points, one return type:

    from epsm_rc.gis_footprint import FootprintImporter

    # 1. OpenStreetMap way/relation ID
    geom = FootprintImporter.from_osm(way_id=90851303)

    # 2. GeoJSON file or dict
    geom = FootprintImporter.from_geojson("building.geojson")
    geom = FootprintImporter.from_geojson({"type": "Feature", ...})

    # 3. Shapefile (EPSG:3006 or auto-detected CRS)
    geoms = FootprintImporter.from_shapefile("footprints.shp")      # all features
    geoms = FootprintImporter.from_shapefile("footprints.shp", fid=0)  # single

All methods return BuildingGeometry (or list thereof) in a local metric
coordinate system with the centroid shifted to the origin.

Coordinate handling:
- WGS84 (EPSG:4326) → UTM auto-zone  (for OSM / GeoJSON in lon/lat)
- EPSG:3006 (SWEREF99 TM) → used directly (already metric)
- Other projected CRS → used directly if units are metres
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import shape as shapely_shape
from shapely.ops import orient

from .building_geometry import BuildingGeometry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _to_local_metres(
    coords: Sequence[Tuple[float, float]],
    source_crs: Optional[str] = None,
) -> List[Tuple[float, float]]:
    """Convert coordinates to local metric system centred on centroid.

    Args:
        coords: List of (x, y) or (lon, lat) tuples (exterior ring, no closing point).
        source_crs: EPSG string like "EPSG:4326" or "EPSG:3006". 
                     If None, auto-detect from coordinate magnitude.

    Returns:
        List of (x, y) in metres, centroid-shifted to origin.
    """
    from pyproj import CRS, Transformer

    arr = np.array(coords, dtype=float)

    # Auto-detect CRS if not given
    if source_crs is None:
        x_range = arr[:, 0].max() - arr[:, 0].min()
        y_range = arr[:, 1].max() - arr[:, 1].min()
        mean_x = arr[:, 0].mean()
        # Heuristic: lon/lat values are small (-180..180 / -90..90),
        # projected coords (SWEREF99 TM) are in the 100k–1M range.
        if abs(mean_x) < 360 and x_range < 10 and y_range < 10:
            source_crs = "EPSG:4326"
        elif mean_x > 100_000:
            source_crs = "EPSG:3006"  # Assume Swedish national grid
        else:
            source_crs = "EPSG:4326"

    crs_obj = CRS(source_crs)

    if crs_obj.is_geographic:
        # Project to UTM zone appropriate for the centroid
        mean_lon = arr[:, 0].mean()
        mean_lat = arr[:, 1].mean()
        utm_zone = int((mean_lon + 180) / 6) + 1
        hemisphere = "north" if mean_lat >= 0 else "south"
        target_crs = CRS(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")
        transformer = Transformer.from_crs(crs_obj, target_crs, always_xy=True)
        xs, ys = transformer.transform(arr[:, 0], arr[:, 1])
        arr = np.column_stack([xs, ys])
    elif not crs_obj.axis_info[0].unit_name.startswith("metre"):
        # Non-metric projected CRS — try reprojecting to UTM
        logger.warning(
            "CRS %s has non-metre units (%s), reprojecting to UTM",
            source_crs, crs_obj.axis_info[0].unit_name,
        )
        mean_x, mean_y = arr[:, 0].mean(), arr[:, 1].mean()
        # Convert to WGS84 first, then UTM
        to_wgs = Transformer.from_crs(crs_obj, CRS("EPSG:4326"), always_xy=True)
        lon, lat = to_wgs.transform(mean_x, mean_y)
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = "north" if lat >= 0 else "south"
        target_crs = CRS(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")
        transformer = Transformer.from_crs(crs_obj, target_crs, always_xy=True)
        xs, ys = transformer.transform(arr[:, 0], arr[:, 1])
        arr = np.column_stack([xs, ys])
    # else: already in a metric projected CRS (e.g. EPSG:3006) — keep as-is

    # Shift centroid to origin for BuildingGeometry
    cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
    arr[:, 0] -= cx
    arr[:, 1] -= cy

    return [(float(x), float(y)) for x, y in arr]


def _polygon_to_geometry(
    poly: ShapelyPolygon,
    source_crs: Optional[str],
    num_floors: int,
    floor_height: float,
) -> BuildingGeometry:
    """Convert a Shapely polygon to BuildingGeometry with CRS handling."""
    oriented = orient(poly, sign=1.0)  # CCW
    coords = list(oriented.exterior.coords)[:-1]  # Drop closing vertex
    local = _to_local_metres(coords, source_crs=source_crs)
    return BuildingGeometry.from_polygon(
        vertices=local,
        num_floors=num_floors,
        floor_height=floor_height,
    )


# ---------------------------------------------------------------------------
# Main importer class
# ---------------------------------------------------------------------------

class FootprintImporter:
    """Import building footprints from OSM, GeoJSON, or Shapefile."""

    # ----- OpenStreetMap -----

    @staticmethod
    def from_osm(
        way_id: Optional[int] = None,
        relation_id: Optional[int] = None,
        url: Optional[str] = None,
        *,
        num_floors: int = 2,
        floor_height: float = 2.5,
    ) -> BuildingGeometry:
        """Fetch a building footprint from OpenStreetMap.

        Args:
            way_id: OSM way ID (e.g. 90851303).
            relation_id: OSM relation ID (for multipolygon buildings).
            url: Full OSM URL — way_id/relation_id extracted automatically.
            num_floors: Number of floors (default 2).
            floor_height: Floor-to-floor height [m] (default 2.5).

        Returns:
            BuildingGeometry in local metric coordinates.

        Example:
            >>> geom = FootprintImporter.from_osm(way_id=90851303)
            >>> geom = FootprintImporter.from_osm(url="https://www.openstreetmap.org/way/90851303")
        """
        import re
        import requests

        # Parse URL if given
        if url is not None:
            m = re.search(r"(way|relation)/(\d+)", url)
            if not m:
                raise ValueError(f"Cannot parse OSM URL: {url}")
            kind, osm_id = m.group(1), int(m.group(2))
            if kind == "way":
                way_id = osm_id
            else:
                relation_id = osm_id

        if way_id is not None:
            element_type, element_id = "way", way_id
        elif relation_id is not None:
            element_type, element_id = "relation", relation_id
        else:
            raise ValueError("Provide way_id, relation_id, or url")

        # Query Overpass API for geometry
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        {element_type}({element_id});
        (._;>;);
        out body;
        """
        logger.info("Fetching OSM %s/%d via Overpass...", element_type, element_id)
        resp = requests.get(overpass_url, params={"data": query}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Build node lookup
        nodes: Dict[int, Tuple[float, float]] = {}
        way_nodes: Optional[List[int]] = None
        tags: Dict[str, str] = {}

        for el in data["elements"]:
            if el["type"] == "node":
                nodes[el["id"]] = (el["lon"], el["lat"])
            elif el["type"] == "way" and el["id"] == element_id:
                way_nodes = el["nodes"]
                tags = el.get("tags", {})
            elif el["type"] == "way" and element_type == "relation":
                # For relations, collect all ways (take outer ring)
                if way_nodes is None:
                    way_nodes = el["nodes"]

        if way_nodes is None:
            raise ValueError(f"OSM {element_type}/{element_id}: no way nodes found")

        # Resolve coordinates (lon, lat)
        coords = []
        for nid in way_nodes:
            if nid in nodes:
                coords.append(nodes[nid])
        # Drop closing vertex if same as first
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]

        if len(coords) < 3:
            raise ValueError(
                f"OSM {element_type}/{element_id}: only {len(coords)} vertices"
            )

        # Try to get floor count from OSM tags
        if num_floors == 2:  # only override if user didn't specify
            for tag in ("building:levels", "levels"):
                if tag in tags:
                    try:
                        num_floors = int(tags[tag])
                        logger.info("OSM tag %s=%d", tag, num_floors)
                    except ValueError:
                        pass
                    break

        poly = ShapelyPolygon(coords)
        geom = _polygon_to_geometry(
            poly, source_crs="EPSG:4326",
            num_floors=num_floors, floor_height=floor_height,
        )

        # Attach OSM metadata
        geom._osm_tags = tags  # type: ignore[attr-defined]
        logger.info(
            "Imported OSM %s/%d: %.0f m² footprint, %d vertices, tags=%s",
            element_type, element_id, geom.footprint_area,
            len(geom.vertices), {k: tags[k] for k in list(tags)[:5]},
        )
        return geom

    # ----- GeoJSON -----

    @staticmethod
    def from_geojson(
        source: Union[str, Path, Dict],
        *,
        feature_index: int = 0,
        num_floors: int = 2,
        floor_height: float = 2.5,
        crs: Optional[str] = None,
    ) -> BuildingGeometry:
        """Load a building footprint from GeoJSON.

        Args:
            source: File path, or a GeoJSON dict (Feature or FeatureCollection).
            feature_index: Which feature to use (if FeatureCollection).
            num_floors: Number of floors.
            floor_height: Floor-to-floor height [m].
            crs: Override CRS (default: auto-detect; GeoJSON is typically EPSG:4326).

        Returns:
            BuildingGeometry in local metric coordinates.

        Example:
            >>> geom = FootprintImporter.from_geojson("building.geojson")
            >>> geom = FootprintImporter.from_geojson({"type": "Feature", ...})
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            with open(path) as f:
                data = json.load(f)
        else:
            data = source

        # Normalise to feature
        if data.get("type") == "FeatureCollection":
            features = data["features"]
            if feature_index >= len(features):
                raise IndexError(
                    f"feature_index={feature_index} but only {len(features)} features"
                )
            feature = features[feature_index]
        elif data.get("type") == "Feature":
            feature = data
        elif data.get("type") in ("Polygon", "MultiPolygon"):
            feature = {"type": "Feature", "geometry": data, "properties": {}}
        else:
            raise ValueError(f"Unsupported GeoJSON type: {data.get('type')}")

        geom_obj = shapely_shape(feature["geometry"])
        if geom_obj.geom_type == "MultiPolygon":
            # Take the largest polygon
            geom_obj = max(geom_obj.geoms, key=lambda g: g.area)
        if geom_obj.geom_type != "Polygon":
            raise ValueError(f"Expected Polygon, got {geom_obj.geom_type}")

        # Try to infer floors from properties
        props = feature.get("properties", {}) or {}
        if num_floors == 2:
            for key in ("building:levels", "levels", "floors", "num_floors"):
                if key in props:
                    try:
                        num_floors = int(props[key])
                    except (ValueError, TypeError):
                        pass
                    break

        # GeoJSON is EPSG:4326 by spec unless overridden
        effective_crs = crs or "EPSG:4326"
        return _polygon_to_geometry(
            geom_obj, source_crs=effective_crs,
            num_floors=num_floors, floor_height=floor_height,
        )

    @staticmethod
    def from_geojson_multi(
        source: Union[str, Path, Dict],
        *,
        num_floors: int = 2,
        floor_height: float = 2.5,
        crs: Optional[str] = None,
        max_features: Optional[int] = None,
    ) -> List[BuildingGeometry]:
        """Load ALL features from a GeoJSON FeatureCollection.

        Returns:
            List of BuildingGeometry, one per polygon feature.
        """
        if isinstance(source, (str, Path)):
            with open(Path(source)) as f:
                data = json.load(f)
        else:
            data = source

        if data.get("type") != "FeatureCollection":
            # Single feature → wrap
            return [FootprintImporter.from_geojson(
                data, num_floors=num_floors, floor_height=floor_height, crs=crs,
            )]

        results = []
        for i, feat in enumerate(data["features"]):
            if max_features and i >= max_features:
                break
            try:
                g = FootprintImporter.from_geojson(
                    feat, num_floors=num_floors, floor_height=floor_height, crs=crs,
                )
                results.append(g)
            except (ValueError, IndexError) as e:
                logger.warning("Skipping feature %d: %s", i, e)
        return results

    # ----- Shapefile -----

    @staticmethod
    def from_shapefile(
        path: Union[str, Path],
        *,
        fid: Optional[int] = None,
        num_floors: int = 2,
        floor_height: float = 2.5,
        crs_override: Optional[str] = None,
        max_features: Optional[int] = None,
    ) -> Union[BuildingGeometry, List[BuildingGeometry]]:
        """Load building footprint(s) from an ESRI Shapefile.

        Handles EPSG:3006 (SWEREF99 TM) and other projected CRS natively.

        Args:
            path: Path to .shp file.
            fid: Feature index to load (returns single BuildingGeometry).
                 If None, loads all features (returns list).
            num_floors: Number of floors.
            floor_height: Floor-to-floor height [m].
            crs_override: Force a specific CRS (e.g. "EPSG:3006").
            max_features: Max features to load (when fid is None).

        Returns:
            Single BuildingGeometry if fid is set, else list.

        Example:
            >>> geom = FootprintImporter.from_shapefile("buildings.shp", fid=0)
            >>> geoms = FootprintImporter.from_shapefile("buildings.shp")
        """
        import fiona

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Shapefile not found: {path}")

        results: List[BuildingGeometry] = []

        with fiona.open(path) as src:
            # Detect CRS
            file_crs = crs_override
            if file_crs is None and src.crs:
                from pyproj import CRS
                try:
                    crs_obj = CRS(src.crs)
                    file_crs = crs_obj.to_epsg()
                    if file_crs:
                        file_crs = f"EPSG:{file_crs}"
                    else:
                        file_crs = crs_obj.to_wkt()
                except Exception:
                    logger.warning("Could not parse shapefile CRS: %s", src.crs)
                    file_crs = None

            logger.info(
                "Shapefile %s: %d features, CRS=%s",
                path.name, len(src), file_crs,
            )

            if fid is not None:
                feat = src[fid]
                geom_obj = shapely_shape(feat["geometry"])
                if geom_obj.geom_type == "MultiPolygon":
                    geom_obj = max(geom_obj.geoms, key=lambda g: g.area)
                return _polygon_to_geometry(
                    geom_obj, source_crs=file_crs,
                    num_floors=num_floors, floor_height=floor_height,
                )

            for i, feat in enumerate(src):
                if max_features and i >= max_features:
                    break
                try:
                    geom_obj = shapely_shape(feat["geometry"])
                    if geom_obj.geom_type == "MultiPolygon":
                        geom_obj = max(geom_obj.geoms, key=lambda g: g.area)
                    if geom_obj.geom_type != "Polygon":
                        continue
                    g = _polygon_to_geometry(
                        geom_obj, source_crs=file_crs,
                        num_floors=num_floors, floor_height=floor_height,
                    )
                    results.append(g)
                except Exception as e:
                    logger.warning("Skipping feature %d: %s", i, e)

        return results
