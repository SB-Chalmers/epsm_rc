"""Compatibility conveniences backed by Core geometry and window placement.

Heights and WWR are explicit inputs. This module supplies no constructions,
glazing, HVAC or operation assumptions. Core loads only when authoring geometry;
install the platform extra to use these entry points. EnergyPlus export belongs
to Core's EnergyPlusCompiler rather than the retired legacy surface emitter.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import ModuleType
from typing import TYPE_CHECKING

from shapely.affinity import rotate, translate
from shapely.geometry import MultiPolygon, Polygon

if TYPE_CHECKING:
    from epsm_core import Building, FacadeSpec, GeometrySpec, GlazingSpec, WindowLayoutSpec
    from epsm_core.geometry import FacadeInfo


def _core() -> ModuleType:
    """Load the coordinated Core API without making the base RC import depend on it."""
    try:
        import epsm_core

        if not hasattr(epsm_core, "GeometrySpec") or not hasattr(epsm_core, "Building"):
            raise ImportError("The installed Core version lacks the shared authoring API")
        return epsm_core
    except ImportError as error:
        raise ImportError(
            "BuildingGeometry now delegates to epsm_core. Install epsm_rc[platform] "
            "with coordinated Core/Data versions; use Core GeometrySpec and FacadeSpec "
            "for new authoring."
        ) from error


@dataclass(frozen=True)
class WallSegment:
    """Compatibility view of one Core facade, with Core's azimuth and endpoints."""

    segment_id: int
    facade: FacadeInfo

    @property
    def facade_id(self) -> str:
        return self.facade.facade_id

    @property
    def start(self) -> tuple[float, float]:
        return self.facade.start_m

    @property
    def end(self) -> tuple[float, float]:
        return self.facade.end_m

    @property
    def length(self) -> float:
        return self.facade.length_m

    @property
    def height(self) -> float:
        return self.facade.storey_elevations_m[-1]

    @property
    def area(self) -> float:
        return self.length * self.height

    @property
    def normal_angle(self) -> float:
        return self.facade.azimuth_deg

    @property
    def cardinal_direction(self) -> str:
        return "NESW"[int((self.normal_angle + 45) // 90) % 4]

    @property
    def midpoint(self) -> tuple[float, float]:
        return tuple((a + b) / 2 for a, b in zip(self.start, self.end))


class BuildingGeometry:
    """Thin compatibility wrapper around an explicitly authored Core GeometrySpec.

    ``core_geometry`` is the shared physical geometry. WWR may remain missing
    while inspecting a footprint; methods that need windows reject incomplete
    WWR. ``to_core_building`` creates an empty physical draft, with only the
    constructions, glazing, HVAC and operation supplied by the caller.
    """

    def __init__(
        self, vertices: Sequence[tuple[float, float]],
        num_floors: int | None = None, floor_height: float | None = None,
    ):
        self._initialize(Polygon(vertices), num_floors, floor_height)

    def _initialize(self, polygon: Polygon | MultiPolygon, num_floors, floor_height):
        if num_floors is None or floor_height is None:
            raise ValueError("Supply explicit num_floors and floor_height; Core does not infer heights")
        core = _core()
        from epsm_core.geometry import facade_catalog

        self._core_geometry = core.GeometrySpec.from_footprint(
            polygon, storeys=num_floors, storey_height_m=floor_height,
        )
        self._wall_segments = tuple(
            WallSegment(index, facade)
            for index, facade in enumerate(facade_catalog(self.core_geometry))
        )
        self.window_layout = core.WindowLayoutSpec(mode="scaled")
        self.height_evidence = {}
        self._wwr_per_segment = {}

    @property
    def core_geometry(self) -> GeometrySpec:
        """Read-only shared geometry; source evidence remains on the wrapper."""
        return self._core_geometry

    @property
    def num_floors(self) -> int:
        return self.core_geometry.parts[0].floors

    @property
    def floor_height(self) -> float:
        elevations = self.core_geometry.parts[0].storey_elevations_m
        return elevations[1] - elevations[0]

    @property
    def window_layout(self) -> WindowLayoutSpec:
        return self._window_layout

    @window_layout.setter
    def window_layout(self, layout: WindowLayoutSpec | dict):
        self._window_layout = _core().WindowLayoutSpec.model_validate(
            layout.model_dump() if hasattr(layout, "model_dump") else layout,
        )

    @property
    def wall_segments(self) -> tuple[WallSegment, ...]:
        return self._wall_segments

    @classmethod
    def from_polygon(
        cls, vertices: Sequence[tuple[float, float]] | str,
        num_floors: int | None = None, floor_height: float | None = None,
    ) -> BuildingGeometry:
        """Delegate metric polygon vertices and explicit heights to Core validation."""
        vertices = ast.literal_eval(vertices) if isinstance(vertices, str) else vertices
        return cls(vertices, num_floors, floor_height)

    @classmethod
    def from_rectangle(
        cls, length: float, width: float, num_floors: int | None = None,
        floor_height: float | None = None, orientation: float = 0.0,
    ) -> BuildingGeometry:
        """Preserve the legacy centred rectangle and counterclockwise rotation convention."""
        polygon = _core().Footprint.rectangle(length, width).polygon
        polygon = translate(polygon, -length / 2, -width / 2)
        return cls.from_shapely(rotate(polygon, orientation, origin=(0, 0)), num_floors, floor_height)

    @classmethod
    def from_shapely(
        cls, polygon: Polygon | MultiPolygon,
        num_floors: int | None = None, floor_height: float | None = None,
    ) -> BuildingGeometry:
        """Preserve valid metric Polygon/MultiPolygon rings and their shared frame in Core."""
        if polygon.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"Core authoring requires Polygon or MultiPolygon, got {polygon.geom_type}")
        result = cls.__new__(cls)
        result._initialize(polygon, num_floors, floor_height)
        return result

    @property
    def shapely_polygon(self) -> Polygon | MultiPolygon:
        polygons = [part.footprint.polygon for part in self.core_geometry.parts]
        return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)

    @property
    def vertices(self) -> list[tuple[float, float]]:
        polygon = self.shapely_polygon
        if polygon.geom_type != "Polygon" or polygon.interiors:
            raise ValueError("A single vertices list cannot preserve this geometry; use core_geometry.parts")
        return list(polygon.exterior.coords)[:-1]

    @property
    def is_valid(self) -> bool:
        return self.shapely_polygon.is_valid

    @property
    def footprint_area(self) -> float:
        return sum(part.footprint.area_m2 for part in self.core_geometry.parts)

    @property
    def floor_area(self) -> float:
        return self.core_geometry.conditioned_floor_area_m2

    @property
    def volume(self) -> float:
        return self.core_geometry.conditioned_volume_m3

    @property
    def perimeter(self) -> float:
        return sum(part.footprint.perimeter_m for part in self.core_geometry.parts)

    @property
    def total_height(self) -> float:
        return max(part.storey_elevations_m[-1] for part in self.core_geometry.parts)

    @property
    def total_wall_area(self) -> float:
        return sum(segment.area for segment in self.wall_segments)

    @property
    def centroid(self) -> tuple[float, float]:
        centroid = self.shapely_polygon.centroid
        return centroid.x, centroid.y

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.shapely_polygon.bounds

    @staticmethod
    def _validate_wwr(value: float) -> float:
        if not math.isfinite(value) or not 0 <= value < 0.95:
            raise ValueError("Core WWR must be finite and in [0, 0.95)")
        return float(value)

    def set_wwr_uniform(self, wwr: float) -> None:
        """Select one explicit WWR for every Core facade."""
        value = self._validate_wwr(wwr)
        self._wwr_per_segment = {segment.segment_id: value for segment in self.wall_segments}

    def set_wwr_cardinal(self, wwr: Mapping[str, float | None]) -> None:
        """Select supplied cardinal values only; omitted directions remain missing."""
        aliases = {"north": "N", "east": "E", "south": "S", "west": "W"}
        selected = {}
        for name, value in wwr.items():
            if value is None:
                continue
            direction = aliases.get(name.lower(), name.upper())
            if direction not in "NESW" or len(direction) != 1:
                raise ValueError(f"Unknown cardinal WWR direction {name}")
            if direction in selected:
                raise ValueError(f"Duplicate cardinal WWR direction {direction}")
            selected[direction] = self._validate_wwr(value)
        self._wwr_per_segment = {
            segment.segment_id: selected[segment.cardinal_direction]
            for segment in self.wall_segments if segment.cardinal_direction in selected
        }

    def set_wwr_per_segment(self, wwr: Mapping[int, float | None]) -> None:
        """Select supplied segment values only; omitted facades remain missing."""
        if set(wwr) - {segment.segment_id for segment in self.wall_segments}:
            raise ValueError("Unknown Core facade segment identifier")
        self._wwr_per_segment = {
            key: self._validate_wwr(value) for key, value in wwr.items() if value is not None
        }

    @property
    def wwr_per_segment(self) -> dict[int, float]:
        return dict(self._wwr_per_segment)

    @property
    def wwr_cardinal(self) -> dict[str, float | None]:
        result = {}
        for direction in "NESW":
            segments = [s for s in self.wall_segments if s.cardinal_direction == direction]
            if not segments or any(s.segment_id not in self._wwr_per_segment for s in segments):
                result[direction] = None
            else:
                result[direction] = sum(
                    s.area * self._wwr_per_segment[s.segment_id] for s in segments
                ) / sum(s.area for s in segments)
        return result

    def _require_wwr(self):
        missing = [s.facade_id for s in self.wall_segments if s.segment_id not in self._wwr_per_segment]
        if missing:
            raise ValueError(f"Missing WWR for Core facades: {', '.join(missing)}; supply explicit values")

    def get_wall_areas_by_cardinal(self) -> dict[str, float]:
        return {
            direction: sum(s.area for s in self.wall_segments if s.cardinal_direction == direction)
            for direction in "NESW"
        }

    def get_window_areas_by_cardinal(self) -> dict[str, float]:
        """Aggregate actual Core window rectangles over explicit storey intervals."""
        from epsm_core.facades import window_rectangles

        self._require_wwr()
        areas = dict.fromkeys("NESW", 0.0)
        for segment in self.wall_segments:
            elevations = segment.facade.storey_elevations_m
            for bottom, top in pairwise(elevations):
                rectangles, _ = window_rectangles(
                    segment.length, top - bottom,
                    self._wwr_per_segment[segment.segment_id], self.window_layout,
                )
                areas[segment.cardinal_direction] += sum(
                    (right - left) * (head - sill) for left, sill, right, head in rectangles
                )
        return areas

    def get_5r1c_params(self) -> dict[str, float]:
        """Return legacy area keys from Core geometry and actual Core window placement."""
        windows = self.get_window_areas_by_cardinal()
        total_windows = sum(windows.values())
        return {
            **{f"window_area_{name}": windows[direction] for direction, name in
               (("N", "north"), ("E", "east"), ("S", "south"), ("W", "west"))},
            "floor_area_m2": self.floor_area, "roof_area_m2": self.footprint_area,
            "total_wall_area_gross": self.total_wall_area,
            "total_wall_area_net": self.total_wall_area - total_windows,
            "total_window_area": total_windows, "perimeter": self.perimeter,
            "num_floors": self.num_floors, "floor_height": self.floor_height,
            "total_height": self.total_height, "volume": self.volume,
            "footprint_area": self.footprint_area,
        }

    def to_core_facade(self, glazing: GlazingSpec | dict) -> FacadeSpec:
        """Transfer complete selected WWR with explicitly supplied glazing; no physical defaults."""
        core = _core()
        self._require_wwr()
        if glazing is None:
            raise ValueError("Supply explicit glazing for Core facade authoring")
        glazing = core.GlazingSpec.model_validate(
            glazing.model_dump() if hasattr(glazing, "model_dump") else glazing,
        )
        return core.FacadeSpec(
            default_wwr=next(iter(self._wwr_per_segment.values())), glazing=glazing,
            layout=self.window_layout,
            overrides=tuple(core.FacadeOverride(
                facade_id=segment.facade_id, wwr=self._wwr_per_segment[segment.segment_id],
            ) for segment in self.wall_segments),
        )

    def to_core_building(
        self, *, glazing: GlazingSpec | dict | None = None,
        construction_set=None, hvac=None, operation=None, provenance=None,
    ) -> Building:
        """Create a Core empty draft containing only explicitly supplied physical inputs."""
        if self._wwr_per_segment and glazing is None:
            raise ValueError("Supply explicit glazing to transfer selected WWR; no glazing is assumed")
        facade = self.to_core_facade(glazing) if glazing is not None else None
        core = _core()
        supplied = core.ProvenanceSpec.model_validate(
            provenance.model_dump(mode="json") if hasattr(provenance, "model_dump") else provenance or {})
        records = list(supplied.records)
        geometry = self.core_geometry
        if self.height_evidence:
            parts = []
            inferred = all(self.height_evidence.get(name) != "caller" for name in (
                "num_floors_source", "floor_height_source"))
            for index, part in enumerate(geometry.parts):
                path = f"geometry.parts.{index}.storey_elevations_m"
                record = next((r for r in records if r.field_path == path), None)
                if record is None:
                    record = core.SourceRecord(field_path=path,
                        status="inferred" if inferred else "assumed", source="gis_height_selection",
                        evidence=self.height_evidence)
                    records.append(record)
                parts.append(part.updated(height_status=record.status))
            geometry = geometry.updated(parts=tuple(parts))
        return core.Building.empty(
            geometry=geometry, facade=facade, construction_set=construction_set,
            hvac=hvac, operation=operation, provenance=core.ProvenanceSpec(records=tuple(records)),
        )

    def get_eplus_surfaces(self):
        """Fail clearly at the retired surface export; use Core EnergyPlusCompiler."""
        raise ValueError(
            "Legacy EnergyPlus surface export is retired. Author an explicit Core Building "
            "and compile its to_spec(weather) with epsm_core.EnergyPlusCompiler; "
            "BuildingGeometry.core_geometry preserves all rings and parts."
        )

    def summary(self) -> str:
        return (
            f"Core-backed BuildingGeometry: {len(self.core_geometry.parts)} parts, "
            f"{self.floor_area:g} m² floor area, {self.volume:g} m³ volume; "
            f"WWR supplied on {len(self._wwr_per_segment)}/{len(self.wall_segments)} facades"
        )

    def __repr__(self):
        return self.summary()
