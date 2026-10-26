# File: src/epsm_rc/building_geometry.py
# Purpose: Compatibility inspection and authoring through Core geometry.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
# License: See LICENSE at the project root.
# Date modified: 2026-10-20
# Changelog:
#   2026-10-20 - Document explicit inputs, units, mutation and I/O boundaries.
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
    """Read-only view of one Core facade in its local metric frame.

    Parameters
    ----------
    segment_id : int
        Compatibility index in the Core facade catalogue.
    facade : epsm_core.geometry.FacadeInfo
        Core facade endpoints, azimuth and explicit storey elevations.

    Notes
    -----
    All lengths use metres and areas use square metres. Inspection performs no I/O;
    the authoritative physical geometry remains in Core.
    """

    segment_id: int
    facade: FacadeInfo

    @property
    def facade_id(self) -> str:
        """Stable Core facade identifier.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self.facade.facade_id

    @property
    def start(self) -> tuple[float, float]:
        """Start (x, y) coordinates in metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self.facade.start_m

    @property
    def end(self) -> tuple[float, float]:
        """End (x, y) coordinates in metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self.facade.end_m

    @property
    def length(self) -> float:
        """Facade length in metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self.facade.length_m

    @property
    def height(self) -> float:
        """Facade top elevation in metres above the footprint plane.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self.facade.storey_elevations_m[-1]

    @property
    def area(self) -> float:
        """Gross facade area in square metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self.length * self.height

    @property
    def normal_angle(self) -> float:
        """Core outward-normal azimuth in degrees clockwise from north.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self.facade.azimuth_deg

    @property
    def cardinal_direction(self) -> str:
        """N/E/S/W group containing the outward-normal azimuth.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return "NESW"[int((self.normal_angle + 45) // 90) % 4]

    @property
    def midpoint(self) -> tuple[float, float]:
        """Midpoint (x, y) coordinates in metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return tuple((a + b) / 2 for a, b in zip(self.start, self.end))


class BuildingGeometry:
    """Inspect explicit metric footprints through Core's physical geometry.

    Parameters
    ----------
    vertices : sequence of tuple of float
        Exterior (x, y) ring in metres. Use ``from_shapely`` to retain holes or parts.
    num_floors : int
        Explicit positive storey count; omission is rejected.
    floor_height : float
        Explicit positive floor-to-floor height in metres; omission is rejected.

    Notes
    -----
    Requires ``epsm_rc[platform]``. Construction performs in-memory Core validation;
    it reads no files or catalogue. Geometry is held on ``core_geometry``. WWR starts
    missing and the setters replace the selected facade assignments. The wrapper
    supplies no construction, glazing, HVAC or operation assumptions.

    Examples
    --------
    >>> from epsm_rc import BuildingGeometry
    >>> geometry = BuildingGeometry.from_rectangle(20, 10, num_floors=2, floor_height=3)
    >>> geometry.floor_area
    400.0
    >>> geometry.set_wwr_uniform(0.2)
    >>> geometry.get_5r1c_params()["total_window_area"] > 0
    True
    """

    def __init__(
        self,
        vertices: Sequence[tuple[float, float]],
        num_floors: int | None = None,
        floor_height: float | None = None,
    ):
        self._initialize(Polygon(vertices), num_floors, floor_height)

    def _initialize(self, polygon: Polygon | MultiPolygon, num_floors, floor_height):
        if num_floors is None or floor_height is None:
            raise ValueError(
                "Supply explicit num_floors and floor_height; Core does not infer heights"
            )
        core = _core()
        from epsm_core.geometry import facade_catalog

        self._core_geometry = core.GeometrySpec.from_footprint(
            polygon,
            storeys=num_floors,
            storey_height_m=floor_height,
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
        """Authoritative Core GeometrySpec; wrapper evidence remains separate.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self._core_geometry

    @property
    def num_floors(self) -> int:
        """Storey count of the first part; wrapper construction uses common heights.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self.core_geometry.parts[0].floors

    @property
    def floor_height(self) -> float:
        """First storey interval height in metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        elevations = self.core_geometry.parts[0].storey_elevations_m
        return elevations[1] - elevations[0]

    @property
    def window_layout(self) -> WindowLayoutSpec:
        """Explicit Core window placement policy; assigning it validates a new selection.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self._window_layout

    @window_layout.setter
    def window_layout(self, layout: WindowLayoutSpec | dict):
        self._window_layout = _core().WindowLayoutSpec.model_validate(
            layout.model_dump() if hasattr(layout, "model_dump") else layout,
        )

    @property
    def wall_segments(self) -> tuple[WallSegment, ...]:
        """Read-only tuple of compatibility views over Core facades.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self._wall_segments

    @classmethod
    def from_polygon(
        cls,
        vertices: Sequence[tuple[float, float]] | str,
        num_floors: int | None = None,
        floor_height: float | None = None,
    ) -> BuildingGeometry:
        """Create geometry from a metric exterior ring with explicit heights.

        Parameters
        ----------
        vertices : sequence of tuple of float or str
            Exterior (x, y) points in metres; a string is parsed as a Python literal.
        num_floors : int
            Explicit positive storey count.
        floor_height : float
            Explicit positive storey height in metres.

        Returns
        -------
        BuildingGeometry
            New Core-backed wrapper; the supplied sequence is not mutated.

        Raises
        ------
        ValueError
            Footprint or height inputs are invalid or missing.
        ImportError
            The coordinated platform extra is unavailable.

        Examples
        --------
        >>> from epsm_rc import BuildingGeometry
        >>> geometry = BuildingGeometry.from_polygon([(0, 0), (10, 0), (10, 5), (0, 5)], 1, 3)
        >>> geometry.footprint_area
        50.0
        """
        vertices = ast.literal_eval(vertices) if isinstance(vertices, str) else vertices
        return cls(vertices, num_floors, floor_height)

    @classmethod
    def from_rectangle(
        cls,
        length: float,
        width: float,
        num_floors: int | None = None,
        floor_height: float | None = None,
        orientation: float = 0.0,
    ) -> BuildingGeometry:
        """Create a centred metric rectangle, rotated counterclockwise.

        Parameters
        ----------
        length, width : float
            Positive side lengths in metres.
        num_floors : int
            Explicit positive storey count.
        floor_height : float
            Explicit positive storey height in metres.
        orientation : float, default 0
            Counterclockwise rotation in degrees around the centred footprint.

        Returns
        -------
        BuildingGeometry
            New geometry with no selected WWR or physical construction assumptions.

        Raises
        ------
        ValueError
            Core rejects the dimensions, footprint or heights.
        ImportError
            The platform extra is unavailable.

        Examples
        --------
        >>> from epsm_rc import BuildingGeometry
        >>> BuildingGeometry.from_rectangle(10, 5, 1, 3).volume
        150.0
        """
        polygon = _core().Footprint.rectangle(length, width).polygon
        polygon = translate(polygon, -length / 2, -width / 2)
        return cls.from_shapely(
            rotate(polygon, orientation, origin=(0, 0)), num_floors, floor_height
        )

    @classmethod
    def from_shapely(
        cls,
        polygon: Polygon | MultiPolygon,
        num_floors: int | None = None,
        floor_height: float | None = None,
    ) -> BuildingGeometry:
        """Preserve valid metric polygon rings and parts in one Core frame.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
            Two-dimensional footprint in metres, including holes and disconnected parts.
        num_floors : int
            Explicit positive storey count applied to each part.
        floor_height : float
            Explicit positive storey height in metres.

        Returns
        -------
        BuildingGeometry
            New wrapper retaining holes and relative part positions; no input is mutated.

        Raises
        ------
        ValueError
            Geometry type, validity, dimensions or height inputs are unsupported.
        ImportError
            The platform extra is unavailable.

        Examples
        --------
        >>> from shapely.geometry import Polygon
        >>> from epsm_rc import BuildingGeometry
        >>> footprint = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])
        >>> BuildingGeometry.from_shapely(footprint, 1, 3).footprint_area
        50.0
        """
        if polygon.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                f"Core authoring requires Polygon or MultiPolygon, got {polygon.geom_type}"
            )
        result = cls.__new__(cls)
        result._initialize(polygon, num_floors, floor_height)
        return result

    @property
    def shapely_polygon(self) -> Polygon | MultiPolygon:
        """New Shapely view preserving every ring and disconnected part in metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        polygons = [part.footprint.polygon for part in self.core_geometry.parts]
        return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)

    @property
    def vertices(self) -> list[tuple[float, float]]:
        """Exterior coordinates in metres; rejects holes/multipart to prevent information loss.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        polygon = self.shapely_polygon
        if polygon.geom_type != "Polygon" or polygon.interiors:
            raise ValueError(
                "A single vertices list cannot preserve this geometry; use core_geometry.parts"
            )
        return list(polygon.exterior.coords)[:-1]

    @property
    def is_valid(self) -> bool:
        """Whether the assembled Shapely footprint is valid.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self.shapely_polygon.is_valid

    @property
    def footprint_area(self) -> float:
        """Total net footprint area in square metres, excluding holes.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return sum(part.footprint.area_m2 for part in self.core_geometry.parts)

    @property
    def floor_area(self) -> float:
        """Core conditioned floor area in square metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self.core_geometry.conditioned_floor_area_m2

    @property
    def volume(self) -> float:
        """Core conditioned volume in cubic metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self.core_geometry.conditioned_volume_m3

    @property
    def perimeter(self) -> float:
        """Sum of footprint boundary lengths in metres, including hole boundaries.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return sum(part.footprint.perimeter_m for part in self.core_geometry.parts)

    @property
    def total_height(self) -> float:
        """Highest part top elevation in metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return max(part.storey_elevations_m[-1] for part in self.core_geometry.parts)

    @property
    def total_wall_area(self) -> float:
        """Sum of gross Core facade areas in square metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return sum(segment.area for segment in self.wall_segments)

    @property
    def centroid(self) -> tuple[float, float]:
        """Footprint area centroid (x, y) in metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        centroid = self.shapely_polygon.centroid
        return centroid.x, centroid.y

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Footprint min-x, min-y, max-x, max-y in metres.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return self.shapely_polygon.bounds

    @staticmethod
    def _validate_wwr(value: float) -> float:
        if not math.isfinite(value) or not 0 <= value < 0.95:
            raise ValueError("Core WWR must be finite and in [0, 0.95)")
        return float(value)

    def set_wwr_uniform(self, wwr: float) -> None:
        """Replace facade assignments with one explicit window-to-wall ratio.

        Parameters
        ----------
        wwr : float
            Finite ratio in [0, 0.95), including an explicit zero-window selection.

        Returns
        -------
        None
            Mutates only the wrapper's WWR selection; performs no I/O.

        Raises
        ------
        ValueError
            The ratio is nonfinite or outside the supported range.
        """
        value = self._validate_wwr(wwr)
        self._wwr_per_segment = {segment.segment_id: value for segment in self.wall_segments}

    def set_wwr_cardinal(self, wwr: Mapping[str, float | None]) -> None:
        """Replace selections using only supplied cardinal facade ratios.

        Parameters
        ----------
        wwr : mapping of str to float or None
            Keys N/E/S/W or their full names; ratios lie in [0, 0.95). Omitted/None
            directions remain missing rather than being assigned defaults.

        Returns
        -------
        None
            Replaces the wrapper's selections without modifying Core geometry.

        Raises
        ------
        ValueError
            A direction is unknown or duplicated, or a supplied ratio is invalid.
        """
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
            for segment in self.wall_segments
            if segment.cardinal_direction in selected
        }

    def set_wwr_per_segment(self, wwr: Mapping[int, float | None]) -> None:
        """Replace WWR selections using explicit compatibility segment IDs.

        Parameters
        ----------
        wwr : mapping of int to float or None
            Known segment IDs and ratios in [0, 0.95). Omitted/None facades stay missing.

        Returns
        -------
        None
            Mutates wrapper selections only; no files or services are accessed.

        Raises
        ------
        ValueError
            An ID is unknown or a ratio is invalid.
        """
        if set(wwr) - {segment.segment_id for segment in self.wall_segments}:
            raise ValueError("Unknown Core facade segment identifier")
        self._wwr_per_segment = {
            key: self._validate_wwr(value) for key, value in wwr.items() if value is not None
        }

    @property
    def wwr_per_segment(self) -> dict[int, float]:
        """Detached mapping of explicitly selected segment ratios.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
        return dict(self._wwr_per_segment)

    @property
    def wwr_cardinal(self) -> dict[str, float | None]:
        """Area-weighted cardinal ratios; None preserves missing facade selections.

        Notes
        -----
        Inspection performs no I/O and does not alter the physical geometry.
        """
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
        missing = [
            s.facade_id for s in self.wall_segments if s.segment_id not in self._wwr_per_segment
        ]
        if missing:
            raise ValueError(
                f"Missing WWR for Core facades: {', '.join(missing)}; supply explicit values"
            )

    def get_wall_areas_by_cardinal(self) -> dict[str, float]:
        """Sum gross facade areas by the Core facade azimuth's cardinal group.

        Returns
        -------
        dict of str to float
            Gross wall areas in square metres for N/E/S/W. No I/O or mutation occurs.
        """
        return {
            direction: sum(s.area for s in self.wall_segments if s.cardinal_direction == direction)
            for direction in "NESW"
        }

    def get_window_areas_by_cardinal(self) -> dict[str, float]:
        """Sum actual Core window rectangles over the explicit storey intervals.

        Returns
        -------
        dict of str to float
            Window areas in square metres for N/E/S/W, using ``window_layout``.

        Raises
        ------
        ValueError
            WWR is missing for any facade. No default ratio is supplied.

        Notes
        -----
        This in-memory calculation delegates placement to Core and performs no I/O.
        """
        from epsm_core.facades import window_rectangles

        self._require_wwr()
        areas = dict.fromkeys("NESW", 0.0)
        for segment in self.wall_segments:
            elevations = segment.facade.storey_elevations_m
            for bottom, top in pairwise(elevations):
                rectangles, _ = window_rectangles(
                    segment.length,
                    top - bottom,
                    self._wwr_per_segment[segment.segment_id],
                    self.window_layout,
                )
                areas[segment.cardinal_direction] += sum(
                    (right - left) * (head - sill) for left, sill, right, head in rectangles
                )
        return areas

    def get_5r1c_params(self) -> dict[str, float]:
        """Calculate historical area keys from Core geometry and actual windows.

        Returns
        -------
        dict
            Floor/roof/window/wall areas in square metres, heights/perimeter in metres,
            volume in cubic metres and integer storey count. These are compatibility
            geometry keys, not a fully authored physical Building or solver parameters.

        Raises
        ------
        ValueError
            Any facade has missing WWR.

        Notes
        -----
        Performs Core window placement in memory. It mutates no inputs and performs no I/O.
        """
        windows = self.get_window_areas_by_cardinal()
        total_windows = sum(windows.values())
        return {
            **{
                f"window_area_{name}": windows[direction]
                for direction, name in (
                    ("N", "north"),
                    ("E", "east"),
                    ("S", "south"),
                    ("W", "west"),
                )
            },
            "floor_area_m2": self.floor_area,
            "roof_area_m2": self.footprint_area,
            "total_wall_area_gross": self.total_wall_area,
            "total_wall_area_net": self.total_wall_area - total_windows,
            "total_window_area": total_windows,
            "perimeter": self.perimeter,
            "num_floors": self.num_floors,
            "floor_height": self.floor_height,
            "total_height": self.total_height,
            "volume": self.volume,
            "footprint_area": self.footprint_area,
        }

    def to_core_facade(self, glazing: GlazingSpec | dict) -> FacadeSpec:
        """Create an explicit Core facade selection from the wrapper's WWR.

        Parameters
        ----------
        glazing : epsm_core.GlazingSpec or dict
            Explicit whole-window glazing properties; no U-value or SHGC is inferred.

        Returns
        -------
        epsm_core.FacadeSpec
            New facade carrying selected ratios and the explicit window layout.

        Raises
        ------
        ValueError
            Glazing or any facade ratio is missing or invalid.

        Notes
        -----
        This is an in-memory conversion. The wrapper is not changed.
        """
        core = _core()
        self._require_wwr()
        if glazing is None:
            raise ValueError("Supply explicit glazing for Core facade authoring")
        glazing = core.GlazingSpec.model_validate(
            glazing.model_dump() if hasattr(glazing, "model_dump") else glazing,
        )
        return core.FacadeSpec(
            default_wwr=next(iter(self._wwr_per_segment.values())),
            glazing=glazing,
            layout=self.window_layout,
            overrides=tuple(
                core.FacadeOverride(
                    facade_id=segment.facade_id,
                    wwr=self._wwr_per_segment[segment.segment_id],
                )
                for segment in self.wall_segments
            ),
        )

    def to_core_building(
        self,
        *,
        glazing: GlazingSpec | dict | None = None,
        construction_set=None,
        hvac=None,
        operation=None,
        provenance=None,
    ) -> Building:
        """Create an incomplete Core draft from only explicitly supplied inputs.

        Parameters
        ----------
        glazing : epsm_core.GlazingSpec or dict, optional
            Required when transferring any selected WWR; no glazing is inferred.
        construction_set : epsm_core.ConstructionSet, optional
            Explicit constructions, retained on the new draft.
        hvac : epsm_core.HVACSpec, optional
            Explicit systems and airflow; omission remains visible in readiness.
        operation : epsm_core.UseSpec, optional
            Explicit operating assumptions; omission remains missing.
        provenance : epsm_core.ProvenanceSpec or dict, optional
            Supplied source records, augmented with selected GIS height evidence.

        Returns
        -------
        epsm_core.Building
            New empty-derived physical draft. Missing inputs remain missing.

        Raises
        ------
        ValueError
            Selected WWR lacks explicit glazing or Core rejects supplied inputs.
        ImportError
            The platform extra is unavailable.

        Notes
        -----
        Performs no file access, catalogue lookup or simulation; the wrapper is unchanged.

        Examples
        --------
        >>> from epsm_rc import BuildingGeometry
        >>> geometry = BuildingGeometry.from_rectangle(10, 5, 1, 3)
        >>> draft = geometry.to_core_building()
        >>> draft.geometry.conditioned_floor_area_m2
        50.0
        """
        if self._wwr_per_segment and glazing is None:
            raise ValueError(
                "Supply explicit glazing to transfer selected WWR; no glazing is assumed"
            )
        facade = self.to_core_facade(glazing) if glazing is not None else None
        core = _core()
        supplied = core.ProvenanceSpec.model_validate(
            provenance.model_dump(mode="json")
            if hasattr(provenance, "model_dump")
            else provenance or {}
        )
        records = list(supplied.records)
        geometry = self.core_geometry
        if self.height_evidence:
            parts = []
            inferred = all(
                self.height_evidence.get(name) != "caller"
                for name in ("num_floors_source", "floor_height_source")
            )
            for index, part in enumerate(geometry.parts):
                path = f"geometry.parts.{index}.storey_elevations_m"
                record = next((r for r in records if r.field_path == path), None)
                if record is None:
                    record = core.SourceRecord(
                        field_path=path,
                        status="inferred" if inferred else "assumed",
                        source="gis_height_selection",
                        evidence=self.height_evidence,
                    )
                    records.append(record)
                parts.append(part.updated(height_status=record.status))
            geometry = geometry.updated(parts=tuple(parts))
        return core.Building.empty(
            geometry=geometry,
            facade=facade,
            construction_set=construction_set,
            hvac=hvac,
            operation=operation,
            provenance=core.ProvenanceSpec(records=tuple(records)),
        )

    def get_eplus_surfaces(self):
        """Report the retired independent EnergyPlus surface-export interface.

        Raises
        ------
        ValueError
            Always raised with migration instructions for Core's EnergyPlusCompiler.

        Notes
        -----
        No surfaces are generated and no files are written. This compatibility error
        remains until a documented removal no earlier than 0.11.0.
        """
        raise ValueError(
            "Legacy EnergyPlus surface export is retired. Author an explicit Core Building "
            "and compile its to_spec(weather) with epsm_core.EnergyPlusCompiler; "
            "BuildingGeometry.core_geometry preserves all rings and parts."
        )

    def summary(self) -> str:
        """Describe part count, areas, volume and selected facade coverage.

        Returns
        -------
        str
            Bounded text with area in square metres and volume in cubic metres.
            Inspection performs no I/O or simulation.
        """
        return (
            f"Core-backed BuildingGeometry: {len(self.core_geometry.parts)} parts, "
            f"{self.floor_area:g} m² floor area, {self.volume:g} m³ volume; "
            f"WWR supplied on {len(self._wwr_per_segment)}/{len(self.wall_segments)} facades"
        )

    def __repr__(self):
        return self.summary()
