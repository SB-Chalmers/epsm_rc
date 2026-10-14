"""Modular Building Geometry System

Handles polygon footprints with flexible window-to-wall ratio (WWR) assignment.
Supports bidirectional conversion between per-segment and cardinal WWR.
Uses Shapely for robust geometric operations (area, CCW winding, validity).

Usage:
    # From polygon
    geom = BuildingGeometry.from_polygon([(0,0), (50,0), (50,20), (0,20)], 
                                          num_floors=5, floor_height=2.6)
    
    # WWR assignment (any of these - auto-converts to other representations)
    geom.set_wwr_uniform(0.20)
    geom.set_wwr_cardinal({'N': 0.15, 'S': 0.25, 'E': 0.20, 'W': 0.20})
    geom.set_wwr_per_segment({0: 0.30, 1: 0.15, 2: 0.25, 3: 0.15})
    
    # Get parameters for different models
    params_5r1c = geom.get_5r1c_params()       # Dict with window areas, envelope areas
    walls_eplus = geom.get_eplus_surfaces()     # List of wall/window vertex dicts
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Sequence
import numpy as np
from math import atan2, degrees, sqrt

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import orient


@dataclass
class WallSegment:
    """A single wall segment of the building envelope."""
    
    segment_id: int
    start: Tuple[float, float]  # (x, y) coordinates
    end: Tuple[float, float]
    height: float  # Total wall height (all floors)
    
    @property
    def length(self) -> float:
        """Horizontal length of wall segment."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return sqrt(dx*dx + dy*dy)
    
    @property
    def area(self) -> float:
        """Total wall area (gross, including windows)."""
        return self.length * self.height
    
    @property
    def normal_angle(self) -> float:
        """Outward normal angle in degrees (0=North, 90=East, 180=South, 270=West).
        
        Assumes polygon vertices are in counter-clockwise order (standard convention).
        """
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        # Wall direction angle
        wall_angle = degrees(atan2(dx, dy))  # Note: atan2(x,y) for compass bearing
        # Outward normal is 90° clockwise from wall direction (for CCW polygon)
        normal = (wall_angle + 90) % 360
        return normal
    
    @property
    def cardinal_direction(self) -> str:
        """Nearest cardinal direction based on normal angle."""
        angle = self.normal_angle
        if 315 <= angle or angle < 45:
            return 'N'
        elif 45 <= angle < 135:
            return 'E'
        elif 135 <= angle < 225:
            return 'S'
        else:  # 225 <= angle < 315
            return 'W'
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        """Midpoint of the wall segment."""
        return (
            (self.start[0] + self.end[0]) / 2,
            (self.start[1] + self.end[1]) / 2
        )
    
    def get_3d_vertices(self, z_base: float = 0.0) -> List[Tuple[float, float, float]]:
        """Get 3D vertices for this wall (counter-clockwise from outside).
        
        EnergyPlus GlobalGeometryRules: UpperLeftCorner, Counterclockwise
        Order: top-left -> bottom-left -> bottom-right -> top-right
        """
        z_top = z_base + self.height
        return [
            (self.start[0], self.start[1], z_top),     # Top-left (start)
            (self.start[0], self.start[1], z_base),    # Bottom-left
            (self.end[0], self.end[1], z_base),        # Bottom-right
            (self.end[0], self.end[1], z_top),         # Top-right (end)
        ]


@dataclass  
class BuildingGeometry:
    """Building geometry with polygon footprint and modular WWR system.
    
    Uses Shapely internally for robust geometric operations (area, winding,
    validity checks). Coordinates are in metres (local / project CRS).
    """
    
    vertices: List[Tuple[float, float]]  # Footprint vertices (CCW order)
    num_floors: int = 1
    floor_height: float = 2.6
    
    # Computed
    wall_segments: List[WallSegment] = field(default_factory=list)
    
    # Internal Shapely polygon (built from vertices)
    _shapely_poly: Optional[ShapelyPolygon] = field(default=None, repr=False)
    
    # WWR storage (both representations maintained in sync)
    _wwr_per_segment: Dict[int, float] = field(default_factory=dict)
    _wwr_cardinal: Dict[str, float] = field(default_factory=lambda: {
        'N': 0.15, 'S': 0.25, 'E': 0.15, 'W': 0.15
    })
    
    def __post_init__(self):
        """Initialize Shapely polygon and wall segments from vertices."""
        self._build_shapely_poly()
        self._build_wall_segments()
        # Initialize per-segment from default cardinal
        self._cardinal_to_per_segment()
    
    def _build_shapely_poly(self):
        """Build the internal Shapely polygon (CCW exterior ring)."""
        self._shapely_poly = orient(ShapelyPolygon(self.vertices), sign=1.0)
        # Re-extract vertices from the oriented polygon (drop closing duplicate)
        coords = list(self._shapely_poly.exterior.coords)[:-1]
        self.vertices = [(float(x), float(y)) for x, y in coords]
    
    def _build_wall_segments(self):
        """Create wall segments from polygon vertices."""
        self.wall_segments = []
        n = len(self.vertices)
        total_height = self.num_floors * self.floor_height
        
        for i in range(n):
            start = self.vertices[i]
            end = self.vertices[(i + 1) % n]
            
            segment = WallSegment(
                segment_id=i,
                start=start,
                end=end,
                height=total_height,
            )
            self.wall_segments.append(segment)
    
    @classmethod
    def from_polygon(
        cls,
        vertices: Union[Sequence[Tuple[float, float]], str],
        num_floors: int = 1,
        floor_height: float = 2.6,
    ) -> 'BuildingGeometry':
        """Create geometry from polygon vertices.
        
        Args:
            vertices: List of (x, y) tuples in any winding order (auto-corrected),
                      or string like "[(0,0), (50,0), (50,20), (0,20)]"
            num_floors: Number of floors
            floor_height: Height per floor [m]
        """
        if isinstance(vertices, str):
            import ast
            parsed: List[Tuple[float, float]] = ast.literal_eval(vertices)
        else:
            parsed = [(float(x), float(y)) for x, y in vertices]
        
        return cls(
            vertices=parsed,
            num_floors=num_floors,
            floor_height=floor_height,
        )
    
    @classmethod
    def from_rectangle(
        cls,
        length: float,
        width: float,
        num_floors: int = 1,
        floor_height: float = 2.6,
        orientation: float = 0.0,
    ) -> 'BuildingGeometry':
        """Create rectangular building geometry.
        
        Args:
            length: Building length in E-W direction [m]
            width: Building width in N-S direction [m]
            num_floors: Number of floors
            floor_height: Height per floor [m]
            orientation: Rotation from north [degrees]
        """
        # Basic rectangle centered at origin
        # Length is E-W (x), Width is N-S (y)
        half_l = length / 2
        half_w = width / 2
        
        vertices = [
            (-half_l, -half_w),  # SW corner
            (half_l, -half_w),   # SE corner
            (half_l, half_w),    # NE corner
            (-half_l, half_w),   # NW corner
        ]
        
        # Apply rotation if needed
        if orientation != 0:
            rad = np.radians(orientation)
            cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
            vertices = [
                (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
                for x, y in vertices
            ]
        
        return cls(
            vertices=vertices,
            num_floors=num_floors,
            floor_height=floor_height,
        )
    
    @classmethod
    def from_shapely(
        cls,
        polygon: ShapelyPolygon,
        num_floors: int = 1,
        floor_height: float = 2.6,
    ) -> 'BuildingGeometry':
        """Create geometry directly from a Shapely Polygon.
        
        Args:
            polygon: shapely.geometry.Polygon (any winding order, auto-corrected)
            num_floors: Number of floors
            floor_height: Height per floor [m]
        """
        oriented = orient(polygon, sign=1.0)  # Ensure CCW
        coords = list(oriented.exterior.coords)[:-1]
        return cls(
            vertices=[(float(x), float(y)) for x, y in coords],
            num_floors=num_floors,
            floor_height=floor_height,
        )
    
    # =========================================================================
    # Properties (delegating area / perimeter to Shapely)
    # =========================================================================
    
    @property
    def shapely_polygon(self) -> ShapelyPolygon:
        """Access the underlying Shapely polygon."""
        if self._shapely_poly is None:
            self._build_shapely_poly()
        assert self._shapely_poly is not None
        return self._shapely_poly
    
    @property
    def is_valid(self) -> bool:
        """Check if the polygon is geometrically valid (no self-intersections)."""
        return self.shapely_polygon.is_valid
    
    @property
    def floor_area(self) -> float:
        """Total floor area [m²]."""
        return self.footprint_area * self.num_floors
    
    @property
    def footprint_area(self) -> float:
        """Single floor footprint area [m²] (via Shapely)."""
        return float(self.shapely_polygon.area)
    
    @property
    def perimeter(self) -> float:
        """Building perimeter [m] (via Shapely)."""
        return float(self.shapely_polygon.length)
    
    @property
    def total_wall_area(self) -> float:
        """Total gross wall area [m²]."""
        return sum(seg.area for seg in self.wall_segments)
    
    @property
    def total_height(self) -> float:
        """Total building height [m]."""
        return self.num_floors * self.floor_height
    
    @property
    def volume(self) -> float:
        """Building volume [m³]."""
        return self.footprint_area * self.total_height
    
    @property
    def centroid(self) -> Tuple[float, float]:
        """Footprint centroid (x, y)."""
        c = self.shapely_polygon.centroid
        return (float(c.x), float(c.y))
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Bounding box (minx, miny, maxx, maxy)."""
        return self.shapely_polygon.bounds
    
    # =========================================================================
    # WWR Assignment Methods
    # =========================================================================
    
    def set_wwr_uniform(self, wwr: float):
        """Set uniform WWR for all walls.
        
        Args:
            wwr: Window-to-wall ratio (0.0 to 1.0)
        """
        self._wwr_cardinal = {'N': wwr, 'S': wwr, 'E': wwr, 'W': wwr}
        self._wwr_per_segment = {seg.segment_id: wwr for seg in self.wall_segments}
    
    def set_wwr_cardinal(self, wwr: Dict[str, float]):
        """Set WWR by cardinal direction, auto-impute per-segment.
        
        Args:
            wwr: Dict with keys 'N', 'S', 'E', 'W' and WWR values
        """
        # Fill any missing directions with 0.15 default
        self._wwr_cardinal = {
            'N': wwr.get('N', wwr.get('north', 0.15)),
            'S': wwr.get('S', wwr.get('south', 0.25)),
            'E': wwr.get('E', wwr.get('east', 0.15)),
            'W': wwr.get('W', wwr.get('west', 0.15)),
        }
        self._cardinal_to_per_segment()
    
    def set_wwr_per_segment(self, wwr: Dict[int, float]):
        """Set WWR per wall segment, auto-compute cardinal aggregates.
        
        Args:
            wwr: Dict mapping segment_id to WWR value
        """
        # Fill missing segments with 0.15 default
        self._wwr_per_segment = {
            seg.segment_id: wwr.get(seg.segment_id, 0.15)
            for seg in self.wall_segments
        }
        self._per_segment_to_cardinal()
    
    def _cardinal_to_per_segment(self):
        """Impute per-segment WWR from cardinal WWR."""
        self._wwr_per_segment = {
            seg.segment_id: self._wwr_cardinal[seg.cardinal_direction]
            for seg in self.wall_segments
        }
    
    def _per_segment_to_cardinal(self):
        """Aggregate per-segment WWR to cardinal directions (area-weighted average)."""
        # Accumulate area-weighted WWR for each direction
        cardinal_area = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        cardinal_wwr_sum = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        
        for seg in self.wall_segments:
            direction = seg.cardinal_direction
            area = seg.area
            wwr = self._wwr_per_segment.get(seg.segment_id, 0.15)
            
            cardinal_area[direction] += area
            cardinal_wwr_sum[direction] += wwr * area
        
        # Compute weighted average
        self._wwr_cardinal = {}
        for direction in ['N', 'S', 'E', 'W']:
            if cardinal_area[direction] > 0:
                self._wwr_cardinal[direction] = cardinal_wwr_sum[direction] / cardinal_area[direction]
            else:
                self._wwr_cardinal[direction] = 0.15  # Default if no walls face this direction
    
    @property
    def wwr_per_segment(self) -> Dict[int, float]:
        """Get per-segment WWR dict."""
        return self._wwr_per_segment.copy()
    
    @property
    def wwr_cardinal(self) -> Dict[str, float]:
        """Get cardinal WWR dict."""
        return self._wwr_cardinal.copy()
    
    # =========================================================================
    # Output for Different Models
    # =========================================================================
    
    def get_wall_areas_by_cardinal(self) -> Dict[str, float]:
        """Get total wall area by cardinal direction [m²]."""
        areas = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        for seg in self.wall_segments:
            areas[seg.cardinal_direction] += seg.area
        return areas
    
    def get_window_areas_by_cardinal(self) -> Dict[str, float]:
        """Get window area by cardinal direction [m²]."""
        window_areas = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        for seg in self.wall_segments:
            direction = seg.cardinal_direction
            wwr = self._wwr_per_segment.get(seg.segment_id, 0.15)
            window_areas[direction] += seg.area * wwr
        return window_areas
    
    def get_5r1c_params(self) -> Dict[str, float]:
        """Get parameters needed for 5R1C model.
        
        Returns:
            Dict with window_area_north, window_area_south, etc., plus envelope areas.
        """
        window_areas = self.get_window_areas_by_cardinal()
        wall_areas = self.get_wall_areas_by_cardinal()
        
        # Net wall area = gross - windows
        net_wall_areas = {
            d: wall_areas[d] - window_areas[d] 
            for d in ['N', 'S', 'E', 'W']
        }
        
        return {
            # Window areas
            'window_area_north': window_areas['N'],
            'window_area_south': window_areas['S'],
            'window_area_east': window_areas['E'],
            'window_area_west': window_areas['W'],
            
            # Total areas
            'floor_area_m2': self.floor_area,
            'roof_area_m2': self.footprint_area,
            'total_wall_area_gross': self.total_wall_area,
            'total_wall_area_net': sum(net_wall_areas.values()),
            'total_window_area': sum(window_areas.values()),
            
            # Geometry
            'perimeter': self.perimeter,
            'num_floors': self.num_floors,
            'floor_height': self.floor_height,
            'total_height': self.total_height,
            'volume': self.volume,
            'footprint_area': self.footprint_area,
        }
    
    def get_eplus_surfaces(self) -> List[Dict]:
        """Get wall and window surface definitions for EnergyPlus IDF.
        
        Returns:
            List of dicts, each with:
                - type: 'wall' or 'window'
                - name: surface name
                - vertices: list of (x, y, z) tuples
                - parent: parent wall name (for windows)
                - segment_id: original segment ID
        """
        surfaces = []
        
        for seg in self.wall_segments:
            wall_name = f"Wall_{seg.segment_id}_{seg.cardinal_direction}"
            wwr = self._wwr_per_segment.get(seg.segment_id, 0.15)
            
            # Full wall vertices
            wall_verts = seg.get_3d_vertices(z_base=0.0)
            
            surfaces.append({
                'type': 'wall',
                'name': wall_name,
                'vertices': wall_verts,
                'segment_id': seg.segment_id,
                'cardinal': seg.cardinal_direction,
                'normal_angle': seg.normal_angle,
            })
            
            # Window (if WWR > 0)
            if wwr > 0.01:
                window_verts = self._get_window_vertices(seg, wwr)
                surfaces.append({
                    'type': 'window',
                    'name': f"Window_{seg.segment_id}_{seg.cardinal_direction}",
                    'vertices': window_verts,
                    'parent': wall_name,
                    'segment_id': seg.segment_id,
                    'wwr': wwr,
                })
        
        return surfaces
    
    def _get_window_vertices(
        self, 
        seg: WallSegment, 
        wwr: float,
        sill_height: float = 0.9,
        head_clearance: float = 0.3,
    ) -> List[Tuple[float, float, float]]:
        """Calculate window vertices centered on wall segment.
        
        Windows are placed with:
        - Sill at sill_height (default 0.9m)
        - Head at (wall_height - head_clearance)
        - Width calculated from WWR and available wall area
        """
        wall_height = seg.height
        wall_length = seg.length
        wall_area = seg.area
        
        # Window dimensions
        window_area = wall_area * wwr
        
        # Window height: from sill to head
        max_window_height = wall_height - sill_height - head_clearance
        window_height = min(max_window_height, 2.4)  # Cap at 2.4m
        
        # Window width from area
        window_width = window_area / window_height
        window_width = min(window_width, wall_length * 0.9)  # Max 90% of wall length
        
        # Recalculate height if width was capped
        if window_width >= wall_length * 0.9:
            window_width = wall_length * 0.9
            window_height = window_area / window_width
            window_height = min(window_height, max_window_height)
        
        # Center the window on the wall
        offset_along_wall = (wall_length - window_width) / 2
        
        # Direction vector along wall
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        length = seg.length
        ux, uy = dx / length, dy / length  # Unit vector along wall
        
        # Window corners (start from bottom-left, going CCW when viewed from outside)
        x1 = seg.start[0] + ux * offset_along_wall
        y1 = seg.start[1] + uy * offset_along_wall
        x2 = x1 + ux * window_width
        y2 = y1 + uy * window_width
        
        z_sill = sill_height
        z_head = sill_height + window_height
        
        # EnergyPlus: UpperLeftCorner, Counterclockwise (viewed from outside)
        # Order: top-left -> bottom-left -> bottom-right -> top-right
        return [
            (x1, y1, z_head),   # Top-left
            (x1, y1, z_sill),   # Bottom-left
            (x2, y2, z_sill),   # Bottom-right
            (x2, y2, z_head),   # Top-right
        ]
    
    def summary(self) -> str:
        """Return a text summary of the building geometry."""
        lines = [
            "Building Geometry Summary",
            "=" * 40,
            f"Footprint: {len(self.vertices)} vertices  "
            f"(valid={self.is_valid})",
            f"Footprint area: {self.footprint_area:.1f} m2",
            f"Volume: {self.volume:.1f} m3",
            f"Floors: {self.num_floors} × {self.floor_height}m = {self.total_height}m",
            f"Floor area: {self.floor_area:.1f} m²",
            f"Perimeter: {self.perimeter:.1f} m",
            f"Total wall area: {self.total_wall_area:.1f} m²",
            "",
            "Wall Segments:",
            "-" * 40,
        ]
        
        for seg in self.wall_segments:
            wwr = self._wwr_per_segment.get(seg.segment_id, 0.15)
            lines.append(
                f"  [{seg.segment_id}] {seg.cardinal_direction}: "
                f"L={seg.length:.1f}m, A={seg.area:.1f}m², "
                f"WWR={wwr:.0%}, θ={seg.normal_angle:.0f}°"
            )
        
        lines.extend([
            "",
            "Cardinal WWR:",
            "-" * 40,
            f"  North: {self._wwr_cardinal['N']:.0%}",
            f"  South: {self._wwr_cardinal['S']:.0%}",
            f"  East:  {self._wwr_cardinal['E']:.0%}",
            f"  West:  {self._wwr_cardinal['W']:.0%}",
            "",
            "Window Areas:",
            "-" * 40,
        ])
        
        window_areas = self.get_window_areas_by_cardinal()
        for d in ['N', 'S', 'E', 'W']:
            lines.append(f"  {d}: {window_areas[d]:.1f} m²")
        lines.append(f"  Total: {sum(window_areas.values()):.1f} m²")
        
        return "\n".join(lines)


def demo():
    """Demonstrate the building geometry system."""
    
    print("=" * 60)
    print("DEMO: Modular Building Geometry System")
    print("=" * 60)
    
    # Example 1: Rectangle
    print("\n1. Simple Rectangle (50m × 20m)")
    geom = BuildingGeometry.from_rectangle(
        length=50, width=20, 
        num_floors=5, floor_height=2.6
    )
    geom.set_wwr_uniform(0.20)
    print(geom.summary())
    
    # Example 2: L-shaped building
    print("\n" + "=" * 60)
    print("2. L-Shaped Building")
    l_shape = [
        (0.0, 0.0), (30.0, 0.0), (30.0, 15.0), (15.0, 15.0), (15.0, 25.0), (0.0, 25.0)
    ]
    geom2 = BuildingGeometry.from_polygon(l_shape, num_floors=3, floor_height=3.0)
    geom2.set_wwr_cardinal({'N': 0.10, 'S': 0.30, 'E': 0.20, 'W': 0.15})
    print(geom2.summary())
    
    # Example 3: Per-segment WWR
    print("\n" + "=" * 60)
    print("3. Custom per-segment WWR")
    geom3 = BuildingGeometry.from_rectangle(40, 30, num_floors=2)
    geom3.set_wwr_per_segment({0: 0.05, 1: 0.40, 2: 0.05, 3: 0.15})
    print(geom3.summary())
    print("\n(Notice cardinal WWR is auto-computed from per-segment)")
    
    # Example 4: From Shapely directly
    print("\n" + "=" * 60)
    print("4. From Shapely Polygon")
    poly = ShapelyPolygon([(0, 0), (20, 0), (20, 10), (0, 10)])
    geom4 = BuildingGeometry.from_shapely(poly, num_floors=2, floor_height=3.0)
    print(geom4.summary())


if __name__ == "__main__":
    demo()
