"""
Parametric Building Generator for ISO 13790 vs EnergyPlus Comparison

Generates matched 5R1C parameters and EnergyPlus IDF files for a simple
shoebox building to enable systematic sensitivity analysis.

Building: Simple rectangular shoebox (single zone)
- Configurable dimensions, orientation, WWR, U-values
- Ideal loads for heating/cooling
- Minimal schedules (constant occupancy/lighting)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from epsm_rc.building_geometry import BuildingGeometry


@dataclass
class ParametricBuilding:
    """Parametric building definition for both 5R1C and E+ models."""
    
    # Geometry
    length_m: float = 20.0          # Building length (E-W direction)
    width_m: float = 10.0           # Building width (N-S direction)
    num_floors: int = 1             # Number of floors
    floor_height_m: float = 3.0     # Floor-to-floor height
    orientation_deg: float = 0.0    # Rotation from north (0 = long axis E-W)
    shape_type: str = "rect"        # rect, l, u, i, skewed, banana
    custom_footprint_vertices: Optional[List[Tuple[float, float]]] = None
    
    # Windows (WWR per facade)
    wwr_north: float = 0.3
    wwr_south: float = 0.3
    wwr_east: float = 0.3
    wwr_west: float = 0.3
    
    # U-values [W/m²K]
    u_walls: float = 0.5
    u_roof: float = 0.3
    u_floor: float = 0.5
    u_windows: float = 2.0
    g_value: float = 0.6            # Solar heat gain coefficient
    
    # Infiltration
    ach_infiltration: float = 0.5   # Air changes per hour at normal pressure
    
    # Internal gains
    occupant_density: float = 0.1   # persons/m²
    lighting_w_m2: float = 10.0     # W/m²
    equipment_w_m2: float = 10.0    # W/m²
    
    # Setpoints
    t_heating: float = 21.0         # Heating setpoint [°C]
    t_cooling: float = 26.0         # Cooling setpoint [°C]
    
    # Thermal mass - now uses construction-based calculation
    # thermal_mass_class is deprecated, use thermal_capacitance_j_m2k instead
    thermal_mass_class: str = "MEDIUM"  # Fallback: VERY_LIGHT, LIGHT, MEDIUM, HEAVY, VERY_HEAVY
    
    # Explicit thermal capacitance [J/(m²·K)] per floor area
    # If set, overrides thermal_mass_class for 5R1C model
    # Typical values: 80,000 (very light) to 370,000 (very heavy)
    thermal_capacitance_j_m2k: Optional[float] = None
    
    # Construction material properties (for E+ and 5R1C matching)
    # Effective depth of thermal mass exposed to interior
    wall_thickness_m: float = 0.10  # Effective thickness for thermal mass
    wall_density_kg_m3: float = 1400  # Material density
    wall_specific_heat_j_kgk: float = 1000  # Specific heat
    
    # Weather
    epw_path: str = "src/data/epw/goteborg.epw"
    
    # Computed properties
    @property
    def geometry(self) -> Optional["BuildingGeometry"]:
        """Return polygon geometry for non-rectangular/custom footprints."""
        shape = (self.shape_type or "rect").lower()
        if shape == "rect" and not self.custom_footprint_vertices:
            return None

        if self.custom_footprint_vertices:
            vertices = [(float(x), float(y)) for x, y in self.custom_footprint_vertices]
        else:
            vertices = self._base_shape_vertices(shape)

        from epsm_rc.building_geometry import BuildingGeometry

        vertices = self._rotate_vertices(vertices, self.orientation_deg)
        geom = BuildingGeometry.from_polygon(
            vertices=vertices,
            num_floors=self.num_floors,
            floor_height=self.floor_height_m,
        )
        geom.set_wwr_cardinal({
            'N': self.wwr_north,
            'S': self.wwr_south,
            'E': self.wwr_east,
            'W': self.wwr_west,
        })
        return geom

    def _rotate_vertices(self, vertices: List[Tuple[float, float]], angle_deg: float) -> List[Tuple[float, float]]:
        """Rotate vertices around footprint center by angle_deg."""
        if abs(angle_deg) < 1e-12:
            return vertices

        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        cx = 0.5 * (min(xs) + max(xs))
        cy = 0.5 * (min(ys) + max(ys))

        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        out: List[Tuple[float, float]] = []
        for x, y in vertices:
            xr = x - cx
            yr = y - cy
            out.append((xr * cos_a - yr * sin_a + cx, xr * sin_a + yr * cos_a + cy))
        return out

    def _base_shape_vertices(self, shape: str) -> List[Tuple[float, float]]:
        """Create unrotated footprint vertices using building length/width envelope."""
        L = float(self.length_m)
        W = float(self.width_m)
        s = shape.lower()

        if s == "rect":
            return [(0.0, 0.0), (L, 0.0), (L, W), (0.0, W)]

        if s == "l":
            lx = 0.58 * L
            ly = 0.56 * W
            return [(0.0, 0.0), (L, 0.0), (L, ly), (lx, ly), (lx, W), (0.0, W)]

        if s == "u":
            x1 = 0.30 * L
            x2 = 0.70 * L
            y_notch = 0.38 * W
            return [
                (0.0, 0.0), (L, 0.0), (L, W), (x2, W),
                (x2, y_notch), (x1, y_notch), (x1, W), (0.0, W),
            ]

        if s == "i":
            y1 = 0.22 * W
            y2 = 0.78 * W
            x1 = 0.34 * L
            x2 = 0.66 * L
            return [
                (0.0, 0.0), (L, 0.0), (L, y1), (x2, y1),
                (x2, y2), (L, y2), (L, W), (0.0, W),
                (0.0, y2), (x1, y2), (x1, y1), (0.0, y1),
            ]

        if s == "skewed":
            return [(0.0, 0.08 * W), (L, 0.0), (0.92 * L, W), (0.10 * L, 0.88 * W)]

        if s == "banana":
            return [
                (0.00 * L, 0.30 * W), (0.12 * L, 0.10 * W), (0.42 * L, 0.00 * W),
                (0.76 * L, 0.10 * W), (1.00 * L, 0.34 * W), (0.92 * L, 0.62 * W),
                (0.62 * L, 0.86 * W), (0.28 * L, 1.00 * W), (0.04 * L, 0.78 * W),
            ]

        raise ValueError(f"Unknown shape_type '{shape}'. Use one of: rect, l, u, i, skewed, banana")

    @property
    def floor_area_m2(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.floor_area
        return self.length_m * self.width_m * self.num_floors
    
    @property
    def total_height_m(self) -> float:
        return self.num_floors * self.floor_height_m
    
    @property
    def wall_area_north(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_wall_areas_by_cardinal()['N']
        return self.length_m * self.total_height_m  # IDF: N/S walls span length_m
    
    @property
    def wall_area_south(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_wall_areas_by_cardinal()['S']
        return self.length_m * self.total_height_m  # IDF: N/S walls span length_m
    
    @property
    def wall_area_east(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_wall_areas_by_cardinal()['E']
        return self.width_m * self.total_height_m  # IDF: E/W walls span width_m
    
    @property
    def wall_area_west(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_wall_areas_by_cardinal()['W']
        return self.width_m * self.total_height_m  # IDF: E/W walls span width_m
    
    @property
    def gross_wall_area(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.total_wall_area
        return 2 * (self.wall_area_north + self.wall_area_east)
    
    @property
    def window_area_north(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_window_areas_by_cardinal()['N']
        return self.wall_area_north * self.wwr_north
    
    @property
    def window_area_south(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_window_areas_by_cardinal()['S']
        return self.wall_area_south * self.wwr_south
    
    @property
    def window_area_east(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_window_areas_by_cardinal()['E']
        return self.wall_area_east * self.wwr_east
    
    @property
    def window_area_west(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_window_areas_by_cardinal()['W']
        return self.wall_area_west * self.wwr_west
    
    @property
    def total_window_area(self) -> float:
        return (self.window_area_north + self.window_area_south + 
                self.window_area_east + self.window_area_west)
    
    @property
    def net_wall_area(self) -> float:
        return self.gross_wall_area - self.total_window_area
    
    @property
    def roof_area(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.footprint_area
        return self.length_m * self.width_m
    
    @property
    def ground_floor_area(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.footprint_area
        return self.length_m * self.width_m
    
    @property
    def average_wwr(self) -> float:
        return self.total_window_area / self.gross_wall_area
    
    @property
    def internal_gains_w_m2(self) -> float:
        """Total internal gains [W/m²] (sensible only).
        
        Uses 70 W/person sensible metabolic heat per ISO 13790 Table G.11,
        not the full 120 W/person total metabolic output. The ~50 W/person
        latent heat does not contribute to the zone heat balance.
        """
        occupant_heat = self.occupant_density * 70  # 70W/person sensible (ISO 13790 Table G.11)
        return occupant_heat + self.lighting_w_m2 + self.equipment_w_m2
    
    @property
    def volume_m3(self) -> float:
        return self.floor_area_m2 * self.floor_height_m
    
    @property
    def thermal_mass_per_surface_area(self) -> float:
        """Thermal mass per unit surface area [J/(m²·K)]."""
        return self.wall_thickness_m * self.wall_density_kg_m3 * self.wall_specific_heat_j_kgk
    
    @property
    def total_internal_surface_area(self) -> float:
        """Total internal surface area exposed to zone [m²].
        
        Includes: walls (interior face), floor, ceiling, interior partitions.
        """
        # External walls (interior face)
        wall_internal = self.net_wall_area
        # Floor and ceiling
        floor_ceiling = 2 * self.ground_floor_area
        # Estimate interior partitions (~0.5 m² per m² floor area)
        partitions = 0.5 * self.floor_area_m2
        return wall_internal + floor_ceiling + partitions
    
    @property
    def calculated_thermal_capacitance_j_m2k(self) -> float:
        """Calculate thermal capacitance per floor area from constructions [J/(m²·K)].
        
        This calculates kappa (κ) = total_Cm / floor_area for use with 5R1C model.
        """
        total_cm = self.thermal_mass_per_surface_area * self.total_internal_surface_area
        return total_cm / self.floor_area_m2 if self.floor_area_m2 > 0 else 0
    
    @property
    def effective_thermal_capacitance_j_m2k(self) -> float:
        """Get the effective thermal capacitance to use [J/(m²·K)].
        
        Priority: explicit value > calculated from constructions > class lookup
        """
        if self.thermal_capacitance_j_m2k is not None:
            return self.thermal_capacitance_j_m2k
        return self.calculated_thermal_capacitance_j_m2k
    
    def summary(self) -> str:
        """Print building summary."""
        lines = [
            "=" * 60,
            "PARAMETRIC BUILDING SUMMARY",
            "=" * 60,
            f"\nGeometry:",
            f"  Shape: {self.shape_type.upper()}",
            f"  Dimensions: {self.length_m} x {self.width_m} x {self.total_height_m} m",
            f"  Floors: {self.num_floors} @ {self.floor_height_m} m each",
            f"  Floor area: {self.floor_area_m2:.0f} m²",
            f"  Volume: {self.volume_m3:.0f} m³",
            f"  Orientation: {self.orientation_deg}° from north",
            f"\nEnvelope:",
            f"  Wall area (gross): {self.gross_wall_area:.0f} m²",
            f"  Wall area (net): {self.net_wall_area:.0f} m²",
            f"  Window area: {self.total_window_area:.0f} m² (WWR={self.average_wwr:.1%})",
            f"  Roof area: {self.roof_area:.0f} m²",
            f"  Ground floor area: {self.ground_floor_area:.0f} m²",
            f"\nU-values [W/m²K]:",
            f"  Walls: {self.u_walls}",
            f"  Roof: {self.u_roof}",
            f"  Floor: {self.u_floor}",
            f"  Windows: {self.u_windows} (g={self.g_value})",
            f"\nOperational:",
            f"  Infiltration: {self.ach_infiltration} ACH",
            f"  Internal gains: {self.internal_gains_w_m2:.1f} W/m²",
            f"  Heating setpoint: {self.t_heating}°C",
            f"\nThermal Mass:",
            f"  Construction: {self.wall_thickness_m*100:.0f}cm @ ρ={self.wall_density_kg_m3}, cp={self.wall_specific_heat_j_kgk}",
            f"  Cm per surface: {self.thermal_mass_per_surface_area/1000:.0f} kJ/(m²·K)",
            f"  Total internal surface: {self.total_internal_surface_area:.0f} m²",
            f"  κ (per floor area): {self.effective_thermal_capacitance_j_m2k/1000:.0f} kJ/(m²·K)",
        ]
        return "\n".join(lines)


def generate_idf(building: ParametricBuilding, output_path: Path) -> Path:
    """
    Generate EnergyPlus IDF file for the parametric building.
    
    Creates a single-zone shoebox model with:
    - Simple geometry (4 walls, roof, floor)
    - Ideal loads HVAC
    - Constant schedules
    """
    
    geometry = building.geometry
    if geometry is not None:
        return generate_polygon_idf(building, geometry, output_path)

    # Calculate vertices based on orientation
    angle_rad = np.radians(building.orientation_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Base rectangle corners (before rotation)
    L, W = building.length_m, building.width_m
    H = building.total_height_m
    
    # Corners: SW, SE, NE, NW (counterclockwise from southwest)
    corners = [
        (0, 0),
        (L, 0),
        (L, W),
        (0, W)
    ]
    
    # Rotate corners
    def rotate(x, y):
        return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
    
    rotated = [rotate(x, y) for x, y in corners]
    
    # Format vertex for IDF
    def fmt_vertex(x, y, z):
        return f"    {x:.4f}, {y:.4f}, {z:.4f}"
    
    # Swedish ground temperatures (monthly)
    ground_temps = [4.0, 3.5, 4.0, 6.0, 9.0, 12.0, 14.5, 14.0, 12.0, 9.0, 6.0, 4.5]
    
    idf_content = f"""!-Generator: Parametric Building Generator for ISO 13790 Comparison
!-Option SortedOrder

Version,
    24.2;                    !- Version Identifier

!-   ===========  SIMULATION PARAMETERS  ===========

SimulationControl,
    No,                      !- Do Zone Sizing Calculation
    No,                      !- Do System Sizing Calculation
    No,                      !- Do Plant Sizing Calculation
    No,                      !- Run Simulation for Sizing Periods
    Yes,                     !- Run Simulation for Weather File Run Periods
    No,                      !- Do HVAC Sizing Simulation for Sizing Periods
    1;                       !- Maximum Number of HVAC Sizing Simulation Passes

Building,
    ParametricBuilding,      !- Name
    {building.orientation_deg},  !- North Axis {{deg}}
    Suburbs,                 !- Terrain
    0.04,                    !- Loads Convergence Tolerance Value {{W}}
    0.4,                     !- Temperature Convergence Tolerance Value {{deltaC}}
    FullInteriorAndExterior, !- Solar Distribution
    25,                      !- Maximum Number of Warmup Days
    6;                       !- Minimum Number of Warmup Days

Timestep,
    6;                       !- Number of Timesteps per Hour

RunPeriod,
    Annual,                  !- Name
    1,                       !- Begin Month
    1,                       !- Begin Day of Month
    ,                        !- Begin Year
    12,                      !- End Month
    31,                      !- End Day of Month
    ,                        !- End Year
    Sunday,                  !- Day of Week for Start Day
    No,                      !- Use Weather File Holidays and Special Days
    No,                      !- Use Weather File Daylight Saving Period
    No,                      !- Apply Weekend Holiday Rule
    Yes,                     !- Use Weather File Rain Indicators
    Yes;                     !- Use Weather File Snow Indicators

Site:GroundTemperature:BuildingSurface,
    {ground_temps[0]}, {ground_temps[1]}, {ground_temps[2]}, {ground_temps[3]},
    {ground_temps[4]}, {ground_temps[5]}, {ground_temps[6]}, {ground_temps[7]},
    {ground_temps[8]}, {ground_temps[9]}, {ground_temps[10]}, {ground_temps[11]};

!-   ===========  SCHEDULES  ===========

ScheduleTypeLimits,
    Fraction,                !- Name
    0,                       !- Lower Limit Value
    1,                       !- Upper Limit Value
    Continuous;              !- Numeric Type

ScheduleTypeLimits,
    Temperature,             !- Name
    -100,                    !- Lower Limit Value
    200,                     !- Upper Limit Value
    Continuous;              !- Numeric Type

Schedule:Constant,
    Always On,               !- Name
    Fraction,                !- Schedule Type Limits Name
    1.0;                     !- Hourly Value

Schedule:Constant,
    Heating Setpoint,        !- Name
    Temperature,             !- Schedule Type Limits Name
    {building.t_heating};    !- Hourly Value

Schedule:Constant,
    Cooling Setpoint,        !- Name
    Temperature,             !- Schedule Type Limits Name
    {building.t_cooling};    !- Hourly Value

!-   ===========  MATERIALS  ===========

! Material properties are set to match 5R1C thermal mass calculation
! Effective thickness: {building.wall_thickness_m} m
! Density: {building.wall_density_kg_m3} kg/m³
! Specific heat: {building.wall_specific_heat_j_kgk} J/(kg·K)
! Thermal mass per surface: {building.thermal_mass_per_surface_area/1000:.0f} kJ/(m²·K)
!
! Conductivity is calculated to achieve target U-values accounting for surface resistances
! Walls: Rsi=0.13, Rse=0.04, total Rs=0.17 m²K/W
! Roof:  Rsi=0.10, Rse=0.04, total Rs=0.14 m²K/W  
! Floor: Rsi=0.17, Rse=0.00, total Rs=0.17 m²K/W (ground)
! Formula: R_material = 1/U - R_surface, k = thickness/R_material

Material,
    WallMaterial,            !- Name
    MediumRough,             !- Roughness
    {building.wall_thickness_m},  !- Thickness {{m}}
    {building.wall_thickness_m / max(0.01, 1.0/building.u_walls - 0.17)},  !- Conductivity {{W/m-K}}
    {building.wall_density_kg_m3},  !- Density {{kg/m3}}
    {building.wall_specific_heat_j_kgk},  !- Specific Heat {{J/kg-K}}
    0.9,                     !- Thermal Absorptance
    0.7,                     !- Solar Absorptance
    0.7;                     !- Visible Absorptance

Material,
    RoofMaterial,            !- Name
    MediumRough,             !- Roughness
    {building.wall_thickness_m},  !- Thickness {{m}}
    {building.wall_thickness_m / max(0.01, 1.0/building.u_roof - 0.14)},   !- Conductivity {{W/m-K}}
    {building.wall_density_kg_m3},  !- Density {{kg/m3}}
    {building.wall_specific_heat_j_kgk},  !- Specific Heat {{J/kg-K}}
    0.9,                     !- Thermal Absorptance
    0.7,                     !- Solar Absorptance
    0.7;                     !- Visible Absorptance

Material,
    FloorMaterial,           !- Name
    MediumRough,             !- Roughness
    {building.wall_thickness_m},  !- Thickness {{m}}
    {building.wall_thickness_m / max(0.01, 1.0/building.u_floor - 0.17)},  !- Conductivity {{W/m-K}}
    {building.wall_density_kg_m3 * 1.2},  !- Density {{kg/m3}} (floor slightly heavier)
    {building.wall_specific_heat_j_kgk},  !- Specific Heat {{J/kg-K}}
    0.9,                     !- Thermal Absorptance
    0.7,                     !- Solar Absorptance
    0.7;                     !- Visible Absorptance

Construction,
    WallConstruction,        !- Name
    WallMaterial;            !- Layer 1

Construction,
    RoofConstruction,        !- Name
    RoofMaterial;            !- Layer 1

Construction,
    FloorConstruction,       !- Name
    FloorMaterial;           !- Layer 1

WindowMaterial:SimpleGlazingSystem,
    SimpleWindow,            !- Name
    {building.u_windows},    !- U-Factor {{W/m2-K}}
    {building.g_value};      !- Solar Heat Gain Coefficient

Construction,
    WindowConstruction,      !- Name
    SimpleWindow;            !- Layer 1

!-   ===========  GEOMETRY  ===========

GlobalGeometryRules,
    UpperLeftCorner,         !- Starting Vertex Position
    Counterclockwise,        !- Vertex Entry Direction
    World;                   !- Coordinate System

Zone,
    Zone1,                   !- Name
    {building.orientation_deg},  !- Direction of Relative North {{deg}}
    0, 0, 0,                 !- X,Y,Z Origin {{m}}
    1,                       !- Type
    1,                       !- Multiplier
    {building.total_height_m},   !- Ceiling Height {{m}}
    {building.length_m * building.width_m * building.total_height_m};    !- Volume {{m3}}

! South Wall (Y=0)
BuildingSurface:Detailed,
    SouthWall,               !- Name
    Wall,                    !- Surface Type
    WallConstruction,        !- Construction Name
    Zone1,                   !- Zone Name
    ,                        !- Space Name
    Outdoors,                !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    SunExposed,              !- Sun Exposure
    WindExposed,             !- Wind Exposure
    autocalculate,           !- View Factor to Ground
    4,                       !- Number of Vertices
    0, 0, {H},
    0, 0, 0,
    {L}, 0, 0,
    {L}, 0, {H};

! North Wall (Y=W)
BuildingSurface:Detailed,
    NorthWall,               !- Name
    Wall,                    !- Surface Type
    WallConstruction,        !- Construction Name
    Zone1,                   !- Zone Name
    ,                        !- Space Name
    Outdoors,                !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    SunExposed,              !- Sun Exposure
    WindExposed,             !- Wind Exposure
    autocalculate,           !- View Factor to Ground
    4,                       !- Number of Vertices
    {L}, {W}, {H},
    {L}, {W}, 0,
    0, {W}, 0,
    0, {W}, {H};

! East Wall (X=L)
BuildingSurface:Detailed,
    EastWall,                !- Name
    Wall,                    !- Surface Type
    WallConstruction,        !- Construction Name
    Zone1,                   !- Zone Name
    ,                        !- Space Name
    Outdoors,                !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    SunExposed,              !- Sun Exposure
    WindExposed,             !- Wind Exposure
    autocalculate,           !- View Factor to Ground
    4,                       !- Number of Vertices
    {L}, 0, {H},
    {L}, 0, 0,
    {L}, {W}, 0,
    {L}, {W}, {H};

! West Wall (X=0)
BuildingSurface:Detailed,
    WestWall,                !- Name
    Wall,                    !- Surface Type
    WallConstruction,        !- Construction Name
    Zone1,                   !- Zone Name
    ,                        !- Space Name
    Outdoors,                !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    SunExposed,              !- Sun Exposure
    WindExposed,             !- Wind Exposure
    autocalculate,           !- View Factor to Ground
    4,                       !- Number of Vertices
    0, {W}, {H},
    0, {W}, 0,
    0, 0, 0,
    0, 0, {H};

! Roof (Z=H, viewed from above counterclockwise)
BuildingSurface:Detailed,
    Roof,                    !- Name
    Roof,                    !- Surface Type
    RoofConstruction,        !- Construction Name
    Zone1,                   !- Zone Name
    ,                        !- Space Name
    Outdoors,                !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    SunExposed,              !- Sun Exposure
    WindExposed,             !- Wind Exposure
    autocalculate,           !- View Factor to Ground
    4,                       !- Number of Vertices
    0, {W}, {H},
    0, 0, {H},
    {L}, 0, {H},
    {L}, {W}, {H};

! Floor (Z=0, viewed from below counterclockwise)
BuildingSurface:Detailed,
    Floor,                   !- Name
    Floor,                   !- Surface Type
    FloorConstruction,       !- Construction Name
    Zone1,                   !- Zone Name
    ,                        !- Space Name
    Ground,                  !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    NoSun,                   !- Sun Exposure
    NoWind,                  !- Wind Exposure
    autocalculate,           !- View Factor to Ground
    4,                       !- Number of Vertices
    0, 0, 0,
    0, {W}, 0,
    {L}, {W}, 0,
    {L}, 0, 0;

!-   ===========  WINDOWS  ===========
"""
    
    # Add windows for each facade if WWR > 0
    def add_window(wall_name, wall_vertices, wwr, wall_width, wall_height):
        if wwr <= 0:
            return ""
        
        # Window dimensions (centered on wall)
        win_height = wall_height * 0.8  # 80% of wall height
        win_width = wall_width * wwr / 0.8  # Adjust width for target WWR
        if win_width > wall_width * 0.9:
            win_width = wall_width * 0.9
            win_height = (wall_width * wall_height * wwr) / win_width
        
        # Window position (centered)
        sill_height = (wall_height - win_height) / 2
        
        return f"""
FenestrationSurface:Detailed,
    {wall_name}Window,       !- Name
    Window,                  !- Surface Type
    WindowConstruction,      !- Construction Name
    {wall_name},             !- Building Surface Name
    ,                        !- Outside Boundary Condition Object
    autocalculate,           !- View Factor to Ground
    ,                        !- Frame and Divider Name
    1,                       !- Multiplier
    4,                       !- Number of Vertices
    {(wall_width - win_width)/2}, {sill_height + win_height}, 0,
    {(wall_width - win_width)/2}, {sill_height}, 0,
    {(wall_width + win_width)/2}, {sill_height}, 0,
    {(wall_width + win_width)/2}, {sill_height + win_height}, 0;
"""
    
    # Use simpler window specification
    if building.wwr_south > 0:
        idf_content += f"""
Window,
    SouthWindow,             !- Name
    WindowConstruction,      !- Construction Name
    SouthWall,               !- Building Surface Name
    ,                        !- Frame and Divider Name
    1,                       !- Multiplier
    {(L - L * building.wwr_south / 0.8) / 2},  !- Starting X Coordinate {{m}}
    {H * 0.1},               !- Starting Z Coordinate {{m}}
    {L * building.wwr_south / 0.8},  !- Length {{m}}
    {H * 0.8};               !- Height {{m}}
"""
    
    if building.wwr_north > 0:
        idf_content += f"""
Window,
    NorthWindow,             !- Name
    WindowConstruction,      !- Construction Name
    NorthWall,               !- Building Surface Name
    ,                        !- Frame and Divider Name
    1,                       !- Multiplier
    {(L - L * building.wwr_north / 0.8) / 2},  !- Starting X Coordinate {{m}}
    {H * 0.1},               !- Starting Z Coordinate {{m}}
    {L * building.wwr_north / 0.8},  !- Length {{m}}
    {H * 0.8};               !- Height {{m}}
"""
    
    if building.wwr_east > 0:
        idf_content += f"""
Window,
    EastWindow,              !- Name
    WindowConstruction,      !- Construction Name
    EastWall,                !- Building Surface Name
    ,                        !- Frame and Divider Name
    1,                       !- Multiplier
    {(W - W * building.wwr_east / 0.8) / 2},  !- Starting X Coordinate {{m}}
    {H * 0.1},               !- Starting Z Coordinate {{m}}
    {W * building.wwr_east / 0.8},   !- Length {{m}}
    {H * 0.8};               !- Height {{m}}
"""
    
    if building.wwr_west > 0:
        idf_content += f"""
Window,
    WestWindow,              !- Name
    WindowConstruction,      !- Construction Name
    WestWall,                !- Building Surface Name
    ,                        !- Frame and Divider Name
    1,                       !- Multiplier
    {(W - W * building.wwr_west / 0.8) / 2},  !- Starting X Coordinate {{m}}
    {H * 0.1},               !- Starting Z Coordinate {{m}}
    {W * building.wwr_west / 0.8},   !- Length {{m}}
    {H * 0.8};               !- Height {{m}}
"""

    # Add infiltration, internal gains, and HVAC
    infiltration_m3_s = building.ach_infiltration * building.volume_m3 / 3600
    
    idf_content += f"""
!-   ===========  INFILTRATION  ===========

ZoneInfiltration:DesignFlowRate,
    Zone1 Infiltration,      !- Name
    Zone1,                   !- Zone Name
    Always On,               !- Schedule Name
    AirChanges/Hour,         !- Design Flow Rate Calculation Method
    ,                        !- Design Flow Rate {{m3/s}}
    ,                        !- Flow Rate per Floor Area {{m3/s-m2}}
    ,                        !- Flow Rate per Exterior Surface Area {{m3/s-m2}}
    {building.ach_infiltration},  !- Air Changes per Hour {{1/hr}}
    1,                       !- Constant Term Coefficient
    0,                       !- Temperature Term Coefficient
    0,                       !- Velocity Term Coefficient
    0;                       !- Velocity Squared Term Coefficient

!-   ===========  INTERNAL GAINS  ===========

People,
    Zone1 People,            !- Name
    Zone1,                   !- Zone Name
    Always On,               !- Number of People Schedule Name
    People,                  !- Number of People Calculation Method
    {building.occupant_density * building.floor_area_m2},  !- Number of People
    ,                        !- People per Floor Area {{person/m2}}
    ,                        !- Floor Area per Person {{m2/person}}
    0.3,                     !- Fraction Radiant
    1.0,                     !- Sensible Heat Fraction
    Activity Schedule;       !- Activity Level Schedule Name

Schedule:Constant,
    Activity Schedule,       !- Name
    ,                        !- Schedule Type Limits Name
    70;                      !- Hourly Value (W/person sensible, ISO 13790 Table G.11)

Lights,
    Zone1 Lights,            !- Name
    Zone1,                   !- Zone Name
    Always On,               !- Schedule Name
    LightingLevel,           !- Design Level Calculation Method
    {building.lighting_w_m2 * building.floor_area_m2},  !- Lighting Level {{W}}
    ,                        !- Watts per Floor Area {{W/m2}}
    ,                        !- Watts per Person {{W/person}}
    0,                       !- Return Air Fraction
    0.4,                     !- Fraction Radiant
    0.2,                     !- Fraction Visible
    1;                       !- Fraction Replaceable

ElectricEquipment,
    Zone1 Equipment,         !- Name
    Zone1,                   !- Zone Name
    Always On,               !- Schedule Name
    EquipmentLevel,          !- Design Level Calculation Method
    {building.equipment_w_m2 * building.floor_area_m2},  !- Design Level {{W}}
    ,                        !- Watts per Floor Area {{W/m2}}
    ,                        !- Watts per Person {{W/person}}
    0,                       !- Fraction Latent
    0.3,                     !- Fraction Radiant
    0;                       !- Fraction Lost

!-   ===========  HVAC (IDEAL LOADS)  ===========

ZoneHVAC:IdealLoadsAirSystem,
    Zone1 Ideal Loads,       !- Name
    Always On,               !- Availability Schedule Name
    Zone1 Inlet Node,        !- Zone Supply Air Node Name
    Zone1 Exhaust Node,      !- Zone Exhaust Air Node Name
    ,                        !- System Inlet Air Node Name
    50,                      !- Maximum Heating Supply Air Temperature {{C}}
    13,                      !- Minimum Cooling Supply Air Temperature {{C}}
    0.0156,                  !- Maximum Heating Supply Air Humidity Ratio {{kgWater/kgDryAir}}
    0.0077,                  !- Minimum Cooling Supply Air Humidity Ratio {{kgWater/kgDryAir}}
    NoLimit,                 !- Heating Limit
    ,                        !- Maximum Heating Air Flow Rate {{m3/s}}
    ,                        !- Maximum Sensible Heating Capacity {{W}}
    NoLimit,                 !- Cooling Limit
    ,                        !- Maximum Cooling Air Flow Rate {{m3/s}}
    ,                        !- Maximum Total Cooling Capacity {{W}}
    Heating Setpoint,        !- Heating Availability Schedule Name
    Cooling Setpoint,        !- Cooling Availability Schedule Name
    ConstantSensibleHeatRatio,  !- Dehumidification Control Type
    0.7,                     !- Cooling Sensible Heat Ratio
    None,                    !- Humidification Control Type
    ,                        !- Design Specification Outdoor Air Object Name
    ,                        !- Outdoor Air Inlet Node Name
    None,                    !- Demand Controlled Ventilation Type
    NoEconomizer,            !- Outdoor Air Economizer Type
    None,                    !- Heat Recovery Type
    0.7,                     !- Sensible Heat Recovery Effectiveness
    0.65;                    !- Latent Heat Recovery Effectiveness

ZoneHVAC:EquipmentConnections,
    Zone1,                   !- Zone Name
    Zone1 Equipment List,    !- Zone Conditioning Equipment List Name
    Zone1 Inlet Nodes,       !- Zone Air Inlet Node or NodeList Name
    Zone1 Exhaust Nodes,     !- Zone Air Exhaust Node or NodeList Name
    Zone1 Air Node,          !- Zone Air Node Name
    Zone1 Return Node;       !- Zone Return Air Node or NodeList Name

ZoneHVAC:EquipmentList,
    Zone1 Equipment List,    !- Name
    SequentialLoad,          !- Load Distribution Scheme
    ZoneHVAC:IdealLoadsAirSystem,  !- Zone Equipment 1 Object Type
    Zone1 Ideal Loads,       !- Zone Equipment 1 Name
    1,                       !- Zone Equipment 1 Cooling Sequence
    1,                       !- Zone Equipment 1 Heating or No-Load Sequence
    ,                        !- Zone Equipment 1 Sequential Cooling Fraction Schedule Name
    ;                        !- Zone Equipment 1 Sequential Heating Fraction Schedule Name

NodeList,
    Zone1 Inlet Nodes,       !- Name
    Zone1 Inlet Node;        !- Node 1 Name

NodeList,
    Zone1 Exhaust Nodes,     !- Name
    Zone1 Exhaust Node;      !- Node 1 Name

!-   ===========  THERMOSTAT  ===========

ZoneControl:Thermostat,
    Zone1 Thermostat,        !- Name
    Zone1,                   !- Zone Name
    Zone1 Thermostat Schedule,  !- Control Type Schedule Name
    ThermostatSetpoint:DualSetpoint,  !- Control 1 Object Type
    Zone1 DualSP;            !- Control 1 Name

Schedule:Constant,
    Zone1 Thermostat Schedule,  !- Name
    ,                        !- Schedule Type Limits Name
    4;                       !- Hourly Value (4 = DualSetpoint)

ThermostatSetpoint:DualSetpoint,
    Zone1 DualSP,            !- Name
    Heating Setpoint,        !- Heating Setpoint Temperature Schedule Name
    Cooling Setpoint;        !- Cooling Setpoint Temperature Schedule Name

!-   ===========  OUTPUT  ===========

Output:VariableDictionary,
    IDF;                     !- Key Field

Output:Variable,
    *,                       !- Key Value
    Zone Ideal Loads Supply Air Total Heating Energy,  !- Variable Name
    Hourly;                  !- Reporting Frequency

Output:Variable,
    *,                       !- Key Value
    Zone Ideal Loads Supply Air Total Cooling Energy,  !- Variable Name
    Hourly;                  !- Reporting Frequency

Output:Variable,
    *,                       !- Key Value
    Zone Mean Air Temperature,  !- Variable Name
    Hourly;                  !- Reporting Frequency

Output:Variable,
    *,                       !- Key Value
    Zone Windows Total Transmitted Solar Radiation Energy,  !- Variable Name
    Hourly;                  !- Reporting Frequency

Output:Variable,
    *,                       !- Key Value
    Zone People Total Heating Energy,  !- Variable Name
    Hourly;                  !- Reporting Frequency

Output:Variable,
    *,                       !- Key Value
    Zone Electric Equipment Total Heating Energy,  !- Variable Name
    Hourly;                  !- Reporting Frequency

Output:Variable,
    *,                       !- Key Value
    Zone Lights Total Heating Energy,  !- Variable Name
    Hourly;                  !- Reporting Frequency

Output:Meter,
    Heating:EnergyTransfer,  !- Key Name
    Hourly;                  !- Reporting Frequency

Output:Meter,
    Cooling:EnergyTransfer,  !- Key Name
    Hourly;                  !- Reporting Frequency

OutputControl:Table:Style,
    HTML,                    !- Column Separator
    JtoKWH;                  !- Unit Conversion

Output:Table:SummaryReports,
    AllSummary;              !- Report 1 Name

Output:SQLite,
    SimpleAndTabular;        !- Option Type
"""
    
    # Write IDF file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(idf_content)
    
    return output_path


def generate_polygon_idf(building: ParametricBuilding, geometry: "BuildingGeometry", output_path: Path) -> Path:
    """Generate EnergyPlus IDF file for polygon-based building."""
    geom_params = geometry.get_5r1c_params()
    surfaces = geometry.get_eplus_surfaces()

    H = geometry.total_height
    floor_area = geom_params['floor_area_m2']
    footprint_area = geom_params['footprint_area']
    volume = footprint_area * H

    ground_temps = [4.0, 3.5, 4.0, 6.0, 9.0, 12.0, 14.5, 14.0, 12.0, 9.0, 6.0, 4.5]

    idf_content = f"""!-Generator: Parametric Building Generator for ISO 13790 Comparison
!-Option SortedOrder

Version,
    24.2;

SimulationControl,
    No, No, No, No, Yes, No, 1;

Building,
    ParametricBuilding,
    0,
    Suburbs,
    0.04, 0.4,
    FullInteriorAndExterior,
    25, 6;

Timestep, 6;

RunPeriod,
    Annual, 1, 1, , 12, 31, , Sunday, No, No, No, Yes, Yes;

Site:GroundTemperature:BuildingSurface,
    {', '.join(f'{t}' for t in ground_temps)};

ScheduleTypeLimits, Fraction, 0, 1, Continuous;
ScheduleTypeLimits, Temperature, -100, 200, Continuous;
Schedule:Constant, Always On, Fraction, 1.0;
Schedule:Constant, Heating Setpoint, Temperature, {building.t_heating};
Schedule:Constant, Cooling Setpoint, Temperature, {building.t_cooling};

Material,
    WallMaterial, MediumRough,
    {building.wall_thickness_m},
    {building.wall_thickness_m / max(0.01, 1.0/building.u_walls - 0.17)},
    {building.wall_density_kg_m3},
    {building.wall_specific_heat_j_kgk},
    0.9, 0.7, 0.7;

Material,
    RoofMaterial, MediumRough,
    {building.wall_thickness_m},
    {building.wall_thickness_m / max(0.01, 1.0/building.u_roof - 0.14)},
    {building.wall_density_kg_m3},
    {building.wall_specific_heat_j_kgk},
    0.9, 0.7, 0.7;

Material,
    FloorMaterial, MediumRough,
    {building.wall_thickness_m},
    {building.wall_thickness_m / max(0.01, 1.0/building.u_floor - 0.17)},
    {building.wall_density_kg_m3 * 1.2},
    {building.wall_specific_heat_j_kgk},
    0.9, 0.7, 0.7;

Construction, WallConstruction, WallMaterial;
Construction, RoofConstruction, RoofMaterial;
Construction, FloorConstruction, FloorMaterial;

WindowMaterial:SimpleGlazingSystem,
    SimpleWindow, {building.u_windows}, {building.g_value};

Construction, WindowConstruction, SimpleWindow;

GlobalGeometryRules,
    UpperLeftCorner, Counterclockwise, World;

Zone,
    Zone1, 0, 0, 0, 0, 1, 1, {geometry.total_height}, {volume};

"""

    for surface in surfaces:
        if surface['type'] != 'wall':
            continue
        verts = surface['vertices']
        idf_content += f"""
BuildingSurface:Detailed,
    {surface['name']},
    Wall,
    WallConstruction,
    Zone1, ,
    Outdoors, ,
    SunExposed, WindExposed,
    autocalculate,
    4,
    {verts[0][0]:.4f}, {verts[0][1]:.4f}, {verts[0][2]:.4f},
    {verts[1][0]:.4f}, {verts[1][1]:.4f}, {verts[1][2]:.4f},
    {verts[2][0]:.4f}, {verts[2][1]:.4f}, {verts[2][2]:.4f},
    {verts[3][0]:.4f}, {verts[3][1]:.4f}, {verts[3][2]:.4f};
"""

    roof_verts = [(v[0], v[1], H) for v in geometry.vertices]
    idf_content += f"""
BuildingSurface:Detailed,
    Roof, Roof, RoofConstruction,
    Zone1, , Outdoors, , SunExposed, WindExposed, autocalculate,
    {len(roof_verts)},
"""
    idf_content += ",\n".join(f"    {v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}" for v in roof_verts) + ";\n"

    floor_verts = [(v[0], v[1], 0) for v in reversed(geometry.vertices)]
    idf_content += f"""
BuildingSurface:Detailed,
    Floor, Floor, FloorConstruction,
    Zone1, , Ground, , NoSun, NoWind, autocalculate,
    {len(floor_verts)},
"""
    idf_content += ",\n".join(f"    {v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}" for v in floor_verts) + ";\n"

    for surface in surfaces:
        if surface['type'] != 'window':
            continue
        verts = surface['vertices']
        idf_content += f"""
FenestrationSurface:Detailed,
    {surface['name']},
    Window,
    WindowConstruction,
    {surface['parent']}, ,
    autocalculate, , 1,
    4,
    {verts[0][0]:.4f}, {verts[0][1]:.4f}, {verts[0][2]:.4f},
    {verts[1][0]:.4f}, {verts[1][1]:.4f}, {verts[1][2]:.4f},
    {verts[2][0]:.4f}, {verts[2][1]:.4f}, {verts[2][2]:.4f},
    {verts[3][0]:.4f}, {verts[3][1]:.4f}, {verts[3][2]:.4f};
"""

    idf_content += f"""
People,
    Zone1 People,
    Zone1,
    Always On,
    People,
    {floor_area * building.occupant_density:.1f},
    ,
    ,
    0.3,
    1.0,
    Activity Schedule;

Schedule:Constant,
    Activity Schedule,
    ,
    70;

Lights,
    Zone1 Lights,
    Zone1,
    Always On,
    LightingLevel,
    {building.lighting_w_m2 * floor_area:.1f},
    ,
    ,
    0,
    0.4,
    0.2,
    1;

ElectricEquipment,
    Zone1 Equipment,
    Zone1,
    Always On,
    EquipmentLevel,
    {building.equipment_w_m2 * floor_area:.1f},
    ,
    ,
    0,
    0.3,
    0;

ZoneInfiltration:DesignFlowRate,
    Zone1 Infiltration,
    Zone1,
    Always On,
    AirChanges/Hour,
    ,
    ,
    ,
    {building.ach_infiltration},
    1.0,
    0.0,
    0.0,
    0.0;

ZoneHVAC:IdealLoadsAirSystem,
    Zone1 Ideal Loads,
    Always On,
    Zone1 Inlet Node,
    Zone1 Exhaust Node,
    ,
    50,
    13,
    0.0156,
    0.0077,
    NoLimit,
    ,
    ,
    NoLimit,
    ,
    ,
    Heating Setpoint,
    Cooling Setpoint,
    ConstantSensibleHeatRatio,
    0.7,
    None,
    ,
    ,
    None,
    NoEconomizer,
    None,
    0.7,
    0.65;

ZoneHVAC:EquipmentConnections,
    Zone1,
    Zone1 Equipment List,
    Zone1 Inlet Nodes,
    Zone1 Exhaust Nodes,
    Zone1 Air Node,
    Zone1 Return Node;

ZoneHVAC:EquipmentList,
    Zone1 Equipment List,
    SequentialLoad,
    ZoneHVAC:IdealLoadsAirSystem,
    Zone1 Ideal Loads,
    1,
    1,
    ,
    ;

NodeList,
    Zone1 Inlet Nodes,
    Zone1 Inlet Node;

NodeList,
    Zone1 Exhaust Nodes,
    Zone1 Exhaust Node;

ZoneControl:Thermostat,
    Zone1 Thermostat,
    Zone1,
    Zone Control Type Sched,
    ThermostatSetpoint:DualSetpoint,
    Zone1 DualSP;

Schedule:Constant,
    Zone Control Type Sched,
    ,
    4;

ThermostatSetpoint:DualSetpoint,
    Zone1 DualSP,
    Heating Setpoint,
    Cooling Setpoint;

Output:VariableDictionary, IDF;
Output:Variable, *, Zone Ideal Loads Supply Air Total Heating Energy, Hourly;
Output:Variable, *, Zone Ideal Loads Supply Air Total Cooling Energy, Hourly;
Output:Variable, *, Zone Mean Air Temperature, Hourly;
Output:Variable, *, Zone Windows Total Transmitted Solar Radiation Energy, Hourly;
Output:Variable, *, Zone People Total Heating Energy, Hourly;
Output:Variable, *, Zone Electric Equipment Total Heating Energy, Hourly;
Output:Variable, *, Zone Lights Total Heating Energy, Hourly;
Output:Meter, Heating:EnergyTransfer, Hourly;
Output:Meter, Cooling:EnergyTransfer, Hourly;
OutputControl:Table:Style, HTML, JtoKWH;
Output:Table:SummaryReports, AllSummary;
Output:SQLite, SimpleAndTabular;
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(idf_content)
    return output_path


def get_5r1c_params(building: ParametricBuilding) -> dict:
    """
    Extract parameters for 5R1C ISO 13790 model.
    
    Returns dict compatible with Building5R1C constructor.
    Uses construction-based thermal mass and orientation-specific windows
    for detailed solar calculation.
    """
    from pathlib import Path
    
    geometry = building.geometry

    # Use construction-based thermal capacitance
    kappa = building.effective_thermal_capacitance_j_m2k

    if geometry is not None:
        geom_params = geometry.get_5r1c_params()
        window_areas = geometry.get_window_areas_by_cardinal()
        return {
            "floor_area_m2": geom_params['floor_area_m2'],
            "thermal_capacitance_j_m2k": kappa,
            "building_class": None,
            "u_walls": building.u_walls,
            "u_roof": building.u_roof,
            "u_floor": building.u_floor,
            "u_windows": building.u_windows,
            "wall_area_m2": geom_params['total_wall_area_net'],
            "roof_area_m2": geom_params['roof_area_m2'],
            "floor_area_ground_m2": geom_params['roof_area_m2'],
            "window_area_m2": geom_params['total_window_area'],
            "window_area_south": window_areas['S'],
            "window_area_north": window_areas['N'],
            "window_area_east": window_areas['E'],
            "window_area_west": window_areas['W'],
            "building_height_m": geom_params['total_height'],
            "ceiling_height_m": geom_params['floor_height'],
            "ach_infiltration": building.ach_infiltration,
            "ach_ventilation": 0.0,
            "heat_recovery_efficiency": 0.0,
            "g_value": building.g_value,
            "frame_factor": 0.8,
            "t_set_heating": building.t_heating,
            "internal_gains_w_per_m2": building.internal_gains_w_m2,
            "temp_adj_ground": 0.5,
            "epw_path": Path(building.epw_path) if building.epw_path else None,
            "solar_method": "auto",
        }
    
    return {
        "floor_area_m2": building.floor_area_m2,
        # Use explicit thermal capacitance instead of deprecated building_class
        "thermal_capacitance_j_m2k": kappa,
        "building_class": None,  # Explicitly disable class-based lookup
        "u_walls": building.u_walls,
        "u_roof": building.u_roof,
        "u_floor": building.u_floor,
        "u_windows": building.u_windows,
        "wall_area_m2": building.net_wall_area,
        "roof_area_m2": building.roof_area,
        "floor_area_ground_m2": building.ground_floor_area,
        "window_area_m2": building.total_window_area,
        # Orientation-specific window areas for detailed solar (pvlib Perez)
        "window_area_south": building.window_area_south,
        "window_area_north": building.window_area_north,
        "window_area_east": building.window_area_east,
        "window_area_west": building.window_area_west,
        "building_height_m": building.total_height_m,
        "ceiling_height_m": building.floor_height_m,
        "ach_infiltration": building.ach_infiltration,
        "ach_ventilation": 0.0,  # No mechanical ventilation
        "heat_recovery_efficiency": 0.0,
        "g_value": building.g_value,
        "frame_factor": 0.8,  # Standard frame factor
        "t_set_heating": building.t_heating,
        "internal_gains_w_per_m2": building.internal_gains_w_m2,
        "temp_adj_ground": 0.5,  # Ground temperature adjustment
        # EPW path for detailed solar
        "epw_path": Path(building.epw_path) if building.epw_path else None,
        # Solar method: auto will use detailed when EPW/orientation available
        "solar_method": "auto",
    }


if __name__ == "__main__":
    # Example: Create a baseline building
    baseline = ParametricBuilding(
        length_m=20,
        width_m=10,
        num_floors=1,
        floor_height_m=3.0,
        orientation_deg=0,
        wwr_north=0.3,
        wwr_south=0.3,
        wwr_east=0.2,
        wwr_west=0.2,
        u_walls=0.5,
        u_roof=0.3,
        u_floor=0.5,
        u_windows=2.0,
        g_value=0.6,
        ach_infiltration=0.5,
        thermal_mass_class="HEAVY",
    )
    
    print(baseline.summary())
    
    # Generate IDF
    output_dir = Path("validation/iso13790_comparison/parametric_models")
    idf_path = generate_idf(baseline, output_dir / "baseline.idf")
    print(f"\nGenerated IDF: {idf_path}")
    
    # Get 5R1C parameters
    params = get_5r1c_params(baseline)
    print("\n5R1C Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
