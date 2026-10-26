"""Historical RC parameter conveniences and an explicit Core comparison CLI.

ParametricBuilding/get_5r1c_params retain standalone compatibility assumptions.
Both independent IDF emitters are retired: new comparisons use a fully authored
Core Building and the shared EnergyPlusCompiler via prepare_engine_comparison.
The CLI requires a physical snapshot, weather and selected RC settings; it
compiles EnergyPlus inputs and simulates RC without running an EnergyPlus year.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from epsm_rc.building_geometry import BuildingGeometry


@dataclass
class ParametricBuilding:
    """Historical standalone RC parameters; explicit Core authoring replaces IDF export."""

    # Geometry
    length_m: float = 20.0  # Building length (E-W direction)
    width_m: float = 10.0  # Building width (N-S direction)
    num_floors: int = 1  # Number of floors
    floor_height_m: float = 3.0  # Floor-to-floor height
    orientation_deg: float = 0.0  # Rotation from north (0 = long axis E-W)
    shape_type: str = "rect"  # rect, l, u, i, skewed, banana
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
    g_value: float = 0.6  # Solar heat gain coefficient

    # Infiltration
    ach_infiltration: float = 0.5  # Air changes per hour at normal pressure

    # Internal gains
    occupant_density: float = 0.1  # persons/m²
    lighting_w_m2: float = 10.0  # W/m²
    equipment_w_m2: float = 10.0  # W/m²

    # Setpoints
    t_heating: float = 21.0  # Heating setpoint [°C]
    t_cooling: float = 26.0  # Cooling setpoint [°C]

    # Thermal mass - now uses construction-based calculation
    # thermal_mass_class is deprecated, use thermal_capacitance_j_m2k instead
    thermal_mass_class: Optional[str] = None  # Deprecated, ignored; accepted through at least 0.10.

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
    epw_path: Optional[str] = None

    def __post_init__(self):
        self._validate_inputs()
        if self.thermal_mass_class is not None:
            warnings.warn(
                "thermal_mass_class is deprecated and has no effect; use "
                "thermal_capacitance_j_m2k or material properties. It will remain "
                "accepted until at least 0.10.",
                DeprecationWarning,
                stacklevel=2,
            )

    def _validate_inputs(self):
        for name in (
            "length_m",
            "width_m",
            "floor_height_m",
            "wall_thickness_m",
            "wall_density_kg_m3",
            "wall_specific_heat_j_kgk",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.num_floors, bool)
            or not isinstance(self.num_floors, (int, np.integer))
            or self.num_floors <= 0
        ):
            raise ValueError("num_floors must be a positive integer")
        for name in (
            "u_walls",
            "u_roof",
            "u_floor",
            "u_windows",
            "ach_infiltration",
            "occupant_density",
            "lighting_w_m2",
            "equipment_w_m2",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("wwr_north", "wwr_south", "wwr_east", "wwr_west", "g_value"):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a finite fraction")
        if not all(
            np.isfinite(getattr(self, name))
            for name in ("orientation_deg", "t_heating", "t_cooling")
        ):
            raise ValueError("Orientation and setpoints must be finite")
        if self.t_heating > self.t_cooling:
            raise ValueError("Heating setpoint must not exceed cooling setpoint")
        if self.thermal_capacitance_j_m2k is not None and (
            not np.isfinite(self.thermal_capacitance_j_m2k) or self.thermal_capacitance_j_m2k <= 0
        ):
            raise ValueError("thermal_capacitance_j_m2k must be finite and positive")

    def __repr__(self):
        """Inspect dimensions without constructing geometry or reading weather."""
        floors = str(self.num_floors)
        floors = floors if len(floors) <= 12 else floors[:9] + "…"
        return (
            f"ParametricBuilding(length_m={self.length_m:.6g}, width_m={self.width_m:.6g}, "
            f"num_floors={floors}, orientation_deg={self.orientation_deg:.6g})"
        )

    # Computed properties
    @property
    def geometry(self) -> Optional["BuildingGeometry"]:
        """Return polygon geometry for non-rectangular/custom footprints."""
        shape = (self.shape_type or "rect").lower()
        if (
            shape == "rect"
            and not self.custom_footprint_vertices
            and abs(self.orientation_deg % 360) < 1e-12
        ):
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
        geom.set_wwr_cardinal(
            {
                "N": self.wwr_north,
                "S": self.wwr_south,
                "E": self.wwr_east,
                "W": self.wwr_west,
            }
        )
        return geom

    def _rotate_vertices(
        self, vertices: List[Tuple[float, float]], angle_deg: float
    ) -> List[Tuple[float, float]]:
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
                (0.0, 0.0),
                (L, 0.0),
                (L, W),
                (x2, W),
                (x2, y_notch),
                (x1, y_notch),
                (x1, W),
                (0.0, W),
            ]

        if s == "i":
            y1 = 0.22 * W
            y2 = 0.78 * W
            x1 = 0.34 * L
            x2 = 0.66 * L
            return [
                (0.0, 0.0),
                (L, 0.0),
                (L, y1),
                (x2, y1),
                (x2, y2),
                (L, y2),
                (L, W),
                (0.0, W),
                (0.0, y2),
                (x1, y2),
                (x1, y1),
                (0.0, y1),
            ]

        if s == "skewed":
            return [(0.0, 0.08 * W), (L, 0.0), (0.92 * L, W), (0.10 * L, 0.88 * W)]

        if s == "banana":
            return [
                (0.00 * L, 0.30 * W),
                (0.12 * L, 0.10 * W),
                (0.42 * L, 0.00 * W),
                (0.76 * L, 0.10 * W),
                (1.00 * L, 0.34 * W),
                (0.92 * L, 0.62 * W),
                (0.62 * L, 0.86 * W),
                (0.28 * L, 1.00 * W),
                (0.04 * L, 0.78 * W),
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
            return geom.get_wall_areas_by_cardinal()["N"]
        return self.length_m * self.total_height_m  # IDF: N/S walls span length_m

    @property
    def wall_area_south(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_wall_areas_by_cardinal()["S"]
        return self.length_m * self.total_height_m  # IDF: N/S walls span length_m

    @property
    def wall_area_east(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_wall_areas_by_cardinal()["E"]
        return self.width_m * self.total_height_m  # IDF: E/W walls span width_m

    @property
    def wall_area_west(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_wall_areas_by_cardinal()["W"]
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
            return geom.get_window_areas_by_cardinal()["N"]
        return self.wall_area_north * self.wwr_north

    @property
    def window_area_south(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_window_areas_by_cardinal()["S"]
        return self.wall_area_south * self.wwr_south

    @property
    def window_area_east(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_window_areas_by_cardinal()["E"]
        return self.wall_area_east * self.wwr_east

    @property
    def window_area_west(self) -> float:
        geom = self.geometry
        if geom is not None:
            return geom.get_window_areas_by_cardinal()["W"]
        return self.wall_area_west * self.wwr_west

    @property
    def total_window_area(self) -> float:
        return (
            self.window_area_north
            + self.window_area_south
            + self.window_area_east
            + self.window_area_west
        )

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

        Priority: explicit value, otherwise calculated from material properties.
        The deprecated thermal_mass_class argument has no effect.
        """
        if self.thermal_capacitance_j_m2k is not None:
            return self.thermal_capacitance_j_m2k
        return self.calculated_thermal_capacitance_j_m2k

    def summary(self) -> str:
        """Return a readable building summary, calculating footprint geometry if needed."""
        lines = [
            "=" * 60,
            "PARAMETRIC BUILDING SUMMARY",
            "=" * 60,
            "\nGeometry:",
            f"  Shape: {self.shape_type.upper()}",
            f"  Dimensions: {self.length_m} x {self.width_m} x {self.total_height_m} m",
            f"  Floors: {self.num_floors} @ {self.floor_height_m} m each",
            f"  Floor area: {self.floor_area_m2:.0f} m²",
            f"  Volume: {self.volume_m3:.0f} m³",
            f"  Orientation: {self.orientation_deg}° from north",
            "\nEnvelope:",
            f"  Wall area (gross): {self.gross_wall_area:.0f} m²",
            f"  Wall area (net): {self.net_wall_area:.0f} m²",
            f"  Window area: {self.total_window_area:.0f} m² (WWR={self.average_wwr:.1%})",
            f"  Roof area: {self.roof_area:.0f} m²",
            f"  Ground floor area: {self.ground_floor_area:.0f} m²",
            "\nU-values [W/m²K]:",
            f"  Walls: {self.u_walls}",
            f"  Roof: {self.u_roof}",
            f"  Floor: {self.u_floor}",
            f"  Windows: {self.u_windows} (g={self.g_value})",
            "\nOperational:",
            f"  Infiltration: {self.ach_infiltration} ACH",
            f"  Internal gains: {self.internal_gains_w_m2:.1f} W/m²",
            f"  Heating setpoint: {self.t_heating}°C",
            "\nThermal Mass:",
            f"  Construction: {self.wall_thickness_m * 100:.0f}cm @ ρ={self.wall_density_kg_m3}, cp={self.wall_specific_heat_j_kgk}",
            f"  Cm per surface: {self.thermal_mass_per_surface_area / 1000:.0f} kJ/(m²·K)",
            f"  Total internal surface: {self.total_internal_surface_area:.0f} m²",
            f"  κ (per floor area): {self.effective_thermal_capacitance_j_m2k / 1000:.0f} kJ/(m²·K)",
        ]
        return "\n".join(lines)


def generate_idf(*args, **kwargs):
    """Retired IDF authoring: supply an explicit Core Building to the shared compiler.

    Any arguments are accepted so every call reports the migration path.
    """
    raise ValueError(
        "Legacy IDF generation is retired. Author an explicit Core Building and use "
        "EnergyPlusCompiler or epsm_rc.prepare_engine_comparison with explicit weather "
        "and RC settings; no material or schedule conversion is inferred."
    )


def generate_polygon_idf(*args, **kwargs):
    """Retired polygon IDF authoring: Core owns geometry and EnergyPlus compilation.

    Any arguments are accepted so every call reports the migration path.
    """
    raise ValueError(
        "Legacy polygon IDF generation is retired. Use an explicit Core Building "
        "and EnergyPlusCompiler or epsm_rc.prepare_engine_comparison."
    )


def get_5r1c_params(building: ParametricBuilding) -> dict:
    """
    Extract parameters for 5R1C ISO 13790 model.

    Returns dict compatible with Building5R1C constructor.
    Uses construction-based thermal mass and orientation-specific windows
    for detailed solar calculation.
    """
    from pathlib import Path

    building._validate_inputs()

    geometry = building.geometry

    # Use construction-based thermal capacitance
    kappa = building.effective_thermal_capacitance_j_m2k

    if geometry is not None:
        geom_params = geometry.get_5r1c_params()
        window_areas = geometry.get_window_areas_by_cardinal()
        return {
            "floor_area_m2": geom_params["floor_area_m2"],
            "thermal_capacitance_j_m2k": kappa,
            "building_class": None,
            "u_walls": building.u_walls,
            "u_roof": building.u_roof,
            "u_floor": building.u_floor,
            "u_windows": building.u_windows,
            "wall_area_m2": geom_params["total_wall_area_net"],
            "roof_area_m2": geom_params["roof_area_m2"],
            "floor_area_ground_m2": geom_params["roof_area_m2"],
            "window_area_m2": geom_params["total_window_area"],
            "window_area_south": window_areas["S"],
            "window_area_north": window_areas["N"],
            "window_area_east": window_areas["E"],
            "window_area_west": window_areas["W"],
            "building_height_m": geom_params["total_height"],
            "ceiling_height_m": geom_params["floor_height"],
            "ach_infiltration": building.ach_infiltration,
            "ach_ventilation": 0.0,
            "heat_recovery_efficiency": 0.0,
            "g_value": building.g_value,
            "frame_factor": 0.8,
            "t_set_heating": building.t_heating,
            "t_set_cooling": building.t_cooling,
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
        "t_set_cooling": building.t_cooling,
        "internal_gains_w_per_m2": building.internal_gains_w_m2,
        "temp_adj_ground": 0.5,  # Ground temperature adjustment
        # EPW path for detailed solar
        "epw_path": Path(building.epw_path) if building.epw_path else None,
        # Solar method: auto will use detailed when EPW/orientation available
        "solar_method": "auto",
    }


def main(argv=None):
    """Compile and calculate RC from explicit physical snapshot/weather/settings files."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building", type=Path, required=True, help="Core Building snapshot JSON")
    parser.add_argument("--weather", type=Path, required=True, help="Complete local EPW")
    parser.add_argument("--rc-settings", type=Path, required=True, help="Explicit RCSettings JSON")
    parser.add_argument("--directory", type=Path, required=True, help="Comparison output directory")
    parser.add_argument("--energyplus", type=Path, help="EnergyPlus 24.2 executable; else env/PATH")
    arguments = parser.parse_args(argv)
    from epsm_core import BuildingSnapshot

    from .comparison import prepare_engine_comparison

    building = BuildingSnapshot.from_dict(json.loads(arguments.building.read_text())).to_building()
    result = prepare_engine_comparison(
        building,
        weather=arguments.weather,
        directory=arguments.directory,
        rc_settings=json.loads(arguments.rc_settings.read_text()),
        energyplus=arguments.energyplus,
    )
    print(f"EnergyPlus compiled inputs: {result.energyplus_model_path}")
    print(f"RC annual heating: {result.rc_result.scalars['space_heating_demand_kwh']:.1f} kWh")
    print(f"Comparison manifest: {result.manifest_path}")
    return result


if __name__ == "__main__":
    main()
