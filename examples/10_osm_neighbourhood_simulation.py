#!/usr/bin/env python3
"""
=============================================================================
Demo: Simulate a Neighbourhood of Buildings from OpenStreetMap
=============================================================================

This demo shows how to:
1. Fetch real building footprints from OpenStreetMap via Overpass API
2. Convert footprints to BuildingGeometry with orientation-aware windows
3. Build ISO 13790 5R1C thermal models for each building
4. Run annual heating demand simulation for the neighbourhood
5. Compare buildings and visualise aggregate results

The workflow uses the FootprintImporter to pull building polygons by OSM
way IDs, then creates Building5R1C models from the derived geometry.

Requirements (install with `pip install epsm_rc[platform,gis]`):
    - pyproj, requests  (for OSM fetching)
    - pvlib              (for detailed solar gains)
    - matplotlib         (for plotting)

Reference: ISO 13790:2008 "Energy performance of buildings"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from pathlib import Path

from epsm_rc import (
    Building5R1C,
    BuildingClass,
    FootprintImporter,
    BuildingGeometry,
)
from epsm_rc.weather import load_epw


# =============================================================================
# 1. FETCH BUILDINGS FROM OPENSTREETMAP
# =============================================================================

# A small neighbourhood in Gothenburg (Majorna).
# Each entry: (OSM way ID, assumed construction era, number of floors override).
# Set num_floors=None to require source OSM levels; absent levels raise an error.
BUILDINGS = [
    {"way_id": 90851303, "era": "1960_1975", "num_floors": None, "label": "A"},
    {"way_id": 90851304, "era": "1976_1990", "num_floors": 3, "label": "B"},
    {"way_id": 90851305, "era": "pre_1960",  "num_floors": None, "label": "C"},
]

# U-values by Swedish construction era [W/(m²K)]
U_VALUES_BY_ERA = {
    "pre_1960":  {"u_walls": 0.60, "u_roof": 0.40, "u_floor": 0.40, "u_windows": 2.8},
    "1960_1975": {"u_walls": 0.40, "u_roof": 0.25, "u_floor": 0.30, "u_windows": 2.5},
    "1976_1990": {"u_walls": 0.25, "u_roof": 0.15, "u_floor": 0.25, "u_windows": 1.8},
    "1991_2010": {"u_walls": 0.20, "u_roof": 0.12, "u_floor": 0.20, "u_windows": 1.4},
    "post_2010": {"u_walls": 0.15, "u_roof": 0.10, "u_floor": 0.15, "u_windows": 1.0},
}

# Ventilation parameters by era — older buildings lack mechanical heat recovery
VENTILATION_BY_ERA = {
    "pre_1960":  {"ach_inf": 0.40, "ach_vent": 0.3, "hr": 0.0},   # natural ventilation, leaky
    "1960_1975": {"ach_inf": 0.30, "ach_vent": 0.5, "hr": 0.0},   # exhaust-only (F-system)
    "1976_1990": {"ach_inf": 0.20, "ach_vent": 0.5, "hr": 0.50},  # early FTX, moderate recovery
    "1991_2010": {"ach_inf": 0.15, "ach_vent": 0.5, "hr": 0.70},  # good FTX
    "post_2010": {"ach_inf": 0.10, "ach_vent": 0.5, "hr": 0.85},  # modern FTX, tight envelope
}


def fetch_buildings(building_defs: list[dict]) -> list[dict]:
    """Fetch building footprints from OSM and attach thermal parameters.

    Returns a list of dicts, each containing:
        - 'geometry': BuildingGeometry
        - 'era': construction era string
        - 'label': short label for plots
        - 'params_5r1c': dict from geometry.get_5r1c_params()
    """
    import time

    results = []
    for i, bdef in enumerate(building_defs):
        way_id = bdef["way_id"]
        # Explicit illustrative assumptions for this example; not source evidence.
        num_floors = bdef.get("num_floors")
        print(f"  Fetching OSM way/{way_id} ...")

        if i > 0:
            time.sleep(2)  # be polite to the Overpass API

        geom = FootprintImporter.from_osm(
            way_id=way_id,
            num_floors=num_floors,
            floor_height=2.5,
        )
        geom.set_wwr_cardinal({"N": 0.12, "S": 0.25, "E": 0.15, "W": 0.15})

        params = geom.get_5r1c_params()
        print(f"    → {params['floor_area_m2']:.0f} m², "
              f"{params['num_floors']} floors, "
              f"{len(geom.wall_segments)} wall segments")

        results.append({
            "geometry": geom,
            "era": bdef["era"],
            "label": bdef["label"],
            "way_id": way_id,
            "params_5r1c": params,
        })

    return results


# =============================================================================
# 2. CREATE THERMAL MODELS
# =============================================================================

def create_thermal_models(buildings: list[dict]) -> list[dict]:
    """Create a Building5R1C for each fetched building.

    Populates each dict with a 'model' key.
    """
    for b in buildings:
        p = b["params_5r1c"]
        u = U_VALUES_BY_ERA[b["era"]]

        window_areas = b["geometry"].get_window_areas_by_cardinal()
        vent = VENTILATION_BY_ERA[b["era"]]

        model = Building5R1C(
            floor_area_m2=p["floor_area_m2"],
            building_class=BuildingClass.MEDIUM,
            u_walls=u["u_walls"],
            u_roof=u["u_roof"],
            u_floor=u["u_floor"],
            u_windows=u["u_windows"],
            wall_area_m2=p["total_wall_area_net"],
            roof_area_m2=p["roof_area_m2"],
            floor_area_ground_m2=p["footprint_area"],
            window_area_m2=p["total_window_area"],
            building_height_m=p["total_height"],
            # Orientation-specific window areas for detailed solar
            window_area_south=window_areas["S"],
            window_area_north=window_areas["N"],
            window_area_east=window_areas["E"],
            window_area_west=window_areas["W"],
            # Era-appropriate ventilation
            ach_infiltration=vent["ach_inf"],
            ach_ventilation=vent["ach_vent"],
            heat_recovery_efficiency=vent["hr"],
            # Solar
            g_value=0.5,
            frame_factor=0.7,
            # Setpoints
            t_set_heating=21.0,
            t_set_cooling=26.0,
            internal_gains_w_per_m2=4.0,
        )
        b["model"] = model
        print(f"  [{b['label']}] H_tr = {model.heat_loss_coefficient:.1f} W/K, "
              f"τ = {model.time_constant_hours:.1f} h")

    return buildings


# =============================================================================
# 3. SIMULATE ALL BUILDINGS
# =============================================================================

def simulate_neighbourhood(
    buildings: list[dict],
    t_outdoor: np.ndarray,
) -> list[dict]:
    """Run annual hourly simulation for every building.

    Populates each dict with a 'results' key containing simulation output.
    """
    n_hours = len(t_outdoor)

    for b in buildings:
        model: Building5R1C = b["model"]

        # Simplified solar gains (sine-wave proxy if no EPW-based solar)
        hours = np.arange(n_hours) % 24
        day_of_year = np.arange(n_hours) // 24
        seasonal = 0.3 + 0.7 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        seasonal = np.clip(seasonal, 0.2, 1.0)
        daylight = np.where(
            (hours >= 6) & (hours <= 18),
            1.0 - ((hours - 12) / 6) ** 2,
            0.0,
        )
        solar_gains = (
            200  # peak W/m² of window
            * model.window_area_m2
            * model.g_value
            * model.frame_factor
            * model.shading_factor
            * seasonal
            * daylight
        )

        results = model.simulate(
            t_outdoor=t_outdoor,
            solar_gains=solar_gains,
            dt_hours=1.0,
        )

        annual_kwh = results["heating_demand"].sum() / 1000
        kwh_m2 = annual_kwh / model.floor_area_m2
        peak_kw = results["heating_demand"].max() / 1000

        b["results"] = results
        b["annual_kwh"] = annual_kwh
        b["kwh_m2"] = kwh_m2
        b["peak_kw"] = peak_kw

        print(f"  [{b['label']}] {annual_kwh:,.0f} kWh/yr  "
              f"({kwh_m2:.1f} kWh/m²)  peak {peak_kw:.1f} kW")

    return buildings


# =============================================================================
# 4. VISUALISATION
# =============================================================================

def plot_footprints(buildings: list[dict], save_path: Path | None = None):
    """Plot all building footprints side-by-side with key metrics."""
    n = len(buildings)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, b in zip(axes, buildings):
        geom: BuildingGeometry = b["geometry"]
        verts = geom.vertices + [geom.vertices[0]]  # close the ring
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        ax.fill(xs, ys, alpha=0.3, color="steelblue")
        ax.plot(xs, ys, "o-", color="steelblue", linewidth=2, markersize=4)

        # Label cardinal directions of each wall segment
        for seg in geom.wall_segments:
            mx, my = seg.midpoint
            ax.annotate(
                seg.cardinal_direction,
                (mx, my),
                fontsize=8,
                ha="center",
                va="center",
                color="navy",
                fontweight="bold",
            )

        ax.set_aspect("equal")
        ax.set_title(
            f"Building {b['label']} (way/{b['way_id']})\n"
            f"{b['params_5r1c']['floor_area_m2']:.0f} m², "
            f"{b['era']}",
            fontsize=10,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Building Footprints from OpenStreetMap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_monthly_comparison(buildings: list[dict], save_path: Path | None = None):
    """Bar chart comparing monthly heating demand across all buildings."""
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    n_buildings = len(buildings)

    monthly = np.zeros((n_buildings, 12))
    for i, b in enumerate(buildings):
        hour = 0
        for m, days in enumerate(month_days):
            h = days * 24
            monthly[i, m] = b["results"]["heating_demand"][hour:hour + h].sum() / 1000
            hour += h

    x = np.arange(12)
    width = 0.8 / n_buildings
    colours = plt.cm.Set2(np.linspace(0, 1, n_buildings))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, b in enumerate(buildings):
        offset = (i - n_buildings / 2 + 0.5) * width
        ax.bar(
            x + offset,
            monthly[i],
            width,
            label=f"{b['label']} – {b['params_5r1c']['floor_area_m2']:.0f} m²",
            color=colours[i],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Heating Demand [kWh]")
    ax.set_title("Monthly Heating Demand by Building", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_annual_summary(buildings: list[dict], save_path: Path | None = None):
    """Horizontal bar chart of annual kWh/m² for each building."""
    labels = [f"{b['label']} (way/{b['way_id']})" for b in buildings]
    values = [b["kwh_m2"] for b in buildings]
    colours = ["#2ca02c" if v < 55 else "#d62728" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.8 * len(buildings) + 1)))
    bars = ax.barh(labels, values, color=colours, alpha=0.85)
    ax.axvline(55, color="gray", linestyle="--", linewidth=1, label="BBR 2020 limit (55 kWh/m²)")
    ax.axvline(15, color="green", linestyle=":", linewidth=1, label="Passive house (15 kWh/m²)")
    ax.set_xlabel("Annual Heating Demand [kWh/m²/year]")
    ax.set_title("Neighbourhood Energy Performance", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def print_neighbourhood_summary(buildings: list[dict]):
    """Print a table summarising every building."""
    total_area = sum(b["params_5r1c"]["floor_area_m2"] for b in buildings)
    total_kwh = sum(b["annual_kwh"] for b in buildings)

    print(f"\n{'='*72}")
    print("NEIGHBOURHOOD SUMMARY")
    print(f"{'='*72}")
    print(f"{'Bldg':>4}  {'OSM Way':>10}  {'Era':>10}  {'Area m²':>8}  "
          f"{'kWh/yr':>10}  {'kWh/m²':>8}  {'Peak kW':>8}")
    print(f"{'-'*72}")
    for b in buildings:
        print(f"{b['label']:>4}  {b['way_id']:>10}  {b['era']:>10}  "
              f"{b['params_5r1c']['floor_area_m2']:>8.0f}  "
              f"{b['annual_kwh']:>10,.0f}  {b['kwh_m2']:>8.1f}  "
              f"{b['peak_kw']:>8.1f}")
    print(f"{'-'*72}")
    print(f"{'Total':>4}  {'':>10}  {'':>10}  {total_area:>8.0f}  "
          f"{total_kwh:>10,.0f}  {total_kwh/total_area:>8.1f}")
    print(f"{'='*72}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  Neighbourhood Thermal Simulation from OpenStreetMap Footprints  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # --- 1. Fetch footprints from OSM ---
    print("[1/4] Fetching building footprints from OpenStreetMap ...")
    buildings = fetch_buildings(BUILDINGS)

    # --- 2. Create thermal models ---
    print("\n[2/4] Creating ISO 13790 5R1C thermal models ...")
    buildings = create_thermal_models(buildings)

    # --- 3. Load weather and simulate ---
    print("\n[3/4] Loading weather & running annual simulations ...")
    epw = load_epw()  # bundled Göteborg TMY
    t_outdoor = epw.temp_c
    print(f"  Weather: {epw.location}, {len(t_outdoor)} hours, "
          f"mean {t_outdoor.mean():.1f} °C")
    buildings = simulate_neighbourhood(buildings, t_outdoor)

    # --- 4. Visualise and summarise ---
    print("\n[4/4] Generating plots ...")
    plot_footprints(buildings, save_path=output_dir / "osm_footprints.png")
    plot_monthly_comparison(buildings, save_path=output_dir / "osm_monthly_comparison.png")
    plot_annual_summary(buildings, save_path=output_dir / "osm_annual_summary.png")
    print_neighbourhood_summary(buildings)

    print("Done! ✓\n")
    return buildings


if __name__ == "__main__":
    buildings = main()
