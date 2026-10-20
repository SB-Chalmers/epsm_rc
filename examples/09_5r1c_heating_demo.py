#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
# License: See LICENSE at the project root.
# Date modified: 2026-10-20
"""
=============================================================================
Demo: ISO 13790 5R1C Building Thermal Model for Heating Demand
=============================================================================

This demo shows how to:
1. Define a building with geometry, shape, and thermal properties
2. Visualize the building envelope
3. Load EPW weather data
4. Simulate annual heating demand
5. Plot hourly, daily, and monthly results with outdoor temperature

The 5R1C model (5 Resistances, 1 Capacitance) from ISO 13790 captures:
- Thermal mass effects (heat storage in walls, floors)
- Solar gains through windows
- Ventilation and infiltration losses
- Transmission losses through the envelope

Reference: ISO 13790:2008 "Energy performance of buildings"
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

# Public standalone RC imports
from epsm_rc.iso13790 import Building5R1C, BuildingClass
from epsm_rc.weather import EPWData, load_epw

# =============================================================================
# 1. BUILDING DEFINITION
# =============================================================================


def define_building() -> tuple[Building5R1C, dict]:
    """
    Define a building with explicit geometry.

    Building: 2-story Swedish villa
    - Footprint: 10m x 8m = 80 m²
    - 2 stories → 160 m² floor area
    - Height per story: 2.5m → total 5m
    """

    # === GEOMETRY ===
    length_m = 10.0  # Building length [m]
    width_m = 8.0  # Building width [m]
    n_stories = 2  # Number of stories
    story_height_m = 2.5  # Height per story [m]

    # Calculated areas
    footprint_m2 = length_m * width_m  # 80 m²
    floor_area_m2 = footprint_m2 * n_stories  # 160 m²
    building_height_m = story_height_m * n_stories  # 5 m

    # External wall area (perimeter × height)
    perimeter_m = 2 * (length_m + width_m)  # 36 m
    gross_wall_area_m2 = perimeter_m * building_height_m  # 180 m²

    # === WINDOW-TO-WALL RATIO ===
    window_to_wall_ratio = 0.20  # 20% glazing (typical Swedish)
    window_area_m2 = gross_wall_area_m2 * window_to_wall_ratio  # 36 m²
    net_wall_area_m2 = gross_wall_area_m2 - window_area_m2  # 144 m²

    # Roof and floor areas
    roof_area_m2 = footprint_m2  # 80 m² (pitched roof simplified)
    ground_floor_area_m2 = footprint_m2  # 80 m²

    # === U-VALUES (W/m²K) - Modern Swedish standard (BBR 2020) ===
    u_walls = 0.18  # Walls: 200mm mineral wool
    u_roof = 0.13  # Roof: 300mm insulation
    u_floor = 0.15  # Floor: 150mm EPS on ground
    u_windows = 1.0  # Triple-glazed windows

    # === VENTILATION ===
    ach_infiltration = 0.1  # Air tightness: 0.1 ACH @ 50 Pa
    ach_ventilation = 0.5  # Mechanical ventilation: 0.5 ACH
    heat_recovery_eff = 0.80  # Heat recovery: 80% efficiency

    # === SOLAR PROPERTIES ===
    g_value = 0.5  # Solar heat gain coefficient
    frame_factor = 0.7  # 70% glass, 30% frame

    # === THERMAL MASS ===
    building_class = BuildingClass.MEDIUM  # Timber frame with some concrete

    # === SETPOINTS ===
    t_set_heating = 21.0  # Heating setpoint [°C]
    t_set_cooling = 26.0  # Cooling setpoint [°C]

    # === INTERNAL GAINS ===
    internal_gains_w_per_m2 = 4.0  # People + appliances + lighting

    # Create Building5R1C instance
    building = Building5R1C(
        floor_area_m2=floor_area_m2,
        building_class=building_class,
        u_walls=u_walls,
        u_roof=u_roof,
        u_floor=u_floor,
        u_windows=u_windows,
        wall_area_m2=net_wall_area_m2,
        roof_area_m2=roof_area_m2,
        floor_area_ground_m2=ground_floor_area_m2,
        window_area_m2=window_area_m2,
        building_height_m=building_height_m,
        ach_infiltration=ach_infiltration,
        ach_ventilation=ach_ventilation,
        heat_recovery_efficiency=heat_recovery_eff,
        g_value=g_value,
        frame_factor=frame_factor,
        t_set_heating=t_set_heating,
        t_set_cooling=t_set_cooling,
        internal_gains_w_per_m2=internal_gains_w_per_m2,
    )

    # Store geometry for plotting
    geometry = {
        "length_m": length_m,
        "width_m": width_m,
        "height_m": building_height_m,
        "n_stories": n_stories,
        "footprint_m2": footprint_m2,
        "floor_area_m2": floor_area_m2,
        "wall_area_m2": net_wall_area_m2,
        "window_area_m2": window_area_m2,
        "roof_area_m2": roof_area_m2,
        "ground_floor_area_m2": ground_floor_area_m2,
        "window_to_wall_ratio": window_to_wall_ratio,
        "u_walls": u_walls,
        "u_roof": u_roof,
        "u_floor": u_floor,
        "u_windows": u_windows,
    }

    return building, geometry


# =============================================================================
# 2. BUILDING VISUALIZATION
# =============================================================================


def plot_building(geometry: dict, save_path: Path = None):
    """
    Create a schematic visualization of the building.

    Shows:
    - 3D-ish front view with windows
    - Floor plan
    - Summary of thermal properties
    """
    fig = plt.figure(figsize=(14, 8))

    # --- Front Elevation (3D-ish) ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(-1, 12)
    ax1.set_ylim(-1, 8)
    ax1.set_aspect("equal")
    ax1.set_title("Front Elevation", fontsize=14, fontweight="bold")
    ax1.axis("off")

    L = geometry["length_m"]
    H = geometry["height_m"]
    W = geometry["width_m"]
    wwr = geometry["window_to_wall_ratio"]

    # Main wall
    wall = FancyBboxPatch(
        (0, 0),
        L,
        H,
        boxstyle="round,pad=0.02",
        facecolor="#D4A574",  # Wood/brick color
        edgecolor="#8B4513",
        linewidth=2,
    )
    ax1.add_patch(wall)

    # Roof (triangle)
    roof_x = [-0.5, L / 2, L + 0.5]
    roof_y = [H, H + 2, H]
    ax1.fill(roof_x, roof_y, color="#8B0000", edgecolor="#5C0000", linewidth=2)

    # Illustrative window symbols; quantitative areas come from geometry inputs.
    n_windows_per_floor = 4
    win_width = 1.2
    win_height = 1.5

    for story in range(geometry["n_stories"]):
        y_base = story * (H / geometry["n_stories"]) + 0.5
        for i in range(n_windows_per_floor):
            x = 1 + i * (L - 2) / (n_windows_per_floor - 1) - win_width / 2
            window = Rectangle(
                (x, y_base),
                win_width,
                win_height,
                facecolor="#87CEEB",
                edgecolor="#4169E1",
                linewidth=1.5,
            )
            ax1.add_patch(window)

    # Door
    door = Rectangle(
        (L / 2 - 0.5, 0), 1.0, 2.0, facecolor="#654321", edgecolor="#3D2314", linewidth=2
    )
    ax1.add_patch(door)

    # Dimensions
    ax1.annotate(
        "", xy=(L, -0.5), xytext=(0, -0.5), arrowprops=dict(arrowstyle="<->", color="black")
    )
    ax1.text(L / 2, -0.8, f"{L:.0f} m", ha="center", fontsize=10)

    ax1.annotate(
        "", xy=(L + 0.8, H), xytext=(L + 0.8, 0), arrowprops=dict(arrowstyle="<->", color="black")
    )
    ax1.text(L + 1.2, H / 2, f"{H:.0f} m", ha="left", va="center", fontsize=10, rotation=90)

    # --- Summary Panel ---
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # Building summary text
    summary = f"""
    ╔══════════════════════════════════════╗
    ║     BUILDING THERMAL PROPERTIES      ║
    ╠══════════════════════════════════════╣
    ║                                      ║
    ║  GEOMETRY                            ║
    ║  ─────────                           ║
    ║  Dimensions:  {L:.0f}m × {W:.0f}m × {H:.0f}m       ║
    ║  Floor Area:  {geometry["floor_area_m2"]:.0f} m²               ║
    ║  Stories:     {geometry["n_stories"]}                    ║
    ║                                      ║
    ║  ENVELOPE AREAS                      ║
    ║  ──────────────                      ║
    ║  Walls:       {geometry["wall_area_m2"]:.0f} m²               ║
    ║  Windows:     {geometry["window_area_m2"]:.0f} m² (WWR: {wwr * 100:.0f}%)     ║
    ║  Roof:        {geometry["roof_area_m2"]:.0f} m²               ║
    ║  Floor:       {geometry["ground_floor_area_m2"]:.0f} m²               ║
    ║                                      ║
    ║  U-VALUES [W/(m²·K)]                 ║
    ║  ─────────────────                   ║
    ║  Walls:       {geometry["u_walls"]:.2f}                 ║
    ║  Roof:        {geometry["u_roof"]:.2f}                 ║
    ║  Floor:       {geometry["u_floor"]:.2f}                 ║
    ║  Windows:     {geometry["u_windows"]:.2f}                 ║
    ║                                      ║
    ╚══════════════════════════════════════╝
    """

    ax2.text(
        0.5,
        0.5,
        summary,
        transform=ax2.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


# =============================================================================
# 3. WEATHER DATA
# =============================================================================


def load_weather(epw_path: str = None, *, preset: str = None) -> tuple[EPWData, np.ndarray]:
    """
    Load EPW weather data.

    Args:
        epw_path: Explicit EPW path; alternatively choose a named preset.

    Returns:
        EPWData object and hourly temperatures array
    """
    epw = load_epw(epw_path, preset=preset)

    print(f"\n{'=' * 60}")
    print("WEATHER DATA")
    print("=" * 60)
    print(f"  Location:    {epw.location}")
    print(f"  Latitude:    {epw.latitude:.2f}°N")
    print(f"  Longitude:   {epw.longitude:.2f}°E")
    print(f"  Elevation:   {epw.elevation:.0f} m")
    print(f"  Data points: {len(epw.temp_c)} hours")
    print(f"  Temp range:  {epw.temp_c.min():.1f}°C to {epw.temp_c.max():.1f}°C")
    print(f"  Annual mean: {epw.temp_c.mean():.1f}°C")
    print("=" * 60)

    return epw, epw.temp_c


# =============================================================================
# 4. SIMULATE ANNUAL HEATING DEMAND
# =============================================================================


def simulate_annual_heating(
    building: Building5R1C,
    t_outdoor: np.ndarray,
    include_solar: bool = True,
) -> dict:
    """
    Simulate annual heating demand.

    Args:
        building: Building5R1C instance
        t_outdoor: Hourly outdoor temperatures [°C]
        include_solar: Include simplified solar gains

    Returns:
        Dictionary with simulation results
    """
    n_hours = len(t_outdoor)

    # Simple solar gain model (sine wave peaking at noon, zero at night)
    # Peak solar gain through windows in winter: ~50 W/m² × window_area
    if include_solar:
        hours = np.arange(n_hours) % 24
        day_of_year = np.arange(n_hours) // 24

        # Seasonal variation (lower in winter, higher in summer)
        seasonal_factor = 0.3 + 0.7 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        seasonal_factor = np.clip(seasonal_factor, 0.2, 1.0)

        # Daily variation (sun up ~6-18h with peak at noon)
        daylight_factor = np.zeros(n_hours)
        for i, h in enumerate(hours):
            if 6 <= h <= 18:
                # Parabolic shape: peak at noon
                daylight_factor[i] = 1.0 - ((h - 12) / 6) ** 2

        # Peak solar gain (W/m² of window)
        peak_solar_w_per_m2 = 200  # Clear day, south-facing average

        solar_gains = (
            peak_solar_w_per_m2
            * building.window_area_m2
            * building.g_value
            * building.frame_factor
            * building.shading_factor
            * seasonal_factor
            * daylight_factor
        )
    else:
        solar_gains = np.zeros(n_hours)

    # Run simulation
    print("\nSimulating annual heating demand...")
    results = building.simulate(
        t_outdoor=t_outdoor,
        solar_gains=solar_gains,
        dt_hours=1.0,
    )

    # Add outdoor temperature to results
    results["t_outdoor"] = t_outdoor
    results["solar_gains"] = solar_gains

    return results


# =============================================================================
# 5. PLOTTING RESULTS
# =============================================================================


def plot_hourly(results: dict, days: int = 7, start_day: int = 0, save_path: Path = None):
    """
    Plot hourly heating demand and temperatures for a period.

    Args:
        results: Simulation results dictionary
        days: Number of days to plot
        start_day: Start day of year (0 = Jan 1)
        save_path: Optional path to save figure
    """
    start_hour = start_day * 24
    end_hour = start_hour + days * 24

    hours = np.arange(start_hour, end_hour)
    t_out = results["t_outdoor"][start_hour:end_hour]
    t_air = results["t_air"][start_hour:end_hour]
    t_m = results["t_m"][start_hour:end_hour]  # Mass temperature
    q_heat = results["heating_demand"][start_hour:end_hour] / 1000  # kW

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Temperature plot
    ax1.plot(hours, t_out, "b-", label="Outdoor", linewidth=1.5, alpha=0.8)
    ax1.plot(hours, t_air, "r-", label="Indoor Air", linewidth=2)
    ax1.plot(hours, t_m, "orange", label="Thermal Mass", linewidth=1.5, linestyle="--", alpha=0.7)
    ax1.axhline(y=21, color="gray", linestyle=":", alpha=0.7, label="Setpoint (21°C)")
    ax1.fill_between(hours, t_out, alpha=0.2, color="blue")
    ax1.fill_between(
        hours, t_air, 21, where=(t_air < 21), alpha=0.2, color="red", label="Below setpoint"
    )
    ax1.set_ylabel("Temperature [°C]", fontsize=12)
    ax1.legend(loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        f"Hourly Simulation: Days {start_day + 1}-{start_day + days}",
        fontsize=14,
        fontweight="bold",
    )

    # Heating demand plot
    ax2.fill_between(hours, q_heat, alpha=0.7, color="orange", label="Heating Demand")
    ax2.plot(hours, q_heat, "darkorange", linewidth=1)
    ax2.set_xlabel("Hour of Year", fontsize=12)
    ax2.set_ylabel("Heating Demand [kW]", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, None)

    # Add day markers
    for d in range(days + 1):
        h = start_hour + d * 24
        ax2.axvline(x=h, color="gray", linestyle=":", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_daily(results: dict, save_path: Path = None):
    """
    Plot daily heating demand and average temperatures.
    """
    n_days = len(results["heating_demand"]) // 24

    # Aggregate to daily
    daily_heat = np.zeros(n_days)
    daily_t_out = np.zeros(n_days)
    daily_t_air = np.zeros(n_days)
    daily_t_air_min = np.zeros(n_days)
    daily_t_air_max = np.zeros(n_days)

    for d in range(n_days):
        start = d * 24
        end = start + 24
        daily_heat[d] = results["heating_demand"][start:end].sum() / 1000  # kWh
        daily_t_out[d] = results["t_outdoor"][start:end].mean()
        daily_t_air[d] = results["t_air"][start:end].mean()
        daily_t_air_min[d] = results["t_air"][start:end].min()
        daily_t_air_max[d] = results["t_air"][start:end].max()

    days = np.arange(1, n_days + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Temperature
    ax1.fill_between(days, daily_t_out, alpha=0.3, color="blue", label="_nolegend_")
    ax1.plot(days, daily_t_out, "b-", label="Outdoor (daily avg)", linewidth=1)
    ax1.fill_between(
        days, daily_t_air_min, daily_t_air_max, alpha=0.3, color="red", label="Indoor range"
    )
    ax1.plot(days, daily_t_air, "r-", label="Indoor (daily avg)", linewidth=1.5)
    ax1.axhline(y=21, color="gray", linestyle=":", alpha=0.7, label="Setpoint (21°C)")
    ax1.set_ylabel("Temperature [°C]", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        "Daily Heating Demand and Indoor/Outdoor Temperature", fontsize=14, fontweight="bold"
    )

    # Monthly markers
    month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Heating demand
    ax2.bar(days, daily_heat, color="orange", alpha=0.7, width=1.0)
    ax2.set_xlabel("Day of Year", fontsize=12)
    ax2.set_ylabel("Heating Demand [kWh/day]", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add month labels
    for i, (start, name) in enumerate(zip(month_days[:-1], month_names)):
        mid = start + (month_days[i + 1] - start) / 2
        ax2.text(mid, ax2.get_ylim()[1] * 0.95, name, ha="center", fontsize=9, alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()

    return daily_heat, daily_t_out


def plot_monthly(results: dict, save_path: Path = None):
    """
    Plot monthly heating demand and average temperature.
    """
    # Monthly aggregation
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    monthly_heat = np.zeros(12)
    monthly_t_out = np.zeros(12)
    monthly_t_air = np.zeros(12)

    hour = 0
    for m, days in enumerate(month_days):
        hours_in_month = days * 24
        monthly_heat[m] = (
            results["heating_demand"][hour : hour + hours_in_month].sum() / 1000
        )  # kWh
        monthly_t_out[m] = results["t_outdoor"][hour : hour + hours_in_month].mean()
        monthly_t_air[m] = results["t_air"][hour : hour + hours_in_month].mean()
        hour += hours_in_month

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(12)
    width = 0.6

    # Bar plot for heating
    bars = ax1.bar(x, monthly_heat, width, color="darkorange", alpha=0.8, label="Heating Demand")
    ax1.set_ylabel("Monthly Heating Demand [kWh]", fontsize=12, color="darkorange")
    ax1.tick_params(axis="y", labelcolor="darkorange")
    ax1.set_ylim(0, max(monthly_heat) * 1.15 if max(monthly_heat) > 0 else 10)

    # Add values on bars
    for bar, val in zip(bars, monthly_heat):
        if val > 10:  # Only label significant values
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(monthly_heat) * 0.02,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Secondary axis for temperature
    ax2 = ax1.twinx()
    ax2.plot(x, monthly_t_out, "b-o", linewidth=2, markersize=8, label="Outdoor Temp")
    ax2.plot(x, monthly_t_air, "r-s", linewidth=2, markersize=7, label="Indoor Temp")
    ax2.axhline(y=21, color="red", linestyle=":", alpha=0.5, label="Setpoint")
    ax2.axhline(y=0, color="lightblue", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Temperature [°C]", fontsize=12)
    ax2.tick_params(axis="y")

    # X-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(month_names, fontsize=11)
    ax1.set_xlabel("Month", fontsize=12)

    # Title and legend
    plt.title("Monthly Heating Demand vs Outdoor Temperature", fontsize=14, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()

    return monthly_heat, monthly_t_out


def print_annual_summary(building: Building5R1C, results: dict, geometry: dict):
    """
    Print annual heating demand summary.
    """
    annual_kwh = results["heating_demand"].sum() / 1000
    floor_area = geometry["floor_area_m2"]
    kwh_per_m2 = annual_kwh / floor_area

    # Heating degree days (base 17°C)
    hdd = np.maximum(17 - results["t_outdoor"], 0).sum() / 24

    # Peak demand
    peak_kw = results["heating_demand"].max() / 1000
    peak_hour = np.argmax(results["heating_demand"])
    peak_temp = results["t_outdoor"][peak_hour]

    print(f"""
{"=" * 60}
ANNUAL HEATING DEMAND SUMMARY
{"=" * 60}

  Building Properties:
  ────────────────────
    Floor Area:           {floor_area:.0f} m²
    Heat Loss Coeff:      {building.heat_loss_coefficient:.1f} W/K
    Time Constant:        {building.time_constant_hours:.1f} hours
    Building Class:       {building.building_class.name}

  Weather:
  ────────
    Annual Mean Temp:     {results["t_outdoor"].mean():.1f} °C
    Min Temperature:      {results["t_outdoor"].min():.1f} °C
    Max Temperature:      {results["t_outdoor"].max():.1f} °C
    Heating Degree Days:  {hdd:.0f} (base 17°C)

  Heating Demand:
  ───────────────
    Annual Total:         {annual_kwh:,.0f} kWh
    Specific Demand:      {kwh_per_m2:.1f} kWh/m²/year
    Peak Demand:          {peak_kw:.1f} kW
    Peak Hour:            {peak_hour} (T_out = {peak_temp:.1f}°C)

  Comparison (Swedish norms):
  ───────────────────────────
    BBR 2020 limit:       ~55 kWh/m²/year (climate zone 3)
    Passive house:        ~15 kWh/m²/year
    This building:        {kwh_per_m2:.1f} kWh/m²/year {"✓ Meets BBR" if kwh_per_m2 < 55 else "✗ Exceeds BBR"}

{"=" * 60}
""")

    return {
        "annual_kwh": annual_kwh,
        "kwh_per_m2": kwh_per_m2,
        "peak_kw": peak_kw,
        "hdd": hdd,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run the complete demo."""

    print("""
╔══════════════════════════════════════════════════════════════╗
║     ISO 13790 5R1C Building Thermal Model Demo               ║
║     ─────────────────────────────────────────────            ║
║     Simulating Annual Heating Demand for a Swedish Villa     ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # 1. Define the building
    print("\n[1/5] Defining building geometry and thermal properties...")
    building, geometry = define_building()

    # 2. Visualize the building
    print("\n[2/5] Visualizing building...")
    plot_building(geometry, save_path=output_dir / "building_schematic.png")

    # 3. Load weather data
    print("\n[3/5] Loading weather data...")
    # You can specify a custom EPW path here:
    # epw_path = "/path/to/your/weather.epw"
    epw, t_outdoor = load_weather(preset="goteborg")  # Explicit illustrative weather assumption.

    # 4. Simulate annual heating
    print("\n[4/5] Running annual simulation...")
    results = simulate_annual_heating(building, t_outdoor, include_solar=True)

    # 5. Plot and summarize results
    print("\n[5/5] Generating plots and summary...")

    # Hourly plot (cold week in January)
    plot_hourly(results, days=7, start_day=15, save_path=output_dir / "heating_hourly.png")

    # Daily plot
    daily_heat, daily_t_out = plot_daily(results, save_path=output_dir / "heating_daily.png")

    # Monthly plot
    monthly_heat, monthly_t_out = plot_monthly(
        results, save_path=output_dir / "heating_monthly.png"
    )

    # Print summary
    summary = print_annual_summary(building, results, geometry)

    print(f"\nOutput saved to: {output_dir}")
    print("\nDemo complete! ✓")

    return building, geometry, results, summary


if __name__ == "__main__":
    building, geometry, results, summary = main()
