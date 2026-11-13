# File: self_shading_energyplus.py
# Purpose: Compare RC self-shading of concave footprints with EnergyPlus.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-13

"""Compare RC self-shading of concave footprints with EnergyPlus.

Rectangle (control), L, U and courtyard (opened by a 2 m slot) footprints of about the same floor area
in four climates, two forms and two orientations, conditioned. EnergyPlus
(``FullExterior`` polygon clipping) shades every facade by the building's own
surfaces. RC runs each ``self_shadowing_policy``: ``ignore``,
``iso13790_hourly_horizon`` and ``hourly_sunlit_fraction``. The comparison is on
annual window incident beam and sky diffuse radiation (the geometry alone),
window transmitted solar, heating and sensible cooling. ``--summary`` writes
``results/self_shading_summary.md``.

Usage::

    python benchmarks/iso52016/self_shading_energyplus.py --directory outputs/self_shading \
        --energyplus /path/to/energyplus --workers 4
    python benchmarks/iso52016/self_shading_energyplus.py --directory outputs/self_shading \
        --summary benchmarks/iso52016/results/self_shading_summary.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from compare_energyplus import SENSIBLE_COOLING, YEAR, energyplus_series
from stress_energyplus import make_building, rc_variants, weather_path

CLIMATES = ("goteborg", "kiruna", "malmo", "chicago")
ARCHETYPES = (("mf_se02", "existing"), ("office_se06", "existing"))
FORMS = ("reference", "tall_glazed")
ORIENTATIONS = (0, 30)
POLICIES = ("ignore", "iso13790_hourly_horizon", "hourly_sunlit_fraction")
SHAPES = {
    "rectangle": {"exterior": ((0, 0), (20, 0), (20, 12), (0, 12))},
    "l_shape": {"exterior": ((0, 0), (20, 0), (20, 8), (8, 8), (8, 20), (0, 20))},
    "u_shape": {
        "exterior": ((0, 0), (28, 0), (28, 20), (20, 20), (20, 8), (8, 8), (8, 20), (0, 20))
    },
    # EnergyPlus Kiva floors cannot hold holes, so the courtyard opens through
    # a 2 m slot in its north wing; it shades as a 14 m x 14 m courtyard.
    "courtyard": {
        "exterior": (
            (0, 0),
            (30, 0),
            (30, 30),
            (16, 30),
            (16, 22),
            (22, 22),
            (22, 8),
            (8, 8),
            (8, 22),
            (14, 22),
            (14, 30),
            (0, 30),
        )
    },
}
BEAM = "Surface Outside Face Incident Beam Solar Radiation Rate per Area"
SKY = "Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area"
TRANSMITTED = "Enclosure Windows Total Transmitted Solar Radiation Energy"


def shaped_building(archetype, state, form, orientation, shape):
    """Stress-test building with one of :data:`SHAPES` as its footprint."""
    from epsm_core.specs.geometry import Footprint

    building = make_building(archetype, state, form, orientation, "conditioned")
    part = building.geometry.parts[0]
    footprint = Footprint(**SHAPES[shape])
    part = part.model_copy(update={"footprint": footprint})
    building.geometry = building.geometry.updated(parts=(part,))
    building.load_reference_area_m2 = footprint.area_m2 * (len(part.storey_elevations_m) - 1)
    return building


def window_areas(building, weather, settings):
    """EnergyPlus window name to area, from the same Core resolution RC uses."""
    from epsm_rc import platform

    preparation = platform._prepare_building(building, weather, platform._settings(settings), None)
    return {pane["name"].upper(): pane["area_m2"] for pane in preparation.resolved.openings}


def run_case(task):
    climate, (archetype, state), form, orientation, shape, directory, energyplus = task
    from epsm import Weather
    from epsm_core.settings import CompilerSettings, HourlyOutputRequest

    from epsm_rc import simulate_building

    case = f"{climate}-{archetype}-{form}-o{orientation}-{shape}"
    weather = Weather.from_epw(weather_path(climate, energyplus), calendar_year=YEAR)
    building = shaped_building(archetype, state, form, orientation, shape)
    area = building.load_reference_area_m2
    outputs = tuple(HourlyOutputRequest(name=name) for name in (SENSIBLE_COOLING, BEAM, SKY))
    eplus = building.simulate(
        engine="energyplus",
        weather=weather,
        energyplus=energyplus,
        directory=Path(directory) / "energyplus" / case,
        settings=CompilerSettings(hourly_outputs=outputs),
    ).energy
    reference = energyplus_series(eplus)
    base = rc_variants()["iso52016_air"]
    panes = window_areas(building, weather, base)

    def incident(name):
        # EnergyPlus 24.2 reports a window's outside-face incident rate per area
        # at exactly half its host wall's value in every hour. On the rectangle
        # (no self-shading) twice the window value matches RC's unshaded plane
        # irradiance within 0.5 %, and transmitted solar agrees, so the factor
        # is a reporting convention, not a physics difference.
        total = 0.0
        for key, values in eplus.hourly.items():
            variable, _, surface = key.partition("::")
            if variable == name and surface in panes:
                total += float(np.sum(values)) * panes[surface]
        return 2.0 * total / 1000 / area

    transmitted = sum(
        float(np.sum(values)) for key, values in eplus.hourly.items() if key.startswith(TRANSMITTED)
    )
    row = {
        "case": case,
        "climate": climate,
        "archetype": archetype,
        "form": form,
        "orientation": orientation,
        "shape": shape,
        "energyplus": {
            "heating_kwh_m2": float(reference["heating"].sum() / area),
            "cooling_kwh_m2": float(reference["cooling"].sum() / area),
            "window_beam_kwh_m2": incident(BEAM),
            "window_sky_kwh_m2": incident(SKY),
            "window_transmitted_kwh_m2": transmitted / 3.6e6 / area,
        },
        "rc": {},
    }
    for policy in POLICIES:
        result = simulate_building(
            building, weather=weather, settings=replace(base, self_shadowing_policy=policy)
        )
        hourly = result.hourly

        def total(key, hourly=hourly):
            return float(np.sum(hourly[key]) / 1000 / area)

        row["rc"][policy] = {
            "heating_kwh_m2": float(np.sum(hourly["space_heating_demand_kwh"]) / area),
            "cooling_kwh_m2": float(np.sum(hourly["cooling_demand_kwh"]) / area),
            "window_beam_kwh_m2": total("window_incident_beam_w"),
            "window_sky_kwh_m2": total("window_incident_sky_diffuse_w"),
            "window_transmitted_kwh_m2": total("window_transmitted_solar_w"),
        }
    return row


def _ratio(rows, policy, key, shapes):
    values = [
        r["rc"][policy][key] / r["energyplus"][key]
        for r in rows
        if r["shape"] in shapes and r["energyplus"][key] > 0.05
    ]
    return statistics.median(values) if values else float("nan")


def summarise(rows):
    """Markdown tables of RC against EnergyPlus by shape and policy."""
    lines = [
        "# Self-shading against EnergyPlus",
        "",
        f"{len(rows)} conditioned cases: {len(CLIMATES)} climates, {len(ARCHETYPES)} archetypes, "
        f"{len(FORMS)} forms (3 storeys WWR 0.2, 6 storeys WWR 0.45), orientations "
        f"{', '.join(map(str, ORIENTATIONS))} degrees. Values are medians of RC / EnergyPlus "
        "annual totals; EnergyPlus uses FullExterior polygon clipping.",
        "",
    ]
    quantities = (
        ("window_beam_kwh_m2", "Window incident beam"),
        ("window_sky_kwh_m2", "Window incident sky diffuse"),
        ("window_transmitted_kwh_m2", "Window transmitted solar"),
        ("heating_kwh_m2", "Heating"),
        ("cooling_kwh_m2", "Sensible cooling"),
    )
    for key, label in quantities:
        lines += [f"## {label}", "", "| Shape | " + " | ".join(POLICIES) + " |"]
        lines.append("|---" * (len(POLICIES) + 1) + "|")
        for shape in SHAPES:
            cells = [f"{_ratio(rows, p, key, {shape}):.3f}" for p in POLICIES]
            lines.append(f"| {shape} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "Ratios exclude cases where the EnergyPlus total is below 0.05 kWh/m².",
        "These are model-to-model comparisons, not validation.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--energyplus")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summary", type=Path, default=None, help="write the table and stop")
    args = parser.parse_args()
    if args.summary:
        rows = json.loads((args.directory / "results.json").read_text())["rows"]
        args.summary.write_text(summarise(rows))
        return
    if not args.energyplus:
        parser.error("--energyplus is required to run the comparison")
    args.directory.mkdir(parents=True, exist_ok=True)
    tasks = [
        (c, a, f, o, s, str(args.directory), args.energyplus)
        for c, a, f, o, s in itertools.product(CLIMATES, ARCHETYPES, FORMS, ORIENTATIONS, SHAPES)
    ]
    rows, failures = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_case, task): task for task in tasks}
        for future, task in futures.items():
            try:
                rows.append(future.result())
                print("done", rows[-1]["case"], flush=True)
            except Exception as error:  # keep every other case
                label = "-".join(str(part) for part in task[:5])
                failures.append({"case": label, "error": repr(error)})
                print("FAILED", label, repr(error), flush=True)
    (args.directory / "results.json").write_text(
        json.dumps({"rows": rows, "failures": failures}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
