# File: overhang_energyplus.py
# Purpose: Compare the RC overhang response with EnergyPlus across climates and forms.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-13

"""Compare the RC overhang response with EnergyPlus across climates and forms.

Uses the stress-test buildings in three Swedish climates (``stress_energyplus.make_building``, conditioned)
with a per-window overhang of 0, 0.5 or 1.0 m. EnergyPlus models the overhang as
a Core shading surface; RC uses ``overhang_shading_method="iso13790_hourly_overhang"``.
The comparison is on the change each overhang makes to annual heating, sensible
cooling and window solar, relative to the same building without one.
``--summary`` writes ``results/overhang_summary.md``.

Usage::

    python benchmarks/iso52016/overhang_energyplus.py --directory outputs/overhang \
        --energyplus /path/to/energyplus --workers 4
    python benchmarks/iso52016/overhang_energyplus.py --directory outputs/overhang --summary \
        benchmarks/iso52016/results/overhang_summary.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from compare_energyplus import YEAR, energyplus_series, energyplus_settings
from stress_energyplus import ARCHETYPES, FORMS, make_building, rc_variants, weather_path

CLIMATES = ("goteborg", "malmo", "kiruna")  # bundled Swedish EPWs
DEPTHS_M = (0.0, 0.5, 1.0)


def run_case(task):
    climate, (archetype, state), form, depth, directory, energyplus = task
    from epsm import Weather
    from epsm_core.specs.facade import OverhangSpec

    case = f"{climate}-{archetype}-{form}-overhang{depth:.1f}"
    weather = Weather.from_epw(weather_path(climate, energyplus), calendar_year=YEAR)
    building = make_building(archetype, state, form, 0, "conditioned")
    building.facade = building.facade.updated(overhang=OverhangSpec(depth_m=depth))
    area = building.load_reference_area_m2
    eplus = building.simulate(
        engine="energyplus",
        weather=weather,
        energyplus=energyplus,
        directory=Path(directory) / "energyplus" / case,
        settings=energyplus_settings(),
    ).energy
    reference = energyplus_series(eplus)
    # epsm.Building.simulate asks Core's readiness first, which still lists facade
    # shading as unsupported for RC; call RC directly so its own gate applies.
    from epsm_rc import simulate_building

    result = simulate_building(building, weather=weather, settings=rc_variants()["iso52016_air"])
    return {
        "case": case,
        "climate": climate,
        "archetype": archetype,
        "form": form,
        "depth_m": depth,
        "energyplus": {
            "heating_kwh_m2": float(reference["heating"].sum() / area),
            "cooling_kwh_m2": float(reference["cooling"].sum() / area),
        },
        "rc": {
            "heating_kwh_m2": float(np.sum(result.hourly["space_heating_demand_kwh"]) / area),
            "cooling_kwh_m2": float(np.sum(result.hourly["cooling_demand_kwh"]) / area),
            "window_solar_kwh_m2": float(
                np.sum(result.hourly["window_solar_heat_gain_w"]) / 1000 / area
            ),
            "overhang_angles_deg": result.diagnostics["overhang_shading"]["overhang_angles_deg"],
        },
    }


def summarise(rows):
    """Markdown table of the overhang effect, RC against EnergyPlus."""
    base = {(r["climate"], r["archetype"], r["form"]): r for r in rows if r["depth_m"] == 0}
    lines = [
        "# Overhang response against EnergyPlus",
        "",
        f"{len(rows)} conditioned cases: {len({r['climate'] for r in rows})} Swedish climates, "
        f"{len(ARCHETYPES)} archetypes, "
        f"{len(FORMS)} forms, overhang depth {', '.join(f'{d:g}' for d in DEPTHS_M)} m. "
        "Changes are relative to the same building without an overhang, in kWh/m².",
        "",
        "| Depth | Quantity | E+ change, median | RC change, median | Same sign | RC / E+ change |",
        "|---|---|---|---|---|---|",
    ]
    for depth in DEPTHS_M[1:]:
        for quantity in ("heating", "cooling"):
            pairs = []
            for r in rows:
                if r["depth_m"] != depth:
                    continue
                b = base[(r["climate"], r["archetype"], r["form"])]
                key = f"{quantity}_kwh_m2"
                pairs.append(
                    (
                        r["energyplus"][key] - b["energyplus"][key],
                        r["rc"][key] - b["rc"][key],
                    )
                )
            meaningful = [(e, c) for e, c in pairs if abs(e) > 0.1]
            same = sum(e * c > 0 for e, c in meaningful)
            ratio = statistics.median(c / e for e, c in meaningful) if meaningful else float("nan")
            lines.append(
                f"| {depth:g} m | {quantity} | {statistics.median(e for e, _ in pairs):+.2f} | "
                f"{statistics.median(c for _, c in pairs):+.2f} | {same}/{len(meaningful)} | "
                f"{ratio:.2f} |"
            )
    lines += [
        "",
        "Same sign and the ratio count cases where EnergyPlus changes by more than 0.1 kWh/m².",
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
        (c, a, f, d, str(args.directory), args.energyplus)
        for c, a, f, d in itertools.product(CLIMATES, ARCHETYPES, FORMS, DEPTHS_M)
    ]
    rows, failures = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_case, task): task for task in tasks}
        for future, task in futures.items():
            try:
                rows.append(future.result())
                print("done", rows[-1]["case"], flush=True)
            except Exception as error:  # keep every other case
                label = f"{task[0]}-{task[1][0]}-{task[2]}-{task[3]}"
                failures.append({"case": label, "error": repr(error)})
                print("FAILED", label, repr(error), flush=True)
    (args.directory / "results.json").write_text(
        json.dumps({"rows": rows, "failures": failures}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
