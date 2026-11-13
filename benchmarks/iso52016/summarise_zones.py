# File: summarise_zones.py
# Purpose: Summarise the basement and attic EnergyPlus comparison as Markdown tables.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-13

"""Summarise ``zones_energyplus.py`` results as Markdown tables.

Energy is summed over cases before taking the RC / EnergyPlus ratio, so small
cooling loads do not dominate. Temperature errors are hourly bias and RMSE per
zone, averaged over cases.

Usage::

    python benchmarks/iso52016/summarise_zones.py outputs/zones/results.json > zones_summary.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

LABELS = {"5r1c": "5R1C", "iso52016": "ISO 52016-1"}
TOPOLOGIES = {
    "flat": "Flat roof, no basement",
    "basement": "Conditioned basement",
    "attic": "Cold attic",
    "basement_attic": "Basement and attic",
}


def main(path):
    data = json.loads(open(path).read())
    rows = data["rows"]
    print("# Conditioned basements and cold attics against EnergyPlus\n")
    print(
        f"{len(rows)} cases: {len({r['climate'] for r in rows})} climates, "
        f"{len({r['archetype'] for r in rows})} archetypes, "
        f"{len({r['topology'] for r in rows})} topologies, "
        "conditioned and free-floating. Failures: "
        f"{len(data['failures'])}.\n"
    )
    print("## Energy (conditioned), RC / EnergyPlus summed over cases\n")
    print("| Topology | Model | Heating | Sensible cooling | Heating hourly CV(RMSE) |")
    print("|---|---|---|---|---|")
    for topology, title in TOPOLOGIES.items():
        selected = [r for r in rows if r["topology"] == topology and r["mode"] == "conditioned"]
        for label, name in LABELS.items():
            heat = sum(r[label]["heating"]["annual_kwh"] for r in selected)
            heat_ref = sum(r[label]["heating"]["reference_kwh"] for r in selected)
            cool = sum(r[label]["cooling"]["annual_kwh"] for r in selected)
            cool_ref = sum(r[label]["cooling"]["reference_kwh"] for r in selected)
            cv = np.mean([r[label]["heating"]["cv_rmse_percent"] for r in selected])
            print(
                f"| {title} | {name} | {heat / heat_ref:.3f} | {cool / cool_ref:.2f} | {cv:.1f} % |"
            )
    for mode in ("conditioned", "free"):
        print(f"\n## Zone air temperature ({mode}), bias / RMSE in K, mean over cases\n")
        print("| Topology | Model | Main | Basement | Attic |")
        print("|---|---|---|---|---|")
        for topology, title in TOPOLOGIES.items():
            selected = [r for r in rows if r["topology"] == topology and r["mode"] == mode]
            for label, name in LABELS.items():
                cells = []
                for zone in ("main", "basement", "attic"):
                    values = defaultdict(list)
                    for row in selected:
                        metrics = row[label].get(zone)
                        if metrics:
                            values["bias"].append(metrics["bias_k"])
                            values["rmse"].append(metrics["rmse_k"])
                    cells.append(
                        f"{np.mean(values['bias']):+.2f} / {np.mean(values['rmse']):.2f}"
                        if values
                        else "–"
                    )
                print(f"| {title} | {name} | " + " | ".join(cells) + " |")
    print("\n## Per climate (conditioned, ISO 52016-1), RC / EnergyPlus\n")
    print("| Climate | Heating | Sensible cooling | Attic RMSE (K) | Basement RMSE (K) |")
    print("|---|---|---|---|---|")
    for climate in dict.fromkeys(r["climate"] for r in rows):
        selected = [r for r in rows if r["climate"] == climate and r["mode"] == "conditioned"]
        heat = sum(r["iso52016"]["heating"]["annual_kwh"] for r in selected)
        heat_ref = sum(r["iso52016"]["heating"]["reference_kwh"] for r in selected)
        cool = sum(r["iso52016"]["cooling"]["annual_kwh"] for r in selected)
        cool_ref = sum(r["iso52016"]["cooling"]["reference_kwh"] for r in selected)
        attic = [r["iso52016"]["attic"]["rmse_k"] for r in selected if "attic" in r["iso52016"]]
        base = [
            r["iso52016"]["basement"]["rmse_k"] for r in selected if "basement" in r["iso52016"]
        ]
        ratio = f"{cool / cool_ref:.2f}" if cool_ref > 0 else "–"
        print(
            f"| {climate} | {heat / heat_ref:.3f} | {ratio} | "
            f"{np.mean(attic):.2f} | {np.mean(base):.2f} |"
        )


if __name__ == "__main__":
    main(sys.argv[1])
