# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
"""Summarise ``compare_energyplus.py`` results as Markdown tables and a JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = (
    "5r1c_v1",
    "5r1c_v2",
    "iso52016_v1",
    "iso52016_v2",
    "5r1c_v1_bypass",
    "5r1c_v2_bypass",
    "iso52016_v1_bypass",
    "iso52016_v1_air_bypass",
    "iso52016_v2_bypass",
)
LABELS = {
    "5r1c_v1": "5R1C v1",
    "5r1c_v2": "5R1C v2",
    "iso52016_v1": "52016 v1",
    "iso52016_v2": "52016 v2",
    "5r1c_v1_bypass": "5R1C v1 +HR bypass",
    "5r1c_v2_bypass": "5R1C v2 +HR bypass",
    "iso52016_v1_bypass": "52016 v1 +HR bypass",
    "iso52016_v1_air_bypass": "52016 v1 air ctrl +HR bypass",
    "iso52016_v2_bypass": "52016 v2 +HR bypass",
}


def _stats(rows, variant, quantity):
    data = [row[variant][quantity] for row in rows]
    model = np.array([d["annual_kwh"] for d in data])
    reference = np.array([d["reference_kwh"] for d in data])
    ratio = model / np.where(reference > 0, reference, np.nan)
    peaks = np.array(
        [d["peak_kw"] / d["reference_peak_kw"] for d in data if d["reference_peak_kw"] > 0]
    )
    cv = np.array([d["cv_rmse_percent"] for d in data if d["cv_rmse_percent"] is not None])
    corr = np.array([d["correlation"] for d in data if d["correlation"] is not None])
    hours = np.array([d["hours"] / max(d["reference_hours"], 1) for d in data])
    return {
        "total_kwh": float(model.sum()),
        "reference_total_kwh": float(reference.sum()),
        "total_ratio": float(model.sum() / reference.sum()),
        "median_ratio": float(np.nanmedian(ratio)),
        "mean_abs_error_kwh_m2": float(np.mean(np.abs(model - reference)) / 720),
        "within_20_percent": int(np.sum(np.abs(ratio - 1) <= 0.2)),
        "median_peak_ratio": float(np.median(peaks)) if len(peaks) else None,
        "median_cv_rmse_percent": float(np.median(cv)) if len(cv) else None,
        "median_correlation": float(np.median(corr)) if len(corr) else None,
        "median_hours_ratio": float(np.median(hours)),
        "cases": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    rows = data["rows"]
    summary = {
        "cases": len(rows),
        "failures": data["failures"],
        "overall": {v: {q: _stats(rows, v, q) for q in ("cooling", "heating")} for v in VARIANTS},
        "by_function": {},
        "by_state": {},
    }
    for key, field in (
        ("by_function", lambda r: r["archetype"].split("_")[0]),
        ("by_state", lambda r: r["state"]),
    ):
        groups = sorted({field(r) for r in rows})
        summary[key] = {
            g: {v: _stats([r for r in rows if field(r) == g], v, "cooling") for v in VARIANTS}
            for g in groups
        }
    lines = [f"# Cooling and heating vs EnergyPlus ({len(rows)} cases)", ""]
    if all("energyplus_cooling_total_kwh" in r for r in rows):
        sensible = sum(r[VARIANTS[0]]["cooling"]["reference_kwh"] for r in rows)
        total = sum(r["energyplus_cooling_total_kwh"] for r in rows)
        lines += [
            "Cooling is the EnergyPlus sensible coil energy, the boundary of both RC models.",
            f"Latent cooling, {1 - sensible / total:.0%} of the E+ total, is excluded.",
            "",
        ]
    for quantity in ("cooling", "heating"):
        lines += [
            f"## {quantity.title()}",
            "",
            "| Model | Total / E+ | Median case ratio | Cases within ±20 % | "
            "Mean abs error kWh/m² | Median peak ratio | Median hourly CV(RMSE) % | "
            "Median hourly r | Median hours ratio |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for v in VARIANTS:
            s = summary["overall"][v][quantity]
            lines.append(
                f"| {LABELS[v]} | {s['total_ratio']:.2f} | {s['median_ratio']:.2f} | "
                f"{s['within_20_percent']}/{s['cases']} | {s['mean_abs_error_kwh_m2']:.2f} | "
                f"{s['median_peak_ratio']:.2f} | {s['median_cv_rmse_percent']:.0f} | "
                f"{s['median_correlation']:.2f} | {s['median_hours_ratio']:.2f} |"
            )
        lines.append("")
    lines += [
        "## Cooling total / E+ by group",
        "",
        "| Group | E+ kWh/m² | " + " | ".join(LABELS[v] for v in VARIANTS) + " |",
        "|---|---|" + "---|" * len(VARIANTS),
    ]
    for key in ("by_function", "by_state"):
        for group, stats in summary[key].items():
            reference = stats[VARIANTS[0]]["reference_total_kwh"] / (
                720 * stats[VARIANTS[0]]["cases"]
            )
            lines.append(
                f"| {group} | {reference:.1f} | "
                + " | ".join(f"{stats[v]['total_ratio']:.2f}" for v in VARIANTS)
                + " |"
            )
    lines += [
        "",
        "## Per case cooling (kWh/m²)",
        "",
        "| Case | E+ | " + " | ".join(LABELS[v] for v in VARIANTS) + " |",
        "|---|---|" + "---|" * len(VARIANTS),
    ]
    for row in sorted(rows, key=lambda r: r["case"]):
        reference = row[VARIANTS[0]]["cooling"]["reference_kwh"] / 720
        lines.append(
            f"| {row['case']} | {reference:.2f} | "
            + " | ".join(f"{row[v]['cooling']['annual_kwh'] / 720:.2f}" for v in VARIANTS)
            + " |"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
