# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
"""Summarise ``stress_energyplus.py`` results as Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = ("5r1c", "iso52016", "iso52016_air")
LABELS = {"5r1c": "5R1C", "iso52016": "52016 (operative)", "iso52016_air": "52016 (air)"}


def _free_row(rows, variant):
    air = [r[variant]["air"] for r in rows]
    return {
        "rmse_k": float(np.median([a["rmse_k"] for a in air])),
        "worst_rmse_k": float(np.max([a["rmse_k"] for a in air])),
        "bias_k": float(np.median([a["bias_k"] for a in air])),
        "summer_bias_k": float(np.median([a["summer_bias_k"] for a in air])),
        "winter_bias_k": float(np.median([a["winter_bias_k"] for a in air])),
        "swing_ratio": float(np.median([a["daily_swing_ratio"] for a in air])),
        "correlation": float(np.median([a["correlation"] for a in air])),
        "max_error_k": float(np.median([a["max_k"] - a["reference_max_k"] for a in air])),
        "hours_above_27_ratio": float(
            sum(a["hours_above_27"] for a in air)
            / max(sum(a["reference_hours_above_27"] for a in air), 1)
        ),
    }


def _load_row(rows, variant, quantity):
    data = [r[variant][quantity] for r in rows]
    model = np.array([d["annual_kwh"] for d in data])
    reference = np.array([d["reference_kwh"] for d in data])
    area = np.array([r["area_m2"] for r in rows])
    significant = reference / area > 1.0  # ratios on near-zero needs are not informative
    ratio = model[significant] / reference[significant]
    return {
        "total_ratio": float(model.sum() / reference.sum()) if reference.sum() else None,
        "median_ratio": float(np.median(ratio)) if len(ratio) else None,
        "within_20": f"{int(np.sum(np.abs(ratio - 1) <= 0.2))}/{len(ratio)}",
        "mae_kwh_m2": float(np.mean(np.abs(model - reference) / area)),
        "peak_ratio": float(
            np.median(
                [d["peak_kw"] / d["reference_peak_kw"] for d, s in zip(data, significant) if s]
            )
        )
        if significant.any()
        else None,
        "correlation": float(
            np.median(
                [d["correlation"] for d, s in zip(data, significant) if s and d["correlation"]]
            )
        )
        if significant.any()
        else None,
    }


def _fmt(value, digits=2):
    return (
        "–"
        if value is None
        else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    rows = data["rows"]
    free = [r for r in rows if r["mode"] == "free"]
    conditioned = [r for r in rows if r["mode"] == "conditioned"]
    lines = [f"# Stress comparison with EnergyPlus ({len(rows)} cases)", ""]
    if data["failures"]:
        lines += [f"Failures: {len(data['failures'])}", ""]

    def free_table(title, subset):
        nonlocal lines
        lines += [
            f"### {title} ({len(subset)} cases)",
            "",
            "| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | "
            "Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for v in VARIANTS:
            s = _free_row(subset, v)
            lines.append(
                f"| {LABELS[v]} | {s['rmse_k']:.2f} | {s['worst_rmse_k']:.2f} | {s['bias_k']:+.2f} | "
                f"{s['summer_bias_k']:+.2f} | {s['winter_bias_k']:+.2f} | {s['swing_ratio']:.2f} | "
                f"{s['correlation']:.3f} | {s['max_error_k']:+.2f} | {s['hours_above_27_ratio']:.2f} |"
            )
        lines.append("")

    lines += ["## Free-floating air temperature", ""]
    free_table("All", free)
    for key in ("climate", "archetype", "form"):
        for group in sorted({r[key] for r in free}):
            free_table(f"{key} = {group}", [r for r in free if r[key] == group])

    def load_table(title, subset, quantity):
        nonlocal lines
        lines += [
            f"### {title}: {quantity} ({len(subset)} cases)",
            "",
            "| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |",
            "|---|---|---|---|---|---|---|",
        ]
        for v in VARIANTS:
            s = _load_row(subset, v, quantity)
            lines.append(
                f"| {LABELS[v]} | {_fmt(s['total_ratio'])} | {_fmt(s['median_ratio'])} | "
                f"{s['within_20']} | {s['mae_kwh_m2']:.2f} | {_fmt(s['peak_ratio'])} | "
                f"{_fmt(s['correlation'])} |"
            )
        lines.append("")

    lines += [
        "## Conditioned loads",
        "",
        "\\* over cases where E+ needs more than 1 kWh/m² of that load.",
        "",
    ]
    for quantity in ("cooling", "heating"):
        load_table("All", conditioned, quantity)
        for group in sorted({r["climate"] for r in conditioned}):
            load_table(
                f"climate = {group}", [r for r in conditioned if r["climate"] == group], quantity
            )
        for group in sorted({r["archetype"] for r in conditioned}):
            load_table(
                f"archetype = {group}",
                [r for r in conditioned if r["archetype"] == group],
                quantity,
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stress_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
