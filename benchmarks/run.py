# File: benchmarks/run.py
# Purpose: Installed-wheel exact regressions and reproducible annual timings.
# SPDX-License-Identifier: MIT
# License: MIT; see the project-root LICENSE.
# Date created: 2026-10-18
"""Compare six explicit physical fixtures using separately installed RC wheels.

Run from a source checkout with an installed package; no source directories are
added to the import path. Reference and candidate interpreters must use identical
numerical dependency versions. A reference is a real calculation, not a result
substituted into the candidate. Output paths are caller-owned new directories.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta
from importlib import metadata
from importlib.resources import files
from pathlib import Path

import numpy as np
from epsm_core import Building, Weather

from epsm_rc import RCSettings, SolarGainsCalculator, simulate_building


def controlled_weather(path):
    """Write synthetic constant 2023 weather; not observed climate evidence."""
    headers = [
        "LOCATION,Unit test,Test,SWE,SYNTHETIC,000000,57,12,1,20",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,Controlled integration weather",
        "COMMENTS 2,Not measured weather",
        "DATA PERIODS,1,1,Data,Sunday,1/1,12/31",
    ]
    with path.open("x") as stream:
        stream.write("\n".join(headers) + "\n")
        for hour in range(8760):
            stamp = datetime(2023, 1, 1) + timedelta(hours=hour)
            row = ["0"] * 35
            row[:10] = [
                "2023",
                str(stamp.month),
                str(stamp.day),
                str(stamp.hour + 1),
                "60",
                "?",
                "10",
                "5",
                "70",
                "101325",
            ]
            row[12], row[20], row[21] = "300", "180", "2"
            stream.write(",".join(row) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--reference-directory", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warm", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 (first call is reported separately)")
    args.directory.mkdir(parents=True, exist_ok=False)
    reference = args.reference_directory
    if args.reference_python:
        reference = args.directory / "reference"
        subprocess.run(
            [
                str(args.reference_python),
                str(Path(__file__).resolve()),
                "--directory",
                str(reference),
                "--repeats",
                str(args.repeats),
            ],
            check=True,
        )
    cache = None
    if args.warm:
        from epsm_rc import PreparationCache

        cache = PreparationCache()
    synthetic = controlled_weather(args.directory / "controlled.epw")
    settings = RCSettings.from_policy(
        "epsm_rc_5r1c_v1",
        airflow_policy="sensible_only_unlimited_supply",
        self_shadowing_policy="ignore",
    )
    report = {
        "python": sys.version,
        "warm_preparation": args.warm,
        "versions": {
            name: metadata.version(name)
            for name in ("epsm_core", "epsm_rc", "epsm_data", "numpy", "pandas", "pvlib")
        },
        "method": "First annual adapter call separate; median of remaining calls, includes preparation and NPZ output, excludes imports and Building reconstruction.",
        "cases": {},
    }
    for path in sorted((Path(__file__).parent / "fixtures").glob("*.json")):
        case = json.loads(path.read_text())
        weather_path = (
            synthetic
            if case["weather"] == "controlled"
            else files("epsm_rc").joinpath("data/epw/goteborg.epw")
        )
        weather = Weather.from_epw(weather_path, calendar_year=case["year"])
        samples = []
        for repeat in range(args.repeats):
            building = Building.from_dict(case["snapshot"])
            before = building.snapshot().sha256
            kwargs = {"preparation_cache": cache} if cache is not None else {}
            started = time.perf_counter()
            result = simulate_building(
                building,
                weather=weather,
                settings=settings,
                directory=args.directory / path.stem / str(repeat),
                **kwargs,
            )
            samples.append(time.perf_counter() - started)
            assert result.diagnostics["accepted"]
            assert building.snapshot().sha256 == before
        np.savez_compressed(
            args.directory / (path.stem + ".npz"), time_axis=result.time_axis, **result.hourly
        )
        summary = {
            "scalars": result.scalars,
            "reduced_properties": result.diagnostics["reduced_properties"],
            "weather_sha256": weather.sha256,
            "snapshot_sha256": before,
            "implementation": result.diagnostics["implementation"],
            "first_seconds": samples[0],
            "repeat_median_seconds": statistics.median(samples[1:]),
            "samples_seconds": samples,
        }
        if reference:
            expected = np.load(reference / (path.stem + ".npz"))
            actual = np.load(args.directory / (path.stem + ".npz"))
            assert set(actual.files) == set(expected.files)
            for key in actual.files:
                np.testing.assert_array_equal(
                    actual[key], expected[key], err_msg=f"{path.stem}: {key}"
                )
            old = json.loads((reference / (path.stem + ".json")).read_text())
            for key in ("scalars", "reduced_properties", "weather_sha256", "snapshot_sha256"):
                assert summary[key] == old[key], f"{path.stem}: {key} changed"
            summary["exact_reference_match"] = True
        (args.directory / (path.stem + ".json")).write_text(json.dumps(summary, indent=2) + "\n")
        report["cases"][path.stem] = summary
        print(
            path.stem,
            summary["repeat_median_seconds"],
            summary.get("exact_reference_match"),
            flush=True,
        )
    # Both public radiation entry points must retain their own legacy arithmetic,
    # including standalone midpoint timing and direct/diffuse angular modifiers.
    calculator = SolarGainsCalculator.from_epw(
        files("epsm_rc").joinpath("data/epw/goteborg.epw"), ground_reflectance=0.2
    )
    standalone = {}
    for label, use_perez, coefficients in (
        ("perez", True, None),
        ("isotropic", False, None),
        ("angular_perez", True, (1.0, -0.189, 0.644, -0.596)),
    ):
        standalone[label] = calculator.calculate_from_epw(
            {"south": 12.0, "north": 8.0, "east": 6.0, "west": 7.0},
            g_value=0.6,
            frame_factor=1.0,
            use_perez=use_perez,
            g_value_angle_coefficients=coefficients,
        )
    np.savez_compressed(args.directory / "standalone.npz", **standalone)
    if reference:
        expected = np.load(reference / "standalone.npz")
        for key, values in standalone.items():
            np.testing.assert_array_equal(values, expected[key], err_msg=f"standalone: {key}")
        report["standalone_exact_reference_match"] = True
    (args.directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
