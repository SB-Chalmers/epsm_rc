# File: benchmark_speed.py
# Purpose: Time 5R1C and ISO 52016-1 annual runs with the Python and Rust recurrences.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-12

"""Time 5R1C and ISO 52016-1 annual runs with the Python and Rust recurrences.

``solve`` is the time inside ``_run_domain`` (hourly recurrence plus result
assembly for the domain); ``total`` is the whole ``simulate_building`` call.

Usage::

    python benchmarks/iso52016/benchmark_speed.py --repeats 5
"""

from __future__ import annotations

import argparse
import contextlib
import time

import numpy as np
from compare_energyplus import YEAR
from stress_energyplus import make_building, rc_variants, weather_path

CASES = (
    ("mf_se02", "reference"),
    ("office_se06", "reference"),
    ("sf_se05", "tall_glazed"),
    ("office_se06", "tall_glazed"),
)


@contextlib.contextmanager
def patched(model, native):
    from epsm_rc import _iso52016_platform as adapter
    from epsm_rc import iso13790, platform

    saved = (platform._run_domain, adapter.run_zone, iso13790._can_use_native)
    timer = {"solve": 0.0}

    def run_domain(*args, **kwargs):
        start = time.perf_counter()
        try:
            return saved[0](*args, **kwargs)
        finally:
            timer["solve"] += time.perf_counter() - start

    platform._run_domain = run_domain
    adapter.run_zone = lambda *a, **k: saved[1](*a, native=native, **k)
    if model == "5r1c" and not native:
        iso13790._can_use_native = lambda *a, **k: False
    try:
        yield timer
    finally:
        platform._run_domain, adapter.run_zone, iso13790._can_use_native = saved


def main():
    from epsm import Weather

    from epsm_rc import simulate_building

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    weather = Weather.from_epw(weather_path("goteborg", ""), calendar_year=YEAR)
    variants = rc_variants()
    print("| Building | Model | Recurrence | Solve s | Total s |")
    print("|---|---|---|---|---|")
    for archetype, form in CASES:
        building = make_building(archetype, "existing", form, 0, "conditioned")
        frozen = building.snapshot().to_building()
        for model, native in (
            ("5r1c", False),
            ("5r1c", True),
            ("iso52016", False),
            ("iso52016", True),
        ):
            solve, total = [], []
            for _ in range(args.repeats + 1):  # first run warms caches and imports
                with patched(model, native) as timer:
                    start = time.perf_counter()
                    simulate_building(frozen, weather=weather, settings=variants[model])
                    total.append(time.perf_counter() - start)
                    solve.append(timer["solve"])
            print(
                f"| {archetype} {form} | {model} | {'Rust' if native else 'Python'} | "
                f"{np.median(solve[1:]):.3f} | {np.median(total[1:]):.3f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
