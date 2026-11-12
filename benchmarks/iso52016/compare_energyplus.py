# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
"""Compare 5R1C, ISO 52016-1 and EnergyPlus over Studio-style building options.

The option grid follows the parameters exposed by EPSM Studio: building function
and construction era (archetype), current fabric state (renovation state) and
footprint orientation. Every case uses the illustrative-reference geometry,
the bundled Gothenburg EPW, the catalogue archetype and an explicitly enabled
26 °C cooling setpoint, so cooling need can be compared. Cooling is compared on
the sensible coil energy, the boundary of both RC models. These are software
comparisons between models, not validation against measured buildings.

Usage::

    python benchmarks/iso52016/compare_energyplus.py --directory outputs/iso52016 \
        --energyplus /path/to/energyplus --workers 4
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date, timedelta
from importlib.resources import files
from pathlib import Path

import numpy as np

ARCHETYPES = (
    "mf_se02",
    "mf_se05",
    "mf_se07",
    "office_se03",
    "office_se06",
    "school_se04",
    "school_se06",
)
# New-build eras carry one catalogue state; it is run once and labelled "existing".
SINGLE_STATE = frozenset({"mf_se07"})
STATES = ("existing", "standard", "advanced")
ORIENTATIONS = (0, 90)
COOLING_SETPOINT_C = 26.0
SOURCE_DAYS = {"office": 225, "school": 220}
YEAR = 2023


def _calendar(category):
    """Synthetic normal-use calendar: source day count spread over the year's weekdays."""
    if category not in SOURCE_DAYS:
        return {"calendar_year": YEAR}
    day, weekdays = date(YEAR, 1, 1), []
    while day.year == YEAR:
        if day.weekday() < 5:
            weekdays.append(day)
        day += timedelta(days=1)
    index = np.round(np.linspace(0, len(weekdays) - 1, SOURCE_DAYS[category])).astype(int)
    return {
        "calendar_year": YEAR,
        "active_dates": [weekdays[i] for i in index],
        "calendar_source": "synthetic evenly spread weekday calendar; not building evidence",
    }


def make_building(archetype, state, orientation):
    from epsm import Building
    from epsm_core.specs import ScheduleSpec
    from epsm_data import Catalogue

    building = Building(preset="illustrative-reference")
    building.geometry = building.geometry.updated(orientation_deg=orientation)
    category = archetype.split("_")[0]
    catalogue = Catalogue.load()
    states = catalogue.renovation_states(archetype)
    if state not in states and states:
        raise ValueError(f"{archetype} has no {state} state")
    building.apply_archetype(
        catalogue.get(archetype, state=state if states else None),
        calendar=_calendar(category),
    )
    building.load_reference_area_m2 = 720  # explicit Atemp, 3 x 20 x 12 m²
    building.operation = building.operation.with_changes(
        cooling_enabled=True, cooling_setpoint=ScheduleSpec.constant(COOLING_SETPOINT_C)
    )
    return building


def rc_variants():
    from epsm_rc import RCSettings

    base = RCSettings.from_policy(
        "epsm_rc_5r1c_v1",
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="iso13370_periodic_approximation",
        iso13370_wall_thickness_m=0.3,
        iso13370_wall_thickness_source="ASSUMED 0.3 m wall at ground level for this comparison",
        self_shadowing_policy="ignore",
    )
    v2 = replace(
        base,
        policy_version="epsm_rc_5r1c_v2",
        solar_time_reference="interval_midpoint",
        window_optics_method="simple_glazing_angular_v1",
    )
    iso = dict(thermal_model="iso52016_1_hourly")
    bypass = dict(heat_recovery_control="load_dependent_bypass")
    return {
        "5r1c_v1": base,
        "5r1c_v2": v2,
        "iso52016_v1": replace(base, **iso),
        "iso52016_v2": replace(v2, **iso),
        "5r1c_v1_bypass": replace(base, **bypass),
        "5r1c_v2_bypass": replace(v2, **bypass),
        "iso52016_v1_bypass": replace(base, **iso, **bypass),
        "iso52016_v1_air_bypass": replace(
            base, iso52016_control_temperature="air", **iso, **bypass
        ),
        "iso52016_v2_bypass": replace(v2, **iso, **bypass),
    }


def _hourly(result, key):
    return np.asarray(result.hourly[key], float)


SENSIBLE_COOLING = "Zone Ideal Loads Supply Air Sensible Cooling Energy"


def energyplus_settings():
    """Request the sensible part of the ideal-loads cooling coil energy."""
    from epsm_core.settings import CompilerSettings, HourlyOutputRequest

    return CompilerSettings(hourly_outputs=(HourlyOutputRequest(name=SENSIBLE_COOLING),))


def energyplus_series(eplus):
    """Hourly E+ series on the RC boundary.

    ``cooling_demand_kwh`` is the total (sensible plus latent) coil energy. Both RC
    models are sensible only, so ``cooling`` is the sensible coil energy and
    ``cooling_total`` keeps the total. In humid climates latent is up to half.
    """
    sensible = [
        _hourly(eplus, key)
        for key in eplus.hourly
        if key.upper().startswith(SENSIBLE_COOLING.upper())
    ]
    if not sensible:
        raise KeyError("EnergyPlus result lacks the sensible cooling output")
    zones = [
        _hourly(eplus, key) for key in eplus.hourly if key.startswith("Zone Mean Air Temperature")
    ]
    return {
        "cooling": np.sum(sensible, axis=0) / 3.6e6,
        "cooling_total": _hourly(eplus, "cooling_demand_kwh"),
        "heating": _hourly(eplus, "space_heating_demand_kwh"),
        "air": np.mean(zones, axis=0),
    }


def _metrics(candidate, reference):
    """Annual, peak and hourly agreement of one hourly series with EnergyPlus."""
    total_c, total_r = candidate.sum(), reference.sum()
    rmse = float(np.sqrt(np.mean((candidate - reference) ** 2)))
    return {
        "annual_kwh": float(total_c),
        "reference_kwh": float(total_r),
        "bias_kwh": float(total_c - total_r),
        "bias_percent": float(100 * (total_c - total_r) / total_r) if total_r > 1e-9 else None,
        "peak_kw": float(candidate.max()),
        "reference_peak_kw": float(reference.max()),
        "hours": int((candidate > 1e-6).sum()),
        "reference_hours": int((reference > 1e-6).sum()),
        "rmse_kw": rmse,
        "cv_rmse_percent": float(100 * rmse / reference.mean())
        if reference.mean() > 1e-9
        else None,
        "correlation": (
            float(np.corrcoef(candidate, reference)[0, 1])
            if candidate.std() > 0 and reference.std() > 0
            else None
        ),
    }


def run_case(task):
    archetype, state, orientation, directory, energyplus = task
    from epsm import Weather

    case = f"{archetype}-{state}-{orientation:03d}"
    weather = Weather.from_epw(files("epsm_rc") / "data/epw/goteborg.epw", calendar_year=YEAR)
    building = make_building(archetype, state, orientation)
    eplus = building.simulate(
        engine="energyplus",
        weather=weather,
        energyplus=energyplus,
        directory=Path(directory) / "energyplus" / case,
        settings=energyplus_settings(),
    ).energy
    hourly = {"energyplus": energyplus_series(eplus)}
    for label, settings in rc_variants().items():
        result = building.simulate(engine="epsm_rc", weather=weather, settings=settings).energy
        hourly[label] = {
            "cooling": _hourly(result, "cooling_demand_kwh"),
            "heating": _hourly(result, "space_heating_demand_kwh"),
            "air": _hourly(result, "indoor_temperature_c"),
        }
    reference = hourly["energyplus"]
    summer = slice(24 * 151, 24 * 243)  # June-August
    row = {
        "case": case,
        "archetype": archetype,
        "state": state,
        "orientation_deg": orientation,
        "energyplus_cooling_unmet_hours": float(eplus.scalars.get("cooling_unmet_zone_hours", 0)),
        "energyplus_heating_unmet_hours": float(eplus.scalars.get("heating_unmet_zone_hours", 0)),
        "energyplus_cooling_total_kwh": float(reference["cooling_total"].sum()),
    }
    for label, series in hourly.items():
        if label == "energyplus":
            continue
        row[label] = {
            "cooling": _metrics(series["cooling"], reference["cooling"]),
            "heating": _metrics(series["heating"], reference["heating"]),
            "summer_air_bias_k": float(np.mean(series["air"][summer] - reference["air"][summer])),
        }
    np.savez_compressed(
        Path(directory) / "hourly" / f"{case}.npz",
        **{
            f"{label}_{name}": values
            for label, data in hourly.items()
            for name, values in data.items()
        },
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--energyplus", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="archetype ids to run")
    args = parser.parse_args()
    (args.directory / "hourly").mkdir(parents=True, exist_ok=True)
    tasks = [
        (a, s, o, str(args.directory), args.energyplus)
        for a, s, o in itertools.product(ARCHETYPES, STATES, ORIENTATIONS)
        if not (a in SINGLE_STATE and s != STATES[0]) and (not args.only or a in args.only)
    ][: args.limit]
    rows, failures = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_case, task): task for task in tasks}
        for future, task in futures.items():
            try:
                rows.append(future.result())
                print("done", rows[-1]["case"], flush=True)
            except Exception as error:  # report every failed option, keep the rest
                failures.append({"case": "-".join(map(str, task[:3])), "error": repr(error)})
                print("FAILED", task[:3], repr(error), flush=True)
    (args.directory / "results.json").write_text(
        json.dumps({"rows": rows, "failures": failures}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
