# File: stress_energyplus.py
# Purpose: Stress the RC thermal response against EnergyPlus across climates and forms.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-12

"""Stress the RC thermal response against EnergyPlus across climates and forms.

Two modes per building: conditioned (heating plus 26 °C cooling) and
free-floating (no heating or cooling), which isolates the thermal response.
Dimensions: climate (Swedish and EnergyPlus-bundled US EPWs), archetype
(heavy multi-family, FTX office, light single-family), form (reference 3 storeys
and 20 % glazing, or 6 storeys and 45 % glazing) and orientation. All RC
variants use v2 solar defaults and the load-dependent heat-recovery bypass.
Cooling is compared on the EnergyPlus sensible coil energy.

Usage::

    python benchmarks/iso52016/stress_energyplus.py --directory outputs/stress \
        --energyplus /path/to/energyplus --workers 4
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import numpy as np
from compare_energyplus import YEAR, _calendar, _metrics, energyplus_series, energyplus_settings

CLIMATES = {
    "goteborg": "rc:goteborg.epw",
    "kiruna": "rc:kiruna.epw",
    "malmo": "rc:malmo.epw",
    "chicago": "eplus:USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
    "golden": "eplus:USA_CO_Golden-NREL.724666_TMY3.epw",
    "tampa": "eplus:USA_FL_Tampa.Intl.AP.722110_TMY3.epw",
}
ARCHETYPES = (("mf_se02", "existing"), ("office_se06", "existing"), ("sf_se05", "existing"))
FORMS = {"reference": (3, 0.2), "tall_glazed": (6, 0.45)}
ORIENTATIONS = (0, 90)
MODES = ("conditioned", "free")
STOREY_HEIGHT_M = 2.8


def weather_path(climate, energyplus):
    source, name = CLIMATES[climate].split(":", 1)
    if source == "rc":
        return files("epsm_rc") / "data/epw" / name
    return Path(energyplus).parent / "WeatherData" / name


def make_building(archetype, state, form, orientation, mode):
    from epsm import Building
    from epsm_core.specs import ScheduleSpec
    from epsm_data import Catalogue

    storeys, wwr = FORMS[form]
    building = Building(preset="illustrative-reference")
    part = building.geometry.parts[0]
    part = part.model_copy(
        update={"storey_elevations_m": tuple(i * STOREY_HEIGHT_M for i in range(storeys + 1))}
    )
    building.geometry = building.geometry.updated(parts=(part,), orientation_deg=orientation)
    building.facade = building.facade.model_copy(update={"default_wwr": wwr})
    catalogue = Catalogue.load()
    states = catalogue.renovation_states(archetype)
    building.apply_archetype(
        catalogue.get(archetype, state=state if states else None),
        calendar=_calendar(archetype.split("_")[0]),
    )
    building.load_reference_area_m2 = 240.0 * storeys
    if mode == "conditioned":
        building.operation = building.operation.with_changes(
            cooling_enabled=True, cooling_setpoint=ScheduleSpec.constant(26.0)
        )
    else:
        # Keep the setpoint (ISO 13370 uses it as the monthly indoor reference) and
        # switch heating availability off, so neither engine conditions the zone.
        building.operation = building.operation.with_changes(cooling_enabled=False)
        airflow = building.hvac.airflow.model_copy(
            update={"heating_availability": ScheduleSpec.constant(0.0)}
        )
        building.hvac = building.hvac.with_changes(airflow=airflow)
    return building


def rc_variants():
    from epsm_rc import RCSettings

    v2 = RCSettings(
        thermal_model="iso13790_5r1c",
        control_temperature="operative",
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="iso13370_periodic_approximation",
        iso13370_wall_thickness_m=0.3,
        iso13370_wall_thickness_source="ASSUMED 0.3 m wall at ground level for this comparison",
        self_shadowing_policy="ignore",
    )
    iso = dict(thermal_model="iso52016_1")
    return {
        "5r1c": v2,
        "iso52016": replace(v2, **iso),
        "iso52016_air": replace(v2, control_temperature="air", **iso),
    }


def _temperature_metrics(model, reference):
    daily_m = model.reshape(-1, 24)
    daily_r = reference.reshape(-1, 24)
    swing_m = (daily_m.max(1) - daily_m.min(1)).mean()
    swing_r = (daily_r.max(1) - daily_r.min(1)).mean()
    return {
        "bias_k": float(np.mean(model - reference)),
        "rmse_k": float(np.sqrt(np.mean((model - reference) ** 2))),
        "summer_bias_k": float(
            np.mean(model[24 * 151 : 24 * 243] - reference[24 * 151 : 24 * 243])
        ),
        "winter_bias_k": float(
            np.mean(
                np.r_[model[: 24 * 59], model[24 * 334 :]]
                - np.r_[reference[: 24 * 59], reference[24 * 334 :]]
            )
        ),
        "daily_swing_ratio": float(swing_m / swing_r) if swing_r > 0 else None,
        "max_k": float(model.max()),
        "reference_max_k": float(reference.max()),
        "hours_above_27": int((model > 27).sum()),
        "reference_hours_above_27": int((reference > 27).sum()),
        "correlation": float(np.corrcoef(model, reference)[0, 1]),
    }


def run_case(task):
    climate, (archetype, state), form, orientation, mode, directory, energyplus, reuse = task
    from epsm import Weather

    case = f"{climate}-{archetype}-{form}-{orientation:03d}-{mode}"
    weather = Weather.from_epw(weather_path(climate, energyplus), calendar_year=YEAR)
    building = make_building(archetype, state, form, orientation, mode)
    area = building.load_reference_area_m2
    previous = Path(reuse) / "hourly" / f"{case}.npz" if reuse else None
    reference = None
    if previous is not None and previous.exists():
        # EnergyPlus inputs depend only on the building and weather, not RC settings.
        # Older arrays stored total cooling; reuse those only where cooling is off.
        names = ("cooling", "cooling_total", "heating", "air")
        try:
            stored = np.load(previous)
            if mode == "free":
                reference = {n: stored[f"energyplus_{n}"] for n in ("cooling", "heating", "air")}
                reference["cooling_total"] = reference["cooling"]
            else:
                reference = {n: stored[f"energyplus_{n}"] for n in names}
        except (OSError, ValueError, KeyError):
            reference = None
    if reference is None:
        eplus = building.simulate(
            engine="energyplus",
            weather=weather,
            energyplus=energyplus,
            directory=Path(directory) / "energyplus" / case,
            settings=energyplus_settings(),
        ).energy
        reference = energyplus_series(eplus)
    row = {
        "case": case,
        "climate": climate,
        "archetype": archetype,
        "form": form,
        "orientation_deg": orientation,
        "mode": mode,
        "area_m2": area,
        "energyplus_cooling_total_kwh": float(reference["cooling_total"].sum()),
    }
    hourly = {"energyplus": reference}
    for label, settings in rc_variants().items():
        result = building.simulate(engine="epsm_rc", weather=weather, settings=settings).energy
        series = {
            "cooling": np.asarray(result.hourly["cooling_demand_kwh"], float),
            "heating": np.asarray(result.hourly["space_heating_demand_kwh"], float),
            "air": np.asarray(result.hourly["indoor_temperature_c"], float),
        }
        hourly[label] = series
        row[label] = {"air": _temperature_metrics(series["air"], reference["air"])}
        if mode == "conditioned":
            row[label]["cooling"] = _metrics(series["cooling"], reference["cooling"])
            row[label]["heating"] = _metrics(series["heating"], reference["heating"])
    np.savez_compressed(
        Path(directory) / "hourly" / f"{case}.npz",
        **{f"{k}_{n}": v for k, d in hourly.items() for n, v in d.items()},
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--energyplus", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--climates", nargs="*", default=None)
    parser.add_argument(
        "--reuse-energyplus", default=None, help="earlier output directory with E+ hourly arrays"
    )
    args = parser.parse_args()
    (args.directory / "hourly").mkdir(parents=True, exist_ok=True)
    tasks = [
        (c, a, f, o, m, str(args.directory), args.energyplus, args.reuse_energyplus)
        for c, a, f, o, m in itertools.product(CLIMATES, ARCHETYPES, FORMS, ORIENTATIONS, MODES)
        if not args.climates or c in args.climates
    ][: args.limit]
    rows, failures = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_case, task): task for task in tasks}
        for future, task in futures.items():
            try:
                rows.append(future.result())
                print("done", rows[-1]["case"], flush=True)
            except Exception as error:  # keep every other case
                label = f"{task[0]}-{task[1][0]}-{task[2]}-{task[3]}-{task[4]}"
                failures.append({"case": label, "error": repr(error)})
                print("FAILED", label, repr(error), flush=True)
    (args.directory / "results.json").write_text(
        json.dumps({"rows": rows, "failures": failures}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
