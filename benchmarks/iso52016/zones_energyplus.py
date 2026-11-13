# File: zones_energyplus.py
# Purpose: Compare conditioned basements and cold attics in the RC models with EnergyPlus.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-13

"""Compare conditioned basements and cold attics in the RC models with EnergyPlus.

Each case is a three-storey building with a conditioned basement (18 °C heating),
a cold gable-roof attic (0.5 air changes per hour), both, or neither (flat roof,
the reference for what the zones add). Two modes:
conditioned (heating plus 26 °C cooling) and free-floating (no heating or
cooling). EnergyPlus models the basement and the attic as their own zones; the
RC variants are 5R1C (coupled basement domain, quasi-steady attic buffer) and
ISO 52016-1 (one multi-zone network with an unconditioned attic zone).
Cooling is compared on the EnergyPlus sensible coil energy, and the main,
basement and attic air temperatures zone by zone.

Usage::

    python benchmarks/iso52016/zones_energyplus.py --directory outputs/zones \
        --energyplus /path/to/energyplus --workers 4
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from compare_energyplus import YEAR, _calendar, _hourly, _metrics, energyplus_settings
from stress_energyplus import CLIMATES, _temperature_metrics, weather_path

ZONE_CLIMATES = ("goteborg", "kiruna", "chicago", "tampa")
ARCHETYPES = (("mf_se02", "existing"), ("sf_se05", "existing"))
TOPOLOGIES = ("flat", "basement", "attic", "basement_attic")
MODES = ("conditioned", "free")
SENSIBLE = "Zone Ideal Loads Supply Air Sensible Cooling Energy"


def make_building(archetype, state, topology, mode):
    from epsm import Building
    from epsm_core import (
        Footprint,
        GeometrySpec,
        RoofSpec,
        ScheduleSpec,
        UnconditionedExchangeSpec,
        ZoneUseSpec,
    )
    from epsm_data import Catalogue

    building = Building(preset="illustrative-reference")
    catalogue = Catalogue.load()
    states = catalogue.renovation_states(archetype)
    building.apply_archetype(
        catalogue.get(archetype, state=state if states else None),
        calendar=_calendar(archetype.split("_")[0]),
    )
    geometry = dict(storeys=3, storey_height_m=2.8)
    if "attic" in topology:
        geometry["roof"] = RoofSpec(shape="gable", pitch_deg=30)
    if "basement" in topology:
        geometry.update(basement_depth_m=2.5, basement_conditioned=True)
    building.geometry = GeometrySpec.from_footprint(Footprint.rectangle(16, 10), **geometry)
    building.load_reference_area_m2 = None
    if "attic" in topology:
        building.unconditioned_exchange = (
            UnconditionedExchangeSpec(
                zone_kind="attic", outdoor_air_ach=0.5, source="ASSUMED ventilated cold attic"
            ),
        )
    if mode == "conditioned":
        building.operation = building.operation.with_changes(
            cooling_enabled=True, cooling_setpoint=ScheduleSpec.constant(26.0)
        )
    else:
        building.operation = building.operation.with_changes(cooling_enabled=False)
        airflow = building.hvac.airflow.model_copy(
            update={"heating_availability": ScheduleSpec.constant(0.0)}
        )
        building.hvac = building.hvac.with_changes(airflow=airflow)
    if "basement" in topology:
        building.zone_uses = (
            ZoneUseSpec(
                zone_kind="basement",
                use=building.operation.updated(heating_setpoint=ScheduleSpec.constant(18.0)),
                source="ASSUMED 18 °C basement",
            ),
        )
    return building


def rc_variants():
    from epsm_rc import RCSettings

    common = dict(
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="iso13370_periodic_approximation",
        iso13370_wall_thickness_m=0.3,
        iso13370_wall_thickness_source="ASSUMED 0.3 m wall at ground level for this comparison",
        self_shadowing_policy="ignore",
    )
    return {
        "5r1c": RCSettings(
            thermal_model="iso13790_5r1c", cold_attic_policy="quasi_steady_buffer", **common
        ),
        "iso52016": RCSettings(**common),
    }


def energyplus_zones(eplus):
    """Heating, sensible cooling and per-zone air temperatures from EnergyPlus."""
    temperatures = {
        key.split("::", 1)[1]: _hourly(eplus, key)
        for key in eplus.hourly
        if key.startswith("Zone Mean Air Temperature")
    }
    main = [v for name, v in temperatures.items() if "_MAIN_" in name.upper()]
    series = {
        "heating": _hourly(eplus, "space_heating_demand_kwh"),
        "cooling": np.sum(
            [_hourly(eplus, k) for k in eplus.hourly if k.upper().startswith(SENSIBLE.upper())],
            axis=0,
        )
        / 3.6e6,
        "main": np.mean(main, axis=0),
    }
    for kind in ("basement", "attic"):
        found = [v for name, v in temperatures.items() if f"_{kind.upper()}_" in name.upper()]
        if found:
            series[kind] = np.mean(found, axis=0)
    return series


def rc_zones(result):
    hourly = result.hourly
    series = {
        "heating": np.asarray(hourly["space_heating_demand_kwh"], float),
        "cooling": np.asarray(hourly["cooling_demand_kwh"], float),
        "main": np.asarray(
            hourly.get("main_indoor_temperature_c", hourly["indoor_temperature_c"]), float
        ),
    }
    if "basement_indoor_temperature_c" in hourly:
        series["basement"] = np.asarray(hourly["basement_indoor_temperature_c"], float)
    if "attic_air_temperature_c" in hourly:
        series["attic"] = np.asarray(hourly["attic_air_temperature_c"], float)
    return series


def run_case(task):
    climate, (archetype, state), topology, mode, directory, energyplus = task
    from epsm import Weather

    case = f"{climate}-{archetype}-{topology}-{mode}"
    weather = Weather.from_epw(weather_path(climate, energyplus), calendar_year=YEAR)
    building = make_building(archetype, state, topology, mode)
    stored = Path(directory) / "hourly" / f"{case}.npz"
    if stored.exists():
        data = np.load(stored)
        reference = {
            key.removeprefix("energyplus_"): data[key]
            for key in data.files
            if key.startswith("energyplus_")
        }
    else:
        eplus = building.simulate(
            engine="energyplus",
            weather=weather,
            energyplus=energyplus,
            directory=Path(directory) / "energyplus" / case,
            settings=energyplus_settings(),
        ).energy
        reference = energyplus_zones(eplus)
    row = {"case": case, "climate": climate, "archetype": archetype, "topology": topology}
    row["mode"] = mode
    hourly = {"energyplus": reference}
    for label, settings in rc_variants().items():
        result = building.simulate(engine="epsm_rc", weather=weather, settings=settings).energy
        series = rc_zones(result)
        hourly[label] = series
        row[label] = {
            zone: _temperature_metrics(series[zone], reference[zone])
            for zone in ("main", "basement", "attic")
            if zone in series and zone in reference
        }
        if mode == "conditioned":
            row[label]["cooling"] = _metrics(series["cooling"], reference["cooling"])
            row[label]["heating"] = _metrics(series["heating"], reference["heating"])
    np.savez_compressed(stored, **{f"{k}_{n}": v for k, d in hourly.items() for n, v in d.items()})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--energyplus", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    (args.directory / "hourly").mkdir(parents=True, exist_ok=True)
    assert set(ZONE_CLIMATES) <= set(CLIMATES)
    tasks = [
        (c, a, t, m, str(args.directory), args.energyplus)
        for c, a, t, m in itertools.product(ZONE_CLIMATES, ARCHETYPES, TOPOLOGIES, MODES)
    ][: args.limit]
    rows, failures = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_case, task): task for task in tasks}
        for future, task in futures.items():
            try:
                rows.append(future.result())
                print("done", rows[-1]["case"], flush=True)
            except Exception as error:  # keep every other case
                label = "-".join(str(part) for part in task[:4])
                failures.append({"case": label, "error": repr(error)})
                print("FAILED", label, repr(error), flush=True)
    (args.directory / "results.json").write_text(
        json.dumps({"rows": rows, "failures": failures}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
