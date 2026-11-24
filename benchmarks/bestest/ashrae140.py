# File: ashrae140.py
# Purpose: Run the ASHRAE Standard 140 Section 5.2 BESTEST case family on both RC models.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-24

"""Run the ASHRAE Standard 140 Section 5.2 BESTEST case family on both RC models.

Each case is a variation of Case 600 (``case600.py``) described by a
:class:`Case`: lightweight or heavyweight fabric, south or east/west glazing,
a south overhang, the thermostat schedule, night ventilation and free-float.
Every case runs through ``iso13790_5r1c`` and ``iso52016_1`` and, with an
executable, EnergyPlus 24.2 on the same Core Building.

Notes
-----
Cases 630 and 930 (side fins) and 960 (sunspace) are not run: RC rejects fins
and has no second conditioned-by-adjacency zone of that kind. Case 650 night
ventilation is a mechanical supply of 1703.16 m³/h; RC applies the site air
heat capacity to it, while EnergyPlus ideal-loads outdoor air uses standard
density.

Examples
--------
::

    python benchmarks/bestest/ashrae140.py --epw USA_CO_Denver.Intl.AP.725650_TMY3.epw \
        --energyplus /path/to/energyplus --output outputs/ashrae140
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import case600 as base  # noqa: E402

HEAVY_WALL = (
    ("wood siding", 0.14, 530.0, 900.0, 0.009),
    ("foam insulation", 0.04, 10.0, 1400.0, 0.0615),
    ("concrete block", 0.51, 1400.0, 1000.0, 0.100),
)
HEAVY_FLOOR = (
    ("floor insulation", 0.04, 1.0, 100.0, 1.007),
    ("concrete slab", 1.13, 1400.0, 1000.0, 0.080),
)
NIGHT_VENT_M3_H = 1703.16
HOURS = range(24)


@dataclass(frozen=True)
class Case:
    """One Section 5.2 case as a variation of Case 600.

    Attributes
    ----------
    name : str
        Standard 140 case identifier.
    heavy : bool
        Heavyweight (900-series) walls and floor.
    windows : str
        ``"south"`` (two 3 m x 2 m) or ``"east_west"`` (one 3 m x 2 m each).
    overhang_m : float
        Depth of a continuous south overhang at roof level.
    thermostat : str
        ``"constant"`` 20/27 °C, ``"setback"`` 10 °C heating 23-07 h,
        ``"night_vent"`` cooling only 07-18 h, or ``"free"``.
    night_vent : bool
        1703.16 m³/h outdoor air 18-07 h.
    """

    name: str
    heavy: bool = False
    windows: str = "south"
    overhang_m: float = 0.0
    thermostat: str = "constant"
    night_vent: bool = False

    @property
    def free_float(self):
        """Whether the case has no heating or cooling."""
        return self.thermostat == "free"


def _cases():
    rows = []
    for series, heavy in (("6", False), ("9", True)):
        rows += [
            Case(f"{series}00", heavy),
            Case(f"{series}10", heavy, overhang_m=1.0),
            Case(f"{series}20", heavy, windows="east_west"),
            Case(f"{series}40", heavy, thermostat="setback"),
            Case(f"{series}50", heavy, thermostat="night_vent", night_vent=True),
            Case(f"{series}00FF", heavy, thermostat="free"),
            Case(f"{series}50FF", heavy, thermostat="free", night_vent=True),
        ]
    return {case.name: case for case in rows}


CASES = _cases()


def _hourly(on):
    return tuple(1.0 if on(h) else 0.0 for h in HOURS)


def make_building(case):
    """Build one Section 5.2 case as an EPSM Core Building.

    Parameters
    ----------
    case : Case
        Case variation.

    Returns
    -------
    epsm_core.Building
        The Case 600 Building with the case's fabric, glazing and controls.
    """
    from epsm_core import HVACSystem, ScheduleSpec
    from epsm_core.specs.facade import FacadeOverride, OverhangSpec, WindowLayoutSpec
    from epsm_core.specs.physics import AirflowSpec
    from epsm_data import construction_set_from_dict, construction_set_to_dict

    building = base.make_building()
    building.identifier = f"bestest-{case.name.lower()}"
    if case.heavy:
        data = construction_set_to_dict(building.construction_set)
        elements = data["elements"]
        elements["EXT_WALL"] = base._construction(
            elements["EXT_WALL"], HEAVY_WALL, outer_absorptance=base.EXTERIOR_SOLAR_ABSORPTANCE
        )
        elements["GROUND_FLOOR"] = base._construction(
            elements["GROUND_FLOOR"], HEAVY_FLOOR, outer_absorptance=0.0
        )
        building.construction_set = construction_set_from_dict(data)

    window = dict(mode="dimensions", window_width_m=3.0, window_height_m=2.0, sill_height_m=0.2)
    if case.windows == "south":
        overhang = OverhangSpec(depth_m=case.overhang_m, continuous=True)
        overrides = (
            FacadeOverride(
                facade_id=base.SOUTH_FACADE,
                wwr=12.0 / (base.FLOOR_LENGTH_M * base.HEIGHT_M),
                layout=WindowLayoutSpec(**window, center_spacing_m=4.0),
                overhang=overhang if case.overhang_m else None,
            ),
        )
    else:
        wwr = 6.0 / (base.FLOOR_WIDTH_M * base.HEIGHT_M)
        overrides = tuple(
            FacadeOverride(
                facade_id=facade, wwr=wwr, layout=WindowLayoutSpec(**window, center_spacing_m=6.0)
            )
            for facade in ("main.r0.e1", "main.r0.e3")
        )
    building.facade = building.facade.updated(overrides=overrides)

    airflow = building.hvac.airflow
    changes = {}
    if case.night_vent:
        area = base.FLOOR_LENGTH_M * base.FLOOR_WIDTH_M
        changes.update(
            topology="F",
            mechanical_vent_l_s_m2=NIGHT_VENT_M3_H / 3.6 / area,
            mechanical_vent_fraction=ScheduleSpec(values=_hourly(lambda h: h >= 18 or h < 7)),
        )
    operation = {}
    if case.thermostat == "setback":
        operation["heating_setpoint"] = ScheduleSpec(
            values=tuple(10.0 if (h >= 23 or h < 7) else 20.0 for h in HOURS)
        )
    elif case.thermostat == "night_vent":
        changes["heating_availability"] = ScheduleSpec.constant(0.0)
        changes["cooling_availability"] = ScheduleSpec(values=_hourly(lambda h: 7 <= h < 18))
    elif case.free_float:
        changes["heating_availability"] = ScheduleSpec.constant(0.0)
        changes["cooling_availability"] = ScheduleSpec.constant(0.0)
        operation["cooling_enabled"] = False
    if changes:
        building.hvac = HVACSystem(airflow=AirflowSpec(**{**airflow.model_dump(), **changes}))
    if operation:
        building.operation = building.operation.with_changes(**operation)
    return building


def summary(hourly, free_float):
    """Standard 140 outputs of one hourly result.

    Parameters
    ----------
    hourly : mapping
        Hourly ``space_heating_demand_kwh``, ``cooling_demand_kwh`` and
        ``indoor_temperature_c``.
    free_float : bool
        Report temperatures instead of loads.

    Returns
    -------
    dict
        Annual loads in MWh and peaks in kW, or air temperature extremes in °C.
    """
    air = np.asarray(hourly["indoor_temperature_c"], float)
    if free_float:
        return {
            "max_temperature_c": float(air.max()),
            "min_temperature_c": float(air.min()),
            "mean_temperature_c": float(air.mean()),
        }
    heating = np.asarray(hourly["space_heating_demand_kwh"], float)
    cooling = np.asarray(hourly["cooling_demand_kwh"], float)
    return {
        "annual_heating_mwh": float(heating.sum() / 1000),
        "annual_cooling_mwh": float(cooling.sum() / 1000),
        "peak_heating_kw": float(heating.max()),
        "peak_cooling_kw": float(cooling.max()),
    }


def run(epw, names, energyplus=None, directory=Path("outputs/ashrae140")):
    """Run the selected cases.

    Parameters
    ----------
    epw : Path
        Denver EPW.
    names : iterable of str
        Case identifiers from :data:`CASES`.
    energyplus : str or None
        EnergyPlus 24.2 executable; None skips EnergyPlus.
    directory : Path
        Output directory; EnergyPlus runs go to ``directory/energyplus/<case>``.

    Returns
    -------
    dict
        Case name to engine name to :func:`summary`.
    """
    from epsm_core import Weather

    from epsm_rc import simulate_building

    weather = Weather.from_epw(Path(epw))
    report = {}
    for name in names:
        case = CASES[name]
        building = make_building(case)
        rows = {}
        if energyplus is not None:
            run_dir = Path(directory) / "energyplus" / name
            eplus = base.run_energyplus(building, weather, energyplus, run_dir)
            rows["energyplus"] = summary(eplus, case.free_float)
            rows["energyplus"]["acceptance"] = eplus["acceptance"]
        for label, settings in base.model_settings().items():
            result = simulate_building(building, weather=weather, settings=settings)
            rows[label] = summary(result.hourly, case.free_float)
        report[name] = rows
        print(
            name,
            json.dumps(
                {
                    k: {m: round(v, 2) for m, v in r.items() if isinstance(v, float)}
                    for k, r in rows.items()
                }
            ),
            flush=True,
        )
    return report


def markdown(report):
    """Format the case table with RC differences from EnergyPlus.

    Parameters
    ----------
    report : dict
        Output of :func:`run`.

    Returns
    -------
    str
        Markdown tables for conditioned and free-float cases.
    """
    titles = {"energyplus": "EnergyPlus", "iso13790_5r1c": "5R1C", "iso52016_1": "ISO 52016-1"}
    loads = [
        ("annual_heating_mwh", "Heating MWh"),
        ("annual_cooling_mwh", "Cooling MWh"),
        ("peak_heating_kw", "Peak heat kW"),
        ("peak_cooling_kw", "Peak cool kW"),
    ]
    temps = [
        ("max_temperature_c", "Max °C"),
        ("min_temperature_c", "Min °C"),
        ("mean_temperature_c", "Mean °C"),
    ]
    lines = ["# ASHRAE 140 Section 5.2 cases", ""]
    for free, metrics in ((False, loads), (True, temps)):
        names = [n for n in report if CASES[n].free_float == free]
        if not names:
            continue
        engines = [e for e in titles if e in report[names[0]]]
        lines += [
            "| Case | Metric | " + " | ".join(titles[e] for e in engines) + " |",
            "|---|---|" + "---|" * len(engines),
        ]
        for name in names:
            for key, label in metrics:
                cells = []
                for engine in engines:
                    value = report[name][engine][key]
                    cell = f"{value:.2f}"
                    if engine != "energyplus" and "energyplus" in report[name]:
                        ref = report[name]["energyplus"][key]
                        if free:
                            cell += f" ({value - ref:+.1f} K)"
                        elif abs(ref) > 0.05:
                            cell += f" ({100 * (value / ref - 1):+.0f} %)"
                    cells.append(cell)
                lines.append(f"| {name} | {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epw", required=True, type=Path)
    parser.add_argument("--energyplus", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/ashrae140"))
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    args = parser.parse_args()
    report = run(args.epw, args.cases, energyplus=args.energyplus, directory=args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ashrae140.json").write_text(json.dumps(report, indent=2))
    text = markdown(report)
    (args.output / "ashrae140.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
