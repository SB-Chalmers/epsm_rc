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

# ASHRAE 140-2020 example-result ranges (min, max) on Denver TMY3 725650 and the
# Addendum b test acceptance criteria for annual heating and cooling, as published
# in the Modelica Buildings Library BESTEST package (Buildings 11,
# ThermalZones.Detailed.Validation.BESTEST). Loads in MWh and kW, temperatures in °C.
REFERENCE = {
    "600": dict(
        heat=(3.993, 4.504),
        cool=(5.432, 6.162),
        peak_heat=(3.020, 3.359),
        peak_cool=(5.422, 6.481),
        heat_criteria=(3.75, 4.98),
        cool_criteria=(5.00, 6.83),
    ),
    "610": dict(
        heat=(4.066, 4.592),
        cool=(4.117, 4.382),
        peak_heat=(3.021, 3.360),
        peak_cool=(5.331, 6.432),
        heat_criteria=(3.61, 5.27),
        cool_criteria=(2.74, 6.03),
    ),
    "620": dict(
        heat=(4.094, 4.719),
        cool=(3.841, 4.404),
        peak_heat=(3.038, 3.385),
        peak_cool=(3.955, 4.797),
        heat_criteria=(3.67, 5.38),
        cool_criteria=(2.76, 5.19),
    ),
    "640": dict(
        heat=(2.403, 2.682),
        cool=(5.237, 5.893),
        peak_heat=(4.039, 4.658),
        peak_cool=(5.365, 6.429),
        heat_criteria=(1.58, 3.76),
        cool_criteria=(4.44, 6.86),
    ),
    "650": dict(
        heat=(0.0, 0.0),
        cool=(4.186, 4.945),
        peak_heat=(0.0, 0.0),
        peak_cool=(5.045, 6.290),
        heat_criteria=(0.0, 0.0),
        cool_criteria=(3.46, 5.88),
    ),
    "900": dict(
        heat=(1.379, 1.814),
        cool=(2.267, 2.714),
        peak_heat=(2.443, 2.778),
        peak_cool=(2.556, 3.376),
        heat_criteria=(1.04, 2.28),
        cool_criteria=(2.35, 2.60),
    ),
    "910": dict(
        heat=(1.648, 2.163),
        cool=(1.191, 1.490),
        peak_heat=(2.469, 2.799),
        peak_cool=(2.081, 2.722),
        heat_criteria=(1.56, 2.30),
        cool_criteria=(0.86, 2.00),
    ),
    "920": dict(
        heat=(2.956, 3.607),
        cool=(2.549, 3.128),
        peak_heat=(2.512, 2.895),
        peak_cool=(2.710, 3.481),
        heat_criteria=(2.55, 4.20),
        cool_criteria=(2.43, 3.08),
    ),
    "940": dict(
        heat=(0.863, 1.389),
        cool=(2.203, 2.613),
        peak_heat=(3.052, 3.882),
        peak_cool=(2.556, 3.376),
        heat_criteria=(0.22, 1.91),
        cool_criteria=(2.24, 3.14),
    ),
    "950": dict(
        heat=(0.0, 0.0),
        cool=(0.586, 0.707),
        peak_heat=(0.0, 0.0),
        peak_cool=(2.054, 2.388),
        heat_criteria=(0.0, 0.0),
        cool_criteria=(0.43, 1.52),
    ),
    "600FF": dict(min_t=(-13.8, -9.9), max_t=(62.4, 68.4), mean_t=(24.3, 26.1)),
    "650FF": dict(min_t=(-17.8, -16.7), max_t=(61.1, 66.8), mean_t=(17.6, 18.9)),
    "900FF": dict(min_t=(0.6, 2.2), max_t=(43.3, 46.0), mean_t=(24.5, 25.7)),
    "950FF": dict(min_t=(-13.4, -12.5), max_t=(36.1, 37.1), mean_t=(14.4, 15.0)),
}
METRIC_KEYS = {
    "annual_heating_mwh": "heat",
    "annual_cooling_mwh": "cool",
    "peak_heating_kw": "peak_heat",
    "peak_cooling_kw": "peak_cool",
    "min_temperature_c": "min_t",
    "max_temperature_c": "max_t",
    "mean_temperature_c": "mean_t",
}
CRITERIA_KEYS = {"annual_heating_mwh": "heat_criteria", "annual_cooling_mwh": "cool_criteria"}
TOLERANCE = 1e-3


def assess(name, metrics):
    """Compare one engine's case result with the Standard 140-2020 tables.

    Parameters
    ----------
    name : str
        Case identifier.
    metrics : dict
        Output of :func:`summary`.

    Returns
    -------
    dict
        ``in_range`` per metric (example-result range, informative) and
        ``meets_criteria`` per annual load (Addendum b acceptance criteria).
    """
    table = REFERENCE[name]
    in_range = {
        key: bool(table[ref][0] - TOLERANCE <= metrics[key] <= table[ref][1] + TOLERANCE)
        for key, ref in METRIC_KEYS.items()
        if key in metrics and ref in table
    }
    criteria = {
        key: bool(table[ref][0] - TOLERANCE <= metrics[key] <= table[ref][1] + TOLERANCE)
        for key, ref in CRITERIA_KEYS.items()
        if key in metrics and ref in table
    }
    return {"in_range": in_range, "meets_criteria": criteria}


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
    # Rebuilding the construction set or window layout resets the window
    # construction, so the Case 600 glazing is assigned again.
    glazing = building.facade.glazing.model_copy(
        update={"u_value_w_m2k": base.WINDOW_U, "solar_heat_gain_coefficient": base.WINDOW_SHGC}
    )
    building.assign_inputs(facade_properties={"glazing": glazing})

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
        for engine in rows.values():
            engine["assessment"] = assess(name, engine)
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


TITLES = {"energyplus": "EnergyPlus", "iso13790_5r1c": "5R1C", "iso52016_1": "ISO 52016-1"}
LABELS = {
    "annual_heating_mwh": "Heating MWh",
    "annual_cooling_mwh": "Cooling MWh",
    "peak_heating_kw": "Peak heat kW",
    "peak_cooling_kw": "Peak cool kW",
    "max_temperature_c": "Max °C",
    "min_temperature_c": "Min °C",
    "mean_temperature_c": "Mean °C",
}


def criteria_failures(report, engine):
    """List the acceptance-criteria checks an engine fails.

    Parameters
    ----------
    report : dict
        Output of :func:`run`.
    engine : str
        Engine label.

    Returns
    -------
    list of str
        ``"<case> <metric>"`` for each failed check.
    """
    return [
        f"{name} {key}"
        for name, rows in report.items()
        if engine in rows
        for key, passed in rows[engine]["assessment"]["meets_criteria"].items()
        if not passed
    ]


def markdown(report):
    """Format results against the Standard 140-2020 ranges and criteria.

    Parameters
    ----------
    report : dict
        Output of :func:`run`.

    Returns
    -------
    str
        Summary and per-case Markdown tables. ``✓``/``✗`` mark the acceptance
        criteria (annual loads); ``¹`` marks a value outside the example-result
        range, which is informative.
    """
    engines = [e for e in TITLES if any(e in rows for rows in report.values())]
    lines = ["# ASHRAE 140-2020 Section 5.2 cases", ""]
    for engine in engines:
        checks = sum(len(r[engine]["assessment"]["meets_criteria"]) for r in report.values())
        failed = criteria_failures(report, engine)
        ranges = [v for r in report.values() for v in r[engine]["assessment"]["in_range"].values()]
        lines.append(
            f"- **{TITLES[engine]}**: {checks - len(failed)}/{checks} acceptance criteria met, "
            f"{sum(ranges)}/{len(ranges)} values inside the example-result range."
        )
    lines += [
        "",
        "✓/✗: Addendum b acceptance criteria (annual loads). ¹: outside the example-result range.",
        "",
        "| Case | Metric | 140-2020 range | Criteria | "
        + " | ".join(TITLES[e] for e in engines)
        + " |",
        "|---|---|---|---|" + "---|" * len(engines),
    ]
    for name, rows in report.items():
        table = REFERENCE[name]
        for key, label in LABELS.items():
            if key not in rows[engines[0]]:
                continue
            low, high = table[METRIC_KEYS[key]]
            crit = table.get(CRITERIA_KEYS.get(key, ""), None)
            cells = []
            for engine in engines:
                value = rows[engine][key]
                assessment = rows[engine]["assessment"]
                cell = f"{value:.2f}"
                if key in assessment["meets_criteria"]:
                    cell += " ✓" if assessment["meets_criteria"][key] else " ✗"
                if not assessment["in_range"].get(key, True):
                    cell += " ¹"
                cells.append(cell)
            crit_text = f"{crit[0]:.2f} – {crit[1]:.2f}" if crit else ""
            lines.append(
                f"| {name} | {label} | {low:.2f} – {high:.2f} | {crit_text} | "
                + " | ".join(cells)
                + " |"
            )
    return "\n".join(lines) + "\n"


def main():
    """Command-line entry point; ``--check`` exits non-zero on a failed criterion."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epw", required=True, type=Path)
    parser.add_argument("--energyplus", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/ashrae140"))
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument(
        "--check",
        nargs="*",
        default=None,
        help="engines that must meet every acceptance criterion (default: both RC models)",
    )
    args = parser.parse_args()
    report = run(args.epw, args.cases, energyplus=args.energyplus, directory=args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ashrae140.json").write_text(json.dumps(report, indent=2))
    text = markdown(report)
    (args.output / "ashrae140.md").write_text(text)
    print(text)
    if args.check is not None:
        engines = args.check or ["iso13790_5r1c", "iso52016_1"]
        failed = {engine: criteria_failures(report, engine) for engine in engines}
        if any(failed.values()):
            print("Acceptance criteria not met:", json.dumps(failed, indent=2))
            raise SystemExit(1)


if __name__ == "__main__":
    main()
