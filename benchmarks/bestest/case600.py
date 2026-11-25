# File: case600.py
# Purpose: Run ASHRAE Standard 140 BESTEST Case 600 through both EPSM RC thermal models.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-24

"""Run ASHRAE Standard 140 BESTEST Case 600 through both EPSM RC thermal models.

Case 600 is built as an EPSM Core ``Building`` from the Standard 140 inputs
(Section 5.2.1): an 8 m x 6 m x 2.7 m single zone, lightweight layered walls,
roof and raised floor, 12 m² of south glazing, 0.5 ACH infiltration, 200 W of
continuous sensible internal gains (60 % radiative), a 20 °C heating and
27 °C cooling setpoint and ideal, fully convective HVAC. The same Building
runs through ``simulate_building`` once per thermal model with every other
setting shared, so the two results differ by the thermal network only.

The same Building also runs in EnergyPlus 24.2 through EPSM Core when an
executable is given, so all three engines see identical inputs and weather.
Annual and peak heating and cooling are compared with that EnergyPlus run and
with the range of the Standard 140-2020 example results, which use the same
Denver TMY3 weather. ``ashrae140.py`` runs the full case family and applies the
Standard 140-2020 Addendum b acceptance criteria.

Notes
-----
Inputs that are assumptions of this script, not Standard 140 values:

- The window is the Core simple glazing model with the U-value and
  normal-incidence SHGC given in ``WINDOW_U`` and ``WINDOW_SHGC``. Standard 140
  specifies the glass layers. The two values are the simple glazing that best
  reproduces EnergyPlus 24.2 running the layered Standard 140 pane in Cases 600
  and 900 (annual heating and cooling within 1 %).
- The floor underside sees outdoor air temperature with no sun or wind, as in
  Standard 140-2020 and the Modelica Buildings Library Case 600FF. Core
  prescribes a monthly boundary, so the floor uses the monthly mean dry-bulb of
  the Denver TMY3 file (``FLOOR_BOUNDARY_C``). Behind 1 m of insulation the
  daily swing does not reach the room.
- Infiltration air uses the air heat capacity at the Denver site altitude
  (``AIR_HEAT_CAPACITY_J_M3_K``), as Standard 140 asks.
- Interior and exterior films come from each RC model, not from Standard 140.

Examples
--------
Download the Denver TMY3 EPW (WMO 725650, the Standard 140-2020 site), then::

    python benchmarks/bestest/case600.py --epw USA_CO_Denver.Intl.AP.725650_TMY3.epw \
        --energyplus /path/to/energyplus --output benchmarks/bestest/results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np

FLOOR_LENGTH_M = 8.0  # east-west
FLOOR_WIDTH_M = 6.0  # north-south
HEIGHT_M = 2.7
WINDOW_AREA_M2 = 12.0  # two 3 m x 2 m south windows
SOUTH_FACADE = "main.r0.e0"  # Core edge from (0, 0) to (8, 0), azimuth 180°
INTERNAL_GAINS_W = 200.0
RADIATIVE_FRACTION = 0.6
INFILTRATION_ACH = 0.5
HEATING_SETPOINT_C = 20.0
COOLING_SETPOINT_C = 27.0
# Standard 140 Case 600 glazing: two 3.048 mm clear panes (tau 0.834, rho 0.075,
# k 1.0, emissivity 0.84) with a 12 mm air gap. At NFRC conditions EnergyPlus
# 24.2 rates it U 2.74 W/(m² K), SHGC 0.760, but as simple glazing those values
# give 4 % more Case 900 cooling than the layered pane in the same run: the
# layered pane loses more heat through its operating-condition U-value. U 2.82,
# SHGC 0.75 match the layered pane's annual heating and cooling in Cases 600
# and 900 within 1 % (EnergyPlus 24.2, Denver TMY3).
WINDOW_U = 2.82
WINDOW_SHGC = 0.75
EXTERIOR_SOLAR_ABSORPTANCE = 0.6
ALTITUDE_M = 1650.0
# Monthly mean dry-bulb (°C) of USA_CO_Denver.Intl.AP.725650_TMY3.epw. The
# Standard 140-2020 floor faces outdoor air; Core takes a monthly boundary.
FLOOR_BOUNDARY_C = (0.79, -0.05, 6.13, 5.83, 15.49, 23.10, 22.27, 22.61, 19.16, 10.04, 2.90, 1.43)
GROUND_SOURCE = "Standard 140-2020 floor to outdoor air: Denver TMY3 monthly mean dry-bulb"
SENSIBLE_COOLING = "Zone Ideal Loads Supply Air Sensible Cooling Energy"

# Standard 140 Case 600 materials, outside to inside: (name, k W/mK, rho kg/m³, cp J/kgK, d m).
WALL = (
    ("wood siding", 0.14, 530.0, 900.0, 0.009),
    ("fiberglass quilt", 0.04, 12.0, 840.0, 0.066),
    ("plasterboard", 0.16, 950.0, 840.0, 0.012),
)
ROOF = (
    ("roof deck", 0.14, 530.0, 900.0, 0.019),
    ("fiberglass quilt", 0.04, 12.0, 840.0, 0.1118),
    ("plasterboard", 0.16, 950.0, 840.0, 0.010),
)
# Standard 140 gives the floor insulation no mass; Core's minimum (1 kg/m³, 100 J/kgK) is used.
FLOOR = (
    ("floor insulation", 0.04, 1.0, 100.0, 1.003),
    ("timber flooring", 0.14, 650.0, 1200.0, 0.025),
)

# Range of the ASHRAE 140-2020 example results for Case 600 (Denver TMY3 725650),
# as published in the Modelica Buildings Library BESTEST validation package.
REFERENCE = {
    "annual_heating_mwh": (3.993, 4.504),
    "annual_cooling_mwh": (5.432, 6.162),
    "peak_heating_kw": (3.020, 3.359),
    "peak_cooling_kw": (5.422, 6.481),
}
ENERGYPLUS_REFERENCE = {
    "annual_heating_mwh": 4.673,
    "annual_cooling_mwh": 6.792,
    "peak_heating_kw": 3.838,
    "peak_cooling_kw": 6.664,
}


def air_heat_capacity_j_m3_k(altitude_m=ALTITUDE_M, temperature_c=20.0):
    """Volumetric heat capacity of air at the site altitude.

    Parameters
    ----------
    altitude_m : float
        Site altitude above sea level in metres.
    temperature_c : float
        Air temperature used for the density, in °C.

    Returns
    -------
    float
        ρ·c_p in J/(m³ K) from the ICAO standard-atmosphere pressure.
    """
    pressure = 101325.0 * (1 - 2.25577e-5 * altitude_m) ** 5.25588
    density = pressure / (287.05 * (temperature_c + 273.15))
    return density * 1006.0


def _construction(template, layers, *, outer_absorptance):
    """Copy a Core construction-set element and replace its layers."""
    element = json.loads(json.dumps(template))
    component = element["components"][0]
    proto = component["layers"][0]
    rows = []
    for position, (name, k, rho, cp, d) in enumerate(layers):
        row = json.loads(json.dumps(proto))
        row["position"] = position
        layer = row["layer"]
        layer["name"] = layer["layer_code"] = f"bestest.{name}"
        layer["thickness"] = d
        material = layer["materials"][0]
        material["thickness"] = d
        material["area_fraction"] = 1
        material["material"].update(
            name=f"BESTEST {name}",
            json_id=f"bestest.{name.replace(' ', '_')}",
            conductivity=k,
            density=rho,
            specific_heat=cp,
            solar_absorptance=outer_absorptance if position == 0 else 0.6,
            thermal_absorptance=0.9,
            uuid=str(uuid5(NAMESPACE_URL, f"epsm:bestest:{name}")),
        )
        rows.append(row)
    component["layers"] = rows
    return element


def make_building():
    """Build BESTEST Case 600 as an EPSM Core Building.

    Returns
    -------
    epsm_core.Building
        Single-zone Case 600 with layered constructions, south glazing,
        infiltration, constant gains and the 20/27 °C thermostat.
    """
    from epsm_core import Building, Footprint, GeometrySpec, HVACSystem, ScheduleSpec
    from epsm_core.specs.facade import FacadeOverride
    from epsm_core.specs.physics import (
        AirflowSpec,
        FoundationSpec,
        GainPartitionSpec,
        GroundTemperatureSpec,
    )
    from epsm_data import construction_set_from_dict, construction_set_to_dict

    geometry = GeometrySpec.from_footprint(
        Footprint.rectangle(FLOOR_LENGTH_M, FLOOR_WIDTH_M), storeys=1, storey_height_m=HEIGHT_M
    )
    building = Building(
        preset="illustrative-reference", identifier="bestest-600", geometry=geometry
    )

    data = construction_set_to_dict(building.construction_set)
    elements = data["elements"]
    elements["EXT_WALL"] = _construction(
        elements["EXT_WALL"], WALL, outer_absorptance=EXTERIOR_SOLAR_ABSORPTANCE
    )
    for role in ("TOP_FLOOR", "ROOF"):
        elements[role] = _construction(
            elements[role], ROOF, outer_absorptance=EXTERIOR_SOLAR_ABSORPTANCE
        )
    elements["GROUND_FLOOR"] = _construction(elements["GROUND_FLOOR"], FLOOR, outer_absorptance=0.0)
    building.construction_set = construction_set_from_dict(data)

    south_wwr = WINDOW_AREA_M2 / (FLOOR_LENGTH_M * HEIGHT_M)
    building.facade = building.facade.updated(
        default_wwr=0.0, overrides=(FacadeOverride(facade_id=SOUTH_FACADE, wwr=south_wwr),)
    )
    # The window construction carries the optics; facade glazing alone does not replace it.
    glazing = building.facade.glazing.model_copy(
        update={"u_value_w_m2k": WINDOW_U, "solar_heat_gain_coefficient": WINDOW_SHGC}
    )
    building.assign_inputs(facade_properties={"glazing": glazing})
    building.hvac = HVACSystem(
        airflow=AirflowSpec(
            topology="S",
            envelope_infiltration_ach=INFILTRATION_ACH,
            infiltration_source="ASHRAE 140 Case 600: 0.5 ACH continuous",
        )
    )
    building.operation = building.operation.with_changes(
        occupancy_category="unspecified",
        occupant_density_per_m2=0.0,
        lighting_gain_w_m2=0.0,
        dhw_kwh_m2_year=0.0,
        equipment_gain_w_m2=INTERNAL_GAINS_W / (FLOOR_LENGTH_M * FLOOR_WIDTH_M),
        equipment_heat_fraction=1.0,
        gain_partitions=GainPartitionSpec(
            policy="explicit",
            people_fraction_radiant=0.6,
            people_sensible_heat_fraction=1.0,
            equipment_fraction_radiant=RADIATIVE_FRACTION,
            equipment_fraction_latent=0.0,
            lighting_fraction_radiant=0.6,
            lighting_fraction_visible=0.0,
            lighting_return_air_fraction=0.0,
        ),
        heating_setpoint=ScheduleSpec.constant(HEATING_SETPOINT_C),
        cooling_setpoint=ScheduleSpec.constant(COOLING_SETPOINT_C),
        cooling_enabled=True,
    )
    building.foundation = FoundationSpec(method="prescribed_temperature", source=GROUND_SOURCE)
    building.ground_temperature = GroundTemperatureSpec(
        monthly_temperatures_c=FLOOR_BOUNDARY_C, source=GROUND_SOURCE
    )
    building.load_reference_area_m2 = FLOOR_LENGTH_M * FLOOR_WIDTH_M
    return building


def model_settings():
    """RC settings for the two thermal models, identical apart from the network.

    Returns
    -------
    dict
        Label to ``RCSettings``.
    """
    from epsm_rc import RCSettings

    base = RCSettings(
        airflow_policy="sensible_only_unlimited_supply",
        # No ground_policy: the Building's prescribed floor boundary is used as given.
        air_volumetric_heat_capacity_j_m3_k=air_heat_capacity_j_m3_k(),
        convective_internal_gains_fraction=1 - RADIATIVE_FRACTION,
        # Cases 610 and 910: exact overhang shadow instead of ISO 13790 G.4-G.6.
        overhang_shading_method="geometric_profile_angle",
    )
    return {
        "iso13790_5r1c": replace(base, thermal_model="iso13790_5r1c"),
        "iso52016_1": replace(base, thermal_model="iso52016_1"),
    }


def _sql_hourly(sql_path, name):
    """Hourly values of one ReportData variable from the annual run period."""
    import sqlite3

    query = (
        "SELECT d.Value FROM ReportData d "
        "JOIN Time t ON d.TimeIndex = t.TimeIndex "
        "JOIN ReportDataDictionary k "
        "ON d.ReportDataDictionaryIndex = k.ReportDataDictionaryIndex "
        "WHERE k.Name = ? AND k.ReportingFrequency = 'Hourly' "
        "AND t.EnvironmentPeriodIndex = (SELECT MAX(EnvironmentPeriodIndex) FROM Time) "
        "ORDER BY d.TimeIndex"
    )
    with sqlite3.connect(sql_path) as connection:
        values = [row[0] for row in connection.execute(query, (name,))]
    if len(values) != 8760:
        raise ValueError(f"{name}: expected 8760 hourly values, found {len(values)}")
    return np.asarray(values, float)


def run_energyplus(building, weather, energyplus, directory):
    """Run the same Building in EnergyPlus 24.2 through EPSM Core.

    Core's acceptance check rejects this run for four psychrometric warnings
    (a wet-bulb iteration in very dry air at the Denver pressure, 82.6 kPa)
    that it cannot waive. The run itself completes with no severe errors, so
    the ideal-loads energies are read from ``eplusout.sql`` and the acceptance
    outcome is recorded with the result.

    Parameters
    ----------
    building : epsm_core.Building
        Case 600 from :func:`make_building`.
    weather : epsm_core.Weather
        The weather used for the RC runs.
    energyplus : str or Path
        EnergyPlus 24.2 executable.
    directory : Path
        Empty run directory.

    Returns
    -------
    dict
        Hourly supply-air (coil) sensible heating and cooling in kWh, which excludes the free cooling of night ventilation air, zone air temperature in
        °C, total cooling coil energy in kWh and the Core acceptance outcome.
    """
    from epsm_core.runner import EnergyPlusRunner, SimulationError
    from epsm_core.settings import CompilerSettings, HourlyOutputRequest

    settings = CompilerSettings(hourly_outputs=(HourlyOutputRequest(name=SENSIBLE_COOLING),))
    directory = Path(directory)
    try:
        EnergyPlusRunner(energyplus, settings=settings).run(building.to_spec(weather), directory)
        acceptance = "accepted by Core"
    except SimulationError:
        log = (directory / "eplusout.err").read_text(errors="replace")
        if "Completed Successfully" not in log or " 0 Severe Errors" not in log:
            raise
        acceptance = "rejected by Core warning review (psychrometric warnings); read from SQL"
    sql = directory / "eplusout.sql"
    joules = 3.6e6
    return {
        "space_heating_demand_kwh": _sql_hourly(
            sql, "Zone Ideal Loads Supply Air Sensible Heating Energy"
        )
        / joules,
        "cooling_demand_kwh": _sql_hourly(
            sql, "Zone Ideal Loads Supply Air Sensible Cooling Energy"
        )
        / joules,
        "cooling_total_kwh": _sql_hourly(sql, "Zone Ideal Loads Supply Air Total Cooling Energy")
        / joules,
        "indoor_temperature_c": _sql_hourly(sql, "Zone Mean Air Temperature"),
        "acceptance": acceptance,
    }


def _summary(hourly):
    heating = np.asarray(hourly["space_heating_demand_kwh"], float)
    cooling = np.asarray(hourly["cooling_demand_kwh"], float)
    air = np.asarray(hourly["indoor_temperature_c"], float)
    return {
        "annual_heating_mwh": float(heating.sum() / 1000),
        "annual_cooling_mwh": float(cooling.sum() / 1000),
        "peak_heating_kw": float(heating.max()),
        "peak_cooling_kw": float(cooling.max()),
        "peak_heating_hour": int(heating.argmax()),
        "peak_cooling_hour": int(cooling.argmax()),
        "air_temperature_c": {
            "min": float(air.min()),
            "max": float(air.max()),
            "mean": float(air.mean()),
        },
    }


def run(epw, energyplus=None, directory=None):
    """Simulate Case 600 with both thermal models and, optionally, EnergyPlus.

    Parameters
    ----------
    epw : str or Path
        Denver EPW file.
    energyplus : str or Path or None
        EnergyPlus 24.2 executable; None skips the EnergyPlus run.
    directory : Path or None
        Output directory; the EnergyPlus run goes to ``directory/energyplus``.

    Returns
    -------
    dict
        Inputs, reference ranges and one summary per thermal model.
    """
    from epsm_core import Weather

    from epsm_rc import simulate_building

    weather = Weather.from_epw(Path(epw))
    building = make_building()
    models, series = {}, {}
    if energyplus is not None:
        run_directory = Path(directory or "outputs/bestest") / "energyplus"
        series["energyplus"] = run_energyplus(building, weather, energyplus, run_directory)
        models["energyplus"] = _summary(series["energyplus"])
        models["energyplus"]["annual_cooling_total_mwh"] = float(
            series["energyplus"]["cooling_total_kwh"].sum() / 1000
        )
        models["energyplus"]["acceptance"] = series["energyplus"]["acceptance"]
    for label, settings in model_settings().items():
        result = simulate_building(building, weather=weather, settings=settings)
        series[label] = result.hourly
        models[label] = _summary(result.hourly)
        models[label]["reduced_properties"] = result.diagnostics.get("reduced_properties")
    for label, summary in models.items():
        summary["within_reference"] = {
            key: bool(low <= summary[key] <= high) for key, (low, high) in REFERENCE.items()
        }
        if "energyplus" in series and label != "energyplus":
            reference = series["energyplus"]
            summary["vs_energyplus"] = {
                key: summary[key] / models["energyplus"][key] - 1 for key in REFERENCE
            }
            summary["air_temperature_rmse_k"] = float(
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(series[label]["indoor_temperature_c"], float)
                            - reference["indoor_temperature_c"]
                        )
                        ** 2
                    )
                )
            )
    return {
        "case": "ASHRAE 140 Case 600",
        "weather": Path(epw).name,
        "inputs": {
            "window_u_w_m2k": WINDOW_U,
            "window_shgc": WINDOW_SHGC,
            "infiltration_ach": INFILTRATION_ACH,
            "air_heat_capacity_j_m3_k": air_heat_capacity_j_m3_k(),
            "internal_gains_w": INTERNAL_GAINS_W,
            "setpoints_c": [HEATING_SETPOINT_C, COOLING_SETPOINT_C],
        },
        "reference_range": {key: list(value) for key, value in REFERENCE.items()},
        "energyplus_standard140_example": ENERGYPLUS_REFERENCE,
        "models": models,
    }


def markdown(report):
    """Format the comparison as a Markdown table.

    Parameters
    ----------
    report : dict
        Output of :func:`run`.

    Returns
    -------
    str
        Markdown summary.
    """
    labels = {
        "annual_heating_mwh": "Annual heating (MWh)",
        "annual_cooling_mwh": "Annual cooling (MWh)",
        "peak_heating_kw": "Peak heating (kW)",
        "peak_cooling_kw": "Peak cooling (kW)",
    }
    models = report["models"]
    names = [n for n in ("energyplus", "iso13790_5r1c", "iso52016_1") if n in models]
    titles = {
        "energyplus": "EnergyPlus 24.2",
        "iso13790_5r1c": "5R1C",
        "iso52016_1": "ISO 52016-1",
    }
    lines = [
        "# BESTEST Case 600",
        "",
        f"Weather: `{report['weather']}`. All engines run the same EPSM Core Building.",
        "Range: ASHRAE 140-2020 example results (Denver TMY3).",
        "RC cells give the difference from the EnergyPlus run in brackets.",
        "",
        "| Metric | Standard 140 range | " + " | ".join(titles[n] for n in names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for key, label in labels.items():
        low, high = report["reference_range"][key]
        cells = []
        for name in names:
            value = models[name][key]
            cell = f"{value:.2f}"
            if "vs_energyplus" in models[name]:
                cell += f" ({100 * models[name]['vs_energyplus'][key]:+.0f} %)"
            if not models[name]["within_reference"][key]:
                cell += " ¹"
            cells.append(cell)
        lines.append(f"| {label} | {low:.2f} – {high:.2f} | {' | '.join(cells)} |")
    rmse = [
        f"{titles[n]} {models[n]['air_temperature_rmse_k']:.2f} K"
        for n in names
        if "air_temperature_rmse_k" in models[n]
    ]
    lines += ["", "¹ outside the Standard 140 range."]
    if rmse:
        lines += ["", "Hourly air temperature RMSE against EnergyPlus: " + ", ".join(rmse) + "."]
    return "\n".join(lines) + "\n"


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epw", required=True, type=Path)
    parser.add_argument("--energyplus", default=None, help="EnergyPlus 24.2 executable")
    parser.add_argument("--output", type=Path, default=Path("outputs/bestest"))
    args = parser.parse_args()
    report = run(args.epw, energyplus=args.energyplus, directory=args.output)
    text = markdown(report)
    print(text)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "case600.json").write_text(json.dumps(report, indent=2, default=str))
    (args.output / "case600.md").write_text(text)


if __name__ == "__main__":
    main()
