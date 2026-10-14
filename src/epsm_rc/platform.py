# File: platform.py
# Purpose: Reduce shared Core Buildings to independent hourly 5R1C evaluations.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# Author: Sanjay Somanath
# Research attribution: Existing numerical solver credits are retained in iso13790.py.
# Date created: 2026-10-14
# Date modified: 2026-10-14
# Changelog:
#   2026-10-14 - Add the Core adapter with explicit reduction and delivery assumptions.
#   2026-10-14 - Correct recovery double counting only in the platform adapter.
"""Reduce a Core Building to independent hourly 5R1C evaluations.

The reduction combines conditioned space into one temperature domain. Material
inventory remains in Core; only transport/storage equivalents enter this model.
Solar radiation uses actual facade azimuths and Perez irradiance with the supplied
scalar window shading factor. Geometric site shading is unsupported. Weather is
read explicitly; optional output writes hourly arrays. No catalogue loads occur
on import. The standalone solver retains its compatibility behavior.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pvlib import atmosphere, irradiance, solarposition
from pvlib.iotools import read_epw

from .iso13790 import Building5R1C, Construction, ConstructionLayer

if TYPE_CHECKING:
    from epsm_core import Building, Weather
    from epsm_core.results import SimulationResult

logger = logging.getLogger(__name__)


class _Platform5R1C(Building5R1C):
    """Apply sensible recovery exactly once, to mechanical conductance.

    The inherited conductance already represents infiltration plus mechanical
    flow multiplied by (1 - effectiveness). Consequently its driving outdoor
    temperature must remain unmodified. The legacy standalone class also warms
    the supply temperature; this adapter correction deliberately leaves that
    separate compatibility API unchanged.
    """

    def calc_supply_air_temperature(self, t_out: float) -> float:
        return t_out


@dataclass(frozen=True)
class RCSettings:
    """Inspectable reduction assumptions; these are not catalogue evidence.

    Parameters
    ----------
    effective_mass_depth_m : float
        Positive room-side active construction depth in metres; default 0.1.
    ground_coupling_factor : float
        Conductance multiplier in [0, 1] when no ground temperature boundary
        is supplied; default 0.5.
    ground_reflectance : float
        Solar ground albedo in [0, 1]; default 0.2.

    Notes
    -----
    These values describe the reduction rather than evidence about materials.
    One-domain reduction cannot represent different zone setpoints, adjacent
    unconditioned spaces, site shading, detailed HVAC or PV delivery.

    Examples
    --------
    >>> RCSettings(effective_mass_depth_m=0.08).effective_mass_depth_m
    0.08
    """

    effective_mass_depth_m: float = 0.1
    ground_coupling_factor: float = 0.5
    ground_reflectance: float = 0.2

    def __post_init__(self):
        if not np.isfinite(self.effective_mass_depth_m) or self.effective_mass_depth_m <= 0:
            raise ValueError("Effective mass depth must be finite and positive")
        if not 0 <= self.ground_coupling_factor <= 1 or not 0 <= self.ground_reflectance <= 1:
            raise ValueError("Ground coupling and reflectance must be fractions")


def simulate_building(
    building: Building,
    *,
    weather: Weather | str | Path,
    directory: str | Path | None = None,
    settings: RCSettings | dict | None = None,
) -> SimulationResult:
    """Calculate one full year from a Core physical Building.

    Parameters
    ----------
    building : epsm_core.Building
        Physical elements, constructions, HVAC and common operating schedules.
        Must pass readiness for ``epsm_rc``. The input is not mutated.
    weather : epsm_core.Weather or path-like
        Local complete hourly EPW. A Weather object preserves its selected
        calendar year and checksum; a path uses Core's default year, 2023.
    directory : path-like, optional
        If supplied, create this directory and write ``hourly.npz``. An existing
        file with that name is overwritten. The caller owns output locations.
    settings : RCSettings or dict, optional
        Explicit reduction assumptions. Defaults to ``RCSettings()``.

    Returns
    -------
    epsm_core.results.SimulationResult
        Hourly heating/cooling demand and delivered carrier/fan energy in kWh,
        indoor air temperature in degrees Celsius, hour-start local-standard
        labels, annual energy sums, assumptions and artifact paths. No LCA runs.

    Raises
    ------
    ValueError
        Building or weather is incomplete/unsupported, the EPW checksum changed,
        or a numerical result is non-finite.
    OSError
        Weather cannot be read or the requested output cannot be written.

    Notes
    -----
    Each call creates fresh thermal state. The five-resistance model uses a
    single temperature domain and sensible heat only. Heat recovery reduces
    mechanical conductance once; infiltration is never recovered. No weather
    padding, stochastic window opening or legacy construction presets are used.
    The standalone Building5R1C API retains its existing numerical behavior.

    Examples
    --------
    Supply a complete local EPW; install this package's ``platform`` extra.

    >>> from epsm_core import Building, Weather
    >>> result = simulate_building(  # doctest: +SKIP
    ...     Building(), weather=Weather.from_epw("weather.epw"))
    >>> result.scalars["space_heating_demand_kwh"] >= 0  # doctest: +SKIP
    True
    """
    from epsm_core import Weather
    from epsm_core.geometry import facade_catalog
    from epsm_core.heating_delivery import heating_delivery
    from epsm_core.results import SimulationResult
    from epsm_core.weather import file_sha256, hourly_axis, validate_epw_calendar

    settings = RCSettings(**settings) if isinstance(settings, dict) else settings or RCSettings()
    weather = weather if isinstance(weather, Weather) else Weather.from_epw(weather)
    if file_sha256(weather.path) != weather.sha256:
        raise ValueError("EPW contents changed after Weather was created")
    building.readiness(engine="epsm_rc", weather=weather).require()
    spec = building.to_spec(weather)
    logger.info(
        "RC calculation started",
        extra={
            "epsm_event": "energy.rc.started",
            "epsm_context": {"building_id": building.identifier},
            "epsm_details": {"calendar_year": weather.calendar_year},
        },
    )
    year = spec.climate.calendar_year
    validate_epw_calendar(weather.path, year)
    if spec.zone_uses or spec.common_use or spec.common_airflow or spec.unconditioned_exchange:
        raise ValueError("epsm_rc currently requires one common operating/airflow domain")
    if spec.delivery.fvp or spec.renewables.pv_systems:
        raise ValueError("epsm_rc adapter does not yet support FVP or PV delivery")
    if any(not p.conditioned for p in spec.geometry.parts):
        raise ValueError("epsm_rc currently supports fully conditioned parts")
    if spec.geometry.basement_depth_m and not spec.geometry.basement_conditioned:
        raise ValueError("epsm_rc cannot aggregate an unconditioned basement")
    area = spec.geometry.conditioned_floor_area_m2
    volume = spec.geometry.conditioned_volume_m3
    reference_area = spec.load_reference_area_m2 or area
    elements = tuple(building.elements)
    grouped = {"wall": [], "roof": [], "ground": [], "window": []}
    constructions = []
    for element in elements:
        role = element.role.value
        key = (
            "wall"
            if role.startswith("EXT_WALL") or role == "BASEMENT_WALL"
            else "roof"
            if role in {"ROOF", "TOP_FLOOR"}
            else "ground"
            if role in {"GROUND_FLOOR", "BASEMENT_FLOOR"}
            else "window"
            if role == "WINDOW"
            else None
        )
        if key:
            grouped[key].append(element)
        assembly = element.construction
        if assembly is None:
            raise ValueError(f"Missing construction for {element.identifier}")
        if not hasattr(assembly, "components"):
            continue
        # Parallel constituents retain separate physical quantities. Only heat
        # capacity per area is averaged here, following the same layer depth.
        layers = []
        for component in assembly.components:
            for slot in component.layers:
                layer = slot.layer
                capacity = (
                    sum(
                        m.area_fraction
                        * m.thickness
                        * m.material.density
                        * m.material.specific_heat
                        for m in layer.materials
                    )
                    / layer.thickness
                )
                layers.append(ConstructionLayer(layer.thickness, capacity, 1.0, layer.name))
        constructions.append(
            Construction(
                element.identifier,
                list(reversed(layers)),
                element.area_m2,
                settings.effective_mass_depth_m,
            )
        )
    areas = {key: sum(e.area_m2 for e in values) for key, values in grouped.items()}

    def u_value(key):
        return (
            sum(e.area_m2 * e.construction.u_value for e in grouped[key]) / areas[key]
            if areas[key]
            else 0.0
        )

    use, airflow = spec.use, spec.airflow

    def annual(schedule):
        return np.asarray(schedule.hourly(year), dtype=float)

    heat_set, cool_set = annual(use.heating_setpoint), annual(use.cooling_setpoint)
    people_activity = (
        annual(use.occupant_activity_schedule)
        if use.occupant_activity_schedule
        else np.full(len(heat_set), use.occupant_activity_w)
    )
    equipment_fraction = (
        annual(use.equipment_heat_fraction_schedule)
        if use.equipment_heat_fraction_schedule
        else np.full(len(heat_set), use.equipment_heat_fraction)
    )
    gains = reference_area * (
        use.occupant_density_per_m2 * people_activity * annual(use.occupancy_fraction)
        + use.equipment_gain_w_m2 * equipment_fraction * annual(use.equipment_fraction)
        + use.lighting_gain_w_m2 * annual(use.lighting_fraction)
    )
    ground = spec.envelope.ground_temperature
    model = _Platform5R1C(
        floor_area_m2=area,
        building_class=None,
        constructions=constructions,
        u_walls=u_value("wall"),
        u_roof=u_value("roof"),
        u_floor=u_value("ground"),
        u_windows=u_value("window"),
        wall_area_m2=areas["wall"],
        roof_area_m2=areas["roof"],
        floor_area_ground_m2=areas["ground"],
        window_area_m2=areas["window"],
        ceiling_height_m=volume / area,
        building_height_m=volume / area,
        ach_infiltration=0,
        ach_ventilation=0,
        heat_recovery_efficiency=airflow.sensible_recovery_effectiveness,
        internal_gains_w_per_m2=0,
        t_set_heating=float(heat_set[0]),
        t_set_cooling=float(cool_set[0]),
        thermal_bridge_class=None,
        ventilation_system=None,
        delta_u_thermal_bridge=0,
        ventilation_behaviour=0,
        temp_adj_ground=settings.ground_coupling_factor,
        ground_temps_monthly=list(ground.monthly_temperatures_c) if ground else None,
    )
    # Apply Core's declared thermal-bridge uplift to the opaque heat transfer,
    # without changing installed materials or thermal storage.
    uplift = 1 + spec.envelope.thermal_bridge_uplift_fraction
    model.u_walls *= uplift
    model.u_roof *= uplift
    model.u_floor *= uplift
    data, _ = read_epw(str(weather.path), coerce_year=year)
    if not np.isfinite(data.ghi).all() or ((data.ghi < 0) | (data.ghi >= 9999)).any():
        raise ValueError("EPW global horizontal radiation must be complete and nonnegative")
    axis = hourly_axis(year)
    times = pd.DatetimeIndex(axis).tz_localize(
        timezone(timedelta(hours=spec.climate.time_zone_hours))
    )
    sun = solarposition.get_solarposition(
        times, spec.climate.latitude_deg, spec.climate.longitude_deg, spec.climate.elevation_m
    )
    extra = irradiance.get_extra_radiation(times)
    air_mass = atmosphere.get_relative_airmass(sun.apparent_zenith)
    facades = {f.facade_id: f for f in facade_catalog(spec.geometry)}
    solar, opaque = np.zeros(len(axis)), np.zeros(len(axis))
    for element in elements:
        role = element.role.value
        if role not in {"WINDOW", "ROOF", "TOP_FLOOR"} and not role.startswith("EXT_WALL"):
            continue
        roof = role in {"ROOF", "TOP_FLOOR"}
        azimuth = 180 if roof else facades[element.facade_id].azimuth_deg
        poa = irradiance.get_total_irradiance(
            0 if roof else 90,
            azimuth,
            sun.apparent_zenith,
            sun.azimuth,
            data.dni.to_numpy(),
            data.ghi.to_numpy(),
            data.dhi.to_numpy(),
            dni_extra=extra,
            airmass=air_mass,
            albedo=settings.ground_reflectance,
            model="perez",
        )["poa_global"].to_numpy()
        # Perez has a 0/0 sky-clearness term when diffuse radiation is zero.
        # The exact zero-diffuse limit is beam plus ground reflection.
        zero_diffuse = data.dhi.to_numpy() == 0
        if zero_diffuse.any():
            limit = irradiance.get_total_irradiance(
                0 if roof else 90,
                azimuth,
                sun.apparent_zenith,
                sun.azimuth,
                data.dni.to_numpy(),
                data.ghi.to_numpy(),
                data.dhi.to_numpy(),
                albedo=settings.ground_reflectance,
                model="isotropic",
            )["poa_global"].to_numpy()
            poa[zero_diffuse] = limit[zero_diffuse]
        no_radiation = (
            (data.ghi.to_numpy() == 0) & (data.dni.to_numpy() == 0) & (data.dhi.to_numpy() == 0)
        )
        poa[no_radiation] = 0.0
        if (np.isnan(poa) & (sun.apparent_zenith.to_numpy() < 90)).any() or np.isinf(poa).any():
            raise ValueError("Solar calculation produced invalid daylight irradiance")
        poa = np.nan_to_num(poa, nan=0.0)
        if role == "WINDOW":
            solar += (
                poa * element.area_m2 * element.construction.shgc * spec.facade.solar_shading_factor
            )
        else:
            outside = element.construction.components[0].layers[0].layer
            alpha = sum(m.area_fraction * m.material.solar_absorptance for m in outside.materials)
            opaque += poa * element.area_m2 * element.construction.u_value * alpha * 0.04 * uplift
    infiltration = (
        airflow.infiltration_design_flow_m3_s(reference_area_m2=reference_area, volume_m3=volume)
        * 3600
        / volume
        * annual(airflow.infiltration_fraction)
    )
    coefficients = airflow.infiltration_coefficients
    wind = data.wind_speed.to_numpy()
    ventilation = (
        airflow.mechanical_vent_l_s_m2
        * reference_area
        * 3.6
        / volume
        * annual(airflow.mechanical_vent_fraction)
    )
    natural = airflow.natural_exchange_ach * annual(airflow.natural_exchange_fraction)
    temperatures = data.temp_air.to_numpy()
    heating_on, cooling_on = (
        annual(airflow.heating_availability),
        annual(airflow.cooling_availability),
    )
    power, indoor = np.zeros(len(axis)), np.zeros(len(axis))
    model.t_m = model.t_air = model.t_s = float(heat_set[0])
    for hour, temperature in enumerate(temperatures):
        model.t_set_heating, model.t_set_cooling = heat_set[hour], cool_set[hour]
        correction = (
            coefficients[0]
            + coefficients[1] * abs(model.t_air - temperature)
            + coefficients[2] * wind[hour]
            + coefficients[3] * wind[hour] ** 2
        )
        model.ach_infiltration = infiltration[hour] * correction + natural[hour]
        model.ach_ventilation = ventilation[hour]
        model._calculate_thermal_properties()
        # The legacy ground helper uses a non-leap month index. Passing a
        # representative hour within the actual month preserves leap-day input.
        month = times[hour].month
        month_hour = sum([744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744][: month - 1])
        power[hour] = model.calc_heating_demand(
            temperature,
            solar[hour],
            gains[hour],
            solar_gains_opaque=opaque[hour],
            max_heating_power=np.inf if heating_on[hour] else 0,
            max_cooling_power=-np.inf if use.cooling_enabled and cooling_on[hour] else 0,
            hour_of_year=month_hour if ground else None,
        )
        indoor[hour] = model.t_air
    hourly = {
        "space_heating_demand_kwh": np.maximum(power, 0) / 1000,
        "cooling_demand_kwh": np.maximum(-power, 0) / 1000,
        "indoor_temperature_c": indoor,
    }
    dhw = annual(use.dhw_fraction)
    hourly["dhw_load_kwh"] = (
        dhw / dhw.sum() * use.dhw_kwh_m2_year * reference_area if dhw.sum() else dhw
    )
    report = SimpleNamespace(
        data={
            "load_reference_area_m2": reference_area,
            "zones": {
                "aggregate": {
                    "conditioned": True,
                    "load_reference_area_m2": reference_area,
                    "airflow_input": airflow.model_dump(mode="json"),
                }
            },
        }
    )
    hourly.update(heating_delivery(spec, hourly["space_heating_demand_kwh"], report))
    scalars = {key: float(values.sum()) for key, values in hourly.items() if key.endswith("kwh")}
    units = {key: "kWh" if key.endswith("kwh") else "C" for key in hourly}
    diagnostics = dict(
        accepted=all(np.isfinite(v).all() for v in hourly.values()),
        engine="epsm_rc",
        reduction=asdict(settings),
        approximations=[
            "One aggregated conditioned thermal domain",
            "Perez solar gains with scalar window shading; no geometric site shading",
            "0.1 m room-side mass depth unless overridden",
            "Ground conductance reduction when no monthly boundary is supplied",
            "Solar gains use fixed whole-window SHGC",
            "No fan heat feedback",
            "Sensible-only model treats all occupant activity as sensible gain",
            "ISO default 50% convective internal gains and internal area 4.5 times floor area",
            "Legacy sky exchange uses a fixed 11 K sky-air difference and 5 W/(m2 K) coefficient",
            "Heat recovery reduces mechanical conductance once; infiltration is unrecovered",
        ],
        empirical_validation="not_established_by_harmonisation",
    )
    if not diagnostics["accepted"]:
        raise ValueError("RC calculation produced non-finite output")
    run_id = hashlib.sha256(json.dumps(spec.physical_dict(), sort_keys=True).encode()).hexdigest()
    artifacts = {}
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "hourly.npz"
        np.savez_compressed(path, **hourly, time_axis=axis)
        artifacts["hourly"] = str(path)
    logger.info(
        "RC calculation completed",
        extra={
            "epsm_event": "energy.rc.completed",
            "epsm_context": {"building_id": building.identifier, "run_id": run_id},
            "epsm_details": {"hours": len(axis)},
        },
    )
    return SimulationResult(run_id, scalars, hourly, units, diagnostics, artifacts, axis)
