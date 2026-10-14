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
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta, timezone
from importlib import import_module, metadata
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
    airflow_policy : str
        ``reject`` (default) rejects unsupported latent recovery and supply
        temperature limits, including Core's default limits. Select
        ``sensible_only_unlimited_supply`` to explicitly omit these fields.
        Omitted requested values are retained in result diagnostics.
    policy_version : str
        Version of the baseline defaults, ``epsm_rc_5r1c_v1``. Explicit
        overrides are retained alongside this baseline identity.
    internal_surface_area_factor, mass_area_factors : float or tuple
        Ratios to conditioned floor area. Mass-area factors correspond to
        construction capacities below, between and above the two thresholds.
    mass_capacity_thresholds_j_m2_k : tuple
        Ordered construction-capacity thresholds in J/(m² K).
    convective_internal_gains_fraction : float
        Fraction of sensible internal gains delivered to the air node.
    surface_air_coefficient_w_m2_k, mass_surface_coefficient_w_m2_k : float
        Positive node transfer coefficients in W/(m² K).
    air_volumetric_heat_capacity_j_m3_k : float
        Positive air volumetric heat capacity used for airflow conductance.
    outside_wall_resistance_m2_k_w, outside_roof_resistance_m2_k_w : float
        Nonnegative outside film resistances used for opaque solar and sky gains.
    sky_radiation_coefficient_w_m2_k, sky_air_temperature_difference_k : float
        Nonnegative sky exchange coefficient and fixed sky-air difference.
    wall_sky_view_factor, roof_sky_view_factor : float
        Sky view fractions in [0, 1]. These are model assumptions.

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
    airflow_policy: str = "reject"
    policy_version: str = "epsm_rc_5r1c_v1"
    internal_surface_area_factor: float = 4.5
    convective_internal_gains_fraction: float = 0.5
    mass_capacity_thresholds_j_m2_k: tuple[float, float] = (165_000.0, 260_000.0)
    mass_area_factors: tuple[float, float, float] = (2.5, 3.0, 3.5)
    surface_air_coefficient_w_m2_k: float = 3.45
    mass_surface_coefficient_w_m2_k: float = 9.1
    air_volumetric_heat_capacity_j_m3_k: float = 1200.0
    outside_wall_resistance_m2_k_w: float = 0.04
    outside_roof_resistance_m2_k_w: float = 0.04
    sky_radiation_coefficient_w_m2_k: float = 5.0
    sky_air_temperature_difference_k: float = 11.0
    wall_sky_view_factor: float = 0.5
    roof_sky_view_factor: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.effective_mass_depth_m) or self.effective_mass_depth_m <= 0:
            raise ValueError("Effective mass depth must be finite and positive")
        if not 0 <= self.ground_coupling_factor <= 1 or not 0 <= self.ground_reflectance <= 1:
            raise ValueError("Ground coupling and reflectance must be fractions")
        if self.airflow_policy not in {"reject", "sensible_only_unlimited_supply"}:
            raise ValueError("Unknown airflow_policy; choose reject or sensible_only_unlimited_supply")
        if self.policy_version != "epsm_rc_5r1c_v1":
            raise ValueError("Unknown reduction policy version")
        positive = (
            "internal_surface_area_factor", "surface_air_coefficient_w_m2_k",
            "mass_surface_coefficient_w_m2_k", "air_volumetric_heat_capacity_j_m3_k",
        )
        nonnegative = (
            "outside_wall_resistance_m2_k_w", "outside_roof_resistance_m2_k_w",
            "sky_radiation_coefficient_w_m2_k", "sky_air_temperature_difference_k",
        )
        for name in (*positive, *nonnegative):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name in positive and value == 0):
                constraint = "positive" if name in positive else "nonnegative"
                raise ValueError(f"{name} must be finite and {constraint}")
        for name in (
            "convective_internal_gains_fraction", "wall_sky_view_factor", "roof_sky_view_factor",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a finite fraction")
        for name, size in (("mass_capacity_thresholds_j_m2_k", 2), ("mass_area_factors", 3)):
            values = tuple(getattr(self, name))
            if len(values) != size or any(not np.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"{name} requires {size} positive finite values")
            object.__setattr__(self, name, values)
        if self.mass_capacity_thresholds_j_m2_k[0] >= self.mass_capacity_thresholds_j_m2_k[1]:
            raise ValueError("Mass capacity thresholds must be strictly increasing")
        if max(self.mass_area_factors) > self.internal_surface_area_factor:
            raise ValueError("Effective mass area cannot exceed the internal surface area")

    @classmethod
    def from_policy(cls, name: str, **overrides) -> RCSettings:
        """Select the named baseline profile with inspectable explicit overrides."""
        return cls(policy_version=name, **overrides)

    def effective_configuration(self) -> dict:
        """Return model assumptions, distinct from the supplied physical evidence."""
        return {
            "origin": "explicit_reduction_policy",
            "coefficients": asdict(self),
            "model_constants": {
                "timestep_hours": 1.0,
                "crank_nicolson_weight": 0.5,
                "load_interpolation_test_power_w_m2": 10.0,
                "load_interpolation_temperature_tolerance_k": 1e-6,
            },
            "methods": {
                "mass_capacity": "room_side_depth_limited_constructions",
                "mass_area": "construction_capacity_thresholds",
                "solar": "perez_with_exact_zero_diffuse_limit",
                "glazing": "supplied_whole_window_shgc",
                "occupant_gains": "all_activity_treated_as_sensible",
                "fan_heat": "omitted",
                "thermal_domain": "one_conditioned_domain",
                "initial_temperature": "first_hour_heating_setpoint",
                "integration": "hourly_crank_nicolson",
                "control": "air_temperature",
                "gain_partition": "area_ratios_with_nonnegative_surface_fraction",
                "load_interpolation": "linear_unlimited_load_interpolation",
                "mass_exterior_coupling": "series_inverse_conductance_with_opaque_fallback",
            },
        }


def _airflow_capabilities(airflow, settings: RCSettings) -> dict:
    """Reject unsupported requests or record the caller-selected omission."""
    omitted = {
        "maximum_heating_supply_temperature_c": airflow.maximum_heating_supply_temperature_c,
        "minimum_cooling_supply_temperature_c": airflow.minimum_cooling_supply_temperature_c,
    }
    if airflow.latent_recovery_effectiveness:
        omitted["latent_recovery_effectiveness"] = airflow.latent_recovery_effectiveness
    if settings.airflow_policy == "reject":
        raise ValueError(
            f"epsm_rc does not support airflow fields: {', '.join(omitted)}. "
            "To acknowledge their omission, pass "
            "settings=RCSettings(airflow_policy='sensible_only_unlimited_supply'). "
            "For epsm.Building.simulate(engine='epsm_rc'), pass these RC settings explicitly."
        )
    return {"effective_model": settings.airflow_policy, "omitted_inputs": omitted}


def _implementation_identity() -> dict:
    """Fingerprint installed source and numerical dependency versions.

    Development checkouts can change without a distribution version bump.
    Source hashes therefore identify the actual adapter, Core delivery and Data
    object behavior. This reads installed source files only when evaluating.
    """
    sources = {}
    for package in ("epsm_rc", "epsm_core", "epsm_data"):
        root = Path(import_module(package).__file__).parent
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        sources[package] = digest.hexdigest()
    return {
        "source_sha256": sources,
        "dependencies": {
            package: metadata.version(package) for package in ("numpy", "pandas", "pvlib")
        },
        "python": sys.version.split()[0],
    }


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
    settings : RCSettings or dict
        Required explicit reduction assumptions. ``RCSettings.from_policy``
        selects versioned baseline defaults and accepts explicit overrides.
        The default airflow policy rejects unsupported latent/supply controls.

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
    Run identity includes the complete physical inputs, weather checksum,
    reduction settings, installed package source hashes and numerical dependency
    versions. Installed Python sources are read for these hashes on evaluation.
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
    ...     Building(), weather=Weather.from_epw("weather.epw"),
    ...     settings=RCSettings.from_policy(
    ...         "epsm_rc_5r1c_v1", airflow_policy="sensible_only_unlimited_supply"))
    >>> result.scalars["space_heating_demand_kwh"] >= 0  # doctest: +SKIP
    True
    """
    if settings is None:
        raise ValueError(
            "epsm_rc requires explicit settings or a named reduction policy: pass "
            "settings=RCSettings.from_policy('epsm_rc_5r1c_v1', ...). "
            "Pass these settings to epsm.Building.simulate(engine='epsm_rc') as well."
        )
    settings = RCSettings(**settings) if isinstance(settings, dict) else settings
    if not isinstance(settings, RCSettings):
        raise TypeError("settings must be RCSettings or an explicit settings dictionary")

    from epsm_core import Weather
    from epsm_core.geometry import facade_catalog
    from epsm_core.heating_delivery import heating_delivery
    from epsm_core.results import SimulationResult
    from epsm_core.weather import file_sha256, hourly_axis, validate_epw_calendar

    weather = weather if isinstance(weather, Weather) else Weather.from_epw(weather)
    if file_sha256(weather.path) != weather.sha256:
        raise ValueError("EPW contents changed after Weather was created")
    building.readiness(engine="epsm_rc", weather=weather).require()
    spec = building.to_spec(weather)
    airflow_capabilities = _airflow_capabilities(spec.airflow, settings)
    implementation = _implementation_identity()
    run_id = hashlib.sha256(
        json.dumps(
            {
                "physical_inputs": spec.physical_dict(),
                "settings": settings.effective_configuration(),
                "implementation": implementation,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
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
    capacity = sum(c.total_thermal_mass_j_k for c in constructions)
    if not constructions or not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("RC reduction requires positive thermal mass from explicit constructions")
    capacity_per_area = capacity / area
    lower, upper = settings.mass_capacity_thresholds_j_m2_k
    mass_area_factor = settings.mass_area_factors[
        2 if capacity_per_area >= upper else 1 if capacity_per_area >= lower else 0
    ]
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
        a_tot_surfaces_m2=settings.internal_surface_area_factor * area,
        mass_area_factor_override=mass_area_factor,
        convective_fraction_internal=settings.convective_internal_gains_fraction,
        h_is=settings.surface_air_coefficient_w_m2_k,
        h_ms=settings.mass_surface_coefficient_w_m2_k,
        air_volumetric_heat_capacity_j_m3_k=settings.air_volumetric_heat_capacity_j_m3_k,
        r_se_wall=settings.outside_wall_resistance_m2_k_w,
        r_se_roof=settings.outside_roof_resistance_m2_k_w,
        h_r=settings.sky_radiation_coefficient_w_m2_k,
        delta_theta_er=settings.sky_air_temperature_difference_k,
        form_factor_wall=settings.wall_sky_view_factor,
        form_factor_roof=settings.roof_sky_view_factor,
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
        )["poa_global"].to_numpy(copy=True)
        # Own the array before applying physical limits: pandas may expose a
        # read-only view of the irradiance provider's storage.
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
            outside_resistance = (
                settings.outside_roof_resistance_m2_k_w if roof
                else settings.outside_wall_resistance_m2_k_w
            )
            opaque += (
                poa * element.area_m2 * element.construction.u_value
                * alpha * outside_resistance * uplift
            )
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
        effective_configuration=settings.effective_configuration(),
        reduced_properties={
            "internal_surface_area_m2": model._a_tot,
            "effective_mass_area_m2": model._mass_area,
            "thermal_capacitance_j_k": model._c_m,
            "surface_air_conductance_w_k": model._h_tr_is,
            "mass_surface_conductance_w_k": model._h_tr_ms,
            "opaque_transmission_conductance_w_k": model._h_tr_op,
            "mass_exterior_conductance_w_k": model._h_tr_em,
            "mass_exterior_coupling_fallback": bool(
                model._h_tr_op > 0 and model._h_tr_ms > 0 and model._h_tr_op >= model._h_tr_ms
            ),
            "sky_radiation_loss_w": model.calc_sky_radiation_loss(),
        },
        implementation=implementation,
        airflow_capabilities=airflow_capabilities,
        approximations=[
            "One aggregated conditioned thermal domain",
            "Perez solar gains with scalar window shading; no geometric site shading",
            "Room-side mass depth and construction-capacity mass-area rule from explicit policy",
            "Ground conductance reduction when no monthly boundary is supplied",
            "Solar gains use fixed whole-window SHGC",
            "No fan heat feedback",
            "Sensible-only model treats all occupant activity as sensible gain",
            "Internal area, gain split, node transfer and air heat capacity from explicit policy",
            "Fixed sky-air difference and sky exchange coefficients from explicit policy",
            "Heat recovery reduces mechanical conductance once; infiltration is unrecovered",
        ],
        empirical_validation="not_established_by_harmonisation",
    )
    if not diagnostics["accepted"]:
        raise ValueError("RC calculation produced non-finite output")
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
