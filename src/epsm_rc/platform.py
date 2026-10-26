# File: platform.py
# Purpose: Reduce shared Core Buildings to independent hourly 5R1C evaluations.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# Author: Sanjay Somanath
# Research attribution: Existing numerical solver credits are retained in iso13790.py.
# Date created: 2026-10-14
# Date modified: 2026-10-21
# Changelog:
#   2026-10-19 - Preserve conditioned-basement controls and finite cold-attic buffers.
#   2026-10-14 - Add the Core adapter with explicit reduction and delivery assumptions.
#   2026-10-14 - Use the common solver with sensible recovery applied once.
"""Reduce a Core Building to independent hourly 5R1C evaluations.

The reduction prepares each conditioned thermal domain once. Material
inventory remains in Core; only transport/storage equivalents enter this model.
Solar radiation uses actual facade azimuths and Perez irradiance with the supplied
scalar window shading factor. Context buildings use the ISO 13790 hourly
average-horizon approximation, with geometry-only preparation reuse.
Core's physical element boundaries determine exposure. Adiabatic/shared and
internal opaque elements retain their depth-limited thermal storage but contribute
no outdoor transmission, absorbed solar radiation or sky exchange.
Self-shadowing for courtyards, concave footprints or multiple parts requires an
explicit omission policy, reported in diagnostics and calculation identity.
Weather is read explicitly; optional output writes hourly arrays. No catalogue loads occur
on import. Sensible recovery uses the common standalone solver implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib import atmosphere, irradiance, solarposition
from pvlib.iotools import read_epw
from shapely.geometry import Polygon

from epsm_rc._logging import bind_context, context_logger
from epsm_rc._logging import operation as logged_operation

from ._identity import implementation_identity
from .iso13790 import Building5R1C, Construction, ConstructionLayer
from .optics import SimpleGlazingOptics
from .preparation import PreparationCache
from .radiation import plane_irradiance

logger = context_logger(__name__)


@dataclass(frozen=True)
class RCSettings:
    """Inspectable reduction assumptions; these are not catalogue evidence.

    Parameters
    ----------
    effective_mass_depth_m : float
        Positive room-side active construction depth in metres; default 0.1.
    ground_coupling_factor : float
        Conductance multiplier in [0, 1] for the explicitly selected
        ``conductance_multiplier`` ground policy; baseline value 0.5.
    ground_reflectance : float
        Solar ground albedo in [0, 1]; default 0.2.
    airflow_policy : str
        ``reject`` (default) rejects unsupported latent recovery and supply
        temperature limits, including Core's default limits. Select
        ``sensible_only_unlimited_supply`` to explicitly omit these fields.
        Omitted requested values are retained in result diagnostics.
    policy_version : str
        Version of the baseline defaults, ``epsm_rc_5r1c_v1`` or v2. Explicit
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
    ground_policy : str
        ``reject`` rejects unsupported foundation methods such as Kiva.
        Select ``conductance_multiplier`` or ``prescribed_monthly_temperature``
        to explicitly approximate an unsupported foundation. The opt-in
        ``iso13370_periodic_approximation`` derives slab/heated-basement ground
        conductance and monthly annual-periodic forcing from geometry, material
        resistance and soil properties. Its monthly indoor reference is the
        heating setpoint; it is not a detailed hourly ground solver.
    iso13370_wall_thickness_m : float or None
        Explicit full wall thickness at ground level, in metres, for ISO 13370.
        Required by the ISO policy; footing-block width is not wall thickness.
    iso13370_wall_thickness_source : str or None
        Evidence or explicit assumption for that wall thickness.
    monthly_ground_temperatures_c : tuple or None
        Twelve finite Celsius boundary values for an explicitly selected
        monthly approximation. A source must also be supplied.
    monthly_ground_temperature_source : str or None
        Source or assumption identifying the supplied monthly boundary.
    self_shadowing_policy : str
        ``reject`` (default) requires an explicit decision for courtyards,
        concave footprints and multiple parts. ``ignore`` acknowledges omitted
        self-shadowing; context horizon shading is a separate method.
    solar_time_reference : str
        ``hour_start`` retains v1; ``interval_midpoint`` samples solar position
        half an hour into each weather interval without changing result labels.
    window_optics_method : str
        ``constant_shgc`` retains v1; ``simple_glazing_angular_v1`` selects the
        documented specular equivalent-layer approximation with beam/diffuse
        optics and nominal inward absorption. Explicit v2 selects both upgrades.
    context_shading_method : str
        ``iso13790_hourly_horizon`` (default) applies Annex G.5.4.2 equation G.3
        using a representative point and average front-horizon angle per receiver.
        Only direct radiation is reduced. ``reject`` retains an explicit refusal
        of context buildings. Facade shading details remain unsupported.
    horizon_azimuth_step_deg : float
        Maximum midpoint-quadrature spacing for the average horizon angle,
        between 1 and 30 degrees; default 10. Numerical choice, not an ISO requirement.

    cold_attic_policy : str
        ``reject`` keeps omitted attic assumptions visible. The explicit
        ``quasi_steady_buffer`` uses supplied attic outdoor exchange and physical
        shell/ceiling conductances, retaining shell solar/sky but omitting shell storage.
    interzone_floor_surface_resistance_m2_k_w : float
        Positive combined room-side films for the main/basement separating floor,
        in m² K/W; initial assumption 0.27 (0.17 + 0.10).

    Notes
    -----
    These values describe reduction assumptions rather than material evidence.
    Main and conditioned basement may have distinct resolved controls. Other
    mixed uses, unconditioned basements, facade shading details, detailed HVAC and PV remain
    unsupported.

    Examples
    --------
    >>> from epsm_rc import RCSettings
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
    ground_policy: str = "reject"
    iso13370_wall_thickness_m: float | None = None
    iso13370_wall_thickness_source: str | None = None
    monthly_ground_temperatures_c: tuple[float, ...] | None = None
    monthly_ground_temperature_source: str | None = None
    self_shadowing_policy: str = "reject"
    solar_time_reference: str | None = None
    window_optics_method: str | None = None
    cold_attic_policy: str = "reject"
    interzone_floor_surface_resistance_m2_k_w: float = 0.27
    context_shading_method: str = "iso13790_hourly_horizon"
    horizon_azimuth_step_deg: float = 10.0

    def __post_init__(self):
        if not np.isfinite(self.effective_mass_depth_m) or self.effective_mass_depth_m <= 0:
            raise ValueError("Effective mass depth must be finite and positive")
        if not 0 <= self.ground_coupling_factor <= 1 or not 0 <= self.ground_reflectance <= 1:
            raise ValueError("Ground coupling and reflectance must be fractions")
        if self.airflow_policy not in {"reject", "sensible_only_unlimited_supply"}:
            raise ValueError(
                "Unknown airflow_policy; choose reject or sensible_only_unlimited_supply"
            )
        if self.policy_version not in {"epsm_rc_5r1c_v1", "epsm_rc_5r1c_v2"}:
            raise ValueError("Unknown reduction policy version")
        angular = self.policy_version == "epsm_rc_5r1c_v2"
        if self.solar_time_reference is None:
            object.__setattr__(
                self, "solar_time_reference", "interval_midpoint" if angular else "hour_start"
            )
        if self.window_optics_method is None:
            object.__setattr__(
                self,
                "window_optics_method",
                "simple_glazing_angular_v1" if angular else "constant_shgc",
            )
        if self.solar_time_reference not in {"hour_start", "interval_midpoint"}:
            raise ValueError("Unknown solar_time_reference")
        if self.window_optics_method not in {"constant_shgc", "simple_glazing_angular_v1"}:
            raise ValueError("Unknown window_optics_method")
        positive = (
            "internal_surface_area_factor",
            "surface_air_coefficient_w_m2_k",
            "mass_surface_coefficient_w_m2_k",
            "air_volumetric_heat_capacity_j_m3_k",
        )
        nonnegative = (
            "outside_wall_resistance_m2_k_w",
            "outside_roof_resistance_m2_k_w",
            "sky_radiation_coefficient_w_m2_k",
            "sky_air_temperature_difference_k",
        )
        for name in (*positive, *nonnegative):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name in positive and value == 0):
                constraint = "positive" if name in positive else "nonnegative"
                raise ValueError(f"{name} must be finite and {constraint}")
        for name in (
            "convective_internal_gains_fraction",
            "wall_sky_view_factor",
            "roof_sky_view_factor",
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
        if self.ground_policy not in {
            "reject",
            "conductance_multiplier",
            "prescribed_monthly_temperature",
            "iso13370_periodic_approximation",
        }:
            raise ValueError("Unknown ground_policy")
        if self.ground_policy == "iso13370_periodic_approximation":
            width = self.iso13370_wall_thickness_m
            if isinstance(width, bool) or width is None or not np.isfinite(width) or width <= 0:
                raise ValueError("ISO 13370 ground policy requires positive wall thickness")
            if not isinstance(self.iso13370_wall_thickness_source, str) or not (
                self.iso13370_wall_thickness_source.strip()
            ):
                raise ValueError(
                    "ISO 13370 wall thickness requires an explicit source or assumption"
                )
        elif self.iso13370_wall_thickness_m is not None or (
            self.iso13370_wall_thickness_source is not None
        ):
            raise ValueError("Select ISO 13370 ground policy for its wall-thickness inputs")
        if self.ground_policy == "prescribed_monthly_temperature":
            values = tuple(self.monthly_ground_temperatures_c or ())
            if len(values) != 12 or any(not np.isfinite(value) for value in values):
                raise ValueError("Monthly ground boundary requires 12 finite temperatures")
            if not isinstance(self.monthly_ground_temperature_source, str) or not (
                self.monthly_ground_temperature_source.strip()
            ):
                raise ValueError(
                    "Monthly ground boundary requires an explicit source or assumption"
                )
            object.__setattr__(self, "monthly_ground_temperatures_c", values)
        elif self.monthly_ground_temperatures_c is not None or (
            self.monthly_ground_temperature_source is not None
        ):
            raise ValueError(
                "Select ground_policy='prescribed_monthly_temperature' for monthly inputs"
            )
        if self.cold_attic_policy not in {"reject", "quasi_steady_buffer"}:
            raise ValueError("Unknown cold_attic_policy")
        if (
            not np.isfinite(self.interzone_floor_surface_resistance_m2_k_w)
            or self.interzone_floor_surface_resistance_m2_k_w <= 0
        ):
            raise ValueError("Interzone floor surface resistance must be finite and positive")
        if self.self_shadowing_policy not in {"reject", "ignore"}:
            raise ValueError("Unknown self_shadowing_policy; choose reject or ignore")
        if self.context_shading_method not in {"reject", "iso13790_hourly_horizon"}:
            raise ValueError("Unknown context_shading_method")
        if (
            isinstance(self.horizon_azimuth_step_deg, bool)
            or not np.isfinite(self.horizon_azimuth_step_deg)
            or not 1 <= self.horizon_azimuth_step_deg <= 30
        ):
            raise ValueError("horizon_azimuth_step_deg must be finite and between 1 and 30")

    def __repr__(self):
        return f"RCSettings(policy={self.policy_version!r}, optics={self.window_optics_method!r})"

    __str__ = __repr__

    def summary(self) -> dict:
        """Return detached settings and method assumptions without I/O.

        Examples
        --------
        >>> RCSettings().summary()["coefficients"]["effective_mass_depth_m"]
        0.1
        """
        return self.effective_configuration()

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
                "glazing": self.window_optics_method,
                "solar_time_reference": self.solar_time_reference,
                "occupant_gains": "all_activity_treated_as_sensible",
                "fan_heat": "omitted",
                "thermal_domain": "one_conditioned_domain",
                "boundaries": "core_physical_element_boundaries",
                "initial_temperature": "first_hour_heating_setpoint",
                "integration": "hourly_crank_nicolson",
                "control": "air_temperature",
                "gain_partition": "area_ratios_with_nonnegative_surface_fraction",
                "load_interpolation": "linear_unlimited_load_interpolation",
                "mass_exterior_coupling": "series_inverse_conductance_with_opaque_fallback",
                "self_shadowing": "geometry_capability_gate_with_explicit_omission",
                "context_shading": self.context_shading_method,
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


def _gain_inputs(use, reference_area, year, settings):
    """Resolve supplied heat channels or the explicitly selected legacy RC policy."""

    def annual(schedule):
        return np.asarray(schedule.hourly(year), dtype=float)

    partitions = use.gain_partitions
    if partitions is not None and partitions.policy == "explicit":
        from epsm_core.gains import resolve_internal_gains

        channels = resolve_internal_gains(use, reference_area_m2=reference_area, calendar_year=year)
        if any(channels.unresolved_people_w):
            raise ValueError(
                "epsm_rc requires an explicit people sensible heat fraction for active people gains"
            )
        if any(channels.return_air_w):
            raise ValueError("epsm_rc does not support active lighting return-air heat")
        convective = np.asarray(channels.convective_w)
        radiant = np.asarray(channels.radiant_w)
        report = {
            "policy": "shared_explicit",
            "aggregate_split_applied": False,
            "supplied_partitions": partitions.model_dump(mode="json"),
            "visible_lighting_treatment": "radiant_heat_deposited_on_room_surfaces",
            "latent_treatment": "excluded_from_sensible_thermal_network",
            "channel_energy_kwh": {
                name: sum(getattr(channels, name)) / 1000
                for name in ("convective_w", "radiant_w", "latent_w", "lost_w", "return_air_w")
            },
        }
        return convective + radiant, convective, radiant, report

    activity = (
        annual(use.occupant_activity_schedule)
        if use.occupant_activity_schedule
        else use.occupant_activity_w
    )
    heat = (
        annual(use.equipment_heat_fraction_schedule)
        if use.equipment_heat_fraction_schedule
        else use.equipment_heat_fraction
    )
    gains = reference_area * (
        use.occupant_density_per_m2 * activity * annual(use.occupancy_fraction)
        + use.equipment_gain_w_m2 * heat * annual(use.equipment_fraction)
        + use.lighting_gain_w_m2 * annual(use.lighting_fraction)
    )
    convective = gains * settings.convective_internal_gains_fraction
    return (
        gains,
        convective,
        gains - convective,
        {
            "policy": "legacy_sensible_from_explicit_rc_policy",
            "aggregate_split_applied": True,
            "core_policy": partitions.policy if partitions is not None else "legacy_default",
            "occupant_treatment": "all_activity_treated_as_sensible",
        },
    )


def _ground_reduction(
    envelope,
    settings: RCSettings,
    *,
    building=None,
    spec=None,
    resolved=None,
    domain=None,
    outdoor_temperatures=None,
    time_axis=None,
) -> tuple[tuple | None, float, dict]:
    """Preserve prescribed monthly boundaries or require an explicit approximation."""
    requested = envelope.foundation.model_dump(mode="json")
    boundary = envelope.ground_temperature
    report = {"requested_foundation": requested, "requested_policy": settings.ground_policy}
    if boundary is not None:
        if settings.monthly_ground_temperatures_c is not None:
            raise ValueError("Two monthly ground boundaries supplied; select one source explicitly")
        months = tuple(boundary.monthly_temperatures_c)
        source = boundary.source
    elif settings.ground_policy == "prescribed_monthly_temperature":
        months = settings.monthly_ground_temperatures_c
        source = settings.monthly_ground_temperature_source
    elif settings.ground_policy == "conductance_multiplier":
        report.update(
            effective_method="conductance_multiplier",
            effective_multiplier=settings.ground_coupling_factor,
            monthly_temperatures_c=None,
            boundary_source="caller-selected conductance multiplier approximation",
        )
        return None, settings.ground_coupling_factor, report
    elif settings.ground_policy == "iso13370_periodic_approximation":
        return _iso13370_reduction(
            report,
            building=building,
            spec=spec,
            resolved=resolved,
            settings=settings,
            domain=domain,
            outdoor_temperatures=outdoor_temperatures,
            time_axis=time_axis,
        )
    else:
        raise ValueError(
            f"epsm_rc does not implement {requested['method']} foundations. Select an explicit "
            "ground_policy='conductance_multiplier', 'prescribed_monthly_temperature' or "
            "'iso13370_periodic_approximation' "
            "in RCSettings to acknowledge the approximation."
        )
    report.update(
        effective_method="prescribed_monthly_temperature",
        effective_multiplier=1.0,
        monthly_temperatures_c=months,
        boundary_source=source,
    )
    return months, 1.0, report


def _iso13370_reduction(
    report, *, building, spec, resolved, settings, domain, outdoor_temperatures, time_axis
):
    """Map an explicit physical foundation to monthly equivalent ground forcing.

    The ISO coefficient refers to the internal/external environments, not a
    measured outside-slab soil temperature. Never apply it to a separately
    prescribed soil boundary. Existing prescribed boundaries take precedence in
    ``_ground_reduction`` and retain their original raw assembly conductance.
    """
    from .ground import ground_coefficients, monthly_ground_temperatures
    from .ground_inputs import resolve_ground_inputs

    if building is None or spec is None or resolved is None:
        raise ValueError("ISO 13370 requires resolved physical foundation inputs")
    inputs = resolve_ground_inputs(building, spec, resolved, settings, domain=domain)
    report.update(
        effective_method="iso13370_periodic_approximation",
        effective_multiplier=1.0,
        boundary_source="ISO 13370 equivalent environment; not prescribed outside-slab soil",
        monthly_temperatures_c=None,
        junction_psi_w_mk=0.0,
        thermal_bridge_method="Core equivalent material resistance before ground calculation",
        forcing_limitations=[
            "Monthly annual-periodic approximation; not a detailed hourly soil solver",
            "Annual-periodic phase offsets omitted",
            "Monthly indoor reference is the supplied heating-setpoint average",
            "Within-month indoor changes and free-floating temperatures retain 5R1C coupling",
            "No separate wall/floor junction line correction beyond Core's declared material uplift",
        ],
    )
    if inputs is None:
        report.update(ground_contact=False, steady_conductance_w_k=0.0)
        return None, 1.0, report
    coefficients = ground_coefficients(**inputs.coefficient_inputs())
    report.update(
        ground_contact=True,
        physical_inputs=inputs.report,
        coefficients=asdict(coefficients),
        steady_conductance_w_k=coefficients.steady_conductance_w_k,
    )
    # Preflight validates geometry/materials without parsing weather or starting
    # a thermal model. Execution supplies the already-captured annual buffers.
    if outdoor_temperatures is None and time_axis is None:
        return None, 1.0, report
    if outdoor_temperatures is None or time_axis is None:
        raise ValueError("ISO 13370 forcing needs both weather and its calendar")
    temperatures = np.asarray(outdoor_temperatures, dtype=float)
    months = pd.DatetimeIndex(time_axis).month.to_numpy()
    if temperatures.shape != months.shape or not np.isfinite(temperatures).all():
        raise ValueError("ISO 13370 weather must be finite and match the annual calendar")
    use = domain.use if domain is not None else spec.use
    indoor = np.asarray(use.heating_setpoint.hourly(spec.climate.calendar_year), dtype=float)
    if indoor.shape != months.shape or not np.isfinite(indoor).all():
        raise ValueError("ISO 13370 indoor reference must match the annual calendar")
    weights = [int(np.count_nonzero(months == m)) for m in range(1, 13)]
    if not all(weights):
        raise ValueError("ISO 13370 forcing needs all twelve calendar months")
    outdoor_monthly = tuple(float(temperatures[months == m].mean()) for m in range(1, 13))
    indoor_monthly = tuple(float(indoor[months == m].mean()) for m in range(1, 13))
    ground = tuple(
        monthly_ground_temperatures(
            coefficients,
            outdoor_monthly,
            indoor_monthly,
            month_weights=weights,
        )
    )
    report.update(
        monthly_temperatures_c=ground,
        monthly_outdoor_reference_c=outdoor_monthly,
        monthly_indoor_reference_c=indoor_monthly,
        month_weights_hours=weights,
    )
    return ground, 1.0, report


def _self_shadowing_capabilities(geometry, settings: RCSettings) -> dict:
    """Gate geometry that can occlude its own facades before irradiance calculation."""
    affected = []
    if len(geometry.parts) > 1:
        affected.append("multiple building parts")
    for part in geometry.parts:
        if part.footprint.holes:
            affected.append(f"courtyard in part {part.identifier}")
        exterior = Polygon(part.footprint.exterior)
        if not exterior.equals(exterior.convex_hull):
            affected.append(f"concave footprint in part {part.identifier}")
    if affected and settings.self_shadowing_policy == "reject":
        raise ValueError(
            f"epsm_rc does not model self-shadowing for {', '.join(affected)}. "
            "Select self_shadowing_policy='ignore' in RCSettings to acknowledge its omission."
        )
    return {
        "policy": settings.self_shadowing_policy,
        "omitted": bool(affected),
        "affected_geometry": affected,
        "geometry_occlusion_applied": False,
    }


def _implementation_identity() -> dict:
    """Return detached calculation identity captured lazily once per process."""
    return implementation_identity()


@dataclass
class _BuildingPreparation:
    """Owned inputs shared by all domains of one captured building."""

    building: object
    spec: object
    resolved: object
    weather: object
    settings: RCSettings
    cache: PreparationCache | None
    implementation: dict
    physical_hash: str
    values: dict
    axis: np.ndarray | None = None
    data: dict | None = None

    def prepared(self, namespace, inputs, compute, *, readonly=False):
        key = (namespace, json.dumps(inputs, sort_keys=True, allow_nan=False), readonly)
        if key not in self.values:
            if self.cache is None:
                value = compute()
            elif readonly:
                value = self.cache._get_or_compute_readonly(namespace, inputs, compute)
            else:
                value = self.cache.get_or_compute(namespace, inputs, compute)
            self.values[key] = value
        return self.values[key]


def _validate_rc_spec(spec):
    """Reject unsupported topology and delivery before weather/numeric preparation."""
    if (
        spec.common_use
        or spec.common_airflow
        or (not spec.geometry.basement_conditioned and spec.zone_uses)
        or any(exchange.zone_kind != "attic" for exchange in spec.unconditioned_exchange)
    ):
        raise ValueError("epsm_rc currently requires one common operating/airflow domain")
    if spec.delivery.fvp or spec.renewables.pv_systems:
        raise ValueError("epsm_rc adapter does not yet support FVP or PV delivery")


def _rc_readiness_engine(building, weather):
    """Retain older Core releases' supported physical handoff for context inputs.

    Older Core reports a blanket context restriction because RC did not yet
    implement shading. RC now owns that capability. Remove only that exact
    obsolete restriction, requiring all remaining missing/unsupported checks,
    then use Core's generic physical resolution. New Core keeps its RC route.
    """
    if building.context.buildings:
        report = building.readiness(engine="epsm_rc", weather=weather)
        obsolete = "epsm_rc does not model explicit context shading"
        if obsolete in report.unsupported:
            replace(
                report,
                unsupported=tuple(issue for issue in report.unsupported if issue != obsolete),
            ).require()
            return None
    return "epsm_rc"


def _settings(settings):
    if settings is None:
        raise ValueError("epsm_rc requires explicit settings or a named reduction policy")
    settings = RCSettings(**settings) if isinstance(settings, dict) else settings
    if not isinstance(settings, RCSettings):
        raise TypeError("settings must be RCSettings or an explicit settings dictionary")
    return settings


def _authoring_checks(building, settings):
    """Keep settings-dependent authoring gates shared by inspection and execution."""
    if building.context.buildings and settings.context_shading_method == "reject":
        yield "rc_context", "settings.context_shading_method", lambda: _reject_context()
    if building.hvac is not None:
        yield (
            "rc_airflow",
            "settings.airflow_policy",
            lambda: _airflow_capabilities(building.hvac.airflow, settings),
        )
        if building.hvac.delivery.fvp or building.hvac.renewables.pv_systems:
            yield "rc_delivery", "hvac", lambda: _reject_delivery()
    if building.geometry is not None:
        yield (
            "rc_self_shadowing",
            "settings.self_shadowing_policy",
            lambda: _self_shadowing_capabilities(building.geometry, settings),
        )


def _reject_context():
    raise ValueError("Context buildings require context_shading_method='iso13790_hourly_horizon'")


def _reject_delivery():
    raise ValueError("epsm_rc adapter does not yet support FVP or PV delivery")


def preflight(building, *, weather=None, settings=None):
    """Inspect physical inputs and explicit RC policies without running a solver.

    Parameters
    ----------
    building : epsm_core.Building
        Authoring inputs to inspect; they are not changed.
    weather : epsm_core.Weather or path-like, optional
        Explicit annual EPW. A path is inspected; a Weather object's file
        checksum/calendar is verified. Missing weather remains a diagnostic.
    settings : RCSettings or dict, optional
        Explicit versioned reduction policies. Missing/invalid choices produce
        diagnostics instead of silently choosing approximations.

    Returns
    -------
    dict
        ``ready``, ``engine``, structured ``diagnostics`` (code, field, message,
        severity), and effective ``settings``. A ready report means these
        input/capability checks passed; it does not assert scientific accuracy
        or successful engine output. No thermal model or solver is initialized.

    Examples
    --------
    >>> from epsm_core import Building
    >>> preflight(Building())["ready"]
    False
    """
    from epsm_core import Building, Weather
    from epsm_core.resolved import resolve_snapshot_inputs

    diagnostics = []

    def issue(code, field, message):
        diagnostics.append(dict(code=code, field=field, message=str(message), severity="error"))

    def check(code, field, operation):
        try:
            return operation()
        except (ValueError, TypeError, OSError) as error:
            issue(code, field, error)
            return None

    if not isinstance(building, Building):
        issue("invalid_input", "building", "building must be an epsm_core.Building")
        return dict(ready=False, engine="epsm_rc", diagnostics=diagnostics, settings=None)
    settings = check("rc_settings", "settings", lambda: _settings(settings))
    report = building.readiness(engine="epsm_rc", weather=weather)
    for field in report.missing:
        issue("missing_input", field, f"Required input is missing: {field}")
    for message in report.unsupported:
        # RC owns the newer context capability while retaining older Core support.
        if (
            building.context.buildings
            and message == "epsm_rc does not model explicit context shading"
        ):
            continue
        issue("unsupported_input", "building", message)
    if weather is None:
        issue("missing_input", "weather", "Explicit annual weather is required")
    elif not isinstance(weather, Weather):
        weather = check("weather", "weather", lambda: Weather.from_epw(weather))
    if weather is not None:
        check("weather", "weather", weather.validate)
    if settings is not None:
        for code, field, operation in _authoring_checks(building, settings):
            check(code, field, operation)
    physical_issues = any(
        row["code"] in {"missing_input", "unsupported_input", "weather"} for row in diagnostics
    )
    if settings is not None and not physical_issues:
        inputs = check(
            "physical_inputs",
            "building",
            lambda: resolve_snapshot_inputs(
                building.snapshot(), weather, engine=_rc_readiness_engine(building, weather)
            ),
        )
        if inputs is not None:
            spec = inputs.spec
            check("rc_topology", "geometry", lambda: _validate_rc_spec(spec))
            check(
                "rc_ground",
                "settings.ground_policy",
                lambda: _ground_reduction(
                    spec.envelope,
                    settings,
                    building=inputs.building,
                    spec=spec,
                    resolved=inputs.resolved,
                ),
            )
            check(
                "rc_gains",
                "operation",
                lambda: _gain_inputs(
                    spec.use,
                    spec.load_reference_area_m2 or spec.geometry.conditioned_floor_area_m2,
                    spec.climate.calendar_year,
                    settings,
                ),
            )
            from .attic import reduce_attics

            check(
                "rc_attic",
                "settings.cold_attic_policy",
                lambda: reduce_attics(
                    inputs.building, spec, weather, settings, resolved=inputs.resolved
                ),
            )
            if spec.geometry.basement_conditioned:
                from .domains import select_domains

                check(
                    "rc_domains",
                    "zone_uses",
                    lambda: select_domains(inputs.building, inputs.resolved, settings),
                )
    return dict(
        ready=not diagnostics,
        engine="epsm_rc",
        diagnostics=diagnostics,
        settings=settings.effective_configuration() if settings is not None else None,
    )


def _prepare_building(building, weather, settings, preparation_cache):
    """Capture and check a building once, with fresh state reserved for each domain."""
    from epsm_core import Building, Weather
    from epsm_core.resolved import resolve_snapshot_inputs
    from epsm_core.weather import hourly_axis, validate_epw_calendar

    settings = _settings(settings)
    if preparation_cache is not None and not isinstance(preparation_cache, PreparationCache):
        raise TypeError("preparation_cache must be a PreparationCache")
    if not isinstance(building, Building):
        raise TypeError("building must be an epsm_core.Building")
    for _, _, operation in _authoring_checks(building, settings):
        operation()
    snapshot = building.snapshot()
    inputs = resolve_snapshot_inputs(
        snapshot,
        weather,
        engine=_rc_readiness_engine(building, weather),
        geometry_cache=preparation_cache.geometry_cache if preparation_cache is not None else None,
    )
    weather = weather if isinstance(weather, Weather) else Weather(inputs.spec.climate)
    # Bind calendar validation and parsed arrays to precisely these captured bytes.
    # A later edit cannot alter the inputs used by this run.
    raw_weather = weather.path.read_bytes()
    if hashlib.sha256(raw_weather).hexdigest() != weather.sha256:
        raise ValueError("EPW contents changed after Weather was created")
    weather_text = raw_weather.decode("utf-8-sig")
    _validate_rc_spec(inputs.spec)
    preparation = _BuildingPreparation(
        inputs.building,
        inputs.spec,
        inputs.resolved,
        weather,
        settings,
        preparation_cache,
        _implementation_identity(),
        inputs.resolved.source_spec_hash,
        {},
    )
    key = {"weather_sha256": weather.sha256, "year": weather.calendar_year}
    preparation.prepared(
        "weather_calendar",
        key,
        lambda: validate_epw_calendar(StringIO(weather_text), weather.calendar_year),
    )

    def weather_arrays():
        frame, _ = read_epw(StringIO(weather_text), coerce_year=weather.calendar_year)
        data = {
            name: frame[name].to_numpy() for name in ("temp_air", "wind_speed", "dni", "ghi", "dhi")
        }
        for name, values in data.items():
            if not np.isfinite(values).all():
                raise ValueError(f"EPW {name} must be finite")
            if name in {"dni", "ghi", "dhi"} and ((values < 0) | (values >= 9999)).any():
                raise ValueError(f"EPW {name} must be complete and nonnegative")
        return data

    preparation.data = preparation.prepared("weather_data", key, weather_arrays, readonly=True)
    preparation.axis = hourly_axis(weather.calendar_year)
    logger.debug(
        "Prepared one captured building",
        extra={
            "epsm_details": {
                "domains": 2 if inputs.spec.geometry.basement_conditioned else 1,
                "hours": len(preparation.axis),
                "physical_inputs_sha256": preparation.physical_hash,
            }
        },
    )
    return preparation


def _prepare_radiation(preparation, model, elements, attic, areas, optical_models):
    """Prepare solar and sky forcing using the building's captured weather buffers."""
    from epsm_core.geometry import facade_catalog

    spec, settings = preparation.spec, preparation.settings
    uplift = 1 + spec.envelope.thermal_bridge_uplift_fraction
    year = spec.climate.calendar_year
    data, axis = preparation.data, preparation.axis
    prepared = preparation.prepared
    preparation_key = {
        "weather_sha256": preparation.weather.sha256,
        "year": year,
        "climate": spec.climate.model_dump(mode="json"),
        "implementation": preparation.implementation,
        "solar_time_reference": settings.solar_time_reference,
    }
    ghi = data["ghi"]
    if not np.isfinite(ghi).all() or ((ghi < 0) | (ghi >= 9999)).any():
        raise ValueError("EPW global horizontal radiation must be complete and nonnegative")
    axis = preparation.axis
    times = pd.DatetimeIndex(axis).tz_localize(
        timezone(timedelta(hours=spec.climate.time_zone_hours))
    )

    sun_times = (
        times + pd.Timedelta(minutes=30)
        if settings.solar_time_reference == "interval_midpoint"
        else times
    )
    if settings.window_optics_method == "simple_glazing_angular_v1":
        for name in ("dni", "dhi"):
            values = data[name]
            if not np.isfinite(values).all() or ((values < 0) | (values >= 9999)).any():
                raise ValueError(f"EPW {name} must be complete and nonnegative for angular optics")

    def solar_geometry():
        position = solarposition.get_solarposition(
            sun_times,
            spec.climate.latitude_deg,
            spec.climate.longitude_deg,
            spec.climate.elevation_m,
        )
        return (
            position.apparent_zenith.to_numpy(),
            position.azimuth.to_numpy(),
            np.asarray(irradiance.get_extra_radiation(sun_times)),
            np.asarray(atmosphere.get_relative_airmass(position.apparent_zenith)),
        )

    zenith, sun_azimuth, extra, air_mass = prepared(
        "solar_geometry", preparation_key, solar_geometry, readonly=True
    )
    dni, dhi = data["dni"], data["dhi"]
    facades = {f.facade_id: f for f in facade_catalog(spec.geometry)}
    horizons, context_shading = (
        {},
        {
            "method": settings.context_shading_method,
            "applied": bool(spec.context.buildings),
            "standard": "ISO 13790:2008 11.4.4 and Annex G.5.4.2, equation G.3",
            "obstacle_count": len(spec.context.buildings),
            "horizon_azimuth_step_deg": settings.horizon_azimuth_step_deg,
            "receiving_point": "area_weighted_resolved_face_or_pane_centroid_per_physical_element",
            "horizon_average": "uniform_azimuth_midpoints_over_facade_front_half_plane",
            "horizontal_surface_extension": "full_360_degree_average_horizon",
            "inclined_surface_extension": "front_azimuth_half_plane_average_horizon",
            "diffuse_and_ground_reflection": "unchanged_ISO_horizon_approximation",
            "self_shadowing_included": False,
            "horizon_angles_deg": {},
        },
    )
    if spec.context.buildings:
        from .shading import context_edges, prepare_horizons, receiver_points

        orientations = {}
        for element in elements:
            role = element.role.value
            if element.identifier in attic.shell_element_ids or element.boundary != "Outdoors":
                continue
            if role in {"ROOF", "TOP_FLOOR"}:
                # Anchor full-horizon quadrature to the scene so rotating the
                # entire scene does not change its numerical sampling error.
                orientations[element.identifier] = (0, spec.geometry.orientation_deg)
            elif role == "WINDOW" or role.startswith("EXT_WALL"):
                orientations[element.identifier] = (90, facades[element.facade_id].azimuth_deg)
        receivers = receiver_points(preparation.resolved, orientations, attic.shell_terms)
        horizon_key = {
            "receivers": receivers,
            "context": spec.context.model_dump(mode="json", exclude={"generation"}),
            "origin_m": list(preparation.resolved.geometry["origin_m"]),
            "orientation_deg": spec.geometry.orientation_deg,
            "step_deg": settings.horizon_azimuth_step_deg,
            "method": settings.context_shading_method,
            "implementation": preparation.implementation,
        }
        horizons = prepared(
            "context_horizon",
            horizon_key,
            lambda: prepare_horizons(
                receivers,
                context_edges(
                    spec.context,
                    preparation.resolved.geometry["origin_m"],
                    spec.geometry.orientation_deg,
                ),
                settings.horizon_azimuth_step_deg,
            ),
        )
        context_shading["horizon_angles_deg"] = dict(horizons)

    def shaded(channels, identifier):
        if not horizons:
            return channels
        from .shading import shade_irradiance

        return shade_irradiance(channels, 90 - zenith, horizons[identifier])

    solar, opaque = np.zeros(len(axis)), np.zeros(len(axis))
    orientation_irradiance = {}
    from ._results import ANGULAR_WINDOW_OUTPUTS

    radiation_hourly = (
        {name: np.zeros(len(axis)) for name in ANGULAR_WINDOW_OUTPUTS}
        if settings.window_optics_method == "simple_glazing_angular_v1"
        else {}
    )
    for element in elements:
        if element.identifier in attic.shell_element_ids:
            continue
        role = element.role.value
        if element.boundary != "Outdoors":
            continue
        if role not in {"WINDOW", "ROOF", "TOP_FLOOR"} and not role.startswith("EXT_WALL"):
            continue
        roof = role in {"ROOF", "TOP_FLOOR"}
        azimuth = 180 if roof else facades[element.facade_id].azimuth_deg
        orientation = (0 if roof else 90, azimuth)
        if orientation in orientation_irradiance:
            channels = orientation_irradiance[orientation]
        else:
            radiation_key = {
                **preparation_key,
                "tilt": orientation[0],
                "azimuth": azimuth,
                "albedo": settings.ground_reflectance,
            }

            def calculate_plane(tilt=orientation[0], direction=azimuth):
                return plane_irradiance(
                    tilt,
                    direction,
                    zenith,
                    sun_azimuth,
                    dni,
                    ghi,
                    dhi,
                    dni_extra=extra,
                    airmass=air_mass,
                    albedo=settings.ground_reflectance,
                )

            channels = prepared("plane_irradiance", radiation_key, calculate_plane, readonly=True)
            orientation_irradiance[orientation] = channels
        channels = shaded(channels, element.identifier)
        poa = channels["poa_global"]
        if role == "WINDOW":
            if settings.window_optics_method == "constant_shgc":
                solar += (
                    poa
                    * element.area_m2
                    * element.construction.shgc
                    * spec.facade.solar_shading_factor
                )
            else:
                optics = optical_models[(element.construction.u_value, element.construction.shgc)]
                # At night solar geometry remains finite and incident beam is zero.
                incidence = irradiance.aoi(orientation[0], azimuth, zenith, sun_azimuth)
                bt, _, ba = optics.beam_properties(incidence)
                dt, _, da = optics.diffuse_properties()
                beam = channels["poa_direct"]
                sky, reflected = channels["poa_sky_diffuse"], channels["poa_ground_diffuse"]
                diffuse = sky + reflected
                area, shade = element.area_m2, spec.facade.solar_shading_factor
                transmitted = (beam * bt + diffuse * dt) * area * shade
                absorbed = (beam * ba + diffuse * da) * optics.inward_fraction * area * shade
                solar += transmitted + absorbed
                radiation_hourly["window_incident_beam_w"] += beam * area
                radiation_hourly["window_incident_sky_diffuse_w"] += sky * area
                radiation_hourly["window_incident_ground_diffuse_w"] += reflected * area
                radiation_hourly["window_transmitted_solar_w"] += transmitted
                radiation_hourly["window_inward_absorbed_solar_w"] += absorbed
        else:
            outside = element.construction.components[0].layers[0].layer
            alpha = sum(m.area_fraction * m.material.solar_absorptance for m in outside.materials)
            outside_resistance = (
                settings.outside_roof_resistance_m2_k_w
                if roof
                else settings.outside_wall_resistance_m2_k_w
            )
            opaque += (
                poa
                * element.area_m2
                * element.construction.u_value
                * alpha
                * outside_resistance
                * uplift
            )
    for term in attic.shell_terms:
        orientation = (term["tilt_deg"], term["azimuth_deg"])
        channels = orientation_irradiance.get(orientation)
        if channels is None:
            radiation_key = {
                **preparation_key,
                "tilt": orientation[0],
                "azimuth": orientation[1],
                "albedo": settings.ground_reflectance,
            }
            channels = prepared(
                "plane_irradiance",
                radiation_key,
                lambda tilt=orientation[0], direction=orientation[1]: plane_irradiance(
                    tilt,
                    direction,
                    zenith,
                    sun_azimuth,
                    dni,
                    ghi,
                    dhi,
                    dni_extra=extra,
                    airmass=air_mass,
                    albedo=settings.ground_reflectance,
                ),
            )
            orientation_irradiance[orientation] = channels
        channels = shaded(channels, term["surface_name"])
        opaque += (
            channels["poa_global"] * term["solar_coefficient_m2"] * term["transmission_fraction"]
        )
    if radiation_hourly:
        radiation_hourly["window_solar_heat_gain_w"] = solar.copy()
        radiation_hourly["opaque_solar_heat_gain_w"] = opaque.copy()
        radiation_hourly["sky_radiation_loss_w"] = np.full(
            len(axis), model.calc_sky_radiation_loss()
        )
    return solar, opaque, radiation_hourly, times, context_shading


def _prepare_domain(preparation, *, domain=None):
    """Reduce one domain from the already checked building without repeating I/O."""

    building, spec, weather, settings = (
        preparation.building,
        preparation.spec,
        preparation.weather,
        preparation.settings,
    )
    operating_use = domain.use if domain is not None else spec.use
    operating_airflow = domain.airflow if domain is not None else spec.airflow
    operating_reference = (
        domain.reference_area_m2
        if domain is not None
        else spec.load_reference_area_m2 or spec.geometry.conditioned_floor_area_m2
    )
    gains, convective_gains, radiant_gains, gain_reduction = _gain_inputs(
        operating_use,
        operating_reference,
        spec.climate.calendar_year,
        settings,
    )
    airflow_capabilities = _airflow_capabilities(operating_airflow, settings)
    ground, ground_multiplier, ground_reduction = _ground_reduction(
        spec.envelope,
        settings,
        building=building,
        spec=spec,
        resolved=preparation.resolved,
        domain=domain,
        outdoor_temperatures=preparation.data["temp_air"],
        time_axis=preparation.axis,
    )
    self_shadowing = _self_shadowing_capabilities(spec.geometry, settings)
    effective_configuration = settings.effective_configuration()
    effective_configuration["gain_reduction"] = gain_reduction
    if not gain_reduction["aggregate_split_applied"]:
        effective_configuration["inactive_coefficients"] = ["convective_internal_gains_fraction"]
        effective_configuration["methods"]["occupant_gains"] = "supplied_sensible_fraction"
        effective_configuration["methods"]["gain_partition"] = "shared_hourly_sensible_channels"
    implementation = preparation.implementation
    run_id = hashlib.sha256(
        json.dumps(
            {
                "physical_inputs_sha256": preparation.physical_hash,
                "settings": effective_configuration,
                "implementation": implementation,
                **({"domain": domain.identity()} if domain is not None else {}),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    bind_context(run_id=run_id)
    year = spec.climate.calendar_year
    area = domain.area_m2 if domain is not None else spec.geometry.conditioned_floor_area_m2
    volume = domain.volume_m3 if domain is not None else spec.geometry.conditioned_volume_m3
    reference_area = operating_reference
    elements = domain.elements if domain is not None else tuple(building.elements)
    from .attic import reduce_attics

    attic = reduce_attics(
        building,
        spec,
        weather,
        settings,
        resolved=preparation.resolved,
        enabled=domain is None or domain.name == "main",
    )
    optical_models = {}
    if settings.window_optics_method == "simple_glazing_angular_v1":
        for element in elements:
            if (
                element.role.value == "WINDOW"
                and element.boundary == "Outdoors"
                and element.area_m2 > 0
            ):
                glass = element.construction
                key = (glass.u_value, glass.shgc)
                if key not in optical_models:
                    optical_models[key] = SimpleGlazingOptics(*key)
    grouped = {"wall": [], "roof": [], "ground": [], "window": []}
    constructions = []
    for element in elements:
        if element.identifier in attic.shell_element_ids:
            continue
        role = element.role.value
        key = (
            "ground"
            if element.boundary == "Ground" and hasattr(element.construction, "components")
            else "wall"
            if role.startswith("EXT_WALL") or role == "BASEMENT_WALL"
            else "roof"
            if role in {"ROOF", "TOP_FLOOR"}
            else "ground"
            if role in {"GROUND_FLOOR", "BASEMENT_FLOOR"}
            else "window"
            if role == "WINDOW"
            else None
        )
        # The role selects the construction category; the boundary selects the
        # heat-transfer path. A shared facade keeps its physical construction
        # below, including mass, without becoming an outdoor wall in the RC model.
        if key and element.boundary == ("Ground" if key == "ground" else "Outdoors"):
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
                element.area_m2
                * (
                    domain.capacity_weights.get(element.identifier, 1.0)
                    if domain is not None
                    else 1.0
                ),
                settings.effective_mass_depth_m,
            )
        )
    areas = {key: sum(e.area_m2 for e in values) for key, values in grouped.items()}

    iso_ground = ground_reduction["effective_method"] == "iso13370_periodic_approximation"
    if iso_ground:
        ground_u = (
            ground_reduction["steady_conductance_w_k"] / areas["ground"] if areas["ground"] else 0.0
        )
    else:
        ground_u = None

    def u_value(key):
        return (
            sum(e.area_m2 * e.construction.u_value for e in grouped[key]) / areas[key]
            if areas[key]
            else 0.0
        )

    use, airflow = operating_use, operating_airflow

    def annual(schedule):
        return np.asarray(schedule.hourly(year), dtype=float)

    heat_set, cool_set = annual(use.heating_setpoint), annual(use.cooling_setpoint)
    capacity = sum(c.total_thermal_mass_j_k for c in constructions)
    if not constructions or not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("RC reduction requires positive thermal mass from explicit constructions")
    capacity_per_area = capacity / area
    lower, upper = settings.mass_capacity_thresholds_j_m2_k
    mass_area_factor = settings.mass_area_factors[
        2 if capacity_per_area >= upper else 1 if capacity_per_area >= lower else 0
    ]
    model = Building5R1C(
        floor_area_m2=area,
        building_class=None,
        constructions=constructions,
        u_walls=u_value("wall"),
        u_roof=(
            u_value("roof") * areas["roof"] / (areas["roof"] + attic.ceiling_area_m2)
            if areas["roof"] + attic.ceiling_area_m2
            else 0.0
        ),
        u_floor=ground_u if iso_ground else u_value("ground"),
        u_windows=u_value("window"),
        wall_area_m2=areas["wall"],
        roof_area_m2=areas["roof"] + attic.ceiling_area_m2,
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
        temp_adj_ground=ground_multiplier,
        ground_temps_monthly=list(ground) if ground is not None else None,
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
    if attic.ceiling_area_m2:
        model.u_roof += attic.equivalent_conductance_w_k / model.roof_area_m2
    if not iso_ground:
        model.u_floor *= uplift
    model._prepare_deterministic_hourly()
    # The finite attic node receives actual shell solar/sky forcing. A fictitious
    # exposed ceiling must not also receive the flat-roof sky term.
    if attic.ceiling_area_m2:
        model._prepared_sky = (
            settings.outside_wall_resistance_m2_k_w
            * model.u_walls
            * areas["wall"]
            * settings.sky_radiation_coefficient_w_m2_k
            * settings.sky_air_temperature_difference_k
            * settings.wall_sky_view_factor
            + settings.outside_roof_resistance_m2_k_w
            * u_value("roof")
            * uplift
            * areas["roof"]
            * settings.sky_radiation_coefficient_w_m2_k
            * settings.sky_air_temperature_difference_k
            * settings.roof_sky_view_factor
            + sum(term["sky_loss_w"] * term["transmission_fraction"] for term in attic.shell_terms)
        )
    data = preparation.data
    solar, opaque, radiation_hourly, times, context_shading = _prepare_radiation(
        preparation, model, elements, attic, areas, optical_models
    )
    axis = preparation.axis
    infiltration = (
        airflow.infiltration_design_flow_m3_s(reference_area_m2=reference_area, volume_m3=volume)
        * 3600
        / volume
        * annual(airflow.infiltration_fraction)
    )
    coefficients = airflow.infiltration_coefficients
    wind = data["wind_speed"]
    ventilation = (
        airflow.mechanical_vent_l_s_m2
        * reference_area
        * 3.6
        / volume
        * annual(airflow.mechanical_vent_fraction)
    )
    natural = airflow.natural_exchange_ach * annual(airflow.natural_exchange_fraction)
    temperatures = data["temp_air"]
    heating_on, cooling_on = (
        annual(airflow.heating_availability),
        annual(airflow.cooling_availability),
    )
    model.t_m = model.t_air = model.t_s = float(heat_set[0])
    if ground:
        month_offsets = np.array(
            (0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016),
            dtype=np.int64,
        )
        month_hours = np.ascontiguousarray(
            month_offsets[times.month.to_numpy(dtype=np.int64) - 1], dtype=np.int64
        )
    else:
        month_hours = None

    from functools import partial

    from ._results import finish_domain

    finish = partial(
        finish_domain,
        airflow=airflow,
        airflow_capabilities=airflow_capabilities,
        areas=areas,
        attic=attic,
        axis=axis,
        convective_gains=convective_gains,
        domain=domain,
        effective_configuration=effective_configuration,
        elements=elements,
        gain_reduction=gain_reduction,
        ground_multiplier=ground_multiplier,
        ground_reduction=ground_reduction,
        grouped=grouped,
        implementation=implementation,
        model=model,
        optical_models=optical_models,
        physical_hash=preparation.physical_hash,
        radiant_gains=radiant_gains,
        radiation_hourly=radiation_hourly,
        reference_area=reference_area,
        run_id=run_id,
        self_shadowing=self_shadowing,
        context_shading=context_shading,
        settings=settings,
        spec=spec,
        use=use,
    )

    from .domains import HourlyInputs, PreparedDomain, _immutable_columns, _immutable_series

    forcing = HourlyInputs(
        model,
        *_immutable_columns(
            temperatures,
            gains,
            convective_gains,
            heat_set,
            cool_set,
            wind,
            infiltration,
            natural,
            ventilation,
            solar,
            opaque,
        ),
        _immutable_series(month_hours, np.int64) if month_hours is not None else None,
        coefficients,
        gain_reduction["aggregate_split_applied"],
    )
    heating_limits_w = np.where(heating_on != 0.0, np.inf, 0.0)
    cooling_limits_w = (
        np.where(cooling_on != 0.0, -np.inf, 0.0)
        if use.cooling_enabled
        else np.zeros(len(cooling_on), dtype=np.float64)
    )
    heating_limits_w = _immutable_series(heating_limits_w)
    cooling_limits_w = _immutable_series(cooling_limits_w)
    return PreparedDomain(
        domain.name if domain is not None else "aggregate",
        model,
        forcing.at,
        finish,
        heating_limits_w,
        cooling_limits_w,
        axis,
        reference_area,
        volume,
        forcing=forcing,
    )


def _run_domain(domain):
    """Advance a fresh prepared domain using the same forcing as coupled runs."""
    from .iso13790 import _can_use_native

    if domain.forcing is None or not _can_use_native(
        domain.model, needs_demand=True, prepared=True
    ):
        power, indoor = np.zeros(len(domain.time_axis)), np.zeros(len(domain.time_axis))
        calculate, advance = domain.model._calculate_demand, domain.model._step
        for hour in range(len(power)):
            power[hour] = calculate(
                **domain.input_at(hour),
                max_heating_power=domain.heating_limits_w[hour],
                max_cooling_power=domain.cooling_limits_w[hour],
                advance=advance,
            )
            indoor[hour] = domain.model.t_air
        return domain.finish(power, indoor)

    from ._core_rs import run_single_domain_hourly

    f = domain.forcing
    power, indoor = run_single_domain_hourly(
        domain.model,
        f.temperatures,
        f.gains,
        f.convective,
        f.heating_setpoints,
        f.cooling_setpoints,
        f.wind,
        f.infiltration,
        f.natural,
        f.ventilation,
        f.solar,
        f.opaque,
        f.month_hours,
        tuple(float(c) for c in f.coefficients),
        bool(f.aggregate_split),
        np.ascontiguousarray(domain.heating_limits_w, dtype=np.float64),
        np.ascontiguousarray(domain.cooling_limits_w, dtype=np.float64),
    )
    return domain.finish(power, indoor)


@logged_operation("energy.rc")
def simulate_building(building, *, weather, directory=None, settings=None, preparation_cache=None):
    """Calculate one full year from a Core physical Building.

    Parameters
    ----------
    building : epsm_core.Building
        Physical elements, constructions, HVAC and explicit operating schedules.
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
    preparation_cache : PreparationCache or None
        Explicit bounded preparation reuse across independent runs. Weather bytes are
        verified per building; each run creates fresh thermal state. Source identity
        is captured lazily once per process. Restart after installed code changes.
        ``batch()`` remains a compatibility context manager for this same cache.
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
    versions. Installed Python sources are read lazily once per process.
    Restart after changing installed source or numerical dependencies.
    Each call creates fresh thermal state. Buildings without basements retain
    one conditioned 5R1C domain. Conditioned basements use separately controlled
    main/basement domains with conservative physical floor exchange. Cold attics
    require an explicit finite quasi-steady buffer policy and outdoor exchange;
    attic shell/air storage is omitted. All paths model sensible heat only. Heat recovery reduces
    mechanical conductance once; infiltration is never recovered. No weather
    padding, stochastic window opening or legacy construction presets are used.
    Standalone correctness fixes and migration details are recorded in the release notes.

    Examples
    --------
    Supply a complete local EPW; install this package's ``platform`` extra.

    The bundled Göteborg EPW is a teaching input, not measured-building validation.

    >>> from importlib.resources import files
    >>> from epsm_core import Building, Weather
    >>> from epsm_rc import RCSettings, simulate_building
    >>> weather = Weather.from_epw(files("epsm_rc") / "data/epw/goteborg.epw")
    >>> result = simulate_building(
    ...     Building(preset="illustrative-reference"),
    ...     weather=weather,
    ...     settings=RCSettings.from_policy(
    ...         "epsm_rc_5r1c_v1", airflow_policy="sensible_only_unlimited_supply",
    ...         ground_policy="conductance_multiplier"))
    >>> result.scalars["space_heating_demand_kwh"] >= 0
    True
    """
    from epsm_core.specs.schedules import hourly_schedule_scope

    with hourly_schedule_scope():
        preparation = _prepare_building(building, weather, settings, preparation_cache)
        if preparation.spec.geometry.basement_conditioned:
            from .domains import simulate_conditioned_domains

            result = simulate_conditioned_domains(preparation)
        else:
            result = _run_domain(_prepare_domain(preparation))
    return _write_result(result, directory)


def _write_result(result, directory):
    """Write optional caller-owned output only after all domains have succeeded."""
    if directory is not None:
        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        path = output_directory / "hourly.npz"
        np.savez_compressed(path, **result.hourly, time_axis=result.time_axis)
        result.artifacts["hourly"] = str(path)
    return result
