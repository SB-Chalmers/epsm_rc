# File: src/epsm_rc/_iso52016_platform.py
# Purpose: Reduce Core elements to an ISO 52016-1 zone and run it on the shared RC forcing.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
# License: See LICENSE at the project root.
"""Shared-Building adapter for the ISO 52016-1 hourly zone.

The 5R1C preparation (weather, solar per orientation, context shading, window
optics, gains, airflow and ground reduction) is reused unchanged. Only the
thermal network differs: elements keep their own node chains instead of the
single 5R1C mass node.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .iso52016 import (
    R_SE,
    R_SI,
    SOIL_CONDUCTIVITY_W_M_K,
    SOIL_THICKNESS_M,
    SOIL_VOLUMETRIC_CAPACITY_J_M3_K,
    Zone52016,
    ZoneElement,
    classify_mass,
    simulate_zone,
)

MINIMUM_RESISTANCE_M2_K_W = 0.01
WALL_ROLES = ("EXT_WALL", "BASEMENT_WALL")
ROOF_ROLES = frozenset({"ROOF", "TOP_FLOOR"})
GROUND_ROLES = frozenset({"GROUND_FLOOR", "BASEMENT_FLOOR"})


def _layers(construction):
    """Series (resistance, areal capacity) layers, outside first."""
    layers = []
    for component in construction.components:
        for slot in component.layers:
            layer = slot.layer
            conductivity = sum(m.area_fraction * m.material.conductivity for m in layer.materials)
            capacity = sum(
                m.area_fraction * m.thickness * m.material.density * m.material.specific_heat
                for m in layer.materials
            )
            layers.append((layer.thickness / conductivity, capacity))
    return layers


def _absorptance(construction):
    outside = construction.components[0].layers[0].layer
    return float(sum(m.area_fraction * m.material.solar_absorptance for m in outside.materials))


def _category(element):
    role = element.role.value
    if role == "WINDOW":
        return "window", "horizontal"
    if role.startswith(WALL_ROLES):
        return "wall", "horizontal"
    if role in ROOF_ROLES:
        return "roof", "upwards"
    if role in GROUND_ROLES:
        return "ground_floor", "downwards"
    if "FLOOR" in role or "CEILING" in role:
        return "internal_floor", None
    return "internal_wall", "horizontal"


def reduce_elements(elements, attic, settings, ground_reduction, uplifts, element_irradiance):
    """Return zone elements, per-group hourly irradiance (W/m²) and bridge conductance."""
    method = ground_reduction["effective_method"]
    multiplier = float(ground_reduction.get("effective_multiplier", 1.0))
    ground_area = sum(
        e.area_m2
        for e in elements
        if e.boundary == "Ground"
        and e.identifier not in attic.shell_element_ids
        and hasattr(e.construction, "components")
    )
    groups = {}
    irradiance = defaultdict(lambda: 0.0)
    bridge = 0.0
    notes = []

    def add(key, element_kwargs, area, poa=None):
        if key in groups:
            previous = groups[key]
            groups[key] = dict(previous, area_m2=previous["area_m2"] + area)
        else:
            groups[key] = dict(element_kwargs, area_m2=area)
        if poa is not None:
            irradiance[key] = irradiance[key] + poa * area

    for element in elements:
        if element.identifier in attic.shell_element_ids or element.area_m2 <= 0:
            continue
        category, direction = _category(element)
        construction = element.construction
        boundary = element.boundary
        if category == "window":
            if boundary != "Outdoors":
                continue
            u = float(construction.u_value)
            resistance = max(1.0 / u - R_SI["horizontal"] - R_SE, MINIMUM_RESISTANCE_M2_K_W)
            facade = element.facade_id
            key = ("window", facade, round(resistance, 9))
            add(
                key,
                dict(
                    name=f"window:{facade}",
                    kind="window",
                    boundary="exterior",
                    direction="horizontal",
                    resistance_m2_k_w=resistance,
                    capacity_j_m2_k=0.0,
                    mass_class="D",
                    sky_view_factor=settings.wall_sky_view_factor,
                ),
                element.area_m2,
            )
            continue
        if not hasattr(construction, "components"):
            continue
        layers = _layers(construction)
        resistance = max(sum(r for r, _ in layers), MINIMUM_RESISTANCE_M2_K_W)
        capacity = sum(k for _, k in layers)
        if boundary == "Outdoors" and category in {"wall", "roof"}:
            roof = category == "roof"
            mass_class = classify_mass(layers)
            alpha = _absorptance(construction)
            poa = element_irradiance.get(element.identifier)
            facade = "roof" if roof else element.facade_id
            key = (
                "opaque",
                category,
                facade,
                round(resistance, 9),
                round(capacity, 3),
                mass_class,
                alpha,
            )
            add(
                key,
                dict(
                    name=f"{category}:{facade}",
                    kind="opaque",
                    boundary="exterior",
                    direction=direction,
                    resistance_m2_k_w=resistance,
                    capacity_j_m2_k=capacity,
                    mass_class=mass_class,
                    solar_absorptance=alpha,
                    sky_view_factor=(
                        settings.roof_sky_view_factor if roof else settings.wall_sky_view_factor
                    ),
                ),
                element.area_m2,
                poa if poa is not None else 0.0,
            )
            bridge += uplifts["top" if roof else "wall"] * element.area_m2 * construction.u_value
            continue
        if boundary == "Ground" and category in {"ground_floor", "wall"}:
            direction = direction if category == "ground_floor" else "horizontal"
            layers_g = list(layers)
            if method == "iso13370_periodic_approximation":
                # Clause 6.5.8: 0.5 m soil layer plus a virtual layer so the
                # steady path equals the ISO 13370 conductance (reading of
                # ISO 13370 Annex F: construction, not R_fl;eff, in the chain).
                soil = (
                    SOIL_THICKNESS_M / SOIL_CONDUCTIVITY_W_M_K,
                    SOIL_THICKNESS_M * SOIL_VOLUMETRIC_CAPACITY_J_M3_K,
                )
                layers_g = [soil] + layers_g
                u_eff = ground_reduction["steady_conductance_w_k"] / ground_area
                chain = R_SI[direction] + sum(r for r, _ in layers_g)
                virtual = 1.0 / u_eff - chain
                if virtual < MINIMUM_RESISTANCE_M2_K_W:
                    notes.append("iso13370_virtual_layer_clamped")
                    virtual = MINIMUM_RESISTANCE_M2_K_W
                boundary_kind = "ground_monthly"
            else:
                # Prescribed monthly boundary (b = 1) or outdoor air scaled by b.
                # Core's ground bridge uplift scales this element's own path, as
                # in 5R1C, so it reaches the same boundary temperature.
                b = 1.0 if method == "prescribed_monthly_temperature" else multiplier
                if b <= 0:
                    boundary_kind = "adiabatic"
                    virtual = None
                else:
                    u = 1.0 / (R_SI[direction] + resistance + R_SE)
                    target = b * u * (1 + uplifts["ground"])
                    virtual = 1.0 / target - R_SI[direction] - resistance
                    if virtual < MINIMUM_RESISTANCE_M2_K_W:
                        notes.append("ground_boundary_resistance_clamped")
                        virtual = MINIMUM_RESISTANCE_M2_K_W
                    boundary_kind = (
                        "ground_monthly"
                        if method == "prescribed_monthly_temperature"
                        else "ground_outdoor"
                    )
            resistance_g = max(sum(r for r, _ in layers_g), MINIMUM_RESISTANCE_M2_K_W)
            capacity_g = sum(k for _, k in layers_g)
            mass_class = classify_mass(layers_g)
            key = (
                "ground",
                category,
                boundary_kind,
                round(resistance_g, 9),
                round(capacity_g, 3),
                mass_class,
                None if virtual is None else round(virtual, 9),
            )
            add(
                key,
                dict(
                    name=f"ground:{category}:{boundary_kind}",
                    kind="opaque",
                    boundary="adiabatic" if boundary_kind == "adiabatic" else "ground",
                    direction=direction,
                    resistance_m2_k_w=resistance_g,
                    capacity_j_m2_k=capacity_g,
                    mass_class=mass_class,
                    boundary_resistance_m2_k_w=virtual,
                ),
                element.area_m2,
            )
            continue
        # Elements inside the conditioned zone (intermediate floors, partitions)
        # and shared walls: adiabatic. An element with both faces in the zone is
        # split at its mid-plane into two halves (Clause 6.5.6.3, internal elements).
        if boundary == "Internal" or category in {"internal_floor", "internal_wall"}:
            faces = (
                (("downwards", "floor-face"), ("upwards", "ceiling-face"))
                if category == "internal_floor"
                else (("horizontal", "face-a"), ("horizontal", "face-b"))
            )
            for face_direction, face in faces:
                key = ("internal", category, face, round(resistance, 9), round(capacity, 3))
                add(
                    key,
                    dict(
                        name=f"internal:{category}:{face}",
                        kind="opaque",
                        boundary="adiabatic",
                        direction=face_direction,
                        resistance_m2_k_w=max(resistance / 2, MINIMUM_RESISTANCE_M2_K_W),
                        capacity_j_m2_k=capacity / 2,
                        mass_class="D",
                    ),
                    element.area_m2,
                )
            continue
        mass_class = classify_mass(layers)
        key = ("adiabatic", category, round(resistance, 9), round(capacity, 3), mass_class)
        add(
            key,
            dict(
                name=f"adiabatic:{category}",
                kind="opaque",
                boundary="adiabatic",
                direction=direction or "horizontal",
                resistance_m2_k_w=resistance,
                capacity_j_m2_k=capacity,
                mass_class=mass_class,
            ),
            element.area_m2,
        )
    keys = list(groups)
    zone_elements = tuple(ZoneElement(**groups[key]) for key in keys)
    ground_kinds = tuple(key[2] if key[0] == "ground" else None for key in keys)
    hourly = []
    for key, element in zip(keys, zone_elements):
        value = irradiance.get(key, 0.0)
        hourly.append(np.asarray(value, float) / element.area_m2 if np.ndim(value) else None)
    return zone_elements, hourly, ground_kinds, bridge, notes


def build_zone(domain_inputs):
    """Create the zone and hourly closure from a prepared 5R1C domain reduction."""
    (elements, attic, settings, ground_reduction, uplifts, element_irradiance, area, volume) = (
        domain_inputs
    )
    zone_elements, hourly, ground_kinds, bridge, notes = reduce_elements(
        elements, attic, settings, ground_reduction, uplifts, element_irradiance
    )
    zone = Zone52016(
        zone_elements,
        floor_area_m2=area,
        air_volume_m3=volume,
        thermal_bridge_w_k=bridge,
        air_volumetric_heat_capacity_j_m3_k=settings.air_volumetric_heat_capacity_j_m3_k,
        control=settings.iso52016_control_temperature,
        delta_theta_er_k=settings.sky_air_temperature_difference_k,
    )
    return zone, hourly, ground_kinds, notes


def _native_kernel():
    try:
        from ._core_rs import run_iso52016_hourly
    except ImportError:
        return None
    return run_iso52016_hourly


def run_zone(
    domain, zone, hourly, ground_kinds, months, monthly_ground, warmup_passes=1, native=None
):
    """Run the zone hour by hour from the shared 5R1C forcing (airflow, gains, setpoints).

    The Rust kernel is used when installed (``native=None``); ``native=False``
    selects the Python recurrence, which defines the same arithmetic.
    """
    forcing = domain.forcing
    model = forcing.model
    count = len(domain.time_axis)
    temperatures = forcing.temperatures
    absorbed = np.zeros((count, len(zone.elements)))
    for index, (element, series) in enumerate(zip(zone.elements, hourly)):
        if series is not None and element.kind == "opaque" and element.boundary == "exterior":
            absorbed[:, index] = series
    # One boundary column serves all ground groups. Mixed kinds use separate
    # meanings, so reject that combination rather than averaging it silently.
    kinds = {kind for kind in ground_kinds if kind in {"ground_monthly", "ground_outdoor"}}
    if len(kinds) > 1:
        raise ValueError("ISO 52016-1 adapter needs one ground boundary kind per zone")
    if "ground_monthly" in kinds:
        ground = np.asarray(monthly_ground, float)[np.asarray(months) - 1]
    else:
        ground = np.asarray(temperatures, float)
    heating_limits, cooling_limits = domain.heating_limits_w, domain.cooling_limits_w
    kernel = _native_kernel() if native is not False else None
    if native and kernel is None:
        raise ImportError("The native ISO 52016-1 kernel is not installed")
    if kernel is not None:
        return _run_native(
            kernel,
            zone,
            forcing,
            model,
            absorbed,
            ground,
            heating_limits,
            cooling_limits,
            warmup_passes,
        )

    class Hours:
        def __len__(self):
            return count

        def __call__(self, hour, air, last_power=0.0):
            if air is not None:
                model.t_air = air
            model._last_hvac_power = last_power
            values = forcing.at(hour)
            vector = zone.forcing_vector(
                values["t_out"],
                ground[hour],
                values["internal_gains"],
                values["solar_gains"],
                absorbed[hour],
            )
            return (
                vector,
                model._h_ve,
                model.t_set_heating,
                model.t_set_cooling,
                heating_limits[hour],
                cooling_limits[hour],
            )

    power, air, operative, _ = simulate_zone(zone, Hours(), warmup_passes=warmup_passes)
    return power, air, operative


def _run_native(
    kernel, zone, forcing, model, absorbed, ground, heating_limits, cooling_limits, warmup
):
    """Pass contiguous network and hourly arrays to the Rust recurrence."""

    def column(values):
        return np.ascontiguousarray(values, dtype=np.float64).ravel()

    built = zone._built
    network = {
        name: column(built[name])
        for name in ("m", "mp", "m_air", "m_hvac", "c_dt", "control", "areas")
    }
    network["inner"] = column(built["inner"])
    hourly = {
        "theta_e": column(forcing.temperatures),
        "theta_gr": column(ground),
        "phi_int": column(forcing.gains),
        "phi_sol": column(forcing.solar),
        "absorbed": column(absorbed),
        "set_heating": column(forcing.heating_setpoints),
        "set_cooling": column(forcing.cooling_setpoints),
        "heat_limit": column(heating_limits),
        "cool_limit": column(cooling_limits),
        "wind": column(forcing.wind),
        "infiltration": column(forcing.infiltration),
        "natural": column(forcing.natural),
        "ventilation": column(forcing.ventilation),
    }
    recovery = (
        forcing.recovery_effectiveness
        if forcing.recovery_bypass
        else float(model.heat_recovery_efficiency)
    )
    return kernel(
        network,
        hourly,
        tuple(float(c) for c in forcing.coefficients),
        float(model._building_vol),
        float(model.air_volumetric_heat_capacity_j_m3_k),
        float(recovery),
        bool(forcing.recovery_bypass),
        int(warmup),
    )
