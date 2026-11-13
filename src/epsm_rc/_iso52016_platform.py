# File: _iso52016_platform.py
# Purpose: Reduce Core elements to an ISO 52016-1 zone and run it on the shared RC forcing.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-12

"""Shared-Building adapter for the ISO 52016-1 hourly multi-zone network.

The 5R1C preparation (weather, solar per orientation, context shading, window
optics, gains, airflow and ground reduction) is reused unchanged for each
conditioned domain. Only the thermal network differs: elements keep their own
node chains instead of the single 5R1C mass node, and the zones of one
building form one network.

Notes
-----
Results add ``attic_air_temperature_c`` and, with a basement,
``interzone_heat_to_main_w``/``interzone_heat_to_basement_w``. The Rust kernel
(``run_iso52016_network``) runs the same recurrence when installed.

Zones
-----
- ``main``: the conditioned aggregate above grade.
- ``basement``: a conditioned basement, when declared. The separating floor is
  one element whose upper surface node faces ``main`` and whose lower surface
  node faces ``basement``; each zone keeps its own gains, airflow and thermostat.
- one zone per cold attic, unconditioned, with the declared outdoor exchange.
  Its roof and gables carry per-face solar and sky forcing; the ceiling joins
  it to ``main``.

Examples
--------
The adapter is internal; ``simulate_building`` selects it with the default
``thermal_model="iso52016_1"``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .iso52016 import (
    R_SE,
    R_SI,
    SOIL_CONDUCTIVITY_W_M_K,
    SOIL_THICKNESS_M,
    SOIL_VOLUMETRIC_CAPACITY_J_M3_K,
    Network52016,
    ThermalZone,
    ZoneElement,
    classify_mass,
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


def reduce_elements(
    elements,
    attic,
    settings,
    ground_reduction,
    uplifts,
    element_irradiance,
    *,
    zone=0,
    interfaces=None,
    skip=frozenset(),
):
    """Return zone elements, per-group hourly irradiance (W/m²) and bridge conductance.

    ``interfaces`` maps element identifiers to ``(adjacent_zone, direction,
    adjacent_direction)``; those elements join ``zone`` to another zone. The
    construction's outside layer faces the adjacent zone. Identifiers in
    ``skip`` belong to another zone and are left out.
    """
    interfaces = interfaces or {}
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
        element_kwargs = dict(element_kwargs, zone=zone)
        if key in groups:
            previous = groups[key]
            groups[key] = dict(previous, area_m2=previous["area_m2"] + area)
        else:
            groups[key] = dict(element_kwargs, area_m2=area)
        if poa is not None:
            irradiance[key] = irradiance[key] + poa * area

    for element in elements:
        if (
            element.identifier in attic.shell_element_ids
            or element.identifier in skip
            or element.area_m2 <= 0
        ):
            continue
        category, direction = _category(element)
        construction = element.construction
        if element.identifier in interfaces and hasattr(construction, "components"):
            adjacent, inner_direction, outer_direction = interfaces[element.identifier]
            layers = _layers(construction)
            resistance = max(sum(r for r, _ in layers), MINIMUM_RESISTANCE_M2_K_W)
            capacity = sum(k for _, k in layers)
            mass_class = classify_mass(layers)
            key = (
                "interzone",
                category,
                adjacent,
                inner_direction,
                outer_direction,
                round(resistance, 9),
                round(capacity, 3),
                mass_class,
            )
            add(
                key,
                dict(
                    name=f"interzone:{category}:{adjacent}",
                    kind="opaque",
                    boundary="zone",
                    direction=inner_direction,
                    resistance_m2_k_w=resistance,
                    capacity_j_m2_k=capacity,
                    mass_class=mass_class,
                    zone=zone,
                    adjacent_zone=adjacent,
                    adjacent_direction=outer_direction,
                ),
                element.area_m2,
            )
            continue
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


@dataclass(frozen=True)
class ZonePart:
    """Prepared inputs of one conditioned domain, kept until the network is built."""

    name: str
    elements: tuple
    attic: object
    settings: object
    ground_reduction: dict
    uplifts: dict
    element_irradiance: dict
    floor_area_m2: float
    volume_m3: float
    months: np.ndarray
    monthly_ground: object


@dataclass
class NetworkRun:
    """Hourly results of a network run; arrays are hours × zones.

    ``exchanges`` holds, per conditioned domain, the heat conducted through the
    floor between main and basement (W, positive into the domain), or None.
    """

    network: Network52016
    power: np.ndarray
    air: np.ndarray
    operative: np.ndarray
    exchanges: np.ndarray | None
    attic_names: tuple
    notes: tuple

    def attic_hourly(self):
        """Attic air temperatures (°C) keyed ``attic_air_temperature_c`` (or numbered)."""
        first = self.network.zone_count - len(self.attic_names)
        single = len(self.attic_names) == 1
        return {
            (
                "attic_air_temperature_c" if single else f"attic_{offset}_air_temperature_c"
            ): self.air[:, first + offset].copy()
            for offset in range(len(self.attic_names))
        }


def _attic_elements(attic_zone, index, by_id, element_irradiance):
    """Group one attic's shell faces and storage into zone elements with irradiance."""
    groups, irradiance = {}, defaultdict(lambda: 0.0)
    for face in attic_zone["faces"]:
        owner = by_id[face["physical_element_id"]]
        layers = _layers(owner.construction)
        resistance = max(sum(r for r, _ in layers), MINIMUM_RESISTANCE_M2_K_W)
        capacity = sum(k for _, k in layers)
        mass_class = classify_mass(layers)
        roof = face["is_roof"]
        key = (
            "attic",
            "roof" if roof else "wall",
            round(face["tilt_deg"], 6),
            round(face["azimuth_deg"], 6),
            round(resistance, 9),
            round(capacity, 3),
            mass_class,
            face["solar_absorptance"],
            round(face["sky_view_factor"], 9),
        )
        area = face["net_area_m2"]
        previous = groups.get(key)
        groups[key] = dict(
            name=f"attic:{'roof' if roof else 'wall'}:{face['azimuth_deg']:.0f}",
            kind="opaque",
            boundary="exterior",
            direction="upwards" if roof else "horizontal",
            resistance_m2_k_w=resistance,
            capacity_j_m2_k=capacity,
            mass_class=mass_class,
            solar_absorptance=face["solar_absorptance"],
            sky_view_factor=face["sky_view_factor"],
            zone=index,
            area_m2=area + (previous["area_m2"] if previous else 0.0),
        )
        irradiance[key] = (
            irradiance[key] + element_irradiance[("attic", face["surface_name"])] * area
        )
    for identifier in attic_zone["storage_elements"]:
        owner = by_id.get(identifier)
        if owner is None or not hasattr(owner.construction, "components"):
            continue
        layers = _layers(owner.construction)
        resistance = max(sum(r for r, _ in layers), MINIMUM_RESISTANCE_M2_K_W)
        capacity = sum(k for _, k in layers)
        for face in ("face-a", "face-b"):
            key = ("attic-internal", face, round(resistance, 9), round(capacity, 3))
            previous = groups.get(key)
            groups[key] = dict(
                name=f"attic:internal:{face}",
                kind="opaque",
                boundary="adiabatic",
                direction="horizontal",
                resistance_m2_k_w=max(resistance / 2, MINIMUM_RESISTANCE_M2_K_W),
                capacity_j_m2_k=capacity / 2,
                mass_class="D",
                zone=index,
                area_m2=owner.area_m2 + (previous["area_m2"] if previous else 0.0),
            )
    elements = [ZoneElement(**groups[key]) for key in groups]
    hourly = [
        np.asarray(irradiance[key], float) / element.area_m2 if key in irradiance else None
        for key, element in zip(groups, elements)
    ]
    return elements, hourly


def build_network(parts):
    """Assemble one network from the main part, an optional basement part and attics.

    Returns the network, per-element hourly irradiance, per-zone ground kinds,
    the attic zones and notes.
    """
    main = parts[0]
    settings = main.settings
    by_id = {element.identifier: element for part in parts for element in part.elements}
    interfaces, skip = {}, frozenset()
    if len(parts) == 2:
        shared = {e.identifier for e in main.elements} & {e.identifier for e in parts[1].elements}
        # Separating floor: upper face in main, lower face in the basement.
        interfaces.update({identifier: (1, "downwards", "downwards") for identifier in shared})
        skip = frozenset(shared)
    attic_zones = tuple(main.attic.zones)
    zone_bridges = []
    for offset, attic_zone in enumerate(attic_zones):
        index = len(parts) + offset
        bridge = 0.0
        for identifier, area in attic_zone["ceilings"]:
            # Ceiling: lower face in main (heat flow up), upper face in the attic.
            interfaces[identifier] = (index, "upwards", "upwards")
            bridge += main.uplifts["top"] * area * by_id[identifier].construction.u_value
        if bridge > 0:
            zone_bridges.append((0, index, bridge))
    elements, hourly, kinds, zones, notes = [], [], [], [], []
    for index, part in enumerate(parts):
        part_elements, part_hourly, ground_kinds, bridge, part_notes = reduce_elements(
            part.elements,
            part.attic,
            settings,
            part.ground_reduction,
            part.uplifts,
            part.element_irradiance,
            zone=index,
            interfaces=interfaces if index == 0 else None,
            skip=frozenset() if index == 0 else skip,
        )
        elements += part_elements
        hourly += part_hourly
        notes += part_notes
        part_kinds = {kind for kind in ground_kinds if kind in {"ground_monthly", "ground_outdoor"}}
        if len(part_kinds) > 1:
            raise ValueError("ISO 52016-1 adapter needs one ground boundary kind per zone")
        kinds.append(next(iter(part_kinds), None))
        zones.append(
            ThermalZone(
                part.name,
                part.floor_area_m2,
                part.volume_m3,
                thermal_bridge_w_k=bridge,
                control=settings.control_temperature.value,
            )
        )
    for offset, attic_zone in enumerate(attic_zones):
        index = len(parts) + offset
        attic_elements, attic_hourly = _attic_elements(
            attic_zone, index, by_id, main.element_irradiance
        )
        elements += attic_elements
        hourly += attic_hourly
        kinds.append(None)
        zones.append(
            ThermalZone(
                f"attic:{attic_zone['name']}",
                sum(area for _, area in attic_zone["ceilings"]),
                attic_zone["volume_m3"],
                conditioned=False,
                internal_capacity_j_m2_k=0.0,
                control="air",
            )
        )
    network = Network52016(
        elements,
        zones,
        zone_bridges=zone_bridges,
        air_volumetric_heat_capacity_j_m3_k=settings.air_volumetric_heat_capacity_j_m3_k,
        delta_theta_er_k=settings.sky_air_temperature_difference_k,
    )
    return network, hourly, kinds, attic_zones, tuple(notes)


def _native_kernel():
    try:
        from ._core_rs import run_iso52016_network
    except ImportError:
        return None
    return run_iso52016_network


def _forcing_matrix(network, domains, hourly, kinds, parts):
    """Hourly forcing in the network column order (hours × columns)."""
    first = domains[0].forcing
    count = len(first.temperatures)
    z = network.zone_count
    matrix = np.zeros((count, network.column_count))
    matrix[:, 0] = first.temperatures
    matrix[:, 1] = 1.0
    for index, (domain, part, kind) in enumerate(zip(domains, parts, kinds)):
        forcing = domain.forcing
        if kind == "ground_monthly":
            ground = np.asarray(part.monthly_ground, float)[np.asarray(part.months) - 1]
        else:
            ground = np.asarray(forcing.temperatures, float)
        matrix[:, 2 + 3 * index] = ground
        matrix[:, 3 + 3 * index] = forcing.gains
        matrix[:, 4 + 3 * index] = forcing.solar
    for index, (element, series) in enumerate(zip(network.elements, hourly)):
        if series is not None and element.kind == "opaque" and element.boundary == "exterior":
            matrix[:, 2 + 3 * z + index] = series
    return matrix


def _probes(network, count_domains):
    """Links (node, node, W/K) summed into the floor exchange towards main."""
    links = []
    built = network._built
    for index, element in enumerate(network.elements):
        if element.boundary == "zone" and element.adjacent_zone < count_domains:
            start = built["outer"][index]
            h = element.internal_conductances()[1] * element.area_m2
            links.append((start + 1, start + 2, h))  # middle link, towards the inner face
    return links


def run_network(domains, *, warmup_passes=1, native=None):
    """Build and run the network for prepared domains (main first, then basement).

    The Rust kernel is used when installed (``native=None``); ``native=False``
    selects the Python recurrence, which defines the same arithmetic.
    """
    parts = [domain.iso52016 for domain in domains]
    network, hourly, kinds, attic_zones, notes = build_network(parts)
    d = len(domains)
    matrix = _forcing_matrix(network, domains, hourly, kinds, parts)
    capacity = parts[0].settings.air_volumetric_heat_capacity_j_m3_k
    attic_h_ve = [
        capacity * attic["outdoor_air_ach"] * attic["fraction"] * attic["volume_m3"] / 3600.0
        for attic in attic_zones
    ]
    links = _probes(network, d)
    kernel = _native_kernel() if native is not False else None
    if native and kernel is None:
        raise ImportError("The native ISO 52016-1 kernel is not installed")
    if kernel is not None:
        power, air, operative, probe = _run_native(
            kernel, network, domains, matrix, attic_h_ve, links, warmup_passes
        )
    else:
        power, air, operative, probe = _run_python(
            network, domains, matrix, attic_h_ve, links, warmup_passes
        )
    exchanges = None
    if d == 2:
        exchanges = np.vstack([probe, -probe])
    return NetworkRun(
        network,
        power,
        air,
        operative,
        exchanges,
        tuple(zone.name for zone in network.zones[d:]),
        notes,
    )


def _run_python(network, domains, matrix, attic_h_ve, links, warmup_passes):
    count, z, d = matrix.shape[0], network.zone_count, len(domains)
    nodes_a = np.array([a for a, _, _ in links], dtype=int)
    nodes_b = np.array([b for _, b, _ in links], dtype=int)
    conductance = np.array([h for _, _, h in links])
    zero = np.zeros(count)
    limits = [(domain.heating_limits_w, domain.cooling_limits_w) for domain in domains]
    limits += [(zero, zero)] * (z - d)
    state, last = None, np.zeros(z)
    for _ in range(warmup_passes + 1):
        power, air, operative = (np.zeros((count, z)) for _ in range(3))
        probe = np.zeros(count)
        for hour in range(count):
            h_ve = np.zeros(z)
            set_h, set_c = np.zeros(z), np.zeros(z)
            for index, domain in enumerate(domains):
                model = domain.forcing.model
                if state is not None:
                    model.t_air = float(state[index])
                model._last_hvac_power = float(last[index])
                domain.forcing.at(hour)
                h_ve[index] = model._h_ve
                set_h[index], set_c[index] = model.t_set_heating, model.t_set_cooling
            set_h[d:], set_c[d:] = set_h[0], set_c[0]
            for offset, series in enumerate(attic_h_ve):
                h_ve[d + offset] = series[hour]
            if state is None:
                state = network.initial_state(set_h[0])
            state, power[hour], air[hour], operative[hour] = network.step(
                state,
                matrix[hour],
                h_ve,
                set_h,
                set_c,
                np.array([limit[0][hour] for limit in limits]),
                np.array([limit[1][hour] for limit in limits]),
            )
            probe[hour] = float(conductance @ (state[nodes_a] - state[nodes_b]))
            last = power[hour]
    return power, air, operative, probe


def _run_native(kernel, network, domains, matrix, attic_h_ve, links, warmup):
    """Pass contiguous network, forcing and per-zone arrays to the Rust recurrence."""

    def column(values):
        return np.ascontiguousarray(values, dtype=np.float64).ravel()

    built = network._built
    payload = {
        name: column(built[name])
        for name in ("m", "mp", "m_air", "m_hvac", "c_dt", "control", "operative")
    }
    payload["forcing"] = column(matrix)
    payload["links"] = column([value for link in links for value in link])
    count = matrix.shape[0]
    zones = []
    for domain in domains:
        forcing, model = domain.forcing, domain.forcing.model
        recovery = (
            forcing.recovery_effectiveness
            if forcing.recovery_bypass
            else float(model.heat_recovery_efficiency)
        )
        zones.append(
            {
                "set_heating": column(forcing.heating_setpoints),
                "set_cooling": column(forcing.cooling_setpoints),
                "heat_limit": column(domain.heating_limits_w),
                "cool_limit": column(domain.cooling_limits_w),
                "wind": column(forcing.wind),
                "infiltration": column(forcing.infiltration),
                "natural": column(forcing.natural),
                "ventilation": column(forcing.ventilation),
                "coefficients": column(forcing.coefficients),
                "volume_m3": float(model._building_vol),
                "air_capacity_j_m3_k": float(model.air_volumetric_heat_capacity_j_m3_k),
                "recovery_effectiveness": float(recovery),
                "recovery_bypass": bool(forcing.recovery_bypass),
            }
        )
    first = zones[0]
    for series in attic_h_ve:
        zones.append(
            {
                "set_heating": first["set_heating"],
                "set_cooling": first["set_cooling"],
                "heat_limit": column(np.zeros(count)),
                "cool_limit": column(np.zeros(count)),
                "h_ve": column(series),
            }
        )
    power, air, operative, probe = kernel(payload, zones, int(warmup))
    z = network.zone_count
    return (
        np.asarray(power).reshape(count, z),
        np.asarray(air).reshape(count, z),
        np.asarray(operative).reshape(count, z),
        np.asarray(probe),
    )


def attach_attic_results(result, run):
    """Add attic air temperatures (°C) to a result's hourly outputs."""
    hourly = run.attic_hourly()
    result.hourly.update(hourly)
    result.units.update({key: "C" for key in hourly})
    return result
