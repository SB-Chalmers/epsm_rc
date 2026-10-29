# SPDX-License-Identifier: MIT
"""Finite, zero-capacity attic nodes reduced from Core's resolved physical shell.

Core owns all areas, ownership, geometry and declared outdoor exchange. This
module eliminates an unconditioned attic air node in steady state; the retained
ceiling capacity belongs to conditioned space. Attic air/shell storage is omitted
explicitly, rather than assigning the attic outdoor temperature.
"""

from dataclasses import dataclass
from math import acos, atan2, cos, degrees, radians

import numpy as np


@dataclass(frozen=True)
class AtticReduction:
    shell_element_ids: frozenset
    ceiling_element_ids: frozenset
    equivalent_conductance_w_k: float
    ceiling_area_m2: float
    shell_terms: tuple
    diagnostics: dict


def eliminate_attic_node(ceiling_w_k, shell_w_k, ventilation_w_k):
    """Return series conductance and shell-forcing transmission fraction.

    For ceiling Hc, outdoor shell Hs and ventilation Hv, the eliminated node
    obeys Hc(Tc-Ta) + (Hs+Hv)(To-Ta) + Qshell = 0. Therefore the conditioned
    loss is Heq(Tc-To) - fraction*Qshell. No outdoor-air rate is inferred.
    """
    values = (ceiling_w_k, shell_w_k, ventilation_w_k)
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("Attic conductances must be finite and nonnegative")
    total = sum(values)
    if total <= 0:
        raise ValueError("Attic requires positive finite conductance")
    fraction = ceiling_w_k / total
    return fraction * (shell_w_k + ventilation_w_k), fraction


def _orientation(vertices):
    points = np.asarray(vertices, dtype=float)
    normal = np.zeros(3)
    for index in range(1, len(points) - 1):
        normal += np.cross(points[index] - points[0], points[index + 1] - points[0])
    length = np.linalg.norm(normal)
    if length <= 0:
        raise ValueError("Attic shell has a degenerate surface normal")
    normal /= length
    return degrees(acos(float(np.clip(normal[2], -1, 1)))), degrees(
        atan2(normal[0], normal[1])
    ) % 360


def reduce_attics(
    building, spec, weather, settings, *, geometry_cache=None, enabled=True, resolved=None
):
    """Resolve explicit cold attics, rejecting omitted exchange or unsupported topology."""
    from epsm_core.bridges import thermal_bridge_uplifts

    parts = [
        part
        for part in spec.geometry.parts
        if spec.geometry.roof_for(part).topology == "cold_attic"
    ]
    if not parts or not enabled:
        return AtticReduction(
            frozenset(),
            frozenset(),
            0.0,
            0.0,
            (),
            {"effective_method": "not_required", "attics": []},
        )
    if settings.cold_attic_policy != "quasi_steady_buffer":
        raise ValueError("Cold attics require explicit cold_attic_policy='quasi_steady_buffer'")
    exchanges = [
        exchange for exchange in spec.unconditioned_exchange if exchange.zone_kind == "attic"
    ]
    if len(exchanges) != 1:
        raise ValueError("Cold attic reduction requires explicit attic outdoor exchange and source")
    exchange = exchanges[0]
    fractions = exchange.fraction.hourly(spec.climate.calendar_year)
    if min(fractions) != max(fractions):
        raise ValueError(
            "Quasi-steady attic reduction currently requires constant exchange fraction"
        )
    actual_ach = exchange.outdoor_air_ach * fractions[0]
    if resolved is None:
        resolved = building.snapshot().resolve(weather, geometry_cache=geometry_cache)
    elements = {element.identifier: element for element in building.elements}
    surfaces = resolved.surfaces
    zones = resolved.zones
    shell_ids, ceiling_ids, terms, reports = set(), set(), [], []
    equivalent, area = 0.0, 0.0
    for name, zone in zones.items():
        if zone.get("physical_zone_kind") != "attic":
            continue
        if zone["conditioned"]:
            raise ValueError("Cold attic unexpectedly belongs to conditioned space")
        local_faces = [face for face in surfaces if face["zone"] == name]
        if any(
            face["boundary"] == "Surface"
            and not (
                face["kind"] == "Floor"
                and face["construction_role"].removesuffix("_reverse") == "top"
            )
            for face in local_faces
        ):
            raise ValueError(
                "Quasi-steady attic reduction does not support vertical or inter-attic interfaces"
            )
        if any(face["kind"] == "Floor" and face["boundary"] != "Surface" for face in local_faces):
            raise ValueError("Quasi-steady attic reduction does not support exposed attic floors")
        shell = [face for face in local_faces if face["boundary"] == "Outdoors"]
        shell_ids.update(
            face["physical_element_id"]
            for face in surfaces
            if face["zone"] == name
            and face["kind"] in {"Roof", "Wall"}
            and face["physical_element_id"] is not None
        )
        # Whole-building zoning may represent an installed attic partition as
        # internal mass rather than a pair of mesh faces. Its storage belongs to
        # the omitted unconditioned shell, just like attic walls with surfaces.
        shell_ids.update(
            item["physical_element_id"]
            for item in resolved.internal_mass
            if item["zone"] == name and item["physical_element_id"] is not None
        )
        ceilings = [
            face
            for face in surfaces
            if face["zone"] == name
            and face["boundary"] == "Surface"
            and face["construction_role"].removesuffix("_reverse") == "top"
        ]
        hc, hs, local_area = 0.0, 0.0, 0.0
        top_uplift = thermal_bridge_uplifts(spec)["top"] if ceilings else 0.0
        for face in ceilings:
            owner = elements.get(face["physical_element_id"])
            if owner is None or owner.role.value != "TOP_FLOOR":
                raise ValueError("Attic ceiling requires an explicit physical TOP_FLOOR owner")
            ceiling_ids.add(owner.identifier)
            hc += face["net_area_m2"] * owner.construction.u_value * (1 + top_uplift)
            local_area += face["net_area_m2"]
        local_terms = []
        for face in shell:
            owner = elements.get(face["physical_element_id"])
            if owner is None:
                raise ValueError("Attic shell requires an explicit physical construction owner")
            if face["opening_area_m2"]:
                raise ValueError("Quasi-steady attic reduction does not support attic glazing")
            shell_ids.add(owner.identifier)
            tilt, azimuth = _orientation(face["vertices"])
            is_roof = face["kind"] == "Roof"
            resistance = (
                settings.outside_roof_resistance_m2_k_w
                if is_roof
                else settings.outside_wall_resistance_m2_k_w
            )
            sky_view = (
                settings.roof_sky_view_factor * (1 + cos(radians(tilt))) / 2
                if is_roof
                else settings.wall_sky_view_factor
            )
            ua = face["net_area_m2"] * owner.construction.u_value
            outside = owner.construction.components[0].layers[0].layer
            alpha = sum(
                share.area_fraction * share.material.solar_absorptance
                for share in outside.materials
            )
            hs += ua
            local_terms.append(
                {
                    "tilt_deg": tilt,
                    "azimuth_deg": azimuth,
                    "solar_coefficient_m2": ua * resistance * alpha,
                    "sky_loss_w": ua
                    * resistance
                    * settings.sky_radiation_coefficient_w_m2_k
                    * settings.sky_air_temperature_difference_k
                    * sky_view,
                    "physical_element_id": owner.identifier,
                    "surface_name": face["name"],
                    "receiver_vertices": face["vertices"],
                }
            )
        if not ceilings or not shell or hc <= 0 or hs <= 0:
            raise ValueError("Cold attic requires a complete ceiling and outdoor weather shell")
        hv = settings.air_volumetric_heat_capacity_j_m3_k * actual_ach * zone["volume_m3"] / 3600
        heq, fraction = eliminate_attic_node(hc, hs, hv)
        for term in local_terms:
            terms.append({**term, "transmission_fraction": fraction})
        equivalent += heq
        area += local_area
        reports.append(
            {
                "zone": name,
                "volume_m3": zone["volume_m3"],
                "ceiling_area_m2": local_area,
                "ceiling_conductance_w_k": hc,
                "shell_conductance_w_k": hs,
                "ventilation_conductance_w_k": hv,
                "equivalent_conductance_w_k": heq,
                "shell_forcing_transmission_fraction": fraction,
            }
        )
    if not reports:
        raise ValueError("Cold attic geometry did not resolve into an unconditioned attic domain")
    return AtticReduction(
        frozenset(shell_ids),
        frozenset(ceiling_ids),
        equivalent,
        area,
        tuple(terms),
        {
            "effective_method": "quasi_steady_buffer",
            "exchange": exchange.model_dump(mode="json"),
            "effective_outdoor_air_ach": actual_ach,
            "attics": reports,
            "shell_storage": "omitted",
            "air_storage": "omitted",
            "ceiling_storage": "retained_on_conditioned_side",
            "solar_and_sky": "finite_node_shell_forcing_transmitted_to_conditioned_domain",
        },
    )
