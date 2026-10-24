# SPDX-License-Identifier: MIT
"""Resolve explicitly scoped ISO 13370 inputs from Core's physical snapshot.

This adapter never changes an installed construction. Core's conductivity-based
bridge convention is applied to material resistance before the soil calculation;
nominal material storage remains the responsibility of the ordinary RC adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite


@dataclass(frozen=True)
class ResolvedGroundInputs:
    kind: str
    area_m2: float
    exposed_perimeter_m: float
    wall_thickness_m: float
    floor_resistance_m2k_w: float
    soil_conductivity_w_mk: float
    soil_volumetric_heat_capacity_j_m3k: float
    basement_depth_m: float
    basement_wall_resistance_m2k_w: float | None
    report: dict

    def coefficient_inputs(self):
        """Return independent keyword inputs for ``ground.ground_coefficients``."""
        return {
            name: getattr(self, name)
            for name in (
                "area_m2", "exposed_perimeter_m", "wall_thickness_m",
                "floor_resistance_m2k_w", "soil_conductivity_w_mk",
                "soil_volumetric_heat_capacity_j_m3k", "basement_depth_m",
                "basement_wall_resistance_m2k_w",
            )
        }


def _resistance(element, uplift):
    from epsm_core.building_adapter import thermal_construction
    from epsm_core.constructions import bridge_equivalent

    nominal = thermal_construction(element.construction)
    applied = bridge_equivalent(nominal, uplift)

    def layer_r(assembly):
        return sum(
            layer.thickness_m / layer.material.conductivity_w_mk for layer in assembly.layers
        )

    original_r, equivalent_r = layer_r(nominal), layer_r(applied)
    return equivalent_r, {
        "physical_element_id": element.identifier,
        "area_m2": element.area_m2,
        "nominal_layer_resistance_m2k_w": original_r,
        "equivalent_layer_resistance_m2k_w": equivalent_r,
        "construction_surface_resistance_m2k_w": nominal.surface_resistance_m2k_w,
        "nominal_reporting_u_w_m2k": nominal.u_value_w_m2k,
        "equivalent_reporting_u_w_m2k": applied.u_value_w_m2k,
        "thermal_bridge_uplift_fraction": uplift,
        "method": "Core bridge_equivalent before soil calculation; nominal storage retained",
    }


def _uniform_resistance(elements, uplift, label):
    values = [_resistance(element, uplift) for element in elements]
    reference = values[0][0]
    if any(not isclose(value, reference, rel_tol=1e-9, abs_tol=1e-12) for value, _ in values):
        raise ValueError(f"ISO13370 currently requires uniform {label} material resistance")
    return reference, [report for _, report in values]


def _floor_polygon(face):
    from shapely.geometry import Polygon

    return Polygon([(point[0], point[1]) for point in face["vertices"]])


def resolve_ground_inputs(building, spec, resolved, settings, *, domain=None):
    """Resolve one slab or fully heated basement, or None for a domain without soil.

    Supported foundations have one simple connected floor, one physical building
    part and uniform floor/buried-wall material resistances. Footing blocks,
    nonzero plinths, extra buried footing-wall depth, groundwater and party-wall
    assumptions are rejected rather than silently discarded. Required wall width
    and its source are independent of Core's footing-block width field.
    """
    selected = domain.elements if domain is not None and hasattr(domain, "elements") else building.elements
    ground = tuple(
        element for element in selected
        if element.boundary == "Ground" and hasattr(element.construction, "components")
    )
    if not ground:
        return None

    foundation = spec.envelope.foundation
    if foundation.method != "kiva" or spec.envelope.ground_temperature is not None:
        raise ValueError("ISO13370 ground policy cannot combine with prescribed ground temperatures")
    if getattr(settings, "monthly_ground_temperatures_c", None) is not None:
        raise ValueError("ISO13370 ground policy cannot add a second monthly soil boundary")
    width = getattr(settings, "iso13370_wall_thickness_m", None)
    source = getattr(settings, "iso13370_wall_thickness_source", None)
    if width is None or not isfinite(width) or width <= 0:
        raise ValueError("ISO13370 requires explicit positive iso13370_wall_thickness_m")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("ISO13370 requires an explicit iso13370_wall_thickness_source")
    if len(spec.geometry.parts) != 1:
        raise ValueError("ISO13370 currently supports one physical foundation part/domain")
    if spec.geometry.party_walls:
        raise ValueError("ISO13370 party-wall foundation exposure is not yet supported")
    if spec.geometry.basement_depth_m and not spec.geometry.basement_conditioned:
        raise ValueError("ISO13370 unconditioned or partly heated basement is not supported")
    if foundation.footing_material is not None or foundation.footing_wall is not None:
        raise ValueError("ISO13370 explicit footing blocks/wall details are not yet supported")
    if foundation.wall_height_above_grade_m or foundation.wall_depth_below_slab_m:
        raise ValueError("ISO13370 nonzero foundation plinth/extra buried wall depth is not supported")
    if foundation.deep_ground_boundary != "ZeroFlux":
        raise ValueError("ISO13370 flowing/deep groundwater treatment is not supported")

    floors = tuple(element for element in ground if element.role.value in {"GROUND_FLOOR", "BASEMENT_FLOOR"})
    walls = tuple(element for element in ground if element.role.value == "BASEMENT_WALL")
    if not floors or len(floors) + len(walls) != len(ground):
        raise ValueError("ISO13370 requires explicit physical soil-contact floor/wall ownership")
    ids = {element.identifier for element in floors}
    faces = [
        face for face in resolved.surfaces
        if face["kind"] == "Floor" and face["boundary"] == "Ground"
        and face["physical_element_id"] in ids
    ]
    if not faces:
        raise ValueError("ISO13370 soil floor is absent from resolved Core geometry")
    if len({face["zone"] for face in faces}) != 1:
        raise ValueError("ISO13370 currently supports one physical foundation part/domain")
    heights = {round(point[2], 8) for face in faces for point in face["vertices"]}
    if len(heights) != 1:
        raise ValueError("ISO13370 ground floor must be coplanar")

    from shapely.geometry import shape
    from shapely.ops import unary_union

    pieces = [_floor_polygon(face) for face in faces]
    floor = unary_union(pieces)
    if floor.geom_type != "Polygon" or floor.interiors:
        raise ValueError("ISO13370 foundation floor cannot contain holes/disconnected pieces")
    area = floor.area
    if not isclose(area, sum(piece.area for piece in pieces), rel_tol=1e-9, abs_tol=1e-7):
        raise ValueError("ISO13370 resolved foundation floors overlap")
    if not isclose(area, sum(element.area_m2 for element in floors), rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("ISO13370 physical/resolved foundation floor areas disagree")
    footprints = [shape(footprint) for footprint in resolved.geometry["local_footprints"]]
    exterior = unary_union(footprints).boundary
    # Match Core Kiva's resolved exterior, without treating zone partitions as exposure.
    geometric_perimeter = min(exterior.intersection(floor.buffer(1e-8)).length, floor.length)
    perimeter = geometric_perimeter * foundation.exposed_perimeter_fraction
    if not isfinite(perimeter) or perimeter <= 0:
        raise ValueError("ISO13370 requires a positive exposed foundation perimeter")
    depth = max(0.0, -next(iter(heights)))
    expected_depth = spec.geometry.basement_depth_m if spec.geometry.basement_conditioned else 0.0
    if not isclose(depth, expected_depth, abs_tol=1e-7):
        raise ValueError("ISO13370 resolved ground depth differs from declared basement depth")

    uplift = spec.envelope.thermal_bridge_uplift_fraction
    floor_r, floor_reports = _uniform_resistance(floors, uplift, "slab")
    wall_r, wall_reports = None, []
    if depth:
        if not walls or not isclose(foundation.exposed_perimeter_fraction, 1.0, abs_tol=1e-12):
            raise ValueError("ISO13370 currently requires a fully exposed, fully heated basement")
        wall_ids = {element.identifier for element in walls}
        wall_faces = [
            face for face in resolved.surfaces
            if face["kind"] == "Wall" and face["boundary"] == "Ground"
            and face["physical_element_id"] in wall_ids
        ]
        if not wall_faces or any(
            not isclose(min(point[2] for point in face["vertices"]), -depth, abs_tol=1e-7)
            or not isclose(max(point[2] for point in face["vertices"]), 0.0, abs_tol=1e-7)
            for face in wall_faces
        ):
            raise ValueError("ISO13370 requires uniform full-depth heated-basement walls")
        wall_area = sum(element.area_m2 for element in walls)
        if not isclose(wall_area, geometric_perimeter * depth, rel_tol=1e-8, abs_tol=1e-5):
            raise ValueError("ISO13370 partly buried/nonuniform basement walls are not supported")
        wall_r, wall_reports = _uniform_resistance(walls, uplift, "basement wall")
    elif walls:
        raise ValueError("ISO13370 slab-on-ground cannot include buried wall elements")

    report = {
        "source": "Core resolved physical ground geometry and nominal material assemblies",
        "kind": "heated_basement" if depth else "slab_on_ground",
        "foundation_method_requested": foundation.method,
        "floor_area_m2": area,
        "geometric_exterior_perimeter_m": geometric_perimeter,
        "exposed_perimeter_fraction": foundation.exposed_perimeter_fraction,
        "exposed_perimeter_m": perimeter,
        "characteristic_dimension_m": area / (0.5 * perimeter),
        "basement_depth_m": depth,
        "buried_wall_area_m2": sum(element.area_m2 for element in walls),
        "wall_thickness_m": width,
        "wall_thickness_source": source,
        "wall_width_not_inferred_from": "footing_width_m or above-grade facade retrofit layers",
        "floor_material_resistances": floor_reports,
        "basement_wall_material_resistances": wall_reports,
        "bridge_treatment": "Core equivalent material resistance applied once before soil; no final ground-H uplift",
        "storage_treatment": "Nominal installed material storage retained by ordinary RC preparation",
        "junction_psi_w_mk": 0.0,
        "junction_scope": "No separately quantified wall/floor linear junction or edge insulation; generic Core material bridge convention retained",
        "dimension_scope": "Core resolved metric floor/exterior footprint dimensions; explicit wall thickness supplied independently",
        "ground_boundary_scope": "ISO soil reduction/periodic approximation; no independent prescribed slab-outside temperature",
        "kiva_details_not_reproduced": [
            "Finite deep/far-field boundaries and multidimensional local surface exchange",
            "Concave-footprint boundary-layer adjustment and explicit footing/edge blocks",
        ],
    }
    return ResolvedGroundInputs(
        report["kind"], area, perimeter, float(width), floor_r,
        foundation.soil_conductivity_w_mk,
        foundation.soil_density_kg_m3 * foundation.soil_specific_heat_j_kgk,
        depth, wall_r, report,
    )
