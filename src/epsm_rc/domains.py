# SPDX-License-Identifier: MIT
"""Prepare and combine coupled main/conditioned-basement 5R1C domains.

Domain use, airflow, reference areas and schedules come from Core's public
resolved snapshot. A physical internal floor is installed once: its existing
room-side active capacity is shared equally between the two reduced domains.
The layer resistance plus declared combined surface resistance connects air nodes.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np

from ._logging import bind_context


@dataclass(frozen=True)
class DomainSelection:
    name: str
    area_m2: float
    volume_m3: float
    reference_area_m2: float
    use: object
    airflow: object
    elements: tuple
    capacity_weights: dict

    def identity(self):
        return {
            "name": self.name,
            "area_m2": self.area_m2,
            "volume_m3": self.volume_m3,
            "reference_area_m2": self.reference_area_m2,
            "use": self.use.model_dump(mode="json"),
            "airflow": self.airflow.model_dump(mode="json"),
            "capacity_weights": self.capacity_weights,
        }


@dataclass
class PreparedDomain:
    name: str
    model: object
    input_at: Callable
    finish: Callable
    heating_limits_w: object
    cooling_limits_w: object
    time_axis: tuple
    reference_area_m2: float
    volume_m3: float
    forcing: object = None


@dataclass
class HourlyInputs:
    """Owned hourly forcing used by ordinary and coupled numerical execution."""

    model: object
    temperatures: np.ndarray
    gains: np.ndarray
    convective: np.ndarray
    heating_setpoints: np.ndarray
    cooling_setpoints: np.ndarray
    wind: np.ndarray
    infiltration: np.ndarray
    natural: np.ndarray
    ventilation: np.ndarray
    solar: np.ndarray
    opaque: np.ndarray
    month_hours: np.ndarray | None
    coefficients: tuple
    aggregate_split: bool

    def at(self, hour):
        model = self.model
        temperature = float(self.temperatures[hour])
        gain = float(self.gains[hour])
        if not self.aggregate_split:
            model.convective_fraction_internal = (
                float(self.convective[hour]) / gain if gain > 0 else 0.0
            )
        model.t_set_heating = float(self.heating_setpoints[hour])
        model.t_set_cooling = float(self.cooling_setpoints[hour])
        w = float(self.wind[hour])
        correction = (
            self.coefficients[0]
            + self.coefficients[1] * abs(model.t_air - temperature)
            + self.coefficients[2] * w
            + self.coefficients[3] * w**2
        )
        model.ach_infiltration = float(self.infiltration[hour]) * correction + float(
            self.natural[hour]
        )
        model.ach_ventilation = float(self.ventilation[hour])
        if not isfinite(model.ach_infiltration) or model.ach_infiltration < 0:
            raise ValueError("Hourly infiltration must be finite and nonnegative")
        model._update_ventilation_coefficient()
        return {
            "t_out": temperature,
            "solar_gains": float(self.solar[hour]),
            "internal_gains": gain,
            "solar_gains_opaque": float(self.opaque[hour]),
            "hour_of_year": int(self.month_hours[hour]) if self.month_hours is not None else None,
        }


def select_domains(building, resolved, settings):
    """Select exactly main and basement, retaining all declared controls and areas."""
    from epsm_core import AirflowSpec, UseSpec

    grouped = {"main": [], "basement": []}
    for zone in resolved.zones.values():
        if not zone["conditioned"]:
            if zone.get("physical_zone_kind") != "attic":
                raise ValueError("RC supports only cold attics outside its conditioned domains")
            continue
        kind = zone.get("physical_zone_kind")
        if kind not in grouped or zone.get("zone_kind") in {"common", "stairwell", "dwelling"}:
            raise ValueError(
                "Coupled RC requires separately resolved main and basement controls; "
                "use per-storey zoning without apartment/common-space controls"
            )
        grouped[kind].append(zone)
    if any(not values for values in grouped.values()):
        raise ValueError("Coupled RC requires explicit main and conditioned-basement domains")
    interfaces = [
        element
        for element in building.elements
        if element.facade_id is None
        and element.elevation_m == 0
        and element.identifier.endswith(".ground_floor")
        and element.boundary == "Internal"
    ]
    if not interfaces:
        raise ValueError("Conditioned basement requires an explicit separating physical floor")
    basement = [
        element
        for element in building.elements
        if element.role.value == "BASEMENT_WALL" or element.identifier.endswith(".basement_floor")
    ]
    base_ids = {element.identifier for element in basement}
    selected_elements = {
        "main": tuple(e for e in building.elements if e.identifier not in base_ids),
        "basement": tuple(basement + interfaces),
    }
    capacity_weights = {element.identifier: 0.5 for element in interfaces}
    selections = []
    for name, zones in grouped.items():
        uses = [UseSpec.model_validate(zone["operation"]) for zone in zones]
        flows = [AirflowSpec.model_validate(zone["airflow"]) for zone in zones]
        if any(use != uses[0] for use in uses) or any(flow != flows[0] for flow in flows):
            raise ValueError(
                f"Coupled RC currently requires uniform operation and airflow within {name}"
            )
        selections.append(
            DomainSelection(
                name,
                sum(zone["floor_area_m2"] for zone in zones),
                sum(zone["volume_m3"] for zone in zones),
                sum(zone["load_reference_area_m2"] for zone in zones),
                uses[0],
                flows[0],
                selected_elements[name],
                capacity_weights,
            )
        )
    conductance = 0.0
    for element in interfaces:
        layer_resistance = sum(
            slot.layer.r_value
            for component in element.construction.components
            for slot in component.layers
        )
        conductance += element.area_m2 / (
            layer_resistance + settings.interzone_floor_surface_resistance_m2_k_w
        )
    report = {
        "method": "two_coupled_5r1c_domains",
        "domains": [selection.identity() for selection in selections],
        "interzone_floor_area_m2": sum(element.area_m2 for element in interfaces),
        "interzone_floor_conductance_w_k": conductance,
        "interzone_surface_resistance_m2_k_w": settings.interzone_floor_surface_resistance_m2_k_w,
        "interzone_capacity": "existing_one_room_side_active_capacity_split_equally_between_domains",
        "interzone_heat_exchange": "symmetric_air_node_conductance_equal_and_opposite",
        "interzone_floor_thermal_bridge_uplift": "not_applied_to_internal_boundary",
    }
    return selections, conductance, report


def simulate_conditioned_domains(preparation):
    """Run conservative coupled controls from one captured building preparation."""
    from epsm_core.results import SimulationResult

    from ._core_rs import run_coupled_domains_hourly
    from .platform import _prepare_domain

    building, settings = preparation.building, preparation.settings
    selections, conductance, reduction = select_domains(building, preparation.resolved, settings)
    prepared = [_prepare_domain(preparation, domain=selection) for selection in selections]
    count = len(prepared[0].time_axis)
    f0, f1 = prepared[0].forcing, prepared[1].forcing
    powers, temperatures, exchanges, residuals = run_coupled_domains_hourly(
        prepared[0].model,
        prepared[1].model,
        f0.temperatures,
        f0.gains,
        f0.convective,
        f0.heating_setpoints,
        f0.cooling_setpoints,
        f0.wind,
        f0.infiltration,
        f0.natural,
        f0.ventilation,
        f0.solar,
        f0.opaque,
        f0.month_hours,
        tuple(float(c) for c in f0.coefficients),
        bool(f0.aggregate_split),
        np.asarray(prepared[0].heating_limits_w, dtype=np.float64),
        np.asarray(prepared[0].cooling_limits_w, dtype=np.float64),
        f1.temperatures,
        f1.gains,
        f1.convective,
        f1.heating_setpoints,
        f1.cooling_setpoints,
        f1.wind,
        f1.infiltration,
        f1.natural,
        f1.ventilation,
        f1.solar,
        f1.opaque,
        f1.month_hours,
        tuple(float(c) for c in f1.coefficients),
        bool(f1.aggregate_split),
        np.asarray(prepared[1].heating_limits_w, dtype=np.float64),
        np.asarray(prepared[1].cooling_limits_w, dtype=np.float64),
        float(conductance),
    )
    results = [
        domain.finish(powers[index], temperatures[index]) for index, domain in enumerate(prepared)
    ]
    if (
        results[0].diagnostics["physical_inputs_sha256"]
        != results[1].diagnostics["physical_inputs_sha256"]
    ):
        raise ValueError("Coupled domains consumed different physical Building inputs")
    hourly = {}
    for key in dict.fromkeys((*results[0].hourly, *results[1].hourly)):
        if key == "indoor_temperature_c":
            hourly[key] = sum(
                result.hourly[key] * domain.volume_m3 for result, domain in zip(results, prepared)
            ) / sum(domain.volume_m3 for domain in prepared)
        else:
            hourly[key] = sum(result.hourly.get(key, np.zeros(count)) for result in results)
    for index, domain in enumerate(prepared):
        hourly[f"{domain.name}_indoor_temperature_c"] = temperatures[index]
        hourly[f"{domain.name}_space_heating_demand_kwh"] = np.maximum(powers[index], 0) / 1000
        hourly[f"interzone_heat_to_{domain.name}_w"] = exchanges[index]
    scalars = {key: float(values.sum()) for key, values in hourly.items() if key.endswith("kwh")}
    units = {
        key: "kWh" if key.endswith("kwh") else "W" if key.endswith("_w") else "C" for key in hourly
    }
    diagnostics = dict(results[0].diagnostics)
    diagnostics["accepted"] = all(result.diagnostics["accepted"] for result in results)
    diagnostics["effective_configuration"] = {
        **diagnostics["effective_configuration"],
        "methods": {
            **diagnostics["effective_configuration"]["methods"],
            "thermal_domain": "coupled_main_and_conditioned_basement",
        },
    }
    diagnostics["domain_reduction"] = {
        **reduction,
        "maximum_control_balance_residual_w": float(residuals.max()),
        "maximum_interzone_exchange_sum_w": float(np.max(np.abs(exchanges.sum(axis=0)))),
    }
    diagnostics["domains"] = {
        domain.name: result.diagnostics for domain, result in zip(prepared, results)
    }
    numeric_properties = [
        "internal_surface_area_m2",
        "effective_mass_area_m2",
        "thermal_capacitance_j_k",
        "surface_air_conductance_w_k",
        "mass_surface_conductance_w_k",
        "opaque_transmission_conductance_w_k",
        "effective_ground_conductance_w_k",
        "mass_exterior_conductance_w_k",
        "sky_radiation_loss_w",
    ]
    diagnostics["reduced_properties"] = {
        key: sum(result.diagnostics["reduced_properties"][key] for result in results)
        for key in numeric_properties
    }
    diagnostics["reduced_properties"]["mass_exterior_coupling_fallback"] = any(
        result.diagnostics["reduced_properties"]["mass_exterior_coupling_fallback"]
        for result in results
    )
    diagnostics["boundary_reduction"] = {
        **diagnostics["boundary_reduction"],
        **{
            key: sum(result.diagnostics["boundary_reduction"][key] for result in results)
            for key in (
                "outdoor_opaque_area_m2",
                "ground_area_m2",
                "ground_floor_area_m2",
                "ground_wall_area_m2",
                "conditioned_basement_floor_area_m2",
                "outdoor_window_area_m2",
                "adiabatic_opaque_area_m2",
            )
        },
    }
    diagnostics["boundary_reduction"]["internal_opaque_area_m2"] = sum(
        element.area_m2
        for element in building.elements
        if element.boundary == "Internal" and hasattr(element.construction, "components")
    )
    diagnostics["approximations"] = [
        "Two coupled conditioned domains with separately resolved main and basement schedules",
        "Separating floor layer resistance plus declared surface films connects the air nodes",
        "Separating floor active storage is divided equally, preserving existing one-side total capacity",
        *[
            item
            for item in diagnostics["approximations"]
            if item != "One aggregated conditioned thermal domain"
        ],
    ]
    run_id = hashlib.sha256(
        json.dumps(
            {"domain_runs": [result.run_id for result in results], "coupling": reduction},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    bind_context(run_id=run_id)
    return SimulationResult(run_id, scalars, hourly, units, diagnostics, {}, results[0].time_axis)
