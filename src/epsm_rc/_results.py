# File: _results.py
# Purpose: Assemble owned RC numerical outputs and inspectable assumptions.
# SPDX-License-Identifier: MIT
# License: MIT; see the project-root LICENSE.
"""Result accounting has no file access; the caller owns optional output writing."""

from dataclasses import asdict

import numpy as np

BASE_OUTPUT_UNITS = {
    "space_heating_demand_kwh": "kWh",
    "cooling_demand_kwh": "kWh",
    "indoor_temperature_c": "C",
    "internal_gains_convective_w": "W",
    "internal_gains_radiant_w": "W",
    "dhw_load_kwh": "kWh",
}
ANGULAR_WINDOW_OUTPUTS = (
    "window_incident_beam_w", "window_incident_sky_diffuse_w",
    "window_incident_ground_diffuse_w", "window_transmitted_solar_w",
    "window_inward_absorbed_solar_w",
)
ANGULAR_DERIVED_OUTPUTS = (
    "window_solar_heat_gain_w", "opaque_solar_heat_gain_w", "sky_radiation_loss_w",
)


def finish_domain(
    power,
    indoor,
    *,
    airflow,
    airflow_capabilities,
    areas,
    attic,
    axis,
    convective_gains,
    context_shading,
    domain,
    effective_configuration,
    elements,
    gain_reduction,
    ground_multiplier,
    ground_reduction,
    grouped,
    implementation,
    model,
    optical_models,
    physical_hash,
    radiant_gains,
    radiation_hourly,
    reference_area,
    run_id,
    self_shadowing,
    settings,
    spec,
    use,
):
    """Assemble one domain result after the numerical recurrence has completed."""
    from epsm_core.heating_delivery import heating_delivery_for_domain
    from epsm_core.results import SimulationResult

    hourly = {
        "space_heating_demand_kwh": np.maximum(power, 0) / 1000,
        "cooling_demand_kwh": np.maximum(-power, 0) / 1000,
        "indoor_temperature_c": indoor,
        "internal_gains_convective_w": convective_gains,
        "internal_gains_radiant_w": radiant_gains,
    }
    hourly.update(radiation_hourly)
    dhw = np.asarray(use.dhw_fraction.hourly(spec.climate.calendar_year), dtype=float)
    hourly["dhw_load_kwh"] = (
        dhw / dhw.sum() * use.dhw_kwh_m2_year * reference_area if dhw.sum() else dhw
    )
    hourly.update(
        heating_delivery_for_domain(
            spec,
            hourly["space_heating_demand_kwh"],
            reference_area_m2=reference_area,
            airflow=airflow,
        )
    )
    scalars = {key: float(values.sum()) for key, values in hourly.items() if key.endswith("kwh")}
    units = {
        key: BASE_OUTPUT_UNITS.get(key, "kWh" if key.endswith("kwh") else "W" if key.endswith("_w") else "C") for key in hourly
    }
    diagnostics = {
        "accepted": all(np.isfinite(v).all() for v in hourly.values()),
        "engine": "epsm_rc",
        "reduction": asdict(settings),
        "effective_configuration": effective_configuration,
        "gain_reduction": gain_reduction,
        "reduced_properties": {
            "internal_surface_area_m2": model._a_tot,
            "effective_mass_area_m2": model._mass_area,
            "thermal_capacitance_j_k": model._c_m,
            "surface_air_conductance_w_k": model._h_tr_is,
            "mass_surface_conductance_w_k": model._h_tr_ms,
            "opaque_transmission_conductance_w_k": model._h_tr_op,
            "effective_ground_conductance_w_k": model._h_tr_ground * ground_multiplier,
            "mass_exterior_conductance_w_k": model._h_tr_em,
            "mass_exterior_coupling_fallback": bool(
                model._h_tr_op > 0 and model._h_tr_ms > 0 and model._h_tr_op >= model._h_tr_ms
            ),
            "sky_radiation_loss_w": model.calc_sky_radiation_loss(),
        },
        "implementation": implementation,
        "physical_inputs_sha256": physical_hash,
        "boundary_reduction": {
            "source": "Core physical element boundaries",
            "outdoor_opaque_area_m2": areas["wall"] + areas["roof"],
            "ground_area_m2": areas["ground"],
            "ground_floor_area_m2": sum(
                e.area_m2 for e in grouped["ground"] if e.role.value != "BASEMENT_WALL"
            ),
            "ground_wall_area_m2": sum(
                e.area_m2 for e in grouped["ground"] if e.role.value == "BASEMENT_WALL"
            ),
            "conditioned_basement_floor_area_m2": (
                spec.geometry.footprint_area_m2
                if spec.geometry.basement_conditioned
                and (domain is None or domain.name == "basement")
                else 0.0
            ),
            "outdoor_window_area_m2": areas["window"],
            "adiabatic_opaque_area_m2": sum(
                element.area_m2
                for element in elements
                if element.boundary == "Adiabatic" and hasattr(element.construction, "components")
            ),
            "internal_opaque_area_m2": sum(
                element.area_m2
                for element in elements
                if element.boundary == "Internal" and hasattr(element.construction, "components")
            ),
            "opaque_capacity_basis": "one room-side effective depth per installed opaque element",
            "internal_active_depth_faces": 1,
            "neighbor_temperature_modelled": False,
        },
        "airflow_capabilities": airflow_capabilities,
        "ground_reduction": ground_reduction,
        "self_shadowing": self_shadowing,
        "context_shading": context_shading,
        "attic_reduction": attic.diagnostics,
        "approximations": [
            "One aggregated conditioned thermal domain",
            "Shared walls are adiabatic; shared/internal depth-limited opaque storage is retained",
            "Perez solar gains with scalar window shading and reported ISO hourly context horizon shading",
            "Room-side mass depth and construction-capacity mass-area rule from explicit policy",
            "Ground boundary or conductance multiplier explicitly selected and reported",
            (
                "Solar gains use fixed whole-window SHGC"
                if settings.window_optics_method == "constant_shgc"
                else "Solar gains use angular transmission and nominal inward absorption"
            ),
            "No fan heat feedback",
            "Explicit gain partitions use supplied sensible heat; legacy gain treatment is separately reported",
            "Internal area, gain split, node transfer and air heat capacity from explicit policy",
            "Fixed sky-air difference and sky exchange coefficients from explicit policy",
            "Heat recovery reduces mechanical conductance once; infiltration is unrecovered",
        ],
        "empirical_validation": "not_established_by_harmonisation",
    }
    if radiation_hourly:
        diagnostics["radiation"] = {
            "solar_time_reference": settings.solar_time_reference,
            "window_optics_method": settings.window_optics_method,
            "diffuse_optics": "isotropic_hemispherical_average_of_angular_model",
            "inward_absorption": "nominal_summer_inward_fraction",
            "longwave": "unchanged_fixed_sky_air_difference",
            "annual_kwh": {
                name: float(values.sum()) / 1000 for name, values in radiation_hourly.items()
            },
            "optical_models": [
                {
                    "u_value_w_m2k": u,
                    "shgc": g,
                    "normal_transmittance": optics.normal_transmittance,
                    "normal_reflectance": optics.normal_reflectance,
                    "inward_fraction": optics.inward_fraction,
                }
                for (u, g), optics in optical_models.items()
            ],
        }
    if not diagnostics["accepted"]:
        raise ValueError("RC calculation produced non-finite output")
    return SimulationResult(run_id, scalars, hourly, units, diagnostics, {}, axis)
