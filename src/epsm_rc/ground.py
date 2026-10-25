"""Explicit ISO 13370 ground coefficients and a monthly boundary approximation.

Steady slab and heated-basement coefficients follow ISO 13370:2017, 7.1/7.3.
Periodic coefficients use the all-over-insulation formulae in Annex H (Annex F,
F.4/F.7 in the 2007 edition). No edge insulation, moving groundwater, unheated
basement or partially heated basement model is implemented here.

The monthly heat-flow equation omits phase differences, as permitted by the
monthly method. Its equivalent boundary reproduces that equation *at the
declared monthly indoor reference temperatures*. Feeding this boundary into
an hourly 5R1C model is an approximation: its instantaneous indoor response and
surface/mass node are not the ISO 13370 dynamic model in Annex F. The boundary
is an equivalent environment temperature, not a measured soil temperature.

Sources: https://www.iso.org/standard/65716.html ; ISO 13370:2007 F.4/F.7;
https://www.gov.uk/government/publications/home-energy-model-technical-documentation
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, fsum, isfinite, log1p, pi, sqrt


@dataclass(frozen=True)
class GroundCoefficients:
    """Whole-foundation coefficients, in SI units; positive flow is heat loss.

    Floor/wall U values are approximate allocations of the coupled basement
    heat flow, not separately resolved physical surface conductances.
    """

    steady_conductance_w_k: float
    periodic_internal_w_k: float
    periodic_external_w_k: float
    floor_u_w_m2k: float
    wall_u_w_m2k: float
    characteristic_dimension_m: float
    equivalent_floor_thickness_m: float
    equivalent_wall_thickness_m: float | None
    penetration_depth_m: float
    area_m2: float
    exposed_perimeter_m: float
    basement_depth_m: float
    junction_conductance_w_k: float


def _number(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _floor_u(conductivity: float, dimension: float, thickness: float) -> float:
    if thickness < dimension:
        return 2 * conductivity / (pi * dimension + thickness) * log1p(
            pi * dimension / thickness
        )
    return conductivity / (0.457 * dimension + thickness)


def _periodic_internal(area: float, conductivity: float, thickness: float, delta: float):
    return area * conductivity / thickness * sqrt(2 / ((1 + delta / thickness) ** 2 + 1))


def ground_coefficients(
    *,
    area_m2: float,
    exposed_perimeter_m: float,
    wall_thickness_m: float,
    floor_resistance_m2k_w: float,
    soil_conductivity_w_mk: float,
    soil_volumetric_heat_capacity_j_m3k: float,
    basement_depth_m: float = 0.0,
    basement_wall_resistance_m2k_w: float | None = None,
    rsi_floor_m2k_w: float = 0.17,
    rsi_wall_m2k_w: float = 0.13,
    rse_m2k_w: float = 0.04,
    junction_psi_w_mk: float = 0.0,
) -> GroundCoefficients:
    """Calculate a slab or fully heated basement with uniform constructions.

    ``area_m2`` is footprint floor area; ``exposed_perimeter_m`` excludes shared
    heated party walls. ``wall_thickness_m`` is the *full* wall thickness at
    grade. Construction resistances contain material layers only, without
    surface films or soil resistance. The defaults are downward floor Rsi,
    horizontal wall Rsi and outdoor-air Rse from ISO 6946. If external *ground
    surface* temperatures replace outdoor air, callers must set Rse to zero.

    Soil properties and dimensions are explicit; no TABULA calibration or
    arbitrary conductance multiplier is used. Junction psi is a separate
    linear bridge correction, added once to all three coefficients. A caller
    that already represents bridges in material R must leave psi at zero.
    """
    area = _number("area_m2", area_m2, positive=True)
    perimeter = _number("exposed_perimeter_m", exposed_perimeter_m, positive=True)
    width = _number("wall_thickness_m", wall_thickness_m, positive=True)
    floor_r = _number("floor_resistance_m2k_w", floor_resistance_m2k_w)
    conductivity = _number("soil_conductivity_w_mk", soil_conductivity_w_mk, positive=True)
    capacity = _number(
        "soil_volumetric_heat_capacity_j_m3k", soil_volumetric_heat_capacity_j_m3k, positive=True
    )
    depth = _number("basement_depth_m", basement_depth_m)
    floor_film = _number("rsi_floor_m2k_w", rsi_floor_m2k_w)
    wall_film = _number("rsi_wall_m2k_w", rsi_wall_m2k_w)
    outdoor_film = _number("rse_m2k_w", rse_m2k_w)
    # Negative junction corrections are meaningful, but total coefficients
    # must still represent a passive heat-transfer path.
    try:
        psi = float(junction_psi_w_mk)
    except (TypeError, ValueError) as error:
        raise ValueError("junction_psi_w_mk must be finite") from error
    if isinstance(junction_psi_w_mk, bool) or not isfinite(psi):
        raise ValueError("junction_psi_w_mk must be finite")
    dimension = 2 * area / perimeter
    floor_thickness = width + conductivity * (floor_film + floor_r + outdoor_film)
    # The standard uses 3.15e7 seconds for an annual cycle.
    delta = sqrt(conductivity * 3.15e7 / (pi * capacity))
    floor_u = _floor_u(conductivity, dimension, floor_thickness + 0.5 * depth)
    wall_u = 0.0
    wall_thickness = None
    internal = _periodic_internal(area, conductivity, floor_thickness, delta)
    external = 0.37 * perimeter * conductivity * log1p(delta / floor_thickness)
    if depth > 0:
        if basement_wall_resistance_m2k_w is None:
            raise ValueError("A heated basement requires basement_wall_resistance_m2k_w")
        wall_r = _number("basement_wall_resistance_m2k_w", basement_wall_resistance_m2k_w)
        wall_thickness = conductivity * (wall_film + wall_r + outdoor_film)
        if wall_thickness <= 0:
            raise ValueError("Basement wall resistance plus surface films must be positive")
        limited_floor = min(floor_thickness, wall_thickness)
        wall_u = (
            2 * conductivity / (pi * depth)
            * (1 + 0.5 * limited_floor / (limited_floor + depth))
            * log1p(depth / wall_thickness)
        )
        internal += _periodic_internal(depth * perimeter, conductivity, wall_thickness, delta)
        attenuation = exp(-depth / delta)
        external = 0.37 * perimeter * conductivity * (
            attenuation * log1p(delta / floor_thickness)
            + 2 * (1 - attenuation) * log1p(delta / wall_thickness)
        )
    elif basement_wall_resistance_m2k_w is not None:
        _number("basement_wall_resistance_m2k_w", basement_wall_resistance_m2k_w)
    junction = perimeter * psi
    steady = area * floor_u + depth * perimeter * wall_u + junction
    internal += junction
    external += junction
    if steady <= 0 or internal < 0 or external < 0:
        raise ValueError("Junction correction produces nonpositive ground conductance")
    return GroundCoefficients(
        steady_conductance_w_k=steady,
        periodic_internal_w_k=internal,
        periodic_external_w_k=external,
        floor_u_w_m2k=floor_u,
        wall_u_w_m2k=wall_u,
        characteristic_dimension_m=dimension,
        equivalent_floor_thickness_m=floor_thickness,
        equivalent_wall_thickness_m=wall_thickness,
        penetration_depth_m=delta,
        area_m2=area,
        exposed_perimeter_m=perimeter,
        basement_depth_m=depth,
        junction_conductance_w_k=junction,
    )


def _months(name: str, values: Sequence[float]) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain 12 finite numbers") from error
    if len(result) != 12 or not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain 12 finite numbers")
    return result


def monthly_ground_heat_flows(
    coefficients: GroundCoefficients,
    outdoor_monthly_temperatures_c: Sequence[float],
    indoor_monthly_temperatures_c: Sequence[float],
    month_weights: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Return 12 monthly mean heat losses (W), with phase differences omitted.

    Uses ISO 13370:2017 C.4. ``month_weights`` should be actual hours in each
    month of the weather calendar. Equal-month weighting is used only when
    weights are omitted. Annual means and periodic deviations use the same
    weights, preserving the annual mean heat-flow identity.
    """
    outdoor = _months("outdoor_monthly_temperatures_c", outdoor_monthly_temperatures_c)
    indoor = _months("indoor_monthly_temperatures_c", indoor_monthly_temperatures_c)
    weights = (1.0,) * 12 if month_weights is None else _months("month_weights", month_weights)
    if any(weight <= 0 for weight in weights):
        raise ValueError("month_weights must contain 12 positive numbers")
    total_weight = fsum(weights)
    mean_outdoor = fsum(t * w for t, w in zip(outdoor, weights)) / total_weight
    mean_indoor = fsum(t * w for t, w in zip(indoor, weights)) / total_weight
    annual = coefficients.steady_conductance_w_k * (mean_indoor - mean_outdoor)
    return tuple(
        annual
        + coefficients.periodic_internal_w_k * (ti - mean_indoor)
        - coefficients.periodic_external_w_k * (te - mean_outdoor)
        for ti, te in zip(indoor, outdoor)
    )


def monthly_ground_temperatures(
    coefficients: GroundCoefficients,
    outdoor_monthly_temperatures_c: Sequence[float],
    indoor_monthly_temperatures_c: Sequence[float],
    month_weights: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Equivalent monthly environments matching the declared reference flux.

    ``H_g * (T_indoor_reference - T_equivalent)`` equals the monthly ISO heat
    flow. These temperatures are a forcing approximation for an hourly 5R1C
    solver, not the Annex F dynamic method or physical interface temperatures.
    Free-floating or intermittent indoor temperatures need special caution:
    their instantaneous coupling is H_g, not the periodic internal coefficient.
    """
    indoor = _months("indoor_monthly_temperatures_c", indoor_monthly_temperatures_c)
    flows = monthly_ground_heat_flows(
        coefficients, outdoor_monthly_temperatures_c, indoor, month_weights
    )
    return tuple(
        temperature - flow / coefficients.steady_conductance_w_k
        for temperature, flow in zip(indoor, flows)
    )
