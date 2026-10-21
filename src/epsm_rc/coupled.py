# SPDX-License-Identifier: MIT
"""Conservative two-domain hourly control for main space and a heated basement.

Each domain keeps its own 5R1C storage and controls. The physical separating floor
provides a symmetric air-node conductance. Linear response trials start from the
same previous state; only the final coupled solution advances thermal storage.
"""

from dataclasses import dataclass

import numpy as np

from ._core_rs import solve_controls_rs


@dataclass(frozen=True)
class CoupledStep:
    power_w: tuple
    temperature_c: tuple
    exchange_w: tuple
    balance_residual_w: float


def solve_controls(
    free_c,
    response_k_w,
    conductance_w_k,
    heat_c,
    cool_c,
    heating_limit_w=(np.inf, np.inf),
    cooling_limit_w=(0.0, 0.0),
):
    """Solve thermostat active sets, including disabled or capacity-limited HVAC.

    A free domain has zero HVAC. Controlled domains hold their requested air
    temperature until the finite power limit is reached. A saturated domain uses
    its limiting power. Heat exchange is equal and opposite for every state.
    """
    free, response = np.asarray(free_c, float), np.asarray(response_k_w, float)
    if (
        free.shape != (2,)
        or response.shape != (2,)
        or not np.isfinite(free).all()
        or (not np.isfinite(response).all() or (response <= 0).any())
    ):
        raise ValueError("Two finite free temperatures and positive responses are required")
    if not np.isfinite(conductance_w_k) or conductance_w_k < 0:
        raise ValueError("Inter-domain conductance must be finite and nonnegative")
    heat, cool = np.asarray(heat_c, float), np.asarray(cool_c, float)
    maximum, minimum = np.asarray(heating_limit_w, float), np.asarray(cooling_limit_w, float)
    if (
        heat.shape != (2,)
        or cool.shape != (2,)
        or (heat > cool).any()
        or not np.isfinite(heat).all()
        or not np.isfinite(cool).all()
    ):
        raise ValueError("Finite ordered heating and cooling setpoints are required")
    if (
        maximum.shape != (2,)
        or minimum.shape != (2,)
        or np.isnan(maximum).any()
        or np.isnan(minimum).any()
        or (maximum < 0).any()
        or (minimum > 0).any()
    ):
        raise ValueError("Heating limits must be nonnegative; cooling limits nonpositive")
    powers, temperatures, exchange, residual = solve_controls_rs(
        [float(free[0]), float(free[1])],
        [float(response[0]), float(response[1])],
        float(conductance_w_k),
        [float(heat[0]), float(heat[1])],
        [float(cool[0]), float(cool[1])],
        [float(maximum[0]), float(maximum[1])],
        [float(minimum[0]), float(minimum[1])],
    )
    return CoupledStep(powers, temperatures, exchange, residual)


def step_domains(models, inputs, conductance_w_k, *, heating_limit_w, cooling_limit_w):
    """Advance exactly two owned models once after solving their coupled controls."""
    states = [(model.t_air, model.t_s, model.t_m) for model in models]
    free, response = [], []
    for model, state, arguments in zip(models, states, inputs):
        model._hour_heat_flows = None
        model.step(**arguments, energy_demand=0.0)
        baseline = model.t_air
        model.t_air, model.t_s, model.t_m = state
        reference_w = 10 * model.floor_area_m2
        model.step(**arguments, energy_demand=reference_w)
        free.append(baseline)
        response.append((model.t_air - baseline) / reference_w)
        model.t_air, model.t_s, model.t_m = state
    solution = solve_controls(
        free,
        response,
        conductance_w_k,
        tuple(model.t_set_heating for model in models),
        tuple(model.t_set_cooling for model in models),
        heating_limit_w,
        cooling_limit_w,
    )
    for index, (model, arguments) in enumerate(zip(models, inputs)):
        model.step(**arguments, energy_demand=solution.power_w[index] + solution.exchange_w[index])
        if not np.isclose(model.t_air, solution.temperature_c[index], rtol=0, atol=1e-7):
            raise ValueError("Coupled control failed its committed-state temperature balance")
    return solution
