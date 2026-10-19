# SPDX-License-Identifier: MIT
"""Conservative two-domain hourly control for main space and a heated basement.

Each domain keeps its own 5R1C storage and controls. The physical separating floor
provides a symmetric air-node conductance. Linear response trials start from the
same previous state; only the final coupled solution advances thermal storage.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np


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
    if (heat > cool).any() or not np.isfinite(heat).all() or not np.isfinite(cool).all():
        raise ValueError("Finite ordered heating and cooling setpoints are required")
    if (
        np.isnan(maximum).any()
        or np.isnan(minimum).any()
        or (maximum < 0).any()
        or (minimum > 0).any()
    ):
        raise ValueError("Heating limits must be nonnegative; cooling limits nonpositive")
    h = conductance_w_k
    matrix = np.array(
        [[1 + response[0] * h, -response[0] * h], [-response[1] * h, 1 + response[1] * h]]
    )
    choices = []
    for index in range(2):
        modes = ["free"]
        if maximum[index] > 0:
            modes.append("heat")
            if np.isfinite(maximum[index]):
                modes.append("heat_limit")
        if minimum[index] < 0:
            modes.append("cool")
            if np.isfinite(minimum[index]):
                modes.append("cool_limit")
        choices.append(modes)
    tolerance = 1e-7
    for modes in product(*choices):
        equations, rhs = matrix.copy(), free.copy()
        for index, mode in enumerate(modes):
            if mode in {"heat", "cool"}:
                equations[index] = 0
                equations[index, index] = 1
                rhs[index] = heat[index] if mode == "heat" else cool[index]
            elif mode in {"heat_limit", "cool_limit"}:
                rhs[index] += response[index] * (
                    maximum[index] if mode == "heat_limit" else minimum[index]
                )
        temperatures = np.linalg.solve(equations, rhs)
        powers = (matrix @ temperatures - free) / response
        valid = True
        for index, mode in enumerate(modes):
            t, q = temperatures[index], powers[index]
            if mode == "free":
                valid &= (maximum[index] == 0 or t >= heat[index] - tolerance) and (
                    minimum[index] == 0 or t <= cool[index] + tolerance
                )
            elif mode == "heat":
                valid &= -tolerance <= q <= maximum[index] + tolerance
            elif mode == "cool":
                valid &= minimum[index] - tolerance <= q <= tolerance
            elif mode == "heat_limit":
                valid &= t <= heat[index] + tolerance
            else:
                valid &= t >= cool[index] - tolerance
        if valid:
            for index, mode in enumerate(modes):
                if mode == "free":
                    powers[index] = 0.0
                elif mode == "heat_limit":
                    powers[index] = maximum[index]
                elif mode == "cool_limit":
                    powers[index] = minimum[index]
            powers = np.clip(powers, minimum, maximum)
            exchange = h * (temperatures[1] - temperatures[0])
            residual = matrix @ temperatures - free - response * powers
            return CoupledStep(
                tuple(powers),
                tuple(temperatures),
                (exchange, -exchange),
                float(np.max(np.abs(residual / response))),
            )
    raise ValueError("No admissible coupled heating/cooling control state")


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
