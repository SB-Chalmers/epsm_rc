"""Inter-domain exchange conserves energy and preserves independent controls."""

import numpy as np
import pytest

from epsm_rc.coupled import solve_controls, step_domains
from epsm_rc.iso13790 import Building5R1C


def test_two_heated_domains_hold_distinct_setpoints_and_conserve_floor_exchange():
    result = solve_controls((10, 10), (0.01, 0.02), 100, (20, 18), (26, 26))
    assert result.temperature_c == pytest.approx((20, 18))
    assert result.exchange_w == pytest.approx((-200, 200))
    assert result.power_w == pytest.approx((1200, 200))
    assert sum(result.exchange_w) == 0
    assert result.balance_residual_w < 1e-9


def test_equal_setpoints_do_not_create_interdomain_heat():
    result = solve_controls((10, 12), (0.01, 0.02), 1000, (20, 20), (26, 26))
    assert result.exchange_w == pytest.approx((0, 0))
    assert result.power_w == pytest.approx((1000, 400))


def test_disabled_basement_heat_can_receive_main_heat_without_inventing_hvac():
    result = solve_controls(
        (10, 10), (0.01, 0.02), 100, (20, 18), (26, 26), heating_limit_w=(np.inf, 0)
    )
    assert result.power_w[1] == 0
    assert 10 < result.temperature_c[1] < 18
    assert sum(result.exchange_w) == 0


def test_capacity_limit_preserves_unmet_temperature_and_cooling_sign():
    result = solve_controls(
        (10, 35),
        (0.01, 0.02),
        10,
        (20, 18),
        (26, 26),
        heating_limit_w=(100, 0),
        cooling_limit_w=(0, -100),
    )
    assert result.power_w == pytest.approx((100, -100))
    assert result.temperature_c[0] < 20
    assert result.temperature_c[1] > 26
    assert result.balance_residual_w < 1e-9


def test_zero_conductance_matches_two_existing_independent_5r1c_steps():
    independent = [
        Building5R1C(floor_area_m2=80, ceiling_height_m=3),
        Building5R1C(floor_area_m2=50, ceiling_height_m=2.5),
    ]
    coupled = [
        Building5R1C(floor_area_m2=80, ceiling_height_m=3),
        Building5R1C(floor_area_m2=50, ceiling_height_m=2.5),
    ]
    inputs = [
        {"t_out": 0, "solar_gains": 0, "internal_gains": 100},
        {"t_out": 0, "solar_gains": 0, "internal_gains": 50},
    ]
    expected = [
        model.calc_heating_demand(**arguments) for model, arguments in zip(independent, inputs)
    ]
    actual = step_domains(
        coupled, inputs, 0, heating_limit_w=(np.inf, np.inf), cooling_limit_w=(0, 0)
    )
    assert actual.power_w == pytest.approx(expected, abs=1e-7)
    for old, new in zip(independent, coupled):
        assert (old.t_air, old.t_s, old.t_m) == pytest.approx(
            (new.t_air, new.t_s, new.t_m), abs=1e-7
        )
