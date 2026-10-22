"""Prepared native execution retains custom numerical model implementations."""

import numpy as np
import pytest

from epsm_rc import Building5R1C, RCSettings, simulate_building


def simulation_settings():
    return RCSettings(
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
        self_shadowing_policy="ignore",
    )


def test_single_domain_preserves_custom_private_demand(
    reference_building, synthetic_weather, monkeypatch
):
    from epsm_core import Weather

    from epsm_rc import platform

    class CustomDemand(Building5R1C):
        def _calculate_demand(self, *args, **kwargs):
            self.t_air = self.t_s = self.t_m = 19.25
            return 123.0

    monkeypatch.setattr(platform, "Building5R1C", CustomDemand)
    result = simulate_building(
        reference_building,
        weather=Weather.from_epw(synthetic_weather),
        settings=simulation_settings(),
    )
    np.testing.assert_array_equal(result.hourly["space_heating_demand_kwh"], 0.123)
    np.testing.assert_array_equal(result.hourly["indoor_temperature_c"], 19.25)
    assert result.scalars["space_heating_demand_kwh"] == pytest.approx(8760 * 0.123)


@pytest.mark.parametrize("method", ["_step", "calc_t_air"])
@pytest.mark.parametrize("patched_domain", [None, 0, 1], ids=["single", "main", "basement"])
def test_prepared_runs_preserve_overrides_on_either_domain(
    reference_building, synthetic_weather, monkeypatch, method, patched_domain
):
    from epsm_core import Weather

    from epsm_rc import platform

    if patched_domain is not None:
        reference_building.geometry = reference_building.geometry.updated(
            basement_depth_m=2.5, basement_conditioned=True
        )
    original = getattr(Building5R1C, method)

    def tracked(self, *args, **kwargs):
        self.numerical_override_calls = getattr(self, "numerical_override_calls", 0) + 1
        return original(self, *args, **kwargs)

    custom = type("CustomNumericalModel", (Building5R1C,), {method: tracked})
    models = []

    def construct(**kwargs):
        selected = patched_domain is None or len(models) == patched_domain
        model = (custom if selected else Building5R1C)(**kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(platform, "Building5R1C", construct)
    before = reference_building.snapshot().sha256
    result = simulate_building(
        reference_building,
        weather=Weather.from_epw(synthetic_weather),
        settings=simulation_settings(),
    )
    target = models[0 if patched_domain is None else patched_domain]
    assert getattr(target, "numerical_override_calls", 0) >= len(result.time_axis)
    assert reference_building.snapshot().sha256 == before
    assert result.diagnostics["accepted"]
    assert np.isfinite(result.hourly["indoor_temperature_c"]).all()
    if patched_domain is not None:
        np.testing.assert_array_equal(
            result.hourly["interzone_heat_to_main_w"],
            -result.hourly["interzone_heat_to_basement_w"],
        )
