"""The explicit ground policy survives Core preparation and both solver paths."""

from importlib.resources import files

import numpy as np
import pytest

from epsm_rc import RCSettings, preflight, simulate_building


def settings(**overrides):
    return RCSettings(
        **{
            "airflow_policy": "sensible_only_unlimited_supply",
            "ground_policy": "iso13370_periodic_approximation",
            "iso13370_wall_thickness_m": 0.3,
            "iso13370_wall_thickness_source": "Explicit test wall dimension",
            **overrides,
        }
    )


@pytest.mark.parametrize("width", [None, 0, -0.1, float("nan"), float("inf"), True])
def test_iso_ground_requires_a_finite_explicit_wall_dimension(width):
    with pytest.raises(ValueError, match="wall thickness"):
        RCSettings(
            ground_policy="iso13370_periodic_approximation",
            iso13370_wall_thickness_m=width,
            iso13370_wall_thickness_source="test",
        )


def test_iso_wall_dimension_requires_a_source_and_matching_policy():
    with pytest.raises(ValueError, match="source"):
        RCSettings(
            ground_policy="iso13370_periodic_approximation",
            iso13370_wall_thickness_m=0.3,
        )
    with pytest.raises(ValueError, match="ISO 13370 ground policy"):
        RCSettings(iso13370_wall_thickness_m=0.3, iso13370_wall_thickness_source="test")


def test_preflight_checks_ground_without_constructing_a_solver(
    reference_building, synthetic_weather, monkeypatch
):
    from epsm_core import Weather

    from epsm_rc import platform

    def forbidden(*args, **kwargs):
        pytest.fail("Ground preflight constructed a thermal model")

    monkeypatch.setattr(platform, "Building5R1C", forbidden)
    result = preflight(
        reference_building, weather=Weather.from_epw(synthetic_weather), settings=settings()
    )
    assert result["ready"], result["diagnostics"]


@pytest.mark.parametrize("basement", [False, True], ids=["slab", "heated_basement"])
def test_periodic_ground_native_and_python_agree_and_preserve_physical_storage(
    reference_building, monkeypatch, basement
):
    from epsm_core import Weather

    from epsm_rc import iso13790

    if basement:
        reference_building.geometry = reference_building.geometry.updated(
            basement_depth_m=2.5, basement_conditioned=True
        )
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    original = reference_building.snapshot().sha256
    candidate = simulate_building(reference_building, weather=weather, settings=settings())
    legacy = simulate_building(
        reference_building,
        weather=weather,
        settings=RCSettings(
            airflow_policy="sensible_only_unlimited_supply",
            ground_policy="conductance_multiplier",
        ),
    )
    assert candidate.diagnostics["accepted"]
    assert candidate.diagnostics["physical_inputs_sha256"] == legacy.diagnostics[
        "physical_inputs_sha256"
    ]
    for key in ("thermal_capacitance_j_k", "effective_mass_area_m2", "internal_surface_area_m2"):
        assert candidate.diagnostics["reduced_properties"][key] == pytest.approx(
            legacy.diagnostics["reduced_properties"][key]
        )
    if basement:
        ground_report = candidate.diagnostics["domains"]["basement"]["ground_reduction"]
        assert not candidate.diagnostics["domains"]["main"]["ground_reduction"][
            "ground_contact"
        ]
        np.testing.assert_array_equal(
            candidate.hourly["interzone_heat_to_main_w"],
            -candidate.hourly["interzone_heat_to_basement_w"],
        )
    else:
        ground_report = candidate.diagnostics["ground_reduction"]
    assert ground_report["ground_contact"]
    assert len(ground_report["monthly_temperatures_c"]) == 12
    assert np.ptp(ground_report["monthly_temperatures_c"]) > 0
    assert sum(ground_report["month_weights_hours"]) == len(candidate.time_axis)
    assert candidate.run_id != legacy.run_id
    monkeypatch.setattr(iso13790, "_can_use_native", lambda *args, **kwargs: False)
    fallback = simulate_building(reference_building, weather=weather, settings=settings())
    for name in ("space_heating_demand_kwh", "cooling_demand_kwh", "indoor_temperature_c"):
        np.testing.assert_allclose(candidate.hourly[name], fallback.hourly[name], atol=1e-8, rtol=1e-10)
    assert reference_building.snapshot().sha256 == original


def test_monthly_ground_forcing_uses_actual_leap_year_hours(monkeypatch):
    from types import SimpleNamespace

    import pandas as pd
    from epsm_core import ScheduleSpec

    from epsm_rc import ground_inputs, platform

    inputs = ground_inputs.ResolvedGroundInputs(
        "slab_on_ground", 80, 36, 0.3, 1.0, 1.73, 1.8e6, 0, None, {}
    )
    monkeypatch.setattr(ground_inputs, "resolve_ground_inputs", lambda *a, **k: inputs)
    axis = pd.date_range("2024-01-01", "2025-01-01", freq="h", inclusive="left")
    outdoor = axis.month.to_numpy(dtype=float)
    spec = SimpleNamespace(
        use=SimpleNamespace(heating_setpoint=ScheduleSpec.constant(20)),
        climate=SimpleNamespace(calendar_year=2024),
    )
    _, _, report = platform._iso13370_reduction(
        {}, building=object(), spec=spec, resolved=object(), settings=settings(),
        domain=None, outdoor_temperatures=outdoor, time_axis=axis,
    )
    assert report["month_weights_hours"][1] == 29 * 24
    assert sum(report["month_weights_hours"]) == 8784
    assert report["monthly_outdoor_reference_c"] == tuple(range(1, 13))
    assert report["monthly_indoor_reference_c"] == (20,) * 12


def test_ground_assumptions_do_not_reuse_stale_cached_models(reference_building, synthetic_weather):
    from epsm_core import Weather

    from epsm_rc import PreparationCache

    cache = PreparationCache()
    weather = Weather.from_epw(synthetic_weather)
    outputs = [
        simulate_building(
            reference_building, weather=weather, settings=settings(iso13370_wall_thickness_m=width),
            preparation_cache=cache,
        )
        for width in (0.3, 0.6, 0.3)
    ]
    assert outputs[0].run_id == outputs[2].run_id
    assert outputs[0].run_id != outputs[1].run_id
    np.testing.assert_array_equal(
        outputs[0].hourly["space_heating_demand_kwh"],
        outputs[2].hourly["space_heating_demand_kwh"],
    )
    assert outputs[0].diagnostics["reduced_properties"]["effective_ground_conductance_w_k"] > (
        outputs[1].diagnostics["reduced_properties"]["effective_ground_conductance_w_k"]
    )
