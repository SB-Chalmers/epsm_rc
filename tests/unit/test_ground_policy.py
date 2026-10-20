"""Requested foundation physics must survive an explicitly selected RC reduction."""

from importlib.resources import files

import numpy as np
import pytest

from epsm_rc import RCSettings, platform, simulate_building


@pytest.fixture
def inputs(reference_building):
    core = pytest.importorskip("epsm_core")
    return reference_building, core.Weather.from_epw(
        files("epsm_rc").joinpath("data/epw/goteborg.epw")
    )


def test_kiva_requires_ground_approximation_before_solver(inputs, monkeypatch):
    building, weather = inputs

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported Kiva input reached the solver")

    monkeypatch.setattr(platform, "Building5R1C", forbidden)
    with pytest.raises(ValueError, match="kiva.*ground_policy|ground_policy.*kiva"):
        simulate_building(
            building,
            weather=weather,
            settings=RCSettings(airflow_policy="sensible_only_unlimited_supply"),
        )


def test_multiplier_used_once_and_reports_requested_method(inputs):
    building, weather = inputs
    results = [
        simulate_building(
            building,
            weather=weather,
            settings=RCSettings(
                airflow_policy="sensible_only_unlimited_supply",
                ground_policy="conductance_multiplier",
                ground_coupling_factor=factor,
            ),
        )
        for factor in (0.2, 0.8)
    ]
    ground_area_u = sum(
        element.area_m2 * element.construction.u_value
        for element in building.elements
        if element.role.value in {"GROUND_FLOOR", "BASEMENT_FLOOR"}
    )
    for result, factor in zip(results, (0.2, 0.8)):
        report = result.diagnostics["ground_reduction"]
        assert report["requested_foundation"]["method"] == "kiva"
        assert report["effective_method"] == "conductance_multiplier"
        assert report["effective_multiplier"] == factor
        assert result.diagnostics["reduced_properties"]["effective_ground_conductance_w_k"] == (
            pytest.approx(ground_area_u * factor)
        )
    assert results[0].run_id != results[1].run_id
    assert (
        results[0].scalars["space_heating_demand_kwh"]
        < results[1].scalars["space_heating_demand_kwh"]
    )


def test_explicit_monthly_boundary_bypasses_multiplier(inputs):
    building, weather = inputs
    months = tuple(float(month) for month in range(1, 13))
    results = [
        simulate_building(
            building,
            weather=weather,
            settings=RCSettings(
                airflow_policy="sensible_only_unlimited_supply",
                ground_policy="prescribed_monthly_temperature",
                monthly_ground_temperatures_c=months,
                monthly_ground_temperature_source="fixed test observation",
                ground_coupling_factor=factor,
            ),
        )
        for factor in (0.2, 0.8)
    ]
    for result in results:
        report = result.diagnostics["ground_reduction"]
        assert report["requested_foundation"]["method"] == "kiva"
        assert report["effective_method"] == "prescribed_monthly_temperature"
        assert report["monthly_temperatures_c"] == months
        assert report["boundary_source"] == "fixed test observation"
        assert report["effective_multiplier"] == 1.0
    np.testing.assert_array_equal(
        results[0].hourly["space_heating_demand_kwh"],
        results[1].hourly["space_heating_demand_kwh"],
    )


def test_existing_prescribed_spec_boundary_is_preserved(inputs):
    from epsm_core.specs.physics import GroundTemperatureSpec

    building, weather = inputs
    spec = building.to_spec(weather)
    months = (8.0,) * 12
    envelope = spec.envelope.updated(
        ground_temperature=GroundTemperatureSpec(
            monthly_temperatures_c=months,
            source="authored monthly boundary",
        )
    )
    resolved_months, multiplier, report = platform._ground_reduction(
        envelope,
        RCSettings(ground_coupling_factor=0.2),
    )
    assert resolved_months == months
    assert multiplier == 1.0
    assert report["requested_foundation"]["method"] == "prescribed_temperature"
    assert report["boundary_source"] == "authored monthly boundary"


@pytest.mark.parametrize(
    "parameters",
    [
        {"ground_policy": "unknown"},
        {"ground_policy": "prescribed_monthly_temperature"},
        {
            "ground_policy": "prescribed_monthly_temperature",
            "monthly_ground_temperatures_c": (8,) * 11,
        },
        {
            "ground_policy": "prescribed_monthly_temperature",
            "monthly_ground_temperatures_c": (8,) * 12,
        },
        {
            "ground_policy": "prescribed_monthly_temperature",
            "monthly_ground_temperatures_c": (float("nan"),) * 12,
            "monthly_ground_temperature_source": "test",
        },
    ],
)
def test_invalid_or_unsourced_ground_configuration_rejected(parameters):
    with pytest.raises(ValueError, match="ground|monthly|source"):
        RCSettings(**parameters)
