"""Unsupported airflow controls require an explicit RC omission policy."""

from importlib.resources import files

import pytest

from epsm_rc import RCSettings, platform, simulate_building


@pytest.fixture
def core_inputs(reference_building):
    core = pytest.importorskip("epsm_core")
    return reference_building, core.Weather.from_epw(
        files("epsm_rc").joinpath("data/epw/goteborg.epw")
    )


@pytest.mark.parametrize(
    "parameters,field",
    [
        ({}, "maximum_heating_supply_temperature_c"),
        ({"maximum_heating_supply_temperature_c": 40}, "maximum_heating_supply_temperature_c"),
        ({"minimum_cooling_supply_temperature_c": 15}, "minimum_cooling_supply_temperature_c"),
        (
            {"latent_recovery_effectiveness": 0.3, "topology": "FTX"},
            "latent_recovery_effectiveness",
        ),
    ],
)
def test_unsupported_airflow_rejected_before_solver(core_inputs, monkeypatch, parameters, field):
    building, weather = core_inputs
    building.hvac = building.hvac.with_changes(
        airflow=building.hvac.airflow.model_copy(update=parameters),
    )

    def forbidden_solver(*args, **kwargs):
        pytest.fail("Unsupported controls entered the solver")

    monkeypatch.setattr(platform, "Building5R1C", forbidden_solver)
    with pytest.raises(ValueError, match=field):
        simulate_building(building, weather=weather, settings=RCSettings())


def test_explicit_sensible_approximation_records_omitted_fields(core_inputs):
    building, weather = core_inputs
    building.hvac = building.hvac.with_changes(
        airflow=building.hvac.airflow.model_copy(
            update={
                "latent_recovery_effectiveness": 0.3,
                "topology": "FTX",
            }
        ),
    )
    result = simulate_building(
        building,
        weather=weather,
        settings=RCSettings(
            airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
        ),
    )
    assert result.scalars["space_heating_demand_kwh"] > 0
    airflow = result.diagnostics["airflow_capabilities"]
    assert airflow["omitted_inputs"] == {
        "latent_recovery_effectiveness": 0.3,
        "maximum_heating_supply_temperature_c": 50.0,
        "minimum_cooling_supply_temperature_c": 13.0,
    }
    assert airflow["effective_model"] == "sensible_only_unlimited_supply"
    assert result.diagnostics["reduction"]["airflow_policy"] == "sensible_only_unlimited_supply"


def test_airflow_policy_rejects_unknown_value():
    with pytest.raises(ValueError, match="airflow_policy"):
        RCSettings(airflow_policy="unknown")
