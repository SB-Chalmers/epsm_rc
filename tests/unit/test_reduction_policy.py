"""An explicit versioned reduction configures the coefficients actually used."""

from importlib.resources import files

import pytest

from epsm_rc import RCSettings, simulate_building
from epsm_rc.platform import _Platform5R1C


@pytest.fixture
def inputs(reference_building):
    core = pytest.importorskip("epsm_core")
    return reference_building, core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))


def test_missing_reduction_policy_is_rejected_before_io():
    with pytest.raises(ValueError, match="explicit.*settings|reduction policy"):
        simulate_building(None, weather="absent.epw")


def test_named_policy_is_inspectable_and_versioned():
    settings = RCSettings.from_policy("epsm_rc_5r1c_v1")
    assert settings.policy_version == "epsm_rc_5r1c_v1"
    assert settings.internal_surface_area_factor == 4.5
    assert settings.air_volumetric_heat_capacity_j_m3_k == 1200.0
    with pytest.raises(ValueError, match="policy"):
        RCSettings.from_policy("unknown")


@pytest.mark.parametrize("changes,property_name", [
    ({"internal_surface_area_factor": 4.0}, "internal_surface_area_m2"),
    ({"mass_area_factors": (1.5, 2.0, 2.5)}, "effective_mass_area_m2"),
    ({"surface_air_coefficient_w_m2_k": 2.1}, "surface_air_conductance_w_k"),
    ({"mass_surface_coefficient_w_m2_k": 6.0}, "mass_surface_conductance_w_k"),
    ({"sky_air_temperature_difference_k": 0.0}, "sky_radiation_loss_w"),
    ({"sky_radiation_coefficient_w_m2_k": 3.0}, "sky_radiation_loss_w"),
    ({"outside_wall_resistance_m2_k_w": 0.02}, "sky_radiation_loss_w"),
    ({"outside_roof_resistance_m2_k_w": 0.02}, "sky_radiation_loss_w"),
    ({"wall_sky_view_factor": 0.25}, "sky_radiation_loss_w"),
    ({"roof_sky_view_factor": 0.5}, "sky_radiation_loss_w"),
])
def test_effective_coefficients_change_actual_reduced_properties(inputs, changes, property_name):
    building, weather = inputs
    baseline = simulate_building(
        building, weather=weather,
        settings=RCSettings(airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"),
    )
    changed = simulate_building(
        building, weather=weather,
        settings=RCSettings(airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier", **changes),
    )
    assert baseline.run_id != changed.run_id
    assert baseline.diagnostics["reduced_properties"][property_name] != pytest.approx(
        changed.diagnostics["reduced_properties"][property_name],
    )
    for key, value in changes.items():
        assert changed.diagnostics["reduction"][key] == value


@pytest.mark.parametrize("changes", [
    {"convective_internal_gains_fraction": 0.7},
    {"air_volumetric_heat_capacity_j_m3_k": 1000.0},
    {"mass_capacity_thresholds_j_m2_k": (1e9, 2e9)},
])
def test_effective_gain_ventilation_and_mass_methods_change_results(inputs, changes):
    building, weather = inputs
    results = [simulate_building(
        building, weather=weather,
        settings=RCSettings(airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier", **configuration),
    ) for configuration in ({}, changes)]
    assert results[0].run_id != results[1].run_id
    assert results[0].scalars["space_heating_demand_kwh"] != pytest.approx(
        results[1].scalars["space_heating_demand_kwh"],
    )


def test_network_and_gain_split_use_same_mass_surface_coefficient():
    model = _Platform5R1C(
        floor_area_m2=100, a_tot_surfaces_m2=450,
        mass_area_factor_override=2.5, h_ms=6.0,
        delta_theta_er=0.0,
    )
    assert model._h_tr_ms == 1500.0
    air, surface, mass = model.calc_heat_flows(0, 1000, 0, include_sky_radiation=False)
    assert air == 500.0
    assert mass == pytest.approx(250 / 450 * 500)
    assert surface == pytest.approx((1 - 250 / 450 - model._h_tr_w / (6 * 450)) * 500)


def test_complete_constructions_do_not_receive_fallback_class(inputs, monkeypatch):
    from epsm_rc import platform

    models = []
    original = platform.Building5R1C

    def capture(*args, **kwargs):
        model = original(*args, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(platform, "Building5R1C", capture)
    building, weather = inputs
    result = simulate_building(
        building, weather=weather,
        settings=RCSettings.from_policy(
            "epsm_rc_5r1c_v1", airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier",
        ),
    )
    assert models[0].building_class is None
    assert models[0].constructions
    assert result.diagnostics["reduced_properties"]["thermal_capacitance_j_k"] > 0


@pytest.mark.parametrize("parameters", [
    {"internal_surface_area_factor": float("nan")},
    {"air_volumetric_heat_capacity_j_m3_k": 0},
    {"convective_internal_gains_fraction": 1.1},
    {"mass_capacity_thresholds_j_m2_k": (260_000, 165_000)},
    {"mass_area_factors": (2.5, -1, 3.5)},
    {"sky_radiation_coefficient_w_m2_k": -1},
])
def test_invalid_reduction_coefficients_are_rejected(parameters):
    with pytest.raises(ValueError):
        RCSettings(**parameters)


@pytest.mark.parametrize("parameters", [{"h_ms": 0}, {"air_volumetric_heat_capacity_j_m3_k": 0}])
def test_new_solver_coefficients_require_positive_values(parameters):
    with pytest.raises(ValueError):
        _Platform5R1C(**parameters)


def test_mass_exterior_fallback_is_declared_and_reported(inputs):
    building, weather = inputs
    settings = RCSettings(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier", mass_surface_coefficient_w_m2_k=0.01,
    )
    result = simulate_building(building, weather=weather, settings=settings)
    assert result.diagnostics["effective_configuration"]["methods"]["mass_exterior_coupling"] == (
        "series_inverse_conductance_with_opaque_fallback"
    )
    properties = result.diagnostics["reduced_properties"]
    assert properties["mass_exterior_coupling_fallback"] is True
    assert properties["mass_exterior_conductance_w_k"] == properties["opaque_transmission_conductance_w_k"]
