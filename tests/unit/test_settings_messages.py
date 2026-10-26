"""Settings and public-call errors name the field, the value and how to fix it."""

import pytest

from epsm_rc import RCSettings, generate_idf
from epsm_rc._validation import choice, finite, fraction
from epsm_rc.parametric import generate_polygon_idf


@pytest.mark.parametrize(
    ("field", "value", "valid"),
    [
        ("policy_version", "v2", "'epsm_rc_5r1c_v1', 'epsm_rc_5r1c_v2'"),
        ("ground_policy", "kiva", "'iso13370_periodic_approximation'"),
        ("window_optics_method", "angular", "'simple_glazing_angular_v1'"),
        ("cold_attic_policy", "buffer", "'quasi_steady_buffer'"),
        ("context_shading_method", "horizon", "'iso13790_hourly_horizon'"),
        ("solar_time_reference", "midpoint", "'interval_midpoint'"),
    ],
)
def test_unknown_policy_lists_the_received_value_and_valid_options(field, value, valid):
    with pytest.raises(ValueError, match=f"Unknown {field} '{value}'; choose one of .*{valid}"):
        RCSettings(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("effective_mass_depth_m", "0.1", "effective_mass_depth_m must be a real number, not str"),
        ("effective_mass_depth_m", True, "must be a real number, not bool"),
        ("horizon_azimuth_step_deg", 45, "between 1 and 30, not 45"),
        ("wall_sky_view_factor", 2, "wall_sky_view_factor must be a finite fraction"),
        ("surface_air_coefficient_w_m2_k", 0, "must be finite and positive, not 0"),
    ],
)
def test_numeric_settings_reject_nonnumbers_and_out_of_range_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        RCSettings(**{field: value})


def test_repr_shows_the_required_explicit_choices():
    settings = RCSettings.from_policy(
        "epsm_rc_5r1c_v2",
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
    )
    assert repr(settings) == (
        "RCSettings(policy='epsm_rc_5r1c_v2', airflow='sensible_only_unlimited_supply', "
        "ground='conductance_multiplier', optics='simple_glazing_angular_v1')"
    )


def test_requirement_errors_keep_the_policy_and_point_to_preflight():
    pytest.importorskip("epsm_core")
    from importlib.resources import files

    from epsm_core import Building, Weather

    from epsm_rc import simulate_building

    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    with pytest.raises(ValueError) as error:
        simulate_building(
            Building(preset="illustrative-reference"),
            weather=weather,
            settings=RCSettings.from_policy("epsm_rc_5r1c_v2"),
        )
    message = str(error.value)
    assert "RCSettings.from_policy('epsm_rc_5r1c_v2'" in message
    assert "epsm_rc.preflight(" in message


def test_wrapped_public_call_names_itself_on_bad_arguments():
    from epsm_rc import simulate_building

    with pytest.raises(TypeError, match=r"simulate_building\(\): got an unexpected keyword"):
        simulate_building(None, weather=None, setings={})


@pytest.mark.parametrize("retired", [generate_idf, generate_polygon_idf])
def test_retired_exports_always_report_their_migration(retired):
    with pytest.raises(ValueError, match="retired"):
        retired()
    with pytest.raises(ValueError, match="retired"):
        retired("building", "geometry", "out.idf", unexpected=True)


def test_validation_helpers_return_normalised_values():
    assert finite("x", 2, positive=True) == 2.0
    assert fraction("x", 0) == 0.0
    assert choice("mode", "a", {"a", "b"}) == "a"
    with pytest.raises(ValueError, match="must be finite, not inf"):
        finite("x", float("inf"))
