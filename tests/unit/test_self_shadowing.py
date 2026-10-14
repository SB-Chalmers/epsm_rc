"""Affected geometry requires explicit acknowledgment of omitted self-shadowing."""

import math
from importlib.resources import files

import pytest
from shapely.geometry import MultiPolygon, Polygon

from epsm_rc import RCSettings, platform, simulate_building


def geometry_fixture(case):
    if case == "courtyard":
        return Polygon(
            [(0, 0), (20, 0), (20, 20), (0, 20)],
            holes=[[(5, 5), (15, 5), (15, 15), (5, 15)]],
        )
    if case == "concave":
        return Polygon([(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)])
    # Two 3 m tall parts separated by 2 m: a 15° sun casts a shadow beyond the gap.
    assert 3 / math.tan(math.radians(15)) > 2
    return MultiPolygon([
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(12, 0), (22, 0), (22, 10), (12, 10)]),
    ])


@pytest.fixture
def weather():
    core = pytest.importorskip("epsm_core")
    return core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))


def authored_building(case):
    core = pytest.importorskip("epsm_core")
    building = core.Building()
    building.geometry = core.GeometrySpec.from_footprint(
        geometry_fixture(case), storeys=1, storey_height_m=3,
    )
    return building


@pytest.mark.parametrize("case", ["courtyard", "concave", "multipart"])
def test_affected_geometry_rejected_before_solver(case, weather, monkeypatch):
    building = authored_building(case)

    def forbidden(*args, **kwargs):
        pytest.fail("Affected geometry reached solver without an omission policy")

    monkeypatch.setattr(platform, "_Platform5R1C", forbidden)
    with pytest.raises(ValueError, match="self.shadow|self_shadowing_policy"):
        simulate_building(
            building, weather=weather,
            settings=RCSettings(
                airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier",
            ),
        )


@pytest.mark.parametrize("case", ["courtyard", "concave", "multipart"])
def test_explicit_omission_is_reported_for_affected_geometry(case, weather):
    result = simulate_building(
        authored_building(case), weather=weather,
        settings=RCSettings(
            airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier",
            self_shadowing_policy="ignore",
        ),
    )
    report = result.diagnostics["self_shadowing"]
    assert report["policy"] == "ignore"
    assert report["omitted"] is True
    assert report["affected_geometry"]
    assert result.diagnostics["reduction"]["self_shadowing_policy"] == "ignore"
    assert result.scalars["space_heating_demand_kwh"] > 0


def test_convex_unshaded_case_remains_accepted_and_policy_is_in_identity(weather):
    core = pytest.importorskip("epsm_core")
    building = core.Building()
    results = [simulate_building(
        building, weather=weather,
        settings=RCSettings(
            airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier",
            self_shadowing_policy=policy,
        ),
    ) for policy in ("reject", "ignore")]
    assert results[0].diagnostics["self_shadowing"]["omitted"] is False
    assert results[0].diagnostics["self_shadowing"]["affected_geometry"] == []
    assert results[0].run_id != results[1].run_id


def test_unknown_self_shadowing_policy_rejected():
    with pytest.raises(ValueError, match="self_shadowing_policy"):
        RCSettings(self_shadowing_policy="unknown")
