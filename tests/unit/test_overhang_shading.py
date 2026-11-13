# File: test_overhang_shading.py
# Purpose: Independent G.4-G.6 balances and overhang-to-RC integration.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-13

"""Independent G.4-G.6 balances and overhang-to-RC integration."""

from importlib.resources import files
from types import SimpleNamespace

import numpy as np
import pytest
from epsm_core import Weather
from epsm_core.specs.facade import OverhangSpec

from epsm_rc import RCSettings, preflight, simulate_building
from epsm_rc.overhang import overhang_angle_deg, overhang_factors, overhang_panes, shade_irradiance


def settings(**overrides):
    return RCSettings(
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
        **overrides,
    )


def weather():
    return Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))


def channels(direct, sky, ground):
    direct, sky, ground = (np.asarray(v, dtype=float) for v in (direct, sky, ground))
    return {
        "poa_direct": direct,
        "poa_sky_diffuse": sky,
        "poa_ground_diffuse": ground,
        "poa_diffuse": sky + ground,
        "poa_global": direct + sky + ground,
    }


def pane(z_bottom, z_top, facade="f", x0=0.0, x1=4.0):
    # A south-facing pane in the plane y = 0; Core's outward normal is along × up = -y.
    vertices = ((x0, 0, z_bottom), (x1, 0, z_bottom), (x1, 0, z_top), (x0, 0, z_top))
    return {"vertices": vertices, "metadata": {"facade_id": facade, "logical_vertices": vertices}}


def overhang(z, depth, facade="f", x0=-1.0, x1=5.0):
    return {
        "kind": "overhang",
        "facade_id": facade,
        "vertices": ((x0, 0, z), (x1, 0, z), (x1, -depth, z), (x0, -depth, z)),
    }


def test_g4_g5_g6_hand_balance():
    # alpha = 45 deg: tan(alpha) = 1. G.4 at Sh = 30 deg: 1 - 0.5 tan(30) = 0.7113.
    # G.5: 1 - 45/90 = 0.5. Ground reflection is unchanged (G.6).
    original = channels([400.0, 400.0, 0.0], [90.0, 90.0, 0.0], [10.0, 10.0, 0.0])
    result = shade_irradiance(original, np.array([30.0, 70.0, -5.0]), ((1.0, 45.0),))
    f_dir_30 = 1 - 0.5 * np.tan(np.radians(30))
    f_dir_70 = max(0.0, 1 - 0.5 * np.tan(np.radians(70)))
    np.testing.assert_allclose(result["poa_direct"], [400 * f_dir_30, 400 * f_dir_70, 0.0])
    np.testing.assert_allclose(result["poa_sky_diffuse"], [45.0, 45.0, 0.0])
    np.testing.assert_array_equal(result["poa_ground_diffuse"], original["poa_ground_diffuse"])
    np.testing.assert_allclose(
        result["poa_global"],
        result["poa_direct"] + result["poa_sky_diffuse"] + result["poa_ground_diffuse"],
    )
    assert np.isfinite(result["poa_global"]).all()
    # The input is not mutated.
    np.testing.assert_array_equal(original["poa_direct"], [400.0, 400.0, 0.0])


def test_g6_total_factor_matches_standard_form():
    # Fov = (Fdir Rdir + Fdif Rdif + 1 - Rtot) / Rtot with R as fractions of the total.
    direct, sky, ground, altitude, alpha = 300.0, 120.0, 30.0, 40.0, 20.0
    result = shade_irradiance(
        channels([direct], [sky], [ground]), np.array([altitude]), ((1, alpha),)
    )
    total = direct + sky + ground
    f_dir = 1 - 0.5 * np.tan(np.radians(alpha)) / np.tan(np.radians(90 - altitude))
    f_dif = 1 - alpha / 90
    expected = (f_dir * direct / total + f_dif * sky / total + ground / total) * total
    assert result["poa_global"][0] == pytest.approx(expected)


def test_zero_angle_is_identity_and_element_factor_is_area_weighted():
    original = channels([200.0], [50.0], [10.0])
    unshaded = shade_irradiance(original, np.array([35.0]), ((2.0, 0.0),))
    for name in original:
        np.testing.assert_allclose(unshaded[name], original[name])
    direct, diffuse = overhang_factors(np.array([35.0]), ((1.0, 30.0), (3.0, 0.0)))
    single_direct, single_diffuse = overhang_factors(np.array([35.0]), ((1.0, 30.0),))
    assert direct[0] == pytest.approx(0.25 * single_direct[0] + 0.75)
    assert diffuse == pytest.approx(0.25 * single_diffuse + 0.75)


def test_overhang_angle_from_pane_mid_height_to_nearest_edge_above():
    # Pane 1.0-2.0 m (mid 1.5 m); overhang 0.5 m deep at 2.0 m: tan(alpha) = 0.5 / 0.5.
    assert overhang_angle_deg(pane(1.0, 2.0), [overhang(2.0, 0.5)]) == pytest.approx(45.0)
    # Offset above the head, as Figure G.2 measures from mid-height: tan = 0.5 / 1.0.
    assert overhang_angle_deg(pane(1.0, 2.0), [overhang(2.5, 0.5)]) == pytest.approx(
        np.degrees(np.arctan(0.5))
    )
    # The nearest overhang above wins over a deeper, higher one.
    nearest = [overhang(2.0, 0.5), overhang(4.0, 2.0)]
    assert overhang_angle_deg(pane(1.0, 2.0), nearest) == pytest.approx(45.0)


@pytest.mark.parametrize(
    "shade",
    [
        overhang(0.9, 0.5),  # below the pane head
        overhang(2.0, 0.5, facade="other"),  # another facade
        overhang(2.0, 0.5, x0=5.0, x1=8.0),  # does not span the pane centre
    ],
)
def test_overhangs_that_do_not_cover_the_pane_are_ignored(shade):
    assert overhang_angle_deg(pane(1.0, 2.0), [shade]) == 0.0


def test_overhang_panes_keeps_uncovered_panes_for_exact_weighting():
    covered, bare = pane(1.0, 2.0), pane(1.0, 2.0, facade="g")
    resolved = SimpleNamespace(
        shading=[overhang(2.0, 0.5), {"kind": "reveal", "facade_id": "g", "vertices": ()}],
        surfaces=[
            {"name": "wall", "boundary": "Outdoors"},
            {"name": "inner", "boundary": "Internal"},
        ],
        openings=[
            {**covered, "parent": "wall", "area_m2": 4.0, "physical_element_id": "w1"},
            {**bare, "parent": "wall", "area_m2": 2.0, "physical_element_id": "w1"},
            {**bare, "parent": "wall", "area_m2": 2.0, "physical_element_id": "w2"},
            {**covered, "parent": "inner", "area_m2": 4.0, "physical_element_id": "w3"},
        ],
    )
    panes = overhang_panes(resolved)
    assert set(panes) == {"w1"}
    np.testing.assert_allclose(panes["w1"], ((4.0, 45.0), (2.0, 0.0)))


def test_overhang_reduces_window_gain_and_cooling_and_reports_angles(reference_building):
    baseline = simulate_building(reference_building, weather=weather(), settings=settings())
    reference_building.facade = reference_building.facade.updated(
        overhang=OverhangSpec(depth_m=0.6)
    )
    before = reference_building.snapshot().sha256
    shaded = simulate_building(reference_building, weather=weather(), settings=settings())
    assert reference_building.snapshot().sha256 == before
    report = shaded.diagnostics["overhang_shading"]
    assert report["applied"] and report["method"] == "iso13790_hourly_overhang"
    angles = [a for rows in report["overhang_angles_deg"].values() for a in rows]
    assert angles and all(0 < a < 90 for a in angles)
    for name in (
        "window_incident_beam_w",
        "window_incident_sky_diffuse_w",
        "window_solar_heat_gain_w",
    ):
        assert shaded.hourly[name].sum() < baseline.hourly[name].sum()
    np.testing.assert_array_equal(
        shaded.hourly["window_incident_ground_diffuse_w"],
        baseline.hourly["window_incident_ground_diffuse_w"],
    )
    np.testing.assert_array_equal(
        shaded.hourly["opaque_solar_heat_gain_w"], baseline.hourly["opaque_solar_heat_gain_w"]
    )
    assert shaded.scalars["space_heating_demand_kwh"] > baseline.scalars["space_heating_demand_kwh"]
    assert shaded.run_id != baseline.run_id


def test_deeper_overhang_shades_more(reference_building):
    gains = []
    for depth in (0.3, 0.6, 1.2):
        reference_building.facade = reference_building.facade.updated(
            overhang=OverhangSpec(depth_m=depth)
        )
        result = simulate_building(reference_building, weather=weather(), settings=settings())
        gains.append(result.hourly["window_solar_heat_gain_w"].sum())
    assert gains[0] > gains[1] > gains[2]


@pytest.mark.parametrize("thermal_model", ["iso52016_1", "iso13790_5r1c"])
def test_both_thermal_models_accept_overhangs(reference_building, thermal_model):
    reference_building.facade = reference_building.facade.updated(
        overhang=OverhangSpec(depth_m=0.6)
    )
    result = simulate_building(
        reference_building, weather=weather(), settings=settings(thermal_model=thermal_model)
    )
    assert result.diagnostics["overhang_shading"]["applied"]


def test_no_overhang_keeps_existing_results(reference_building):
    plain = simulate_building(reference_building, weather=weather(), settings=settings())
    rejected = simulate_building(
        reference_building,
        weather=weather(),
        settings=settings(overhang_shading_method="reject"),
    )
    assert not plain.diagnostics["overhang_shading"]["applied"]
    for name in plain.hourly:
        np.testing.assert_array_equal(plain.hourly[name], rejected.hourly[name])


def test_reject_and_other_facade_details_remain_unsupported(reference_building):
    reference_building.facade = reference_building.facade.updated(
        overhang=OverhangSpec(depth_m=0.6)
    )
    report = preflight(
        reference_building, weather=weather(), settings=settings(overhang_shading_method="reject")
    )
    assert not report["ready"]
    assert any("facade shading" in row["message"] for row in report["diagnostics"])
    assert preflight(reference_building, weather=weather(), settings=settings())["ready"]
    # Reveals still have no RC method, with or without an overhang.
    reference_building.facade = reference_building.facade.updated(reveal_depth_m=0.2)
    report = preflight(reference_building, weather=weather(), settings=settings())
    assert not report["ready"]
    assert any("facade shading" in row["message"] for row in report["diagnostics"])
    with pytest.raises(ValueError, match="facade shading"):
        simulate_building(reference_building, weather=weather(), settings=settings())


def test_unknown_overhang_method_rejected():
    with pytest.raises(ValueError, match="overhang_shading_method"):
        settings(overhang_shading_method="tables")
