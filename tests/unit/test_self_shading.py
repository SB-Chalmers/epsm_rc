# File: test_self_shading.py
# Purpose: Own-geometry shading of concave, courtyard and multi-part buildings.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-13

"""Own-geometry shading of concave, courtyard and multi-part buildings.

Notes
-----
Geometry kernels are checked against closed forms and an independent ray cast;
the platform tests check both thermal models, the diagnostics and that convex
buildings are unchanged by every policy.
"""

import math
from importlib.resources import files

import numpy as np
import pytest

from epsm_rc import RCSettings, simulate_building
from epsm_rc.self_shading import (
    horizon_profiles,
    occluder_edges,
    sample_points,
    sky_view_ratio,
    sunlit_fraction,
)
from epsm_rc.shading import sky_diffuse_fraction

from .test_self_shadowing import authored_building


def _uniform(alpha_deg, points=1):
    return np.full((points, 360), float(alpha_deg))


@pytest.mark.parametrize("alpha", [0.0, 15.0, 40.0, 75.0])
def test_sky_view_ratio_matches_uniform_horizon_closed_forms(alpha):
    weights, a = np.ones(1), math.radians(alpha)
    vertical = sky_view_ratio(_uniform(alpha), weights, 90.0, 180.0)
    horizontal = sky_view_ratio(_uniform(alpha), weights, 0.0, 0.0)
    # Same elevation in every azimuth: (4 / pi) * integral of cos^2 e from alpha.
    expected = 4 / math.pi * ((math.pi / 2 - a) / 2 - math.sin(2 * a) / 4)
    assert vertical == pytest.approx(expected, abs=1e-4)
    assert horizontal == pytest.approx(sky_diffuse_fraction(alpha, 0.0), abs=1e-4)


@pytest.mark.parametrize("height", [2.0, 6.0, 20.0])
def test_sky_view_ratio_behind_a_long_parallel_wall_is_one_minus_sin(height):
    # Facing a long wall, the 2-D strip result 1 - sin(alpha) holds.
    point = np.array([[0.0, 0.0, 0.0]])
    edges = (np.array([[-5000.0, -4.0, height]]), np.array([[5000.0, -4.0, height]]))
    ratio = sky_view_ratio(horizon_profiles(point, edges, 0.25), np.ones(1), 90.0, 180.0, 0.25)
    alpha = math.degrees(math.atan2(height, 4.0))
    assert ratio == pytest.approx(sky_diffuse_fraction(alpha, 90.0), abs=2e-3)


def test_sky_view_ratio_ignores_obstacles_behind_a_facade():
    profile = np.zeros((1, 360))
    profile[0, 0:60] = 80.0  # northern obstacle behind a south facade
    profile[0, 300:] = 80.0
    assert sky_view_ratio(profile, np.ones(1), 90.0, 180.0) == pytest.approx(1.0)


def test_horizon_profile_of_a_long_wall_is_its_elevation_angle():
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 4.0]])
    edges = (np.array([[-500.0, 3.0, 10.0]]), np.array([[500.0, 3.0, 10.0]]))
    profile = horizon_profiles(points, edges)
    assert profile[:, 0] == pytest.approx(np.degrees(np.arctan2([9.0, 6.0], 3.0)))
    assert (profile[:, 90:271] == 0).all()  # nothing to the south


def _fin_shaded(point, sun, depth, height):
    """Independent ray cast against a fin rectangle in the plane x = 0, y < 0."""
    if sun[0] >= 0:
        return False
    t = -point[0] / sun[0]
    hit = point + t * sun
    return t > 0 and -depth <= hit[1] <= 0 and 0 <= hit[2] <= height


def test_sunlit_fraction_of_inner_corner_facade_matches_ray_cast():
    # South-facing facade y = 0, x in [0, 10], z in [0, 6]; the other wing's wall
    # x = 0 projects 8 m south of it, as in the inner corner of an L.
    facade = [(0, 0, 0), (10, 0, 0), (10, 0, 6), (0, 0, 6)]
    wing = {
        "boundary": "Outdoors",
        "vertices": [(0, -8, 0), (0, 0, 0), (0, 0, 6), (0, -8, 6)],
    }
    edges = occluder_edges([wing])
    points = sample_points(facade, spacing_m=0.25, max_per_axis=60)
    assert np.allclose(points[:, 1], -0.01)
    profiles = horizon_profiles(points, edges)
    weights = np.ones(len(points))
    for altitude, azimuth in [(20.0, 240.0), (35.0, 210.0), (10.0, 260.0), (40.0, 150.0)]:
        a, z = math.radians(altitude), math.radians(azimuth)
        sun = np.array([math.cos(a) * math.sin(z), math.cos(a) * math.cos(z), math.sin(a)])
        expected = 1 - np.mean([_fin_shaded(p, sun, 8.0, 6.0) for p in points])
        found = sunlit_fraction(profiles, weights, np.array([altitude]), np.array([azimuth]))
        assert found[0] == pytest.approx(expected, abs=0.02)


@pytest.fixture
def weather():
    core = pytest.importorskip("epsm_core")
    return core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))


def _settings(**changes):
    return RCSettings(
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
        **changes,
    )


def test_convex_building_is_unchanged_by_every_policy(weather, reference_building):
    results = [
        simulate_building(reference_building, weather=weather, settings=_settings(**c))
        for c in (
            {"self_shadowing_policy": "ignore"},
            {"self_shadowing_policy": "iso13790_hourly_horizon"},
            {"self_shadowing_policy": "hourly_sunlit_fraction"},
        )
    ]
    for result in results[1:]:
        for key in ("space_heating_demand_kwh", "window_solar_heat_gain_w"):
            assert np.array_equal(result.hourly[key], results[0].hourly[key])
        assert result.diagnostics["self_shadowing"]["geometry_occlusion_applied"] is False


@pytest.mark.parametrize("thermal_model", ["iso52016_1", "iso13790_5r1c"])
@pytest.mark.parametrize("case", ["courtyard", "concave", "multipart"])
@pytest.mark.parametrize("method", ["hourly_sunlit_fraction", "iso13790_hourly_horizon"])
def test_own_geometry_reduces_solar_in_both_models(
    case, thermal_model, method, weather, reference_building
):
    building = authored_building(case, reference_building)
    ignored, shaded = (
        simulate_building(
            building,
            weather=weather,
            settings=_settings(self_shadowing_policy=policy, thermal_model=thermal_model),
        )
        for policy in ("ignore", method)
    )
    report = shaded.diagnostics["self_shadowing"]
    assert report["omitted"] is False
    assert report["geometry_occlusion_applied"] is True
    solar = [np.sum(r.hourly["window_solar_heat_gain_w"]) for r in (ignored, shaded)]
    assert solar[1] < solar[0]
    assert shaded.scalars["space_heating_demand_kwh"] >= ignored.scalars["space_heating_demand_kwh"]
