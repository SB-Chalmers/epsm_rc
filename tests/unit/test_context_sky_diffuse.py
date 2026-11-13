# File: test_context_sky_diffuse.py
# Purpose: Opt-in sky-diffuse obstruction behind context buildings.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-13

"""Opt-in sky-diffuse obstruction behind context buildings."""

from importlib.resources import files

import numpy as np
import pytest
from epsm_core import Weather
from epsm_core.specs import ContextSpec

from epsm_rc import RCSettings, simulate_building
from epsm_rc.shading import shade_irradiance, sky_diffuse_fraction

from .test_context_shading import neighbour


def settings(**overrides):
    return RCSettings(
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
        **overrides,
    )


def weather():
    return Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))


@pytest.mark.parametrize(
    ("alpha", "tilt", "expected"),
    [
        (0.0, 90.0, 1.0),
        (0.0, 0.0, 1.0),
        (30.0, 90.0, 0.5),  # vertical: 1 - sin 30
        (60.0, 0.0, 0.25),  # horizontal: cos² 60
        (90.0, 90.0, 0.0),
        (30.0, 45.0, 0.5 * 0.5 + 0.5 * np.cos(np.radians(30)) ** 2),
    ],
)
def test_isotropic_sky_fraction_behind_uniform_horizon(alpha, tilt, expected):
    assert sky_diffuse_fraction(alpha, tilt) == pytest.approx(expected)


def test_sky_diffuse_is_reduced_and_direct_and_ground_follow_g3():
    channels = {
        "poa_direct": np.array([200.0, 200.0, 0.0]),
        "poa_sky_diffuse": np.array([80.0, 80.0, 0.0]),
        "poa_ground_diffuse": np.array([20.0, 20.0, 0.0]),
        "poa_diffuse": np.array([100.0, 100.0, 0.0]),
        "poa_global": np.array([300.0, 300.0, 0.0]),
    }
    originals = {key: value.copy() for key, value in channels.items()}
    result = shade_irradiance(channels, [10.0, 40.0, -5.0], 30.0, tilt_deg=90.0)
    np.testing.assert_array_equal(result["poa_direct"], [0.0, 200.0, 0.0])
    np.testing.assert_allclose(result["poa_sky_diffuse"], [40.0, 40.0, 0.0])
    np.testing.assert_array_equal(result["poa_ground_diffuse"], originals["poa_ground_diffuse"])
    np.testing.assert_allclose(result["poa_diffuse"], [60.0, 60.0, 0.0])
    np.testing.assert_allclose(result["poa_global"], [60.0, 260.0, 0.0])
    for name in channels:
        np.testing.assert_array_equal(channels[name], originals[name])
    # Without a tilt the function is the unchanged G.3 rule.
    plain = shade_irradiance(channels, [10.0, 40.0, -5.0], 30.0)
    np.testing.assert_array_equal(plain["poa_sky_diffuse"], originals["poa_sky_diffuse"])


def test_isotropic_horizon_hides_sky_diffuse_and_adds_heating(reference_building):
    reference_building.context = ContextSpec(buildings=(neighbour(),))
    beam_only = simulate_building(reference_building, weather=weather(), settings=settings())
    sky = simulate_building(
        reference_building,
        weather=weather(),
        settings=settings(context_sky_diffuse="isotropic_horizon"),
    )
    report = sky.diagnostics["context_shading"]
    assert report["sky_diffuse"] == "isotropic_horizon"
    assert (
        report["horizon_angles_deg"]
        == beam_only.diagnostics["context_shading"]["horizon_angles_deg"]
    )
    np.testing.assert_array_equal(
        sky.hourly["window_incident_beam_w"], beam_only.hourly["window_incident_beam_w"]
    )
    np.testing.assert_array_equal(
        sky.hourly["window_incident_ground_diffuse_w"],
        beam_only.hourly["window_incident_ground_diffuse_w"],
    )
    assert (
        sky.hourly["window_incident_sky_diffuse_w"].sum()
        < beam_only.hourly["window_incident_sky_diffuse_w"].sum()
    )
    assert (
        sky.hourly["opaque_solar_heat_gain_w"].sum()
        < beam_only.hourly["opaque_solar_heat_gain_w"].sum()
    )
    assert sky.scalars["space_heating_demand_kwh"] > beam_only.scalars["space_heating_demand_kwh"]
    assert sky.run_id != beam_only.run_id


def test_without_context_the_option_changes_nothing(reference_building):
    plain = simulate_building(reference_building, weather=weather(), settings=settings())
    sky = simulate_building(
        reference_building,
        weather=weather(),
        settings=settings(context_sky_diffuse="isotropic_horizon"),
    )
    for name in plain.hourly:
        np.testing.assert_array_equal(plain.hourly[name], sky.hourly[name])


def test_unknown_sky_diffuse_choice_rejected():
    with pytest.raises(ValueError, match="context_sky_diffuse"):
        settings(context_sky_diffuse="perez")
