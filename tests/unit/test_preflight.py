# File: test_preflight.py
# Purpose: Public capability inspection never enters the RC thermal solver.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-10-29

"""Public capability inspection never enters the RC thermal solver."""

from importlib.resources import files

import pytest


def test_preflight_collects_settings_issues_without_solver(reference_building, monkeypatch):
    from epsm_core import Weather

    from epsm_rc import RCSettings, platform, preflight

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight initialized the solver")

    monkeypatch.setattr(platform, "Building5R1C", forbidden)
    weather = Weather.from_epw(files("epsm_rc") / "data/epw/goteborg.epw")
    before = reference_building.snapshot().sha256
    rejected = preflight(reference_building, weather=weather, settings=RCSettings())
    assert not rejected["ready"]
    assert {row["code"] for row in rejected["diagnostics"]} >= {"rc_airflow", "rc_ground"}
    accepted = preflight(
        reference_building,
        weather=weather,
        settings=RCSettings(
            airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
        ),
    )
    assert accepted["ready"], accepted
    assert reference_building.snapshot().sha256 == before


def test_preflight_reports_missing_settings_and_inputs_without_weather_io():
    from epsm_core import Building

    from epsm_rc import preflight

    report = preflight(Building(), weather=None, settings=None)
    assert not report["ready"]
    assert {row["code"] for row in report["diagnostics"]} >= {"missing_input", "rc_settings"}
