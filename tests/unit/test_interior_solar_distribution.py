# File: test_interior_solar_distribution.py
# Purpose: ISO 52016-1 interior solar distribution: area weighting and beam to floor.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-25

"""ISO 52016-1 interior solar distribution: area weighting and beam to floor."""

from importlib.resources import files

import numpy as np
import pytest

from epsm_rc import RCSettings
from epsm_rc.iso52016 import F_SOL_C, Network52016, ThermalZone, ZoneElement
from epsm_rc.options import InteriorSolarDistribution


def _elements():
    """A wall, a window, a roof and a floor facing one zone."""
    return (
        ZoneElement("wall", 60, "opaque", "exterior", "horizontal", 2.0, 150_000, "I"),
        ZoneElement("window", 12, "window", "exterior", "horizontal", 0.4, 0, "D"),
        ZoneElement("roof", 48, "opaque", "exterior", "upwards", 4.0, 30_000, "D"),
        ZoneElement(
            "floor",
            48,
            "opaque",
            "ground",
            "downwards",
            0.1,
            110_000,
            "I",
            boundary_resistance_m2_k_w=25.0,
        ),
    )


def _network(distribution):
    zone = ThermalZone("zone", 48, 130, control="air")
    return Network52016(_elements(), (zone,), solar_distribution=distribution)


def test_network_default_is_area_weighted_and_adds_no_column():
    default = Network52016(_elements(), (ThermalZone("zone", 48, 130, control="air"),))
    weighted = _network("area_weighted")
    assert default.solar_distribution == "area_weighted"
    assert default.column_count == weighted.column_count == 2 + 3 + 4
    np.testing.assert_array_equal(default._built["mp"], weighted._built["mp"])
    # RCSettings defaults to beam_to_floor, the closer match to EnergyPlus.
    assert RCSettings().interior_solar_distribution == InteriorSolarDistribution.BEAM_TO_FLOOR


def test_beam_column_reaches_floor_and_air_only():
    network = _network("beam_to_floor")
    assert network.column_count == 2 + 3 + 4 + 1
    built = network._built
    beam = built["p"][:, -1]
    floor_inner = built["inner"][3]
    assert beam[0] == pytest.approx(F_SOL_C)
    assert beam[floor_inner] == pytest.approx(1 - F_SOL_C)
    assert beam.sum() == pytest.approx(1.0)


def test_zero_beam_matches_area_weighting():
    weighted, split = _network("area_weighted"), _network("beam_to_floor")
    absorbed = np.zeros(4)
    states = [n.initial_state(20) for n in (weighted, split)]
    for hour in range(48):
        theta_e = 5 + 5 * np.sin(hour / 24 * 2 * np.pi)
        sun = max(0.0, 2000 * np.sin(hour / 24 * 2 * np.pi))
        out = []
        for index, network in enumerate((weighted, split)):
            forcing = network.forcing_vector(theta_e, 10, 300, sun, absorbed)
            states[index], power, air, _ = network.step(
                states[index], forcing, [30.0], [20.0], [26.0], [np.inf], [-np.inf]
            )
            out.append((power, air))
        assert out[0][0] == pytest.approx(out[1][0], abs=1e-9)
        assert out[0][1] == pytest.approx(out[1][1], abs=1e-9)


def test_beam_on_massive_floor_stores_more_sun():
    """All-beam sun: a massive floor stores more of it than area weighting does."""
    results = {}
    for distribution in ("area_weighted", "beam_to_floor"):
        network = _network(distribution)
        state = network.initial_state(20)
        absorbed = np.zeros(4)
        heating = 0.0
        for hour in range(24 * 20):
            sun = max(0.0, 3000 * np.sin((hour % 24 - 6) / 12 * np.pi))
            forcing = network.forcing_vector(0.0, 10, 200, sun, absorbed, phi_sol_beam=sun)
            state, power, _, _ = network.step(
                state, forcing, [30.0], [20.0], [26.0], [np.inf], [-np.inf]
            )
            heating += max(float(power[0]), 0.0)
        results[distribution] = heating
    # Less sun lands on the window and the light roof, so less leaks out.
    assert results["beam_to_floor"] < results["area_weighted"]


def test_unknown_distribution_is_refused():
    with pytest.raises(ValueError):
        _network("everywhere")
    with pytest.raises(ValueError):
        RCSettings(interior_solar_distribution="everywhere")


def test_5r1c_ignores_the_distribution():
    settings = {
        d: RCSettings(thermal_model="iso13790_5r1c", interior_solar_distribution=d)
        for d in ("area_weighted", "beam_to_floor")
    }
    configurations = [s.effective_configuration() for s in settings.values()]
    assert configurations[0] == configurations[1]
    assert "interior_solar_distribution" not in configurations[0]["coefficients"]


def test_beam_to_floor_runs_on_a_building(reference_building):
    core = pytest.importorskip("epsm_core")
    from epsm_rc import simulate_building

    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    common = dict(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    weighted = simulate_building(
        reference_building,
        weather=weather,
        settings=RCSettings(interior_solar_distribution="area_weighted", **common),
    )
    beam = simulate_building(reference_building, weather=weather, settings=RCSettings(**common))
    old = weighted.scalars["space_heating_demand_kwh"]
    new = beam.scalars["space_heating_demand_kwh"]
    assert np.isfinite(new) and new != old
    assert new == pytest.approx(old, rel=0.1)


def test_native_kernel_matches_python_with_beam_column(reference_building, monkeypatch):
    core = pytest.importorskip("epsm_core")
    from epsm_rc import _iso52016_platform as adapter
    from epsm_rc import simulate_building

    if adapter._native_kernel() is None:
        pytest.skip("native extension without the ISO 52016-1 kernel")
    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    settings = RCSettings(
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
        interior_solar_distribution="beam_to_floor",
    )
    original = adapter.run_network
    results = {}
    for native in (False, True):
        monkeypatch.setattr(
            adapter, "run_network", lambda *a, _n=native, **k: original(*a, native=_n, **k)
        )
        results[native] = simulate_building(reference_building, weather=weather, settings=settings)
    for name in ("space_heating_demand_kwh", "cooling_demand_kwh", "indoor_temperature_c"):
        np.testing.assert_allclose(
            results[True].hourly[name], results[False].hourly[name], rtol=0, atol=1e-9
        )
