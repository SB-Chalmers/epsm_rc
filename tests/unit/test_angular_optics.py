"""Published simple-glazing optics, separate from thermal-demand calibration."""

import json
from pathlib import Path

import numpy as np
import pytest

from epsm_rc import RCSettings
from epsm_rc.optics import SimpleGlazingOptics


def test_named_radiation_policy_is_explicit_and_legacy_stays_unchanged():
    legacy = RCSettings.from_policy("epsm_rc_5r1c_v1")
    current = RCSettings.from_policy("epsm_rc_5r1c_v2")
    assert legacy.solar_time_reference == "hour_start"
    assert legacy.window_optics_method == "constant_shgc"
    assert current.solar_time_reference == "interval_midpoint"
    assert current.window_optics_method == "simple_glazing_angular_v1"
    assert RCSettings(policy_version="epsm_rc_5r1c_v2") == current
    assert current.effective_configuration()["methods"]["glazing"] == current.window_optics_method
    for kwargs in ({"solar_time_reference": "guess"}, {"window_optics_method": "guess"}):
        with pytest.raises(ValueError):
            RCSettings(**kwargs)


def test_normal_incidence_reproduces_supplied_total_heat_gain():
    optics = SimpleGlazingOptics(u_value_w_m2k=1.8, shgc=0.6)
    assert optics.normal_transmittance == pytest.approx(0.5242934, abs=1e-12)
    t, r, a = optics.beam_properties(np.array([0.0, 60.0, 90.0, 120.0]))
    np.testing.assert_allclose(t + r + a, 1, atol=1e-14)
    assert t[0] + optics.inward_fraction * a[0] == pytest.approx(0.6, abs=1e-12)
    assert t[1] < t[0]
    assert t[2] == t[3] == 0
    assert np.all((t >= 0) & (r >= 0) & (a >= 0))
    assert np.all(t + optics.inward_fraction * a <= 1)


def test_angular_tables_match_energyplus_24_2_reference():
    reference = json.loads(
        (Path(__file__).parents[1] / "fixtures/angular-optics-24.2.json").read_text()
    )
    for case in reference["cases"]:
        optics = SimpleGlazingOptics(case["u_value_w_m2k"], case["shgc"])
        assert optics.normal_transmittance == pytest.approx(case["normal_transmittance"], abs=2e-14)
        assert optics.normal_reflectance == pytest.approx(case["normal_reflectance"], abs=2e-14)
        assert optics.inward_fraction == pytest.approx(case["inward_fraction"], abs=2e-14)
        t, r, _ = optics.beam_properties(np.array(reference["angles_deg"]))
        np.testing.assert_allclose(t, case["transmittance"], atol=2e-13, rtol=0)
        np.testing.assert_allclose(r, case["reflectance"], atol=2e-13, rtol=0)


def test_diffuse_properties_integrate_same_angular_model():
    optics = SimpleGlazingOptics(1.8, 0.6)
    t, r, a = optics.diffuse_properties()
    assert t + r + a == pytest.approx(1, abs=1e-14)
    # Independent dense integration of the public interpolated beam properties.
    theta = np.linspace(0, np.pi / 2, 100001)
    beam = optics.beam_properties(np.degrees(theta))
    for actual, samples in zip((t, r, a), beam):
        expected = np.trapezoid(samples * np.sin(2 * theta), theta)
        assert actual == pytest.approx(expected, abs=2e-8)


@pytest.mark.parametrize(
    "u,g", [(0, 0.6), (7.1, 0.6), (1.8, -0.1), (1.8, 1.1), (float("nan"), 0.6), (1.8, float("nan"))]
)
def test_unsupported_optical_inputs_fail_explicitly(u, g):
    with pytest.raises(ValueError):
        SimpleGlazingOptics(u, g)
