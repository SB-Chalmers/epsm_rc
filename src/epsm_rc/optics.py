# File: optics.py
# Purpose: Explicit simple-glazing angular optics for reduced thermal models.
# SPDX-License-Identifier: MIT AND LicenseRef-EnergyPlus-24.2
# Date created: 2026-10-18
# Correlations adapted from EnergyPlus version 24.2 Material.cc and WindowManager.cc.
# Upstream copyright and complete terms are retained in notices/EnergyPlus.txt.
"""Nominal angular heat gains from supplied U-value and normal-incidence SHGC.

The model follows EnergyPlus version 24.2's equivalent-layer and 28-cell angular
correlations, including its summer-film interpolation. It retains the nominal
inward-flowing fraction at every angle. This is an explicit reduced approximation:
it does not solve EnergyPlus's temperature-dependent glazing heat balance.

Beam optics interpolate the source's ten-degree angular samples. Diffuse optics
integrate that same piecewise-linear model exactly over an isotropic hemisphere;
this conserves optical energy rather than copying the source's coarse quadrature.
Perez sky and ground diffuse use that hemispherical optical response. No measured
angular performance, detailed sky radiance or frame geometry is inferred.

References
----------
EnergyPlus 24.2 Engineering Reference, Window Calculation Module, Simple Window
Model, steps 1, 4, 5 and 7. Numerical source: v24.2.0 Material.cc and
WindowManager.cc (TransAndReflAtPhi). See docs/radiation.md for source links.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log

import numpy as np

# Tables 2 and 3, curves A-J. R stores the pre-subtraction polynomial;
# the angular reflectance modifier is R(mu) - T(mu).
_TRANS = np.array(
    [
        [0, 3.36, -3.85, 1.49, 0.01],
        [0, 2.83, -2.42, 0.04, 0.55],
        [0, 2.45, -1.58, -0.64, 0.77],
        [0, 2.85, -2.58, 0.40, 0.35],
        [0, 1.51, 2.49, -5.87, 2.88],
        [0, 1.21, 3.14, -6.37, 3.03],
        [0, 1.09, 3.54, -6.84, 3.23],
        [0, 0.98, 3.83, -7.13, 3.33],
        [0, 0.79, 3.93, -6.86, 3.15],
        [0, 0.08, 6.02, -8.84, 3.74],
    ]
)
_REFLECT = np.array(
    [
        [1, -0.70, 2.57, -3.20, 1.33],
        [1, -1.87, 6.50, -7.86, 3.23],
        [1, -2.52, 8.40, -9.86, 3.99],
        [1, -1.85, 6.40, -7.64, 3.11],
        [1, -1.57, 5.60, -6.82, 2.80],
        [1, -3.15, 10.98, -13.14, 5.32],
        [1, -3.25, 11.32, -13.54, 5.49],
        [1, -3.39, 11.70, -13.94, 5.64],
        [1, -4.06, 13.55, -15.74, 6.27],
        [1, -4.35, 14.27, -16.32, 6.39],
    ]
)
_ANGLES = np.arange(0.0, 91.0, 10.0)
for _table in (_TRANS, _REFLECT, _ANGLES):
    _table.flags.writeable = False


def _blend(x, lo, hi, left, right):
    weight = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return left + weight * (right - left)


def _angular_coefficients(u, g, table):
    """Interpolate the source's 28 cells through their three U-boundary families."""
    a, b, c, d, e, f, _, h, _, j = table
    fghi = table[5:9].mean(axis=0)
    fh, bdcd = (f + h) / 2, (b + d + c + d) / 4
    low = _blend(g, 0.35, 0.45, j, e)
    if g >= 0.5:
        middle = _blend(g, 0.5, 0.55, fghi, e)
    elif g >= 0.3:
        middle = fghi
    else:
        middle = _blend(g, 0.25, 0.3, fh, fghi)
    if g >= 0.6:
        high = _blend(g, 0.6, 0.65, bdcd, a)
    elif g >= 0.45:
        high = bdcd
    else:
        high = _blend(g, 0.3, 0.45, d, bdcd)
    if u < 1.4195:
        return low
    if u <= 1.7034:
        return _blend(u, 1.4195, 1.7034, low, middle)
    if u < 3.4068:
        return middle
    return _blend(u, 3.4068, 4.5424, middle, high)


def _normal_properties(u, g):
    ri = 1 / (0.359073 * log(u) + 6.949915) if u < 5.85 else 1 / (1.788041 * u - 2.886625)
    ro = 1 / (0.025342 * u + 29.163853)
    layer_resistance = 1 / u - ri - ro
    if min(ri, ro, layer_resistance) <= 0:
        raise ValueError("Simple glazing requires positive nominal film and layer resistances")
    low = 0.41040 * g if g <= 0.15 else 0.085775 * g * g + 0.963954 * g - 0.084958
    high = 0.939998 * g * g + 0.20332 * g if g < 0.7206 else 1.30415 * g - 0.30515
    t = _blend(u, 3.4, 4.5, low, high)
    delta = g - t
    ri_low = 1 / (199.8208128 * delta**3 - 90.639733 * delta**2 + 19.737055 * delta + 6.766575)
    ri_high = 1 / (29.436546 * delta**3 - 21.943415 * delta**2 + 9.945872 * delta + 7.426151)
    ro_low = 1 / (5.763355 * delta + 20.541528)
    ro_high = 1 / (2.225824 * delta + 20.577080)
    if u < 3.4:
        ri_s, ro_s = ri_low, ro_low
    elif u > 4.5:
        ri_s, ro_s = ri_high, ro_high
    else:
        # Retain the actual 24.2 implementation, including its reversed film
        # interpolation difference; the Engineering Reference only says interpolate.
        weight = (u - 3.4) / (4.5 - 3.4)
        ri_s = weight * (ri_low - ri_high) + ri_low
        ro_s = weight * (ro_low - ro_high) + ro_low
    inward = (ro_s + 0.5 * layer_resistance) / (ro_s + layer_resistance + ri_s)
    r = 1 - t - delta / inward
    if not (0 <= t <= 1 and 0 <= r <= 1 - t and 0 <= inward <= 1):
        raise ValueError(
            "U-value and SHGC are outside the physical simple-glazing correlation domain"
        )
    return t, r, inward


def _hemisphere(values):
    """Exact integral of linearly interpolated angular optics times sin(2 theta)."""
    angles = np.radians(_ANGLES)
    left, right = angles[:-1], angles[1:]
    widths = right - left
    slopes = np.diff(values) / widths
    constant = 0.5 * (np.cos(2 * left) - np.cos(2 * right))
    linear = -0.5 * widths * np.cos(2 * right) + 0.25 * (np.sin(2 * right) - np.sin(2 * left))
    return float(np.sum(values[:-1] * constant + slopes * linear))


@dataclass(frozen=True)
class SimpleGlazingOptics:
    """Explicit nominal angular approximation for specular glazing.

    Parameters
    ----------
    u_value_w_m2k : float
        Supplied whole-window U-value, strictly positive and at most 7 W/(m² K).
    shgc : float
        Supplied normal-incidence total solar heat gain fraction in [0, 1].

    Raises
    ------
    ValueError
        Nonfinite inputs, invalid resistance or unphysical correlated T/R. No
        arbitrary coefficients, missing-data substitutions or fallback model are used.
    """

    u_value_w_m2k: float
    shgc: float
    normal_transmittance: float = field(init=False)
    normal_reflectance: float = field(init=False)
    inward_fraction: float = field(init=False)
    _transmittance: tuple = field(init=False, repr=False)
    _reflectance: tuple = field(init=False, repr=False)

    def __post_init__(self):
        u, g = self.u_value_w_m2k, self.shgc
        if not (np.isfinite(u) and 0 < u <= 7 and np.isfinite(g) and 0 <= g <= 1):
            raise ValueError("Simple glazing requires finite 0 < U <= 7 and 0 <= SHGC <= 1")
        t0, r0, inward = _normal_properties(u, g)
        mu = np.cos(np.radians(_ANGLES))
        tc = _angular_coefficients(u, g, _TRANS)
        rc = _angular_coefficients(u, g, _REFLECT)
        trans = np.polynomial.polynomial.polyval(mu, tc)
        reflect = np.polynomial.polynomial.polyval(mu, rc) - trans
        trans[0], reflect[0] = 1.0, 0.0
        t = np.clip(t0 * trans, 0, 1)
        r = np.maximum(0, np.minimum(0.9999 - t, r0 * (1 - reflect) + reflect))
        # The source's grazing-angle sample is exactly zero cosine.
        t[-1], r[-1] = 0.0, 0.9999
        object.__setattr__(self, "normal_transmittance", t0)
        object.__setattr__(self, "normal_reflectance", r0)
        object.__setattr__(self, "inward_fraction", inward)
        object.__setattr__(self, "_transmittance", tuple(t))
        object.__setattr__(self, "_reflectance", tuple(r))

    def beam_properties(self, incidence_deg):
        """Return detached transmittance, reflectance and absorptance arrays.

        Angles must be finite in [0, 180] degrees. Back-facing samples transmit
        no beam and reflect it all; the incident beam itself is zero there.
        """
        angles = np.asarray(incidence_deg, dtype=float)
        if not np.isfinite(angles).all() or ((angles < 0) | (angles > 180)).any():
            raise ValueError("Incidence angles must be finite in [0, 180] degrees")
        t = np.interp(angles, _ANGLES, self._transmittance)
        r = np.interp(angles, _ANGLES, self._reflectance)
        t, r = np.where(angles > 90, 0.0, t), np.where(angles > 90, 1.0, r)
        return t, r, np.maximum(0.0, 1 - t - r)

    def diffuse_properties(self):
        """Return hemispherical T, R and A of the same angular model; no I/O."""
        t, r = (
            _hemisphere(np.asarray(self._transmittance)),
            _hemisphere(np.asarray(self._reflectance)),
        )
        return t, r, max(0.0, 1 - t - r)
