# File: radiation.py
# Purpose: Shared vectorized radiation operations with explicit legacy policies.
# SPDX-License-Identifier: MIT
# License: MIT; see the project-root LICENSE.
# Date created: 2026-10-18
"""Shared plane-of-array calculations for standalone and Core-adapter radiation.

The legacy standalone path retains its safe-input handling and left-associated
sum. The adapter retains pvlib's sum order, zero-diffuse limit and daylight checks.
Those differences are named here rather than silently changing saved results.
All returned arrays are detached and writable; supplied weather is never changed.
"""

from __future__ import annotations

import numpy as np
from pvlib import irradiance


def plane_irradiance(
    tilt,
    azimuth,
    zenith,
    sun_azimuth,
    dni,
    ghi,
    dhi,
    *,
    albedo,
    dni_extra=None,
    airmass=None,
    model="perez",
    legacy_standalone=False,
) -> dict[str, np.ndarray]:
    """Return detached pvlib irradiance channels in W/m² for one orientation.

    Inputs are hourly arrays and angles in degrees. No file access, cache or
    location defaults occur here. Perez requires explicit extraterrestrial DNI
    and relative air mass. ``legacy_standalone`` preserves the previous standalone
    safe-input and summation rules; it is not a new physical radiation model.
    """
    kwargs = {"model": model, "albedo": albedo}
    if model == "perez":
        kwargs.update(dni_extra=dni_extra, airmass=airmass)
        if legacy_standalone:
            kwargs.update(
                dni_extra=np.where(dni_extra > 0, dni_extra, 1.0),
                airmass=np.where(airmass > 0, airmass, 1.0),
            )
    if legacy_standalone:
        # get_total_irradiance roundtrips projection -> acos -> cos for beam.
        # The legacy standalone API used beam_component's direct projection;
        # preserve it even though the difference is only a few floating-point ulps.
        beam = irradiance.beam_component(tilt, azimuth, zenith, sun_azimuth, dni)
        sky_kwargs = {name: value for name, value in kwargs.items() if name != "albedo"}
        sky = irradiance.get_sky_diffuse(
            tilt, azimuth, zenith, sun_azimuth, dni, ghi, dhi, **sky_kwargs
        )
        if model == "perez":
            sky = np.nan_to_num(sky, nan=0.0)
        ground = irradiance.get_ground_diffuse(tilt, ghi, albedo=albedo)
        return {
            "poa_direct": beam,
            "poa_sky_diffuse": sky,
            "poa_ground_diffuse": ground,
            "poa_diffuse": sky + ground,
            "poa_global": np.maximum(0, beam + sky + ground),
        }
    supplied = irradiance.get_total_irradiance(
        tilt, azimuth, zenith, sun_azimuth, dni, ghi, dhi, **kwargs
    )
    channels = {name: np.asarray(values).copy() for name, values in supplied.items()}
    zero_diffuse = dhi == 0
    if model == "perez" and zero_diffuse.any():
        limit = irradiance.get_total_irradiance(
            tilt, azimuth, zenith, sun_azimuth, dni, ghi, dhi, albedo=albedo, model="isotropic"
        )
        for name, values in channels.items():
            values[zero_diffuse] = np.asarray(limit[name])[zero_diffuse]
    no_radiation = (ghi == 0) & (dni == 0) & (dhi == 0)
    for name, values in channels.items():
        values[no_radiation] = 0.0
        if (np.isnan(values) & (zenith < 90)).any() or np.isinf(values).any():
            raise ValueError("Solar calculation produced invalid daylight irradiance")
        channels[name] = np.nan_to_num(values, nan=0.0)
    return channels
