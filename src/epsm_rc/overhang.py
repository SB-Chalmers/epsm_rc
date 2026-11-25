# File: overhang.py
# Purpose: ISO 13790:2008 Annex G.5.4.3 hourly overhang shading of window panes.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-13

"""ISO 13790:2008 Annex G.5.4.3 hourly overhang shading of window panes.

Notes
-----
The hourly method of Annex G.5.4.3 reduces direct radiation on a window by

    F_ov,dir = max(0, 1 - 0.5 tan(alpha) / tan(90 - S_h))        (G.4)

and sky-diffuse radiation by

    F_ov,dif = 1 - alpha / 90                                     (G.5)

with ground-reflected radiation unchanged, which together give the total factor
of equation G.6. ``alpha`` is the overhang angle of Figure G.2, seen from the
window mid-height, and ``S_h`` the solar height.

Core resolves each overhang as a shading surface. For every window pane, the
nearest overhang above it on the same facade that spans the pane centre sets
``tan(alpha) = depth / (overhang height - pane mid-height)``. A physical window
element can hold several panes; its factor is the area-weighted mean of its
panes' factors, which is exact because every pane of one element lies in the
same plane and so receives the same irradiance. The overhang is treated as
infinitely wide, as in the standard: side extensions and finite-width end
effects are not resolved. Geometry is never intersected inside the hourly
recurrence. This implements that shading clause, not whole-standard or
national certification.

Examples
--------
Example::

    import numpy as np
    from epsm_rc.overhang import shade_irradiance

    channels = {
        "poa_direct": np.array([400.0]),
        "poa_sky_diffuse": np.array([90.0]),
        "poa_ground_diffuse": np.array([10.0]),
    }
    # One pane under a 45 degree overhang, sun at 30 degrees.
    shaded = shade_irradiance(channels, np.array([30.0]), ((1.0, 45.0),))
    assert np.isclose(shaded["poa_direct"][0], 400.0 * (1 - 0.5 * np.tan(np.radians(30))))
    assert np.isclose(shaded["poa_sky_diffuse"][0], 90.0 * 0.5)
"""

from __future__ import annotations

import numpy as np

#: Vertical and plan tolerance, in metres, for matching overhangs to panes.
TOLERANCE_M = 1e-6


def _pane_frame(pane):
    """Facade origin, along-facade unit vector and outward unit normal of a pane."""
    logical = np.asarray(pane["metadata"]["logical_vertices"], dtype=float)
    along = logical[1] - logical[0]
    along[2] = 0.0
    along /= np.linalg.norm(along)
    # Core's convention: outward = along × up (see epsm_core.facades.shading_geometry).
    outward = np.cross(along, [0.0, 0.0, 1.0])
    return logical[0], along, outward


def overhang_angle_deg(pane, overhangs):
    """Overhang angle (Figure G.2) of one pane, or 0 when no overhang covers it.

    Parameters
    ----------
    pane : mapping
        A Core resolved opening with ``vertices`` and ``metadata`` (``facade_id``,
        ``logical_vertices``).
    overhangs : sequence of mapping
        Core resolved shading surfaces of kind ``overhang``.

    Returns
    -------
    float
        Angle in degrees between the horizontal through the pane mid-height and
        the overhang's outer edge, in [0, 90).
    """
    vertices = np.asarray(pane["vertices"], dtype=float)
    top, mid = vertices[:, 2].max(), 0.5 * (vertices[:, 2].max() + vertices[:, 2].min())
    origin, along, outward = _pane_frame(pane)
    centre = float(np.dot(vertices.mean(axis=0) - origin, along))
    best = None
    for shade in overhangs:
        if shade.get("facade_id") != pane["metadata"].get("facade_id"):
            continue
        points = np.asarray(shade["vertices"], dtype=float)
        height = points[:, 2].mean()
        if height < top - TOLERANCE_M:
            continue
        span = (points - origin) @ along
        if not span.min() - TOLERANCE_M <= centre <= span.max() + TOLERANCE_M:
            continue
        depth = float(((points - origin) @ outward).max())
        if depth <= TOLERANCE_M:
            continue
        # Nearest overhang above the pane; the deeper one if two share a height.
        if (
            best is None
            or height < best[0] - TOLERANCE_M
            or (abs(height - best[0]) <= TOLERANCE_M and depth > best[1])
        ):
            best = (height, depth)
    if best is None:
        return 0.0
    height, depth = best
    return float(np.degrees(np.arctan2(depth, max(height - mid, TOLERANCE_M))))


def overhang_panes(resolved):
    """Area and overhang angle of each outdoor window pane, per physical element.

    Returns
    -------
    dict
        ``{physical_element_id: ((area_m2, alpha_deg), ...)}`` for elements with at
        least one pane under an overhang; panes without one keep ``alpha = 0`` so
        the element's area-weighted factor stays exact.
    """
    overhangs = [shade for shade in resolved.shading if shade.get("kind") == "overhang"]
    if not overhangs:
        return {}
    outdoor = {face["name"] for face in resolved.surfaces if face["boundary"] == "Outdoors"}
    panes = {}
    for pane in resolved.openings:
        if pane["parent"] in outdoor and pane["area_m2"] > 0:
            panes.setdefault(pane["physical_element_id"], []).append(
                (float(pane["area_m2"]), overhang_angle_deg(pane, overhangs))
            )
    return {
        element: tuple(rows)
        for element, rows in panes.items()
        if any(alpha > 0 for _, alpha in rows)
    }


def overhang_factors(solar_altitude_deg, panes):
    """Area-weighted direct (G.4) and sky-diffuse (G.5) factors of one element."""
    areas = np.array([area for area, _ in panes], dtype=float)
    alphas = np.radians([alpha for _, alpha in panes])
    weights = areas / areas.sum()
    # tan(90 - S_h) = 1 / tan(S_h); below the horizon there is no beam to shade.
    height = np.radians(np.clip(np.asarray(solar_altitude_deg, dtype=float), 0.0, 90.0))
    tan_height = np.tan(np.minimum(height, np.radians(89.999)))
    direct = np.clip(1.0 - 0.5 * np.tan(alphas)[:, None] * tan_height[None, :], 0.0, 1.0)
    return weights @ direct, float(weights @ (1.0 - np.degrees(alphas) / 90.0))


def shade_irradiance(channels, solar_altitude_deg, panes):
    """Apply G.4 and G.5 to one element's plane irradiance; G.6 follows by summation."""
    direct_factor, diffuse_factor = overhang_factors(solar_altitude_deg, panes)
    result = dict(channels)
    result["poa_direct"] = channels["poa_direct"] * direct_factor
    result["poa_sky_diffuse"] = channels["poa_sky_diffuse"] * diffuse_factor
    result["poa_global"] = (
        result["poa_direct"] + result["poa_sky_diffuse"] + channels["poa_ground_diffuse"]
    )
    return result


def overhang_geometry(pane, overhangs):
    """Depth, head gap and height of the nearest overhang over one pane.

    Parameters
    ----------
    pane : mapping
        A Core resolved opening, as for :func:`overhang_angle_deg`.
    overhangs : sequence of mapping
        Core resolved shading surfaces of kind ``overhang``.

    Returns
    -------
    tuple of float or None
        ``(depth_m, gap_m, height_m)``: outward depth of the overhang, vertical
        distance from the pane head to the overhang, and pane height. None when
        no overhang covers the pane centre.
    """
    vertices = np.asarray(pane["vertices"], dtype=float)
    top, bottom = vertices[:, 2].max(), vertices[:, 2].min()
    alpha = overhang_angle_deg(pane, overhangs)
    if alpha <= 0.0:
        return None
    mid = 0.5 * (top + bottom)
    # Recover the matched overhang from its angle at mid-height: tan(alpha) = depth / (h - mid).
    origin, along, outward = _pane_frame(pane)
    centre = float(np.dot(vertices.mean(axis=0) - origin, along))
    for shade in overhangs:
        if shade.get("facade_id") != pane["metadata"].get("facade_id"):
            continue
        points = np.asarray(shade["vertices"], dtype=float)
        height = points[:, 2].mean()
        span = (points - origin) @ along
        depth = float(((points - origin) @ outward).max())
        if (
            height >= top - TOLERANCE_M
            and span.min() - TOLERANCE_M <= centre <= span.max() + TOLERANCE_M
            and depth > TOLERANCE_M
            and np.isclose(
                np.degrees(np.arctan2(depth, max(height - mid, TOLERANCE_M))), alpha, atol=1e-9
            )
        ):
            return depth, max(height - top, 0.0), top - bottom
    return None


def overhang_pane_geometry(resolved):
    """Area and overhang geometry of each outdoor window pane, per physical element.

    Returns
    -------
    dict
        ``{physical_element_id: ((area_m2, depth_m, gap_m, height_m), ...)}`` for
        elements with at least one covered pane; uncovered panes carry zero depth.
    """
    overhangs = [shade for shade in resolved.shading if shade.get("kind") == "overhang"]
    if not overhangs:
        return {}
    outdoor = {face["name"] for face in resolved.surfaces if face["boundary"] == "Outdoors"}
    panes = {}
    for pane in resolved.openings:
        if pane["parent"] in outdoor and pane["area_m2"] > 0:
            geometry = overhang_geometry(pane, overhangs) or (0.0, 0.0, 1.0)
            panes.setdefault(pane["physical_element_id"], []).append(
                (float(pane["area_m2"]), *map(float, geometry))
            )
    return {element: tuple(rows) for element, rows in panes.items() if any(r[1] > 0 for r in rows)}


def geometric_factors(solar_altitude_deg, solar_azimuth_deg, facade_azimuth_deg, panes):
    """Sunlit beam fraction and sky-diffuse view fraction under infinite overhangs.

    Parameters
    ----------
    solar_altitude_deg, solar_azimuth_deg : array_like
        Hourly solar position in degrees (azimuth clockwise from north).
    facade_azimuth_deg : float
        Outward facade normal, clockwise from north.
    panes : sequence of tuple
        ``(area_m2, depth_m, gap_m, height_m)`` from :func:`overhang_pane_geometry`.

    Returns
    -------
    direct : ndarray
        Area-weighted sunlit fraction of the panes per hour. The shadow edge
        lies ``depth · tan(profile angle)`` below the overhang, with
        ``tan(profile) = tan(altitude) / cos(azimuth difference)``.
    diffuse : float
        Area-weighted share of the isotropic sky-diffuse irradiance a vertical
        pane keeps. A point ``y`` below an infinitely long overhang of depth
        ``d`` keeps ``y / sqrt(y² + d²)``; the pane mean is
        ``(sqrt((g + H)² + d²) - sqrt(g² + d²)) / H``.

    Notes
    -----
    Exact for an overhang that is infinitely wide relative to the pane, which
    is the same width assumption as ISO 13790 Annex G.5.4.3. Unlike equations
    G.4 and G.5, the beam uses the profile angle and the pane's head gap and
    height, and the diffuse share is the isotropic view factor.
    """
    rows = np.asarray(panes, dtype=float)
    areas, depth, gap, height = rows.T
    weights = areas / areas.sum()
    altitude = np.radians(np.clip(np.asarray(solar_altitude_deg, dtype=float), 0.0, 90.0))
    relative = np.radians(np.asarray(solar_azimuth_deg, dtype=float) - facade_azimuth_deg)
    facing = np.cos(relative)
    with np.errstate(divide="ignore", invalid="ignore"):
        tan_profile = np.where(
            facing > 1e-9, np.tan(np.minimum(altitude, np.radians(89.999))) / facing, np.inf
        )
    shadow = depth[:, None] * tan_profile[None, :]
    shaded = np.clip((shadow - gap[:, None]) / height[:, None], 0.0, 1.0)
    # Sun behind the facade gives no beam; keep the factor at 1 so zero stays zero.
    shaded = np.where(facing[None, :] > 1e-9, shaded, 0.0)
    direct = weights @ (1.0 - shaded)
    sky = np.where(
        depth > 0,
        (np.hypot(gap + height, depth) - np.hypot(gap, depth)) / height,
        1.0,
    )
    return direct, float(weights @ sky)


def shade_irradiance_geometric(
    channels, solar_altitude_deg, solar_azimuth_deg, facade_azimuth_deg, panes
):
    """Apply :func:`geometric_factors` to one element's plane irradiance."""
    direct_factor, diffuse_factor = geometric_factors(
        solar_altitude_deg, solar_azimuth_deg, facade_azimuth_deg, panes
    )
    result = dict(channels)
    result["poa_direct"] = channels["poa_direct"] * direct_factor
    result["poa_sky_diffuse"] = channels["poa_sky_diffuse"] * diffuse_factor
    result["poa_global"] = (
        result["poa_direct"] + result["poa_sky_diffuse"] + channels["poa_ground_diffuse"]
    )
    return result
