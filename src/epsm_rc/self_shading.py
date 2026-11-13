# File: self_shading.py
# Purpose: Hourly shading of a building's facades and roofs by its own wings.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-13

"""Hourly shading of a building's facades and roofs by its own wings.

Notes
-----
Concave footprints (L, U, T), courtyards and multi-part buildings of different
heights shade their own envelope. ISO 13790:2008 11.4.4 counts "other elements
of the same building" as external obstacles, and ISO 52016-1 refers solar
shading to ISO 52010-1 without fixing a geometric procedure. Two methods share
the geometry prepared here:

``iso13790_hourly_horizon``
    The building's own outdoor faces are the obstacles of Annex G.5.4.2: one
    average horizon angle per receiver, from its representative point over
    the front half-plane. Direct radiation is removed while the sun is below
    that angle (equation G.3); diffuse radiation is unchanged.

``hourly_sunlit_fraction``
    Every receiver is sampled with a grid of points. Each point gets a horizon
    profile, the highest elevation of the building's own faces in every
    azimuth degree. In each hour a point is sunlit when the sun is above its
    profile at the sun's azimuth, and the receiver's direct radiation is scaled
    by its sunlit area share. Isotropic sky diffuse radiation is scaled by the
    share of the unobstructed sky view factor each point keeps behind its
    profile (exact for the profile, cosine-weighted over the front hemisphere).
    Ground-reflected radiation is unchanged, as in EnergyPlus without
    reflection calculations.

The profile is exact for a point when every obstacle crossing a vertical
half-plane reaches down to at least the point's height, which holds for
extruded parts standing on the same ground. Occluders are the edges of the
resolved outdoor and adiabatic faces, so pitched roofs and parts of different
heights are represented. Geometry is never intersected inside the hourly
recurrence; profiles depend only on geometry and are cached by the caller.

Examples
--------
Example::

    import numpy as np
    from epsm_rc.self_shading import horizon_profiles, sunlit_fraction

    # A point 1 m in front of a 10 m high wall facing it, wall 1 m away.
    point = np.array([[0.0, 0.0, 1.0]])
    edges = (np.array([[-50.0, 1.0, 10.0]]), np.array([[50.0, 1.0, 10.0]]))
    profile = horizon_profiles(point, edges)
    assert profile[0, 0] > 80  # due north is almost fully blocked
    fraction = sunlit_fraction(profile, np.array([1.0]), np.array([30.0]), np.array([0.0]))
    assert fraction[0] == 0.0
"""

from __future__ import annotations

import numpy as np

AZIMUTH_STEP_DEG = 1.0
SAMPLE_SPACING_M = 1.0
MAX_SAMPLES_PER_AXIS = 12
NORMAL_OFFSET_M = 0.01


def occluder_edges(surfaces):
    """Return 3-D start and end points of every edge of the opaque outer faces.

    Parameters
    ----------
    surfaces : list of dict
        Core resolved surfaces; faces with boundary ``Outdoors`` or ``Adiabatic``
        bound the building's volume and can cast shadows on its own envelope.

    Returns
    -------
    tuple of numpy.ndarray
        ``(starts, ends)``, each of shape ``(n, 3)``. Edges that are vertical in
        plan (zero plan length) are dropped: a ray in plan cannot cross them
        except at their end points, which neighbouring edges already supply.
        Of level edges over the same plan segment only the highest is kept.
    """
    starts, ends = [], []
    for face in surfaces:
        if face["boundary"] not in {"Outdoors", "Adiabatic"}:
            continue
        points = np.asarray(face["vertices"], dtype=float)
        starts.append(points)
        ends.append(np.roll(points, -1, axis=0))
    if not starts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    starts, ends = np.concatenate(starts), np.concatenate(ends)
    keep = np.hypot(*(ends[:, :2] - starts[:, :2]).T) > 1e-9
    starts, ends = starts[keep], ends[keep]
    # Orient each edge so the plan key does not depend on traversal direction.
    swap = (ends[:, 0] < starts[:, 0]) | (
        (ends[:, 0] == starts[:, 0]) & (ends[:, 1] < starts[:, 1])
    )
    starts[swap], ends[swap] = ends[swap].copy(), starts[swap].copy()
    level = np.abs(ends[:, 2] - starts[:, 2]) <= 1e-9
    # A ray crossing a plan segment meets every level edge stacked on it at the
    # same distance, so only the highest can set the horizon: storey joints,
    # wall bases and flat-roof outlines below a wall top are dropped. Sloped
    # edges (gables, hips) are always kept.
    index = np.flatnonzero(level)
    plan = np.round(np.hstack((starts[index, :2], ends[index, :2])), 6)
    order = np.lexsort((-starts[index, 2], *plan.T[::-1]))
    plan_sorted = plan[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(plan_sorted[1:] != plan_sorted[:-1], axis=1)
    highest = np.zeros(len(starts), dtype=bool)
    highest[index[order[first]]] = True
    keep = ~level | highest
    return starts[keep], ends[keep]


def face_normal(vertices):
    """Unit outward normal of a planar Core face (counter-clockwise from outside)."""
    points = np.asarray(vertices, dtype=float)
    normal = np.zeros(3)
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        normal += np.cross(a, b)
    length = np.linalg.norm(normal)
    if length <= 0:
        raise ValueError("Self-shading receiver has a degenerate planar face")
    return normal / length


def sample_points(vertices, spacing_m=SAMPLE_SPACING_M, max_per_axis=MAX_SAMPLES_PER_AXIS):
    """Cell-centre grid points inside a planar face, offset 1 cm outwards.

    Parameters
    ----------
    vertices : sequence
        Face vertices in 3-D.
    spacing_m : float
        Target grid spacing; each axis has at least two and at most
        ``max_per_axis`` cells.
    max_per_axis : int
        Upper bound on cells per in-plane axis.

    Returns
    -------
    numpy.ndarray
        Points of shape ``(n, 3)``; every point carries an equal share of area.
    """
    from shapely import contains_xy
    from shapely.geometry import Polygon

    points = np.asarray(vertices, dtype=float)
    normal = face_normal(points)
    # In-plane axes: horizontal along the face where possible, otherwise x.
    reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(reference, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    local = (points - points[0]) @ np.column_stack((u, v))
    low, high = local.min(axis=0), local.max(axis=0)
    counts = np.clip(np.ceil((high - low) / spacing_m), 2, max_per_axis).astype(int)
    grids = [lo + (np.arange(n) + 0.5) * (hi - lo) / n for lo, hi, n in zip(low, high, counts)]
    gu, gv = np.meshgrid(*grids, indexing="ij")
    gu, gv = gu.ravel(), gv.ravel()
    inside = contains_xy(Polygon(local), gu, gv)
    if not inside.any():
        centre = local.mean(axis=0)
        gu, gv, inside = np.array([centre[0]]), np.array([centre[1]]), np.array([True])
    gu, gv = gu[inside], gv[inside]
    return points[0] + np.outer(gu, u) + np.outer(gv, v) + NORMAL_OFFSET_M * normal


def horizon_profiles(points, edges, step_deg=AZIMUTH_STEP_DEG):
    """Highest obstacle elevation, in degrees, seen from each point per azimuth.

    Parameters
    ----------
    points : numpy.ndarray
        Receiving points, shape ``(n, 3)``.
    edges : tuple of numpy.ndarray
        Occluder ``(starts, ends)`` from :func:`occluder_edges`.
    step_deg : float
        Azimuth spacing; azimuths are ``0, step, ...`` clockwise from north.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 360 / step)``, nonnegative. Along a vertical half-plane the
        elevation of a straight edge segment is monotonic, so the maximum over
        edge crossings is the maximum over the obstacle faces.
    """
    starts, ends = edges
    count = int(round(360.0 / step_deg))
    profiles = np.zeros((len(points), count))
    if not len(starts) or not len(points):
        return profiles
    angles = np.radians(np.arange(count) * step_deg)
    rays = np.column_stack((np.sin(angles), np.cos(angles)))  # (a, 2)
    delta = ends - starts  # (e, 3)
    # cross(ray, delta) is shared by every point.
    denominator = rays[:, None, 0] * delta[None, :, 1] - rays[:, None, 1] * delta[None, :, 0]
    nonparallel = np.abs(denominator) > 1e-12
    safe = np.where(nonparallel, denominator, 1.0)
    cross_sd = starts[:, 0] * delta[:, 1] - starts[:, 1] * delta[:, 0]
    # Chunks bound memory at (chunk, azimuths, edges) while keeping numpy loops.
    chunk = max(1, int(2_000_000 // max(1, count * len(starts))))
    for first in range(0, len(points), chunk):
        point = points[first : first + chunk, None, None, :]  # (p, 1, 1, 3)
        # cross(start - point, delta) for every point and edge.
        cross_od = cross_sd[None, None, :] - (
            point[..., 0] * delta[None, None, :, 1] - point[..., 1] * delta[None, None, :, 0]
        )
        distance = cross_od / safe[None]
        ox = starts[None, None, :, 0] - point[..., 0]
        oy = starts[None, None, :, 1] - point[..., 1]
        fraction = (ox * rays[None, :, None, 1] - oy * rays[None, :, None, 0]) / safe[None]
        hit = nonparallel[None] & (distance > 1e-6) & (fraction >= -1e-9) & (fraction <= 1 + 1e-9)
        height = starts[None, None, :, 2] + np.clip(fraction, 0, 1) * delta[None, None, :, 2]
        height = height - point[..., 2]
        elevation = np.where(hit, np.arctan2(height, np.where(hit, distance, 1.0)), 0.0)
        profiles[first : first + chunk] = np.degrees(np.clip(elevation.max(axis=2), 0.0, None))
    return profiles


def _profile_at(profiles, azimuth_deg, step_deg=AZIMUTH_STEP_DEG):
    """Periodic linear interpolation of profiles at hourly sun azimuths."""
    count = profiles.shape[1]
    position = np.mod(np.asarray(azimuth_deg, dtype=float), 360.0) / step_deg
    low = np.floor(position).astype(int) % count
    high = (low + 1) % count
    weight = position - np.floor(position)
    return profiles[:, low] * (1 - weight) + profiles[:, high] * weight


def sunlit_fraction(profiles, weights, solar_altitude_deg, solar_azimuth_deg):
    """Area-weighted share of sample points with the sun above their profile.

    Parameters
    ----------
    profiles : numpy.ndarray
        Horizon profiles of one receiver's points, shape ``(n, a)``.
    weights : numpy.ndarray
        Area weight of each point.
    solar_altitude_deg, solar_azimuth_deg : numpy.ndarray
        Hourly solar position.

    Returns
    -------
    numpy.ndarray
        Hourly sunlit fraction in ``[0, 1]``. Where the sun is behind the
        receiver the plane irradiance carries no beam, so the value there does
        not matter.
    """
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    altitude = np.asarray(solar_altitude_deg, dtype=float)
    azimuth = np.asarray(solar_azimuth_deg, dtype=float)
    fraction = np.ones(altitude.shape)
    # Points with no obstacle in any direction are always lit; night hours carry
    # no beam. Only obstructed points in daylight need the profile lookup.
    blocked = profiles.max(axis=1) > 0
    day = altitude > 0
    if blocked.any() and day.any():
        lit = altitude[None, day] > _profile_at(profiles[blocked], azimuth[day])
        fraction[day] = weights[~blocked].sum() + weights[blocked] @ lit
    return fraction


def sky_view_ratio(profiles, weights, tilt_deg, azimuth_deg, step_deg=AZIMUTH_STEP_DEG):
    """Share of the unobstructed isotropic sky view factor kept behind the profiles.

    For a surface of tilt ``beta`` and azimuth ``phi_n`` a sky direction of
    elevation ``e`` and azimuth ``phi`` contributes ``cos(theta) cos(e) de dphi``
    with ``cos(theta) = sin(beta) cos(e) cos(phi - phi_n) + cos(beta) sin(e)``
    where positive. The elevation integral is closed form; azimuth uses the
    profile resolution. The unobstructed integral reproduces ``(1 + cos beta) / 2``.
    """
    count = profiles.shape[1]
    phi = np.radians(np.arange(count) * step_deg)
    beta = np.radians(tilt_deg)
    a = np.sin(beta) * np.cos(phi - np.radians(azimuth_deg))
    b = np.cos(beta)
    # Lowest elevation at which the direction is in front of the surface.
    with np.errstate(divide="ignore", invalid="ignore"):
        front = np.where(a >= 0, 0.0, np.where(b > 1e-12, np.arctan(-a / b), np.pi / 2))
    front = np.clip(front, 0.0, np.pi / 2)

    def integral(low):
        top = np.pi / 2
        low = np.clip(low, 0.0, top)
        f = lambda e: a * (e / 2 + np.sin(2 * e) / 4) + b * np.sin(e) ** 2 / 2  # noqa: E731
        return f(top) - f(low)

    clear = integral(front).sum()
    if clear <= 0:
        return 1.0
    kept = integral(np.maximum(np.radians(profiles), front[None, :])).sum(axis=1)
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    return float(weights @ (kept / clear))


def average_front_horizon(point, azimuth_deg, tilt_deg, edges, step_deg=10.0):
    """ISO 13790 G.5.4.2 average horizon angle from the building's own faces.

    Same quadrature as the context method: equal-azimuth midpoints over the
    front half-plane (full circle for horizontal receivers).
    """
    width = 360.0 if abs(tilt_deg) < 1e-8 else 180.0
    count = int(np.ceil(width / step_deg))
    azimuths = azimuth_deg - width / 2 + (np.arange(count) + 0.5) * width / count
    profile = horizon_profiles(np.asarray([point], dtype=float), edges, step_deg=1.0)[0]
    return float(_profile_at(profile[None, :], azimuths)[0].mean())


def prepare_receivers(resolved, element_orientations, attic_terms, *, method):
    """Geometry-only self-shading data per receiving element.

    Parameters
    ----------
    resolved : ResolvedBuildingInput
        Core resolved geometry, in the scene frame used by solar positions.
    element_orientations : dict
        ``{element_id: (tilt_deg, azimuth_deg)}`` for every receiver.
    attic_terms : list of dict
        Attic shell terms with ``surface_name`` and ``receiver_vertices``.
    method : str
        ``hourly_sunlit_fraction`` or ``iso13790_hourly_horizon``.

    Returns
    -------
    dict
        Per receiver: ``profiles`` and ``weights`` (sunlit fraction) or
        ``horizon_deg`` (ISO horizon).
    """
    from .shading import receiver_points

    edges = occluder_edges(resolved.surfaces)
    if method == "iso13790_hourly_horizon":
        receivers = receiver_points(resolved, element_orientations, attic_terms)
        result = {}
        for key, value in receivers.items():
            normal = np.zeros(3)
            tilt, azimuth = value["tilt_deg"], value["azimuth_deg"]
            normal[:] = (
                np.sin(np.radians(tilt)) * np.sin(np.radians(azimuth)),
                np.sin(np.radians(tilt)) * np.cos(np.radians(azimuth)),
                np.cos(np.radians(tilt)),
            )
            point = np.asarray(value["point_m"]) + NORMAL_OFFSET_M * normal
            result[key] = {"horizon_deg": average_front_horizon(point, azimuth, tilt, edges)}
        return result
    faces = {}
    for face in resolved.surfaces:
        if face["boundary"] == "Outdoors" and face["physical_element_id"] in element_orientations:
            faces.setdefault(face["physical_element_id"], []).append(face["vertices"])
    outdoor = {face["name"] for face in resolved.surfaces if face["boundary"] == "Outdoors"}
    for pane in resolved.openings:
        if pane["parent"] in outdoor and pane["physical_element_id"] in element_orientations:
            faces.setdefault(pane["physical_element_id"], []).append(pane["vertices"])
    for term in attic_terms:
        faces.setdefault(term["surface_name"], []).append(term["receiver_vertices"])
    result = {}
    for key, polygons in faces.items():
        points, weights = [], []
        for vertices in polygons:
            grid = sample_points(vertices)
            area = 0.5 * np.linalg.norm(
                sum(np.cross(a, b) for a, b in zip(np.asarray(vertices), np.roll(vertices, -1, 0)))
            )
            points.append(grid)
            weights.append(np.full(len(grid), area / len(grid)))
        points = np.concatenate(points)
        profiles = horizon_profiles(points, edges)
        # Points that see no obstacle anywhere add nothing to compute hourly.
        result[key] = {"profiles": profiles, "weights": np.concatenate(weights)}
    missing = set(element_orientations) - set(result)
    if missing:
        raise ValueError(f"Self-shading requires resolved receiver geometry: {sorted(missing)}")
    return result


def shade_irradiance(
    channels, solar_altitude_deg, solar_azimuth_deg, receiver, tilt_deg, azimuth_deg
):
    """Apply one receiver's self-shading to its plane irradiance channels.

    Parameters
    ----------
    channels : dict
        ``poa_direct``, ``poa_sky_diffuse`` and ``poa_ground_diffuse`` arrays.
    solar_altitude_deg, solar_azimuth_deg : numpy.ndarray
        Hourly solar position.
    receiver : dict
        One entry of :func:`prepare_receivers`.
    tilt_deg, azimuth_deg : float
        Receiver orientation, used for the sky view ratio.

    Returns
    -------
    dict
        New channels with ``poa_direct``, ``poa_sky_diffuse`` and ``poa_global``
        reduced; ground reflection unchanged.
    """
    result = dict(channels)
    if "horizon_deg" in receiver:
        blocked = np.asarray(solar_altitude_deg) < receiver["horizon_deg"]
        result["poa_direct"] = np.where(blocked, 0.0, channels["poa_direct"])
    else:
        fraction = sunlit_fraction(
            receiver["profiles"], receiver["weights"], solar_altitude_deg, solar_azimuth_deg
        )
        result["poa_direct"] = channels["poa_direct"] * fraction
        ratio = sky_view_ratio(receiver["profiles"], receiver["weights"], tilt_deg, azimuth_deg)
        result["poa_sky_diffuse"] = channels["poa_sky_diffuse"] * ratio
        if "poa_diffuse" in channels:
            result["poa_diffuse"] = result["poa_sky_diffuse"] + channels["poa_ground_diffuse"]
    result["poa_global"] = (
        result["poa_direct"] + result["poa_sky_diffuse"] + channels["poa_ground_diffuse"]
    )
    return result
