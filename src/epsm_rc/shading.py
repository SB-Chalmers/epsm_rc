# SPDX-License-Identifier: MIT
"""ISO 13790:2008 Annex G.5.4.2 hourly external-horizon approximation.

One average horizon angle is prepared at each receiving element's representative
point. Equal-azimuth midpoint quadrature covers the 180 degrees facing a facade;
horizontal surfaces use 360 degrees (an explicitly reported extension). Geometry
is never intersected inside the hourly thermal recurrence. Only direct radiation
is reduced, as in equation G.3; diffuse and ground reflection stay unchanged.
This implements that shading method, not whole-standard or national certification.
"""

from __future__ import annotations

import numpy as np


def context_edges(context, origin, orientation_deg):
    """Return neighbour ring edges and roof heights in Core's resolved scene frame.

    All rings/components are retained. Rotating clockwise follows Core's scene
    transform; elevations already share the target's datum. No distance cutoff
    silently discards a potentially important distant tall obstacle.
    """
    theta = np.radians(orientation_deg)
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    starts, ends, tops = [], [], []
    for building in context.buildings:
        top = building.base_elevation_m + building.height_m
        for footprint in building.footprints:
            for ring in (footprint.exterior, *footprint.holes):
                points = (np.asarray(ring, dtype=float) - np.asarray(origin[:2])) @ rotation.T
                if np.array_equal(points[0], points[-1]):
                    points = points[:-1]
                starts.extend(points)
                ends.extend(np.roll(points, -1, axis=0))
                tops.extend([top] * len(points))
    return np.asarray(starts).reshape(-1, 2), np.asarray(ends).reshape(-1, 2), np.asarray(tops)


def average_horizon_angle(receiver, azimuth_deg, tilt_deg, edges, step_deg=10.0):
    """Average the highest obstacle elevation at equally spaced azimuth midpoints.

    Unobstructed directions contribute zero. Multiple obstacles are combined by
    their maximum angle at each direction, avoiding duplicate shadow reductions.
    The step and representative-point sampling are numerical choices, not ISO
    prescriptions. A roof uses the full horizon; other surfaces use their front
    azimuth half-plane, including inclined attic shells.
    """
    starts, ends, tops = edges
    if not len(tops):
        return 0.0
    receiver = np.asarray(receiver, dtype=float)
    above = tops > receiver[2]
    if not above.any():
        return 0.0
    starts, ends, tops = starts[above], ends[above], tops[above]
    width = 360.0 if abs(tilt_deg) < 1e-8 else 180.0
    count = int(np.ceil(width / step_deg))
    angles = np.radians(azimuth_deg - width / 2 + (np.arange(count) + 0.5) * width / count)
    rays = np.column_stack((np.sin(angles), np.cos(angles)))
    delta = ends - starts
    offset = starts - receiver[:2]

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    denominator = cross(rays[:, None, :], delta[None, :, :])
    distance = np.full_like(denominator, np.inf)
    fraction = np.full_like(denominator, np.inf)
    nonparallel = np.abs(denominator) > 1e-12
    np.divide(cross(offset, delta)[None, :], denominator, out=distance, where=nonparallel)
    np.divide(
        cross(offset[None, :, :], rays[:, None, :]), denominator, out=fraction, where=nonparallel
    )
    hit = nonparallel & (distance >= -1e-9) & (fraction >= -1e-9) & (fraction <= 1 + 1e-9)
    elevation = np.where(
        hit, np.degrees(np.arctan2(tops[None, :] - receiver[2], np.maximum(distance, 0))), 0
    )
    return float(np.max(elevation, axis=1).mean())


def surface_centroid(vertices):
    """Area centroid of a simple planar Core face, including concave outlines."""
    points = np.asarray(vertices, dtype=float)
    triangles = np.stack(
        (np.repeat(points[:1], len(points) - 2, axis=0), points[1:-1], points[2:]), axis=1
    )
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal = crosses.sum(axis=0)
    # Signed weights preserve a concave face's excluded triangle areas.
    weights = crosses @ normal
    if weights.sum() <= 0:
        raise ValueError("Context shading receiver has a degenerate planar face")
    return np.average(triangles.mean(axis=1), axis=0, weights=weights)


def receiver_points(resolved, element_orientations, attic_terms):
    """Area-weight resolved faces/panes into one point per physical receiver.

    Core supplies all receiving geometry; construction and operating inputs do
    not enter this preparation. Windows retain their physical storey identity.
    Attic shell faces retain separate points/orientations.
    """
    weighted, weights = {}, {}

    def add(identifier, vertices, area):
        if identifier not in element_orientations or area <= 0:
            return
        centre = surface_centroid(vertices)
        weighted[identifier] = weighted.get(identifier, np.zeros(3)) + area * centre
        weights[identifier] = weights.get(identifier, 0.0) + area

    for face in resolved.surfaces:
        if face["boundary"] == "Outdoors":
            add(face["physical_element_id"], face["vertices"], face["net_area_m2"])
    outdoor = {face["name"] for face in resolved.surfaces if face["boundary"] == "Outdoors"}
    for pane in resolved.openings:
        if pane["parent"] in outdoor:
            add(pane["physical_element_id"], pane["vertices"], pane["area_m2"])
    receivers = {
        key: {
            "point_m": (value / weights[key]).tolist(),
            "tilt_deg": element_orientations[key][0],
            "azimuth_deg": element_orientations[key][1],
        }
        for key, value in weighted.items()
    }
    for term in attic_terms:
        receivers[term["surface_name"]] = {
            "point_m": surface_centroid(term["receiver_vertices"]).tolist(),
            "tilt_deg": term["tilt_deg"],
            "azimuth_deg": (
                resolved.geometry["orientation_deg"]
                if abs(term["tilt_deg"]) < 1e-8
                else term["azimuth_deg"]
            ),
        }
    missing = set(element_orientations) - set(receivers)
    if missing:
        raise ValueError(f"Context shading requires resolved receiver geometry: {sorted(missing)}")
    return receivers


def _contacts_obstacle(receiver, edges, tolerance_m=1e-6):
    """Whether a taller obstacle edge passes through the receiver's plan point."""
    starts, ends, tops = edges
    if not len(tops):
        return False
    receiver = np.asarray(receiver, dtype=float)
    above = tops > receiver[2]
    starts, ends = starts[above], ends[above]
    delta = ends - starts
    lengths = np.einsum("ij,ij->i", delta, delta)
    along = np.einsum("ij,ij->i", receiver[:2] - starts, delta)
    fraction = np.clip(np.divide(along, lengths, out=np.zeros_like(along), where=lengths > 0), 0, 1)
    nearest = starts + fraction[:, None] * delta
    return bool((np.hypot(*(nearest - receiver[:2]).T) <= tolerance_m).any())


def prepare_horizons(receivers, edges, step_deg):
    """Prepare geometry-only angles suitable for reuse across weather/retrofits.

    A representative point lying on a taller context building's outline would
    see a 90 degree horizon in every direction and so lose all direct radiation
    on the whole receiver, however small the contact. Such contact is rejected.
    """
    touching = sorted(
        key for key, value in receivers.items() if _contacts_obstacle(value["point_m"], edges)
    )
    if touching:
        raise ValueError(
            "Context buildings touch the representative point of receivers "
            f"{touching}; one average horizon cannot represent a facade in partial "
            "contact. Model the contact as a shared boundary or separate the footprints."
        )
    return {
        key: average_horizon_angle(
            value["point_m"], value["azimuth_deg"], value["tilt_deg"], edges, step_deg
        )
        for key, value in receivers.items()
    }


def shade_irradiance(channels, solar_altitude_deg, horizon_angle_deg):
    """Apply G.3 without dividing by irradiance, including at night/zero totals."""
    blocked = np.asarray(solar_altitude_deg) < horizon_angle_deg
    direct = np.where(blocked, 0.0, channels["poa_direct"])
    result = dict(channels)
    result["poa_direct"] = direct
    result["poa_global"] = direct + channels["poa_sky_diffuse"] + channels["poa_ground_diffuse"]
    return result
