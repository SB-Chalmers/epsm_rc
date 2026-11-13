// File: self_shading.rs
// Purpose: Native horizon-profile and hourly sunlit-fraction kernels for self-shading.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sanjay Somanath
// License: MIT; see the project-root LICENSE for the full permission and warranty terms.
// Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
// Project PI: Alexander Hollberg
// Computational Sustainable Design Group
// Research Area of Sustainable Built Environments
// Chalmers University of Technology, Gothenburg
// Date created: 2026-11-13

//! Native kernels of `epsm_rc.self_shading` (`hourly_sunlit_fraction`).
//!
//! Both functions mirror the NumPy reference in `self_shading.py` and are tested
//! against it; the Python path runs when the extension is absent.
//!
//! - [`horizon_profiles`]: for every receiving point and azimuth degree, the
//!   highest elevation of the building's own face edges. Along a vertical
//!   half-plane the elevation of a straight edge segment is monotonic, so the
//!   maximum over plan crossings of edges is the obstacle silhouette.
//! - [`sunlit_fraction`]: the area-weighted share of points with the sun above
//!   their profile, per hour, using periodic linear interpolation in azimuth.
//!
//! These are geometric extensions of the obstacle treatment of ISO 13790:2008
//! 11.4.4 (own building elements as obstacles), not a clause of the standard.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Horizon profiles in degrees, row-major `(points, count)`, nonnegative.
///
/// `points`, `starts` and `ends` are row-major `(n, 3)` coordinates in metres;
/// azimuths are `k * 360 / count` degrees clockwise from north (`+y`). A plan
/// crossing counts when it lies more than 1 µm along the ray and within the
/// edge (1e-9 tolerance); its height is interpolated along the edge.
pub fn horizon_profiles(points: &[f64], starts: &[f64], ends: &[f64], count: usize) -> Vec<f64> {
    let n_points = points.len() / 3;
    let n_edges = starts.len() / 3;
    let mut profiles = vec![0.0; n_points * count];
    let rays: Vec<(f64, f64)> = (0..count)
        .map(|k| {
            let angle = (k as f64 * 360.0 / count as f64).to_radians();
            (angle.sin(), angle.cos())
        })
        .collect();
    let mut candidates = Vec::with_capacity(n_edges);
    for p in 0..n_points {
        let (px, py, pz) = (points[3 * p], points[3 * p + 1], points[3 * p + 2]);
        // Only edges reaching above the point can raise its horizon above zero.
        candidates.clear();
        for e in 0..n_edges {
            if starts[3 * e + 2] > pz || ends[3 * e + 2] > pz {
                candidates.push(e);
            }
        }
        if candidates.is_empty() {
            continue;
        }
        let row = &mut profiles[p * count..(p + 1) * count];
        for (k, &(rx, ry)) in rays.iter().enumerate() {
            let mut best = 0.0_f64;
            for &e in &candidates {
                let (sx, sy, sz) = (starts[3 * e], starts[3 * e + 1], starts[3 * e + 2]);
                let (dx, dy, dz) = (ends[3 * e] - sx, ends[3 * e + 1] - sy, ends[3 * e + 2] - sz);
                let denominator = rx * dy - ry * dx;
                if denominator.abs() <= 1e-12 {
                    continue;
                }
                let (ox, oy) = (sx - px, sy - py);
                let distance = (ox * dy - oy * dx) / denominator;
                let fraction = (ox * ry - oy * rx) / denominator;
                if distance <= 1e-6 || !(-1e-9..=1.0 + 1e-9).contains(&fraction) {
                    continue;
                }
                let height = sz + fraction.clamp(0.0, 1.0) * dz - pz;
                let elevation = height.atan2(distance);
                if elevation > best {
                    best = elevation;
                }
            }
            row[k] = best.to_degrees();
        }
    }
    profiles
}

/// Hourly area-weighted sunlit share of one receiver's points.
///
/// `profiles` is row-major `(points, count)` with azimuth spacing `step_deg`;
/// `weights` need not be normalised. Night hours (altitude at or below zero)
/// return 1, as the plane irradiance carries no beam then.
pub fn sunlit_fraction(
    profiles: &[f64],
    weights: &[f64],
    altitude_deg: &[f64],
    azimuth_deg: &[f64],
    count: usize,
    step_deg: f64,
) -> Vec<f64> {
    let total: f64 = weights.iter().sum();
    let blocked: Vec<usize> = (0..weights.len())
        .filter(|&p| profiles[p * count..(p + 1) * count].iter().any(|&h| h > 0.0))
        .collect();
    let clear: f64 = total - blocked.iter().map(|&p| weights[p]).sum::<f64>();
    let mut fraction = vec![1.0; altitude_deg.len()];
    if blocked.is_empty() || total <= 0.0 {
        return fraction;
    }
    for (hour, (&altitude, &azimuth)) in altitude_deg.iter().zip(azimuth_deg).enumerate() {
        if altitude <= 0.0 {
            continue;
        }
        let position = azimuth.rem_euclid(360.0) / step_deg;
        let floor = position.floor();
        let low = (floor as usize) % count;
        let high = (low + 1) % count;
        let weight = position - floor;
        let mut lit = clear;
        for &p in &blocked {
            let row = &profiles[p * count..(p + 1) * count];
            if altitude > row[low] * (1.0 - weight) + row[high] * weight {
                lit += weights[p];
            }
        }
        fraction[hour] = lit / total;
    }
    fraction
}

/// Python entry for [`horizon_profiles`]; returns a flat `(points * count)` array.
#[pyfunction]
pub fn self_shading_horizon_profiles<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    starts: PyReadonlyArray1<'py, f64>,
    ends: PyReadonlyArray1<'py, f64>,
    count: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (points, starts, ends) = (points.as_slice()?, starts.as_slice()?, ends.as_slice()?);
    if points.len() % 3 != 0 || starts.len() % 3 != 0 || starts.len() != ends.len() || count == 0 {
        return Err(PyValueError::new_err("self-shading arrays must be (n, 3) and count > 0"));
    }
    let (points, starts, ends) = (points.to_vec(), starts.to_vec(), ends.to_vec());
    let profiles = py.detach(move || horizon_profiles(&points, &starts, &ends, count));
    Ok(PyArray1::from_vec(py, profiles))
}

/// Python entry for [`sunlit_fraction`].
#[pyfunction]
pub fn self_shading_sunlit_fraction<'py>(
    py: Python<'py>,
    profiles: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    altitude_deg: PyReadonlyArray1<'py, f64>,
    azimuth_deg: PyReadonlyArray1<'py, f64>,
    count: usize,
    step_deg: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let profiles = profiles.as_slice()?.to_vec();
    let weights = weights.as_slice()?.to_vec();
    let altitude = altitude_deg.as_slice()?.to_vec();
    let azimuth = azimuth_deg.as_slice()?.to_vec();
    if count == 0 || profiles.len() != weights.len() * count || altitude.len() != azimuth.len() {
        return Err(PyValueError::new_err("self-shading profile, weight or sun arrays mismatch"));
    }
    if !(step_deg > 0.0) {
        return Err(PyValueError::new_err("step_deg must be positive"));
    }
    let fraction = py.detach(move || {
        sunlit_fraction(&profiles, &weights, &altitude, &azimuth, count, step_deg)
    });
    Ok(PyArray1::from_vec(py, fraction))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_wall_profile_is_its_elevation() {
        let profile = horizon_profiles(
            &[0.0, 0.0, 1.0],
            &[-500.0, 3.0, 10.0],
            &[500.0, 3.0, 10.0],
            360,
        );
        assert!((profile[0] - (9.0_f64).atan2(3.0).to_degrees()).abs() < 1e-9);
        assert_eq!(profile[180], 0.0);
    }

    #[test]
    fn night_and_clear_points_are_lit() {
        let mut profiles = vec![0.0; 2 * 360];
        profiles[360..].iter_mut().for_each(|h| *h = 45.0);
        let fraction = sunlit_fraction(&profiles, &[1.0, 3.0], &[-5.0, 30.0, 60.0], &[0.0, 90.0, 90.0], 360, 1.0);
        assert_eq!(fraction, vec![1.0, 0.25, 1.0]);
    }
}
