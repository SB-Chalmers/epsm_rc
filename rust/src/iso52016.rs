// File: iso52016.rs
// Purpose: ISO 52016-1 Clause 6.5 hourly recurrence on a pre-inverted network.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sanjay Somanath
// License: MIT; see the project-root LICENSE for the full permission and warranty terms.
// Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
// Project PI: Alexander Hollberg
// Computational Sustainable Design Group
// Research Area of Sustainable Built Environments
// Chalmers University of Technology, Gothenburg
// Date created: 2026-11-12

//! ISO 52016-1:2017 Clause 6.5 hourly zone recurrence on a pre-inverted network.
//!
//! Python assembles the network (`epsm_rc.iso52016.Zone52016`): five nodes per
//! opaque element, two per window and one air node (Clause 6.5.6 and 6.5.7),
//! with Table 25 surface coefficients. Backward Euler gives, each hour,
//!
//! ```text
//! (A0 + H_ve e0 e0ᵀ) θ_t = C/Δt θ_{t−1} + P f_t + H_ve θ_e e0 + φ_HC h
//! ```
//!
//! where A0 is constant, `e0` selects the air node, `f_t` holds the forcing
//! (outdoor and ground temperature, internal and solar gains, sky term, absorbed
//! solar per element) and `h` distributes HVAC power (fully convective, Annex B).
//! Python inverts A0 once (M = A0⁻¹). Here the hourly H_ve enters through the
//! Sherman-Morrison identity, so no matrix is factorised inside the loop.
//!
//! Because the network is linear, the free-floating state plus the response to
//! one watt gives the exact thermostat power (Clause 6.5.5.2 without trials).
//! The control temperature is the air or operative temperature, set by the
//! `control` row. Airflow and the heat-recovery rule match
//! `epsm_rc._forcing.HourlyInputs.at`. The year runs `1 + warmup_passes` times
//! and the last pass is returned.

/// Constant network matrices from `Zone52016._build`, row-major.
pub struct Network52016 {
    /// Number of nodes; node 0 is the air node.
    pub n: usize,
    /// Number of forcing columns: five shared ones plus one per element.
    pub k: usize,
    /// Row-major n x n inverse of the constant matrix.
    pub m: Vec<f64>,
    /// Row-major n x k product of the inverse and the forcing map.
    pub mp: Vec<f64>,
    /// Column of M for the air node (response to H_ve coupling).
    pub m_air: Vec<f64>,
    /// M times the HVAC distribution vector: response to 1 W of HVAC.
    pub m_hvac: Vec<f64>,
    /// Node capacity divided by the time step, C/Δt (W/K).
    pub c_dt: Vec<f64>,
    /// Weights giving the control temperature from the node vector.
    pub control: Vec<f64>,
    /// Index of each element's inside surface node.
    pub inner: Vec<usize>,
    /// Element areas (m²), for the mean radiant and operative temperatures.
    pub areas: Vec<f64>,
}

/// Hourly forcing columns, each one value per hour.
pub struct Hourly52016<'a> {
    /// Outdoor air temperature (°C).
    pub theta_e: &'a [f64],
    /// Ground boundary temperature (°C).
    pub theta_gr: &'a [f64],
    /// Internal gains (W).
    pub phi_int: &'a [f64],
    /// Solar gains transmitted through windows (W).
    pub phi_sol: &'a [f64],
    /// Row-major hours x elements absorbed solar (W/m²).
    pub absorbed: &'a [f64],
    /// Heating setpoint (°C).
    pub set_heating: &'a [f64],
    /// Cooling setpoint (°C).
    pub set_cooling: &'a [f64],
    /// Heating capacity (W); 0 when heating is unavailable.
    pub heat_limit: &'a [f64],
    /// Cooling capacity as a nonpositive power (W).
    pub cool_limit: &'a [f64],
    /// Wind speed (m/s) for the infiltration correction.
    pub wind: &'a [f64],
    /// Base infiltration air changes per hour before correction.
    pub infiltration: &'a [f64],
    /// Natural ventilation air changes per hour (unrecovered).
    pub natural: &'a [f64],
    /// Mechanical ventilation air changes per hour (recoverable).
    pub ventilation: &'a [f64],
}

/// Constants for H_ve = ρc V n_eff / 3600 and the heat-recovery rule.
pub struct Airflow52016 {
    /// Infiltration correction c0 + c1 |θ_air − θ_e| + c2 w + c3 w².
    pub coefficients: [f64; 4],
    pub volume_m3: f64,
    pub air_capacity_j_m3_k: f64,
    /// Sensible recovery effectiveness of the mechanical airflow.
    pub recovery_effectiveness: f64,
    /// Use recovery only when it reduces the load (`heat_recovery="when_beneficial"`).
    pub recovery_bypass: bool,
}

/// Hourly HVAC power (W, + heating), air and operative temperatures (°C).
pub struct Output52016 {
    pub power: Vec<f64>,
    pub air: Vec<f64>,
    pub operative: Vec<f64>,
}

/// out += matrix · x for a row-major `rows × cols` matrix.
fn matvec_add(matrix: &[f64], rows: usize, cols: usize, x: &[f64], out: &mut [f64]) {
    for (row, value) in out.iter_mut().enumerate().take(rows) {
        let line = &matrix[row * cols..(row + 1) * cols];
        let mut sum = 0.0;
        for (a, b) in line.iter().zip(x) {
            sum += a * b;
        }
        *value += sum;
    }
}

/// Run the hourly recurrence and return the last pass.
///
/// Each hour: infiltration from the previous air temperature; heat recovery
/// active only when it reduces the load if `recovery_bypass` (same rule and
/// tolerances as [`crate::domain::DomainArrays::apply_at`]); the free state
/// y = M (C/Δt θ + P f + H_ve θ_e e0) corrected for H_ve by Sherman-Morrison;
/// the HVAC response r likewise; then the power that brings the control
/// temperature to the violated setpoint, clipped to the capacity.
pub fn run(
    net: &Network52016,
    hours: &Hourly52016,
    air: &Airflow52016,
    warmup_passes: usize,
) -> Result<Output52016, String> {
    let count = hours.theta_e.len();
    let n = net.n;
    let k = net.k;
    let elements = k - 5;
    let area_total: f64 = net.areas.iter().sum();
    let mut state = vec![hours.set_heating[0]; n];
    let mut weighted = vec![0.0; n];
    let mut forcing = vec![0.0; k];
    let mut y = vec![0.0; n];
    let mut response = vec![0.0; n];
    let mut last = 0.0;
    let mut out = Output52016 {
        power: vec![0.0; count],
        air: vec![0.0; count],
        operative: vec![0.0; count],
    };
    let [c0, c1, c2, c3] = air.coefficients;
    for _ in 0..=warmup_passes {
        for hour in 0..count {
            let theta_e = hours.theta_e[hour];
            let t_air = state[0];
            let w = hours.wind[hour];
            let correction = c0 + c1 * (t_air - theta_e).abs() + c2 * w + c3 * w * w;
            let ach_infiltration = hours.infiltration[hour] * correction + hours.natural[hour];
            if !ach_infiltration.is_finite() || ach_infiltration < 0.0 {
                return Err("Hourly infiltration must be finite and nonnegative".into());
            }
            let eta = if air.recovery_bypass {
                let heating = last > 1e-3 || t_air <= hours.set_heating[hour] + 1e-6;
                let cooling = last < -1e-3 || t_air >= hours.set_cooling[hour] - 1e-6;
                let active =
                    (heating && theta_e < t_air - 1e-6) || (cooling && theta_e > t_air + 1e-6);
                if active {
                    air.recovery_effectiveness
                } else {
                    0.0
                }
            } else {
                air.recovery_effectiveness
            };
            let ach_effective = ach_infiltration + hours.ventilation[hour] * (1.0 - eta);
            let h_ve = air.air_capacity_j_m3_k * air.volume_m3 * ach_effective / 3600.0;

            forcing[0] = theta_e;
            forcing[1] = hours.theta_gr[hour];
            forcing[2] = hours.phi_int[hour];
            forcing[3] = hours.phi_sol[hour];
            forcing[4] = 1.0;
            forcing[5..].copy_from_slice(&hours.absorbed[hour * elements..(hour + 1) * elements]);
            for i in 0..n {
                weighted[i] = net.c_dt[i] * state[i];
                y[i] = net.m_air[i] * h_ve * theta_e;
            }
            matvec_add(&net.m, n, n, &weighted, &mut y);
            matvec_add(&net.mp, n, k, &forcing, &mut y);

            let denominator = 1.0 + h_ve * net.m_air[0];
            let scale_free = h_ve * y[0] / denominator;
            let scale_hvac = h_ve * net.m_hvac[0] / denominator;
            let mut free_control = 0.0;
            let mut gain = 0.0;
            for i in 0..n {
                y[i] -= net.m_air[i] * scale_free;
                response[i] = net.m_hvac[i] - net.m_air[i] * scale_hvac;
                free_control += net.control[i] * y[i];
                gain += net.control[i] * response[i];
            }
            let (heating, cooling) = (hours.set_heating[hour], hours.set_cooling[hour]);
            let (heat_limit, cool_limit) = (hours.heat_limit[hour], hours.cool_limit[hour]);
            let mut power = 0.0;
            if free_control < heating && heat_limit > 0.0 {
                power = ((heating - free_control) / gain).min(heat_limit);
            } else if free_control > cooling && cool_limit < 0.0 {
                power = ((cooling - free_control) / gain).max(cool_limit);
            }
            for i in 0..n {
                state[i] = y[i] + power * response[i];
            }
            let mut radiant = 0.0;
            for (index, area) in net.inner.iter().zip(&net.areas) {
                radiant += area * state[*index];
            }
            if !state[0].is_finite() || !power.is_finite() {
                return Err("ISO 52016-1 recurrence produced a non-finite value".into());
            }
            out.power[hour] = power;
            out.air[hour] = state[0];
            out.operative[hour] = 0.5 * state[0] + 0.5 * radiant / area_total;
            last = power;
        }
    }
    Ok(out)
}
