// File: iso52016.rs
// Purpose: ISO 52016-1 Clause 6.5 hourly multi-zone recurrence on a pre-inverted network.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sanjay Somanath
// License: MIT; see the project-root LICENSE for the full permission and warranty terms.
// Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
// Project PI: Alexander Hollberg
// Computational Sustainable Design Group
// Research Area of Sustainable Built Environments
// Chalmers University of Technology, Gothenburg
// Date created: 2026-11-12

//! ISO 52016-1:2017 Clause 6.5 hourly multi-zone recurrence on a pre-inverted network.
//!
//! Python assembles the network (`epsm_rc.iso52016.Network52016`): one air node
//! per zone, five nodes per opaque element and two per window (Clause 6.5.6 and
//! 6.5.7), with Table 25 surface coefficients. An element between two zones has
//! its surface nodes in different zones. Backward Euler gives, each hour,
//!
//! ```text
//! (A0 + U D Uᵀ) θ_t = C/Δt θ_{t−1} + P f_t + U D θ_e + H φ_HC
//! ```
//!
//! where A0 is constant, `U` selects the air nodes, `D` holds each zone's
//! ventilation conductance H_ve, `f_t` is the hour's forcing row (outdoor air,
//! sky constant, per-zone ground temperature, internal and window solar gains,
//! absorbed solar per element) and `H` distributes each zone's HVAC power (fully
//! convective, Annex B). Python inverts A0 once (M = A0⁻¹). Here the hourly
//! H_ve enters through the Woodbury identity
//!
//! ```text
//! (A0 + U D Uᵀ)⁻¹ = M − M U (I + D Uᵀ M U)⁻¹ D Uᵀ M
//! ```
//!
//! so only a zones × zones system is solved inside the loop.
//!
//! Because the network is linear, the free-floating state plus the response to
//! one watt in each zone gives the exact thermostat powers. Coupled thermostats
//! take the first consistent active set (free, at a setpoint or at a capacity
//! limit per zone), as `epsm_rc.iso52016.solve_thermostats`; for one zone this
//! is the linear interpolation of Clause 6.5.5.2. Airflow and the
//! heat-recovery rule match `epsm_rc._forcing.HourlyInputs.at`. The year runs
//! `1 + warmup_passes` times and the last pass is returned.

/// Constant network matrices from `Network52016._build`, row-major.
pub struct Network52016 {
    /// Number of nodes; nodes `0..z` are the zone air nodes.
    pub n: usize,
    /// Number of zones.
    pub z: usize,
    /// Number of forcing columns: 2 + 3 z + elements.
    pub k: usize,
    /// n × n inverse of the constant matrix.
    pub m: Vec<f64>,
    /// n × k product of the inverse and the forcing map.
    pub mp: Vec<f64>,
    /// n × z columns of M for the air nodes (response to H_ve coupling).
    pub m_air: Vec<f64>,
    /// n × z: M times each zone's HVAC distribution, the response to 1 W.
    pub m_hvac: Vec<f64>,
    /// Node capacity divided by the time step, C/Δt (W/K).
    pub c_dt: Vec<f64>,
    /// z × n weights giving each zone's control temperature.
    pub control: Vec<f64>,
    /// z × n weights giving each zone's operative temperature.
    pub operative: Vec<f64>,
}

/// Ventilation of one zone: a given hourly H_ve, or the conditioned-domain airflow.
pub enum Airflow<'a> {
    /// Hourly H_ve (W/K), e.g. an attic's declared outdoor exchange.
    Fixed(&'a [f64]),
    /// H_ve = ρc V n_eff / 3600 from the domain's airflow and recovery rule.
    Forcing {
        /// Wind speed (m/s) for the infiltration correction.
        wind: &'a [f64],
        /// Base infiltration air changes per hour before correction.
        infiltration: &'a [f64],
        /// Natural ventilation air changes per hour (unrecovered).
        natural: &'a [f64],
        /// Mechanical ventilation air changes per hour (recoverable).
        ventilation: &'a [f64],
        /// Infiltration correction c0 + c1 |θ_air − θ_e| + c2 w + c3 w².
        coefficients: [f64; 4],
        volume_m3: f64,
        air_capacity_j_m3_k: f64,
        /// Sensible recovery effectiveness of the mechanical airflow.
        recovery_effectiveness: f64,
        /// Use recovery only when it reduces the load (`heat_recovery="when_beneficial"`).
        recovery_bypass: bool,
    },
}

/// Hourly thermostat and airflow of one zone.
pub struct ZoneHourly<'a> {
    /// Heating setpoint (°C).
    pub set_heating: &'a [f64],
    /// Cooling setpoint (°C).
    pub set_cooling: &'a [f64],
    /// Heating capacity (W); 0 when heating is unavailable or the zone is unconditioned.
    pub heat_limit: &'a [f64],
    /// Cooling capacity as a nonpositive power (W).
    pub cool_limit: &'a [f64],
    pub airflow: Airflow<'a>,
}

/// Hourly results, row-major hours × zones, plus the summed link flows per hour.
pub struct Output52016 {
    /// HVAC power (W, + heating).
    pub power: Vec<f64>,
    /// Air temperature (°C).
    pub air: Vec<f64>,
    /// Operative temperature (°C).
    pub operative: Vec<f64>,
    /// Σ h (θ_a − θ_b) over the requested links (W).
    pub probe: Vec<f64>,
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

/// Solve the row-major `size × size` system `a x = b` for `columns` right-hand
/// sides stored row-major in `b` (size × columns), by Gaussian elimination with
/// partial pivoting. `a` and `b` are overwritten; the solution is left in `b`.
fn solve_small(a: &mut [f64], b: &mut [f64], size: usize, columns: usize) -> Result<(), String> {
    for col in 0..size {
        let mut pivot = col;
        for row in col + 1..size {
            if a[row * size + col].abs() > a[pivot * size + col].abs() {
                pivot = row;
            }
        }
        if a[pivot * size + col] == 0.0 {
            return Err("Singular ISO 52016-1 zone system".into());
        }
        if pivot != col {
            for j in 0..size {
                a.swap(col * size + j, pivot * size + j);
            }
            for j in 0..columns {
                b.swap(col * columns + j, pivot * columns + j);
            }
        }
        for row in col + 1..size {
            let factor = a[row * size + col] / a[col * size + col];
            if factor == 0.0 {
                continue;
            }
            for j in col..size {
                a[row * size + j] -= factor * a[col * size + j];
            }
            for j in 0..columns {
                b[row * columns + j] -= factor * b[col * columns + j];
            }
        }
    }
    for col in (0..size).rev() {
        for j in 0..columns {
            let mut value = b[col * columns + j];
            for k in col + 1..size {
                value -= a[col * size + k] * b[k * columns + j];
            }
            b[col * columns + j] = value / a[col * size + col];
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Free,
    Heat,
    HeatLimit,
    Cool,
    CoolLimit,
}

/// Thermostat powers (W) making the control temperatures `free + gain · p`
/// consistent: each zone free inside its band, at a setpoint within capacity,
/// or at a capacity limit beyond the setpoint. Mirrors
/// `epsm_rc.iso52016.solve_thermostats`, including its search order.
#[allow(clippy::too_many_arguments)]
fn solve_thermostats(
    free: &[f64],
    gain: &[f64],
    set_h: &[f64],
    set_c: &[f64],
    heat: &[f64],
    cool: &[f64],
    power: &mut [f64],
) -> Result<(), String> {
    let z = free.len();
    let tol = 1e-7;
    let mut options: Vec<Vec<Mode>> = Vec::with_capacity(z);
    for i in 0..z {
        let mut modes = vec![Mode::Free];
        if heat[i] > 0.0 {
            modes.push(Mode::Heat);
            if heat[i].is_finite() {
                modes.push(Mode::HeatLimit);
            }
        }
        if cool[i] < 0.0 {
            modes.push(Mode::Cool);
            if cool[i].is_finite() {
                modes.push(Mode::CoolLimit);
            }
        }
        options.push(modes);
    }
    let mut counter = vec![0usize; z];
    let mut modes = vec![Mode::Free; z];
    let mut temperature = vec![0.0; z];
    loop {
        for i in 0..z {
            modes[i] = options[i][counter[i]];
            power[i] = match modes[i] {
                Mode::HeatLimit => heat[i],
                Mode::CoolLimit => cool[i],
                _ => 0.0,
            };
        }
        let target: Vec<usize> = (0..z)
            .filter(|&i| matches!(modes[i], Mode::Heat | Mode::Cool))
            .collect();
        if !target.is_empty() {
            let t = target.len();
            let mut a = vec![0.0; t * t];
            let mut b = vec![0.0; t];
            for (r, &i) in target.iter().enumerate() {
                let mut rest = free[i];
                for j in 0..z {
                    rest += gain[i * z + j] * power[j];
                }
                let setpoint = if modes[i] == Mode::Heat {
                    set_h[i]
                } else {
                    set_c[i]
                };
                b[r] = setpoint - rest;
                for (c, &j) in target.iter().enumerate() {
                    a[r * t + c] = gain[i * z + j];
                }
            }
            solve_small(&mut a, &mut b, t, 1)?;
            for (r, &i) in target.iter().enumerate() {
                power[i] = b[r];
            }
        }
        for i in 0..z {
            let mut value = free[i];
            for j in 0..z {
                value += gain[i * z + j] * power[j];
            }
            temperature[i] = value;
        }
        let consistent = (0..z).all(|i| {
            let (t, p) = (temperature[i], power[i]);
            match modes[i] {
                Mode::Free => !(heat[i] > 0.0 && t < set_h[i]) && !(cool[i] < 0.0 && t > set_c[i]),
                Mode::Heat => -tol <= p && p <= heat[i] * (1.0 + tol) + tol,
                Mode::Cool => cool[i] * (1.0 + tol) - tol <= p && p <= tol,
                Mode::HeatLimit => t <= set_h[i] + tol,
                Mode::CoolLimit => t >= set_c[i] - tol,
            }
        });
        if consistent {
            for i in 0..z {
                power[i] = power[i].max(cool[i]).min(heat[i]);
            }
            return Ok(());
        }
        // Advance the mixed-radix counter; the last zone varies fastest.
        let mut index = z;
        loop {
            if index == 0 {
                return Err("No consistent ISO 52016-1 thermostat state".into());
            }
            index -= 1;
            counter[index] += 1;
            if counter[index] < options[index].len() {
                break;
            }
            counter[index] = 0;
        }
    }
}

/// Run the hourly recurrence and return the last pass.
///
/// Each hour, per zone: infiltration from the previous air temperature; heat
/// recovery active only when it reduces the load if `recovery_bypass` (same
/// rule and tolerances as [`crate::domain::DomainArrays::apply_at`]). Then the
/// free state y = M (C/Δt θ + P f + U D θ_e) and the HVAC responses, both
/// corrected for H_ve by the Woodbury identity, and the consistent thermostat
/// powers. `forcing` is row-major hours × k; `links` are (node a, node b, W/K).
pub fn run(
    net: &Network52016,
    forcing: &[f64],
    zones: &[ZoneHourly],
    links: &[(usize, usize, f64)],
    warmup_passes: usize,
) -> Result<Output52016, String> {
    let (n, z, k) = (net.n, net.z, net.k);
    let count = forcing.len() / k;
    let mut state = vec![zones[0].set_heating[0]; n];
    let mut weighted = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut response = vec![0.0; n * z];
    let mut last = vec![0.0; z];
    let mut h_ve = vec![0.0; z];
    let mut power = vec![0.0; z];
    let mut free_t = vec![0.0; z];
    let mut gain = vec![0.0; z * z];
    let (mut set_h, mut set_c) = (vec![0.0; z], vec![0.0; z]);
    let (mut heat, mut cool) = (vec![0.0; z], vec![0.0; z]);
    let mut out = Output52016 {
        power: vec![0.0; count * z],
        air: vec![0.0; count * z],
        operative: vec![0.0; count * z],
        probe: vec![0.0; count],
    };
    for _ in 0..=warmup_passes {
        for hour in 0..count {
            let row = &forcing[hour * k..(hour + 1) * k];
            let theta_e = row[0];
            for (i, zone) in zones.iter().enumerate() {
                set_h[i] = zone.set_heating[hour];
                set_c[i] = zone.set_cooling[hour];
                heat[i] = zone.heat_limit[hour];
                cool[i] = zone.cool_limit[hour];
                h_ve[i] = match &zone.airflow {
                    Airflow::Fixed(series) => series[hour],
                    Airflow::Forcing {
                        wind,
                        infiltration,
                        natural,
                        ventilation,
                        coefficients,
                        volume_m3,
                        air_capacity_j_m3_k,
                        recovery_effectiveness,
                        recovery_bypass,
                    } => {
                        let t_air = state[i];
                        let [c0, c1, c2, c3] = *coefficients;
                        let w = wind[hour];
                        let correction = c0 + c1 * (t_air - theta_e).abs() + c2 * w + c3 * w * w;
                        let ach = infiltration[hour] * correction + natural[hour];
                        if !ach.is_finite() || ach < 0.0 {
                            return Err("Hourly infiltration must be finite and nonnegative".into());
                        }
                        let eta = if *recovery_bypass {
                            let heating = last[i] > 1e-3 || t_air <= set_h[i] + 1e-6;
                            let cooling = last[i] < -1e-3 || t_air >= set_c[i] - 1e-6;
                            let active = (heating && theta_e < t_air - 1e-6)
                                || (cooling && theta_e > t_air + 1e-6);
                            if active {
                                *recovery_effectiveness
                            } else {
                                0.0
                            }
                        } else {
                            *recovery_effectiveness
                        };
                        let effective = ach + ventilation[hour] * (1.0 - eta);
                        air_capacity_j_m3_k * volume_m3 * effective / 3600.0
                    }
                };
            }
            for i in 0..n {
                weighted[i] = net.c_dt[i] * state[i];
                let mut coupling = 0.0;
                for (a, h) in net.m_air[i * z..(i + 1) * z].iter().zip(&h_ve) {
                    coupling += a * (h * theta_e);
                }
                y[i] = coupling;
            }
            matvec_add(&net.m, n, n, &weighted, &mut y);
            matvec_add(&net.mp, n, k, row, &mut y);

            // Woodbury: S = I + D M_aa; free = y − M_a S⁻¹ D y_a; R = M_h − M_a S⁻¹ D M_h,a.
            let mut s = vec![0.0; z * z];
            let mut rhs = vec![0.0; z * (1 + z)];
            for i in 0..z {
                for j in 0..z {
                    s[i * z + j] = if i == j { 1.0 } else { 0.0 } + h_ve[i] * net.m_air[i * z + j];
                }
                rhs[i * (1 + z)] = h_ve[i] * y[i];
                for j in 0..z {
                    rhs[i * (1 + z) + 1 + j] = h_ve[i] * net.m_hvac[i * z + j];
                }
            }
            solve_small(&mut s, &mut rhs, z, 1 + z)?;
            for i in 0..n {
                let mut free_correction = 0.0;
                for j in 0..z {
                    free_correction += net.m_air[i * z + j] * rhs[j * (1 + z)];
                }
                y[i] -= free_correction;
                for c in 0..z {
                    let mut correction = 0.0;
                    for j in 0..z {
                        correction += net.m_air[i * z + j] * rhs[j * (1 + z) + 1 + c];
                    }
                    response[i * z + c] = net.m_hvac[i * z + c] - correction;
                }
            }
            for r in 0..z {
                let weights = &net.control[r * n..(r + 1) * n];
                let mut value = 0.0;
                for i in 0..n {
                    value += weights[i] * y[i];
                }
                free_t[r] = value;
                for c in 0..z {
                    let mut value = 0.0;
                    for i in 0..n {
                        value += weights[i] * response[i * z + c];
                    }
                    gain[r * z + c] = value;
                }
            }
            solve_thermostats(&free_t, &gain, &set_h, &set_c, &heat, &cool, &mut power)?;
            for i in 0..n {
                let mut value = y[i];
                for c in 0..z {
                    value += response[i * z + c] * power[c];
                }
                state[i] = value;
            }
            let mut probe = 0.0;
            for &(a, b, h) in links {
                probe += h * (state[a] - state[b]);
            }
            for r in 0..z {
                if !state[r].is_finite() || !power[r].is_finite() {
                    return Err("ISO 52016-1 recurrence produced a non-finite value".into());
                }
                let weights = &net.operative[r * n..(r + 1) * n];
                let mut value = 0.0;
                for i in 0..n {
                    value += weights[i] * state[i];
                }
                out.power[hour * z + r] = power[r];
                out.air[hour * z + r] = state[r];
                out.operative[hour * z + r] = value;
                last[r] = power[r];
            }
            out.probe[hour] = probe;
        }
    }
    Ok(out)
}
