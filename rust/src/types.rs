// File: types.rs
// Purpose: Plain values shared by the native kernels.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sanjay Somanath
// License: MIT; see the project-root LICENSE for the full permission and warranty terms.
// Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
// Project PI: Alexander Hollberg
// Computational Sustainable Design Group
// Research Area of Sustainable Built Environments
// Chalmers University of Technology, Gothenburg
// Date created: 2026-10-29

//! Values shared by the numerical kernels. No Python objects enter a kernel.

/// Kernel errors are static messages; bindings turn them into `ValueError`.
pub type KernelResult<T> = Result<T, &'static str>;

/// Node temperatures of a 5R1C zone (°C).
#[derive(Clone, Copy, Debug, Default)]
pub struct ThermalState {
    /// Air node θ_air.
    pub t_air: f64,
    /// Central surface node θ_s.
    pub t_s: f64,
    /// Mass node θ_m.
    pub t_m: f64,
}

/// Forcing for one hour: outdoor temperature (°C) and gains (W).
#[derive(Clone, Copy, Debug)]
pub struct StepInputs {
    pub t_out: f64,
    pub solar_gains: f64,
    pub internal_gains: f64,
    pub solar_gains_opaque: f64,
    pub hour_of_year: Option<i64>,
    pub dt_hours: f64,
    pub include_sky_radiation: bool,
}

/// Setpoints (°C) and capacity limits (W) for one hour.
#[derive(Clone, Copy, Debug)]
pub struct Thermostat {
    pub max_heating_power: f64,
    pub max_cooling_power: f64,
    pub t_set_heat: f64,
    pub t_set_cool: f64,
}

impl Thermostat {
    /// Reject non-finite or unordered setpoints and negative capacity.
    pub fn validate(&self, dt_hours: f64) -> KernelResult<()> {
        if !dt_hours.is_finite() || dt_hours <= 0.0 {
            return Err("dt_hours must be positive and finite");
        }
        if self.max_heating_power.is_nan() || self.max_heating_power < 0.0 {
            return Err("max_heating_power must be nonnegative");
        }
        if self.max_cooling_power.is_nan() {
            return Err("max_cooling_power must not be NaN");
        }
        if !self.t_set_heat.is_finite()
            || !self.t_set_cool.is_finite()
            || self.t_set_heat > self.t_set_cool
        {
            return Err("Finite ordered heating and cooling setpoints are required");
        }
        Ok(())
    }
}

/// Coupled two-domain control result: powers (W), air temperatures (°C), the
/// heat each domain receives from the other (W) and the energy-balance residual.
#[derive(Clone, Copy, Debug)]
pub struct ControlSolution {
    pub powers: [f64; 2],
    pub temperatures: [f64; 2],
    pub exchanges: [f64; 2],
    pub residual: f64,
}

/// Hourly power (W) and air temperature (°C) of one domain.
pub struct DomainOutput {
    pub power: Vec<f64>,
    pub indoor: Vec<f64>,
}

/// Hourly results of the coupled main and basement domains.
pub struct CoupledOutput {
    pub powers: [Vec<f64>; 2],
    pub temperatures: [Vec<f64>; 2],
    pub exchanges: [Vec<f64>; 2],
    pub residuals: Vec<f64>,
}

/// Owned snapshots allow the deterministic runner to release the GIL safely.
pub struct SimulationInputs {
    pub outdoor: Vec<f64>,
    pub solar: Vec<f64>,
    pub internal: Vec<f64>,
    pub opaque: Vec<f64>,
    pub occupancy: Option<Vec<bool>>,
    pub dt_hours: f64,
    pub include_sky_radiation: bool,
    pub use_hour_of_year: bool,
}

impl SimulationInputs {
    /// Check lengths and controls before the loop starts.
    pub fn validate(&self, controls: Thermostat) -> KernelResult<()> {
        controls.validate(self.dt_hours)?;
        let count = self.outdoor.len();
        if count == 0 {
            return Err("t_outdoor must not be empty");
        }
        if [self.solar.len(), self.internal.len(), self.opaque.len()]
            .iter()
            .any(|&length| length != count)
        {
            return Err("All gain arrays must match the t_outdoor length");
        }
        let last_hour = ((count - 1) as f64 * self.dt_hours) as usize;
        if self.use_hour_of_year
            && self
                .occupancy
                .as_ref()
                .is_some_and(|schedule| schedule.len() <= last_hour)
        {
            return Err("occupancy_schedule does not cover the simulation elapsed hours");
        }
        Ok(())
    }
}

/// Hourly results of [`crate::iso13790::simulate`].
pub struct SimulationOutput {
    pub heating_demand: Vec<f64>,
    pub t_air: Vec<f64>,
    pub t_m: Vec<f64>,
    pub t_s: Vec<f64>,
    pub t_operative: Vec<f64>,
    pub windows_open: Vec<bool>,
}
