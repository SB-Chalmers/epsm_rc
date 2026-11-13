// File: domain.rs
// Purpose: Platform hourly forcing and loop for one 5R1C domain, with the heat-recovery rule.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sanjay Somanath
// License: MIT; see the project-root LICENSE for the full permission and warranty terms.
// Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
// Project PI: Alexander Hollberg
// Computational Sustainable Design Group
// Research Area of Sustainable Built Environments
// Chalmers University of Technology, Gothenburg
// Date created: 2026-10-29

//! Hourly platform forcing for one 5R1C domain, read from `HourlyInputs` columns.
//!
//! [`DomainArrays::apply_at`] sets one hour's setpoints, gains and airflow on the
//! model exactly as `epsm_rc._forcing.HourlyInputs.at` does in Python, including
//! the wind- and temperature-dependent infiltration correction and the
//! heat-recovery rule. [`run_single_domain`] is the single-zone hourly loop.

use crate::iso13790::Core5R1C;
use crate::types::{DomainOutput, KernelResult, StepInputs, Thermostat};

/// One allocation per domain avoids per-column allocation overhead at the boundary.
pub struct DomainSnapshot {
    storage: Vec<f64>,
    count: usize,
    month_hours: Option<Vec<i64>>,
    coefficients: (f64, f64, f64, f64),
    aggregate_split: bool,
}

impl DomainSnapshot {
    /// Copy the 13 hourly columns into one contiguous allocation.
    pub fn new(
        columns: [&[f64]; 13],
        month_hours: Option<&[i64]>,
        coefficients: (f64, f64, f64, f64),
        aggregate_split: bool,
    ) -> KernelResult<Self> {
        let count = columns[0].len();
        if count == 0 {
            return Err("Hourly inputs must not be empty");
        }
        if columns.iter().any(|column| column.len() != count) {
            return Err("All hourly inputs must have the same length");
        }
        if month_hours.is_some_and(|hours| hours.len() != count) {
            return Err("month_hours must match the hourly inputs length");
        }
        let mut storage = Vec::with_capacity(count * columns.len());
        for column in columns {
            storage.extend_from_slice(column);
        }
        Ok(Self {
            storage,
            count,
            month_hours: month_hours.map(<[i64]>::to_vec),
            coefficients,
            aggregate_split,
        })
    }

    /// Borrow the snapshot as [`DomainArrays`].
    pub fn view(&self) -> DomainArrays<'_> {
        DomainArrays::new(
            std::array::from_fn(|index| {
                &self.storage[index * self.count..(index + 1) * self.count]
            }),
            self.month_hours.as_deref(),
            self.coefficients,
            self.aggregate_split,
        )
    }
}

/// Borrowed hourly columns and coefficients for one domain.
pub struct DomainArrays<'a> {
    pub temperatures: &'a [f64],
    pub gains: &'a [f64],
    pub convective: &'a [f64],
    pub heating_setpoints: &'a [f64],
    pub cooling_setpoints: &'a [f64],
    pub wind: &'a [f64],
    pub infiltration: &'a [f64],
    pub natural: &'a [f64],
    pub ventilation: &'a [f64],
    pub solar: &'a [f64],
    pub opaque: &'a [f64],
    pub month_hours: Option<&'a [i64]>,
    /// Infiltration correction c0 + c1 |θ_air − θ_e| + c2 w + c3 w².
    pub coefficients: (f64, f64, f64, f64),
    pub aggregate_split: bool,
    pub heating_limits_w: &'a [f64],
    pub cooling_limits_w: &'a [f64],
    /// Recovery effectiveness when recovery is used only when beneficial; `None`
    /// keeps the model's effectiveness in every hour (`heat_recovery="always"`).
    pub recovery_bypass: Option<f64>,
}

impl<'a> DomainArrays<'a> {
    /// Name the columns in `FORCING_COLUMNS` order followed by the two limits.
    pub fn new(
        columns: [&'a [f64]; 13],
        month_hours: Option<&'a [i64]>,
        coefficients: (f64, f64, f64, f64),
        aggregate_split: bool,
    ) -> Self {
        let [temperatures, gains, convective, heating_setpoints, cooling_setpoints, wind, infiltration, natural, ventilation, solar, opaque, heating_limits_w, cooling_limits_w] =
            columns;
        Self {
            temperatures,
            gains,
            convective,
            heating_setpoints,
            cooling_setpoints,
            wind,
            infiltration,
            natural,
            ventilation,
            solar,
            opaque,
            heating_limits_w,
            cooling_limits_w,
            month_hours,
            coefficients,
            aggregate_split,
            recovery_bypass: None,
        }
    }

    /// Validate every shape before the first model mutation or indexed access.
    pub fn validate(&self, count: usize) -> KernelResult<()> {
        if count == 0 {
            return Err("Hourly inputs must not be empty");
        }
        for values in [
            &self.temperatures,
            &self.gains,
            &self.convective,
            &self.heating_setpoints,
            &self.cooling_setpoints,
            &self.wind,
            &self.infiltration,
            &self.natural,
            &self.ventilation,
            &self.solar,
            &self.opaque,
            &self.heating_limits_w,
            &self.cooling_limits_w,
        ] {
            if values.len() != count {
                return Err("All hourly inputs must have the same length");
            }
        }
        if self
            .month_hours
            .as_ref()
            .is_some_and(|hours| hours.len() != count)
        {
            return Err("month_hours must match the hourly inputs length");
        }
        Ok(())
    }

    /// Apply hour `hour` to `core` and return its [`StepInputs`].
    ///
    /// When `recovery_bypass` holds an effectiveness, recovery is used only when it
    /// reduces the load, as in EnergyPlus ideal loads: in heating mode with
    /// outdoor air colder than the zone, or in cooling mode with warmer outdoor
    /// air. The mode is lagged one hour: `last_power` was heating (> 1e-3 W) or
    /// cooling (< −1e-3 W), or the air is within 1e-6 K of or beyond a setpoint.
    /// The tolerances keep Python and Rust on the same side of each comparison.
    #[inline]
    pub fn apply_at(
        &self,
        core: &mut Core5R1C,
        hour: usize,
        last_power: f64,
    ) -> KernelResult<StepInputs> {
        let temperature = self.temperatures[hour];
        let gain = self.gains[hour];
        if !self.aggregate_split {
            core.convective_fraction_internal = if gain > 0.0 {
                self.convective[hour] / gain
            } else {
                0.0
            };
        }
        core.t_set_heating = self.heating_setpoints[hour];
        core.t_set_cooling = self.cooling_setpoints[hour];
        let w = self.wind[hour];
        let correction = self.coefficients.0
            + self.coefficients.1 * (core.state.t_air - temperature).abs()
            + self.coefficients.2 * w
            + self.coefficients.3 * w.powi(2);
        core.ach_infiltration = self.infiltration[hour] * correction + self.natural[hour];
        core.ach_ventilation = self.ventilation[hour];
        if let Some(effectiveness) = self.recovery_bypass {
            // Same rule as `HourlyInputs.at`: recover only when it reduces the load.
            let t_air = core.state.t_air;
            let heating = last_power > 1e-3 || t_air <= core.t_set_heating + 1e-6;
            let cooling = last_power < -1e-3 || t_air >= core.t_set_cooling - 1e-6;
            let active =
                (heating && temperature < t_air - 1e-6) || (cooling && temperature > t_air + 1e-6);
            core.heat_recovery_efficiency = if active { effectiveness } else { 0.0 };
        }
        if !core.ach_infiltration.is_finite() || core.ach_infiltration < 0.0 {
            return Err("Hourly infiltration must be finite and nonnegative");
        }
        core.update_ventilation_coefficient(None)?;
        let hour_of_year = self.month_hours.as_ref().map(|mh| mh[hour]);
        Ok(StepInputs {
            t_out: temperature,
            solar_gains: self.solar[hour],
            internal_gains: gain,
            solar_gains_opaque: self.opaque[hour],
            hour_of_year,
            dt_hours: 1.0,
            include_sky_radiation: true,
        })
    }
}

/// Hourly loop of one platform domain; returns thermostat power and air temperature.
///
/// The previous hour's power feeds the heat-recovery mode of the next hour.
pub fn run_single_domain(
    core: &mut Core5R1C,
    arrays: &DomainArrays<'_>,
) -> KernelResult<DomainOutput> {
    let count = arrays.temperatures.len();
    let mut power = Vec::with_capacity(count);
    let mut indoor = Vec::with_capacity(count);
    let mut last = 0.0;
    for hour in 0..count {
        let inputs = arrays.apply_at(core, hour, last)?;
        let controls = Thermostat {
            max_heating_power: arrays.heating_limits_w[hour],
            max_cooling_power: arrays.cooling_limits_w[hour],
            t_set_heat: core.t_set_heating,
            t_set_cool: core.t_set_cooling,
        };
        last = core.calc_heating_demand_with_rng(inputs, controls, || Ok(1.0))?;
        power.push(last);
        indoor.push(core.state.t_air);
    }
    Ok(DomainOutput { power, indoor })
}
