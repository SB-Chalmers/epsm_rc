// File: iso13790.rs
// Purpose: ISO 13790 Annex C 5R1C zone, thermostat and standalone hourly loop.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sanjay Somanath
// License: MIT; see the project-root LICENSE for the full permission and warranty terms.
// Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
// Project PI: Alexander Hollberg
// Computational Sustainable Design Group
// Research Area of Sustainable Built Environments
// Chalmers University of Technology, Gothenburg
// Date created: 2026-10-29

//! ISO 13790:2008 Annex C simple hourly method: the 5R1C network.
//!
//! One air node (θ_air), one central surface node (θ_s) and one mass node (θ_m)
//! with capacity C_m. Conductances: H_ve (ventilation, air node), H_tr,is (air to
//! surface), H_tr,w (windows, surface node), H_tr,ms (surface to mass) and
//! H_tr,em (opaque elements, mass node). Each hour follows Annex C:
//!
//! 1. Heat flows split gains between nodes (Eq. C.1 to C.3).
//! 2. The series conductances H_tr,1..3 combine the air and surface paths
//!    (Eq. C.6 to C.8).
//! 3. The mass node advances with a Crank-Nicolson step (Eq. C.4, C.5), and the
//!    hour uses the mean of the old and new mass temperatures (Eq. C.9).
//! 4. Surface and air temperatures follow (Eq. C.10, C.11); operative temperature
//!    is 0.3 θ_air + 0.7 θ_s (Eq. C.12).
//! 5. The thermostat interpolates between a free-floating trial and a trial with
//!    10 W/m² (Eq. C.13). Without window opening the network is linear, so the
//!    response to unit power ([`Core5R1C::sensitivities`]) gives the same result
//!    without the second trial.
//!
//! Every function mirrors `epsm_rc/iso13790.py`; the Python tests compare both
//! paths hour by hour. Heat recovery reduces the mechanical part of H_ve; the
//! hour-by-hour decision for the `when_beneficial` option is taken in
//! [`crate::domain::DomainArrays::apply_at`].

use crate::types::{
    KernelResult, SimulationInputs, SimulationOutput, StepInputs, ThermalState, Thermostat,
};

// Forced inlining is limited to measured control/airflow hot paths. Keep
// benchmarks/native.py comparisons when changing these boundaries.

/// Run the standalone hourly loop over owned forcing (`Building5R1C.simulate`).
///
/// The state starts at the heating setpoint. Unoccupied hours use the setback
/// setpoints when an occupancy schedule is supplied. `random` supplies uniform
/// draws for stochastic window opening and is not called when it is disabled.
/// Returns heating (+) or cooling (−) power in W and the node temperatures.
pub fn simulate<F>(
    core: &mut Core5R1C,
    forcing: &SimulationInputs,
    controls: Thermostat,
    mut random: F,
) -> KernelResult<SimulationOutput>
where
    F: FnMut() -> KernelResult<f64>,
{
    let count = forcing.outdoor.len();
    let mut output = SimulationOutput {
        heating_demand: Vec::with_capacity(count),
        t_air: Vec::with_capacity(count),
        t_m: Vec::with_capacity(count),
        t_s: Vec::with_capacity(count),
        t_operative: Vec::with_capacity(count),
        windows_open: Vec::with_capacity(count),
    };
    core.state = ThermalState {
        t_air: core.t_set_heating,
        t_s: core.t_set_heating,
        t_m: core.t_set_heating,
    };
    core.windows_open = false;
    for hour in 0..count {
        let hour_of_year = forcing
            .use_hour_of_year
            .then_some((hour as f64 * forcing.dt_hours) as i64);
        let setback = match (&forcing.occupancy, hour_of_year) {
            (Some(schedule), Some(elapsed)) => !schedule[elapsed as usize],
            _ => false,
        };
        let hour_controls = if setback {
            Thermostat {
                t_set_heat: core.t_set_heating_setback.unwrap_or(controls.t_set_heat),
                t_set_cool: core.t_set_cooling_setback.unwrap_or(controls.t_set_cool),
                ..controls
            }
        } else {
            controls
        };
        let inputs = StepInputs {
            t_out: forcing.outdoor[hour],
            solar_gains: forcing.solar[hour],
            internal_gains: forcing.internal[hour],
            solar_gains_opaque: forcing.opaque[hour],
            hour_of_year,
            dt_hours: forcing.dt_hours,
            include_sky_radiation: forcing.include_sky_radiation,
        };
        output
            .heating_demand
            .push(core.calc_heating_demand_with_rng(inputs, hour_controls, &mut random)?);
        output.t_air.push(core.state.t_air);
        output.t_m.push(core.state.t_m);
        output.t_s.push(core.state.t_s);
        output.t_operative.push(core.t_operative());
        output.windows_open.push(core.windows_open);
    }
    Ok(output)
}

/// First hour of each month in a 365-day year, for monthly ground temperatures.
const MONTH_START_HOURS: [i64; 12] = [
    0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016,
];

/// Parameters and state of one 5R1C zone, read from `Building5R1C._native_parameters`.
///
/// Conductances are in W/K, capacity in J/K, temperatures in °C and powers in W.
/// The `prepared_*` fields cache hour-invariant coefficients for the platform
/// path, where airflow changes hourly but the fabric does not.
#[derive(Clone, Debug)]
#[cfg_attr(test, derive(Default))]
pub struct Core5R1C {
    /// Conditioned floor area A_f (m²); scales the 10 W/m² thermostat trial.
    pub floor_area_m2: f64,
    /// Air volume (m³) used for H_ve.
    pub building_vol: f64,
    /// Total internal surface area A_t (m²).
    pub a_tot: f64,
    /// Effective mass area A_m (m²).
    pub mass_area: f64,
    /// Internal heat capacity C_m (J/K).
    pub c_m: f64,
    /// Total opaque transmission conductance H_tr,op (W/K), split below for the boundary temperature.
    pub h_tr_op: f64,
    /// Part of H_tr,op to outdoor air; weights θ_e in the opaque boundary temperature.
    pub h_tr_op_walls_roof: f64,
    /// Part of H_tr,op to the ground; weights the monthly ground temperature.
    pub h_tr_ground: f64,
    /// Thermal-bridge conductance to outdoor air (W/K).
    pub h_tr_tb: f64,
    /// Window transmission H_tr,w, connected to the surface node (W/K).
    pub h_tr_w: f64,
    /// Surface-to-mass coupling H_tr,ms = h_ms A_m (W/K).
    pub h_tr_ms: f64,
    /// Air-to-surface coupling H_tr,is = h_is A_t (W/K).
    pub h_tr_is: f64,
    /// Mass-to-exterior conductance H_tr,em (W/K).
    pub h_tr_em: f64,
    /// Mass-surface heat transfer coefficient h_ms (W/(m² K)), 9.1 by default.
    pub h_ms: f64,
    /// ρ_a c_a of air (J/(m³ K)).
    pub air_volumetric_heat_capacity_j_m3_k: f64,
    /// Unrecovered air changes per hour: infiltration plus natural ventilation.
    pub ach_infiltration: f64,
    /// Mechanical air changes per hour, eligible for heat recovery.
    pub ach_ventilation: f64,
    /// Sensible recovery effectiveness applied to `ach_ventilation` this hour.
    pub heat_recovery_efficiency: f64,
    /// Share of internal gains delivered to the air node (0.5 in Eq. C.1).
    pub convective_fraction_internal: f64,
    /// Standalone constant internal gains (W/m²).
    pub internal_gains_w_per_m2: f64,
    /// Heating setpoint for the current hour (°C).
    pub t_set_heating: f64,
    /// Cooling setpoint for the current hour (°C).
    pub t_set_cooling: f64,
    pub t_set_heating_setback: Option<f64>,
    pub t_set_cooling_setback: Option<f64>,
    /// Control operative instead of air temperature.
    pub use_operative_temperature: bool,
    /// Probability of opening windows when it would cool; 0 disables it.
    pub ventilation_behaviour: f64,
    pub ach_window_open: f64,
    /// Mean sky-air temperature difference Δθ_er (K) for the sky loss.
    pub delta_theta_er: f64,
    /// Sky radiation loss subtracted from the surface node (W).
    pub sky_radiation_loss_w: f64,
    /// Monthly ground boundary temperatures, if a ground policy supplies them.
    pub ground_temps_monthly: Option<[f64; 12]>,
    pub ground_temp_annual_mean: f64,
    /// The platform path: cache hour-invariant coefficients between calls.
    pub prepared_hourly: bool,
    pub prepared_mass_ratio: f64,
    pub prepared_window_ratio: f64,
    /// Node temperatures at the end of the last completed hour.
    pub state: ThermalState,
    pub t_supply: f64,
    pub windows_open: bool,
    /// Window decision held for every thermostat trial of one hour.
    pub window_decision: Option<bool>,
    /// Ventilation conductance H_ve for this hour (W/K).
    pub h_ve: f64,
    pub prepared_airflow: Option<(f64, f64, f64)>,
    pub prepared_network: (f64, f64, f64),
    pub prepared_cn: (f64, f64),
    pub prepared_sens: (f64, f64, f64),
    pub hour_heat_flows: Option<(f64, f64, f64)>,
}

impl Core5R1C {
    /// Month index (0 to 11) of an hour of the year, wrapping outside one year.
    #[inline]
    pub fn month_index_from_hour_of_year(hour_of_year: i64) -> usize {
        let hour_wrapped = hour_of_year.rem_euclid(8760);
        MONTH_START_HOURS.partition_point(|&h| h <= hour_wrapped) - 1
    }

    /// Recompute H_ve = ρ_a c_a V [n_unrecovered + n_mech (1 − η)] / 3600.
    ///
    /// `ach_total_opt` overrides the total air changes (window opening). In the
    /// prepared path the series conductances, Crank-Nicolson factors and unit
    /// responses are recomputed only when airflow or η changed.
    #[inline(always)]
    pub fn update_ventilation_coefficient(
        &mut self,
        ach_total_opt: Option<f64>,
    ) -> Result<(), &'static str> {
        let ach_total = ach_total_opt.unwrap_or(self.ach_infiltration + self.ach_ventilation);
        if self.prepared_hourly {
            let airflow_key = (
                ach_total,
                self.ach_ventilation,
                self.heat_recovery_efficiency,
            );
            if self.prepared_airflow == Some(airflow_key) {
                return Ok(());
            }
            if !ach_total.is_finite() || ach_total < 0.0 {
                return Err("Total air change rate must be finite and nonnegative");
            }
            if ach_total < self.ach_ventilation {
                return Err("Total air change rate cannot omit the declared mechanical flow");
            }
            let ach_unrecovered = ach_total - self.ach_ventilation;
            let ach_effective =
                ach_unrecovered + self.ach_ventilation * (1.0 - self.heat_recovery_efficiency);
            self.h_ve =
                self.air_volumetric_heat_capacity_j_m3_k * self.building_vol * ach_effective
                    / 3600.0;
            self.prepared_airflow = Some(airflow_key);
            let h1 = if self.h_ve > 0.0 && self.h_tr_is > 0.0 {
                1.0 / (1.0 / self.h_ve + 1.0 / self.h_tr_is)
            } else {
                0.0
            };
            let h2 = h1 + self.h_tr_w;
            let h3 = if h2 > 0.0 && self.h_tr_ms > 0.0 {
                1.0 / (1.0 / h2 + 1.0 / self.h_tr_ms)
            } else {
                0.0
            };
            self.prepared_network = (h1, h2, h3);
            let c_dt = self.c_m / 3600.0;
            let h_sum = h3 + self.h_tr_em;
            self.prepared_cn = (c_dt - 0.5 * h_sum, c_dt + 0.5 * h_sum);
            self.prepared_sens = self.compute_sensitivities(1.0);
            return Ok(());
        }

        if !ach_total.is_finite() || ach_total < 0.0 {
            return Err("Total air change rate must be finite and nonnegative");
        }
        if ach_total < self.ach_ventilation {
            return Err("Total air change rate cannot omit the declared mechanical flow");
        }
        let ach_unrecovered = ach_total - self.ach_ventilation;
        let ach_effective =
            ach_unrecovered + self.ach_ventilation * (1.0 - self.heat_recovery_efficiency);
        self.h_ve =
            self.air_volumetric_heat_capacity_j_m3_k * self.building_vol * ach_effective / 3600.0;
        Ok(())
    }

    /// H_tr,1 = 1 / (1/H_ve + 1/H_tr,is), Eq. (C.6).
    #[inline]
    pub fn h_tr_1(&self) -> f64 {
        if self.prepared_hourly {
            return self.prepared_network.0;
        }
        if self.h_ve > 0.0 && self.h_tr_is > 0.0 {
            1.0 / (1.0 / self.h_ve + 1.0 / self.h_tr_is)
        } else {
            0.0
        }
    }

    /// H_tr,2 = H_tr,1 + H_tr,w, Eq. (C.7).
    #[inline]
    pub fn h_tr_2(&self) -> f64 {
        if self.prepared_hourly {
            return self.prepared_network.1;
        }
        self.h_tr_1() + self.h_tr_w
    }

    /// H_tr,3 = 1 / (1/H_tr,2 + 1/H_tr,ms), Eq. (C.8).
    #[inline]
    pub fn h_tr_3(&self) -> f64 {
        if self.prepared_hourly {
            return self.prepared_network.2;
        }
        let h2 = self.h_tr_2();
        if h2 > 0.0 && self.h_tr_ms > 0.0 {
            1.0 / (1.0 / h2 + 1.0 / self.h_tr_ms)
        } else {
            0.0
        }
    }

    /// Operative temperature 0.3 θ_air + 0.7 θ_s, Eq. (C.12).
    #[inline]
    pub fn t_operative(&self) -> f64 {
        0.3 * self.state.t_air + 0.7 * self.state.t_s
    }

    /// Change of (θ_m, θ_s, θ_air) per watt delivered to the air node this hour.
    ///
    /// Differentiates Eq. (C.4), (C.5), (C.10) and (C.11) with respect to φ_HC,
    /// including the H_ve → 0 limit. The thermostat uses it to replace the second
    /// Eq. (C.13) trial when the hour is linear.
    #[inline(always)]
    pub fn compute_sensitivities(&self, dt_hours: f64) -> (f64, f64, f64) {
        let h1 = self.h_tr_1();
        let h2 = self.h_tr_2();
        let h3 = self.h_tr_3();
        let b = if self.prepared_hourly && dt_hours == 1.0 {
            self.prepared_cn.1
        } else {
            let dt_seconds = dt_hours * 3600.0;
            let c_dt = self.c_m / dt_seconds;
            let h_sum = h3 + self.h_tr_em;
            c_dt + 0.5 * h_sum
        };
        // Without ventilation, all air-node heat reaches the surface node.
        let air_to_surface = if self.h_tr_is > 0.0 { 1.0 } else { 0.0 };
        let coupled = self.h_tr_ms + self.h_tr_w;
        let d_phi_m_tot = if self.h_ve > 0.0 && h2 > 0.0 {
            (h3 * h1) / (h2 * self.h_ve)
        } else if self.h_ve <= 0.0 && coupled > 0.0 {
            self.h_tr_ms * air_to_surface / coupled
        } else {
            0.0
        };
        let k_m = if b > 0.0 { d_phi_m_tot / b } else { 0.0 };
        let k_s = if self.h_ve <= 0.0 {
            if coupled > 0.0 {
                (self.h_tr_ms * 0.5 * k_m + air_to_surface) / coupled
            } else {
                k_m * 0.5
            }
        } else {
            let denom = self.h_tr_ms + self.h_tr_w + h1;
            if denom > 0.0 {
                (self.h_tr_ms * 0.5 * k_m + h1 / self.h_ve) / denom
            } else {
                k_m * 0.5
            }
        };
        let denom_air = self.h_tr_is + self.h_ve;
        let k_air = if denom_air > 0.0 {
            (self.h_tr_is * k_s + 1.0) / denom_air
        } else {
            k_s
        };
        (k_m, k_s, k_air)
    }

    /// Cached unit responses in the prepared hourly path, else computed.
    #[inline]
    pub fn sensitivities(&self, dt_hours: f64) -> (f64, f64, f64) {
        if self.prepared_hourly && dt_hours == 1.0 {
            self.prepared_sens
        } else {
            self.compute_sensitivities(dt_hours)
        }
    }

    /// Split gains into φ_ia (air), φ_st (surface) and φ_m (mass), Eq. (C.1)–(C.3).
    ///
    /// Radiant gains go to the mass and surface nodes in proportion to A_m/A_t and
    /// the remaining area after windows. The sky loss is taken from φ_st. HVAC
    /// power `energy_demand` is convective and added to φ_ia.
    #[inline(always)]
    pub fn calc_heat_flows(
        &mut self,
        internal_gains: f64,
        solar_gains: f64,
        energy_demand: f64,
        solar_gains_opaque: f64,
        include_sky_radiation: bool,
    ) -> (f64, f64, f64) {
        if self.prepared_hourly {
            if let Some((phi_ia, phi_st, phi_m)) = self.hour_heat_flows {
                return if energy_demand != 0.0 {
                    (phi_ia + energy_demand, phi_st, phi_m)
                } else {
                    (phi_ia, phi_st, phi_m)
                };
            }
        }
        let phi_sol_total = solar_gains + solar_gains_opaque;
        let mut phi_ia = self.convective_fraction_internal * internal_gains;
        let radiant_fraction = 1.0 - self.convective_fraction_internal;
        let phi_gains_radiant = radiant_fraction * internal_gains + phi_sol_total;

        let (mut phi_st, phi_m) = if self.a_tot > 0.0 {
            let ratio_mass = if self.prepared_hourly {
                self.prepared_mass_ratio
            } else {
                self.mass_area / self.a_tot
            };
            let ratio_window = if self.prepared_hourly {
                self.prepared_window_ratio
            } else {
                self.h_tr_w / (self.h_ms * self.a_tot)
            };
            let rem = 1.0 - ratio_mass - ratio_window;
            let ratio_surface = if rem > 0.0 { rem } else { 0.0 };
            (
                ratio_surface * phi_gains_radiant,
                ratio_mass * phi_gains_radiant,
            )
        } else {
            (0.0, 0.0)
        };

        if include_sky_radiation && self.delta_theta_er > 0.0 {
            phi_st -= self.sky_radiation_loss_w;
        }
        if self.prepared_hourly {
            self.hour_heat_flows = Some((phi_ia + 0.0, phi_st, phi_m));
        }
        phi_ia += energy_demand;
        (phi_ia, phi_st, phi_m)
    }

    /// Total heat flow to the mass node φ_m,tot, Eq. (C.5).
    ///
    /// The opaque boundary temperature weights outdoor air and the monthly ground
    /// temperature by their conductances. Supply air is at outdoor temperature;
    /// heat recovery is already in H_ve.
    #[inline(always)]
    pub fn calc_phi_m_tot(
        &mut self,
        t_out: f64,
        phi_ia: f64,
        phi_st: f64,
        phi_m: f64,
        hour_of_year: Option<i64>,
    ) -> f64 {
        let t_supply = t_out;
        self.t_supply = t_supply;

        let t_e_opaque = match (&self.ground_temps_monthly, hour_of_year) {
            (Some(monthly), Some(hoy)) => {
                let month = Self::month_index_from_hour_of_year(hoy);
                let t_ground = monthly[month];
                if self.h_tr_op > 0.0 {
                    (self.h_tr_op_walls_roof * t_out
                        + self.h_tr_ground * t_ground
                        + self.h_tr_tb * t_out)
                        / self.h_tr_op
                } else {
                    t_out
                }
            }
            (Some(_), None) => {
                let t_ground = self.ground_temp_annual_mean;
                if self.h_tr_op > 0.0 {
                    (self.h_tr_op_walls_roof * t_out
                        + self.h_tr_ground * t_ground
                        + self.h_tr_tb * t_out)
                        / self.h_tr_op
                } else {
                    t_out
                }
            }
            (None, _) => t_out,
        };

        if self.h_ve <= 0.0 {
            // Limit H_ve -> 0 of Eq. (C.5); mirrors iso13790.py calc_phi_m_tot.
            let coupled = self.h_tr_ms + self.h_tr_w;
            if coupled <= 0.0 {
                return phi_m + self.h_tr_em * t_e_opaque;
            }
            let air_heat = if self.h_tr_is > 0.0 { phi_ia } else { 0.0 };
            return phi_m
                + self.h_tr_em * t_e_opaque
                + self.h_tr_ms * (phi_st + self.h_tr_w * t_out + air_heat) / coupled;
        }

        let h1 = self.h_tr_1();
        let h2 = self.h_tr_2();
        let h3 = self.h_tr_3();
        phi_m
            + self.h_tr_em * t_e_opaque
            + h3 * (phi_st + self.h_tr_w * t_out + h1 * ((phi_ia / self.h_ve) + t_supply)) / h2
    }

    /// Mass temperature at the end of the hour by Crank-Nicolson, Eq. (C.4).
    #[inline]
    pub fn calc_t_m_next(&self, t_m_prev: f64, phi_m_tot: f64, dt_hours: f64) -> f64 {
        if self.prepared_hourly && dt_hours == 1.0 {
            let (a, b) = self.prepared_cn;
            return (t_m_prev * a + phi_m_tot) / b;
        }
        let dt_seconds = dt_hours * 3600.0;
        let c_dt = self.c_m / dt_seconds;
        let h_sum = self.h_tr_3() + self.h_tr_em;
        (t_m_prev * (c_dt - 0.5 * h_sum) + phi_m_tot) / (c_dt + 0.5 * h_sum)
    }

    /// Central surface node temperature θ_s, Eq. (C.10).
    #[inline]
    pub fn calc_t_s(&self, t_m: f64, t_out: f64, phi_ia: f64, phi_st: f64) -> f64 {
        let t_supply = t_out;
        if self.h_ve <= 0.0 {
            // Limit H_ve -> 0 of Eq. (C.10); mirrors iso13790.py calc_t_s.
            let h_sum = self.h_tr_ms + self.h_tr_w;
            let air_heat = if self.h_tr_is > 0.0 { phi_ia } else { 0.0 };
            if h_sum > 0.0 {
                return (self.h_tr_ms * t_m + phi_st + self.h_tr_w * t_out + air_heat) / h_sum;
            }
            return t_m;
        }
        let h1 = self.h_tr_1();
        let numerator = self.h_tr_ms * t_m
            + phi_st
            + self.h_tr_w * t_out
            + h1 * (t_supply + phi_ia / self.h_ve);
        let denominator = self.h_tr_ms + self.h_tr_w + h1;
        if denominator > 0.0 {
            numerator / denominator
        } else {
            t_m
        }
    }

    /// Air node temperature θ_air, Eq. (C.11).
    #[inline]
    pub fn calc_t_air(&self, t_s: f64, t_out: f64, phi_ia: f64) -> f64 {
        let t_supply = t_out;
        let h_sum = self.h_tr_is + self.h_ve;
        if h_sum > 0.0 {
            (self.h_tr_is * t_s + self.h_ve * t_supply + phi_ia) / h_sum
        } else {
            t_s
        }
    }

    /// Advance one hour with a fixed HVAC power and return (θ_air, θ_s, θ_m).
    ///
    /// Window opening, if enabled, uses the decision held for this hour so that
    /// every thermostat trial sees the same network.
    #[inline(always)]
    pub fn step_with_rng<F>(
        &mut self,
        inputs: StepInputs,
        energy_demand: f64,
        enable_window_opening: bool,
        mut rng_fn: F,
    ) -> Result<(f64, f64, f64), &'static str>
    where
        F: FnMut() -> Result<f64, &'static str>,
    {
        let StepInputs {
            t_out,
            solar_gains,
            internal_gains,
            solar_gains_opaque,
            hour_of_year,
            dt_hours,
            include_sky_radiation,
        } = inputs;
        if enable_window_opening && self.ventilation_behaviour > 0.0 {
            let open = match self.window_decision {
                Some(open) => open,
                None => self.decide_window(t_out, &mut rng_fn)?,
            };
            self.windows_open = open;
            let ach_window = if open { self.ach_window_open } else { 0.0 };
            self.update_ventilation_coefficient(Some(
                self.ach_infiltration + self.ach_ventilation + ach_window,
            ))?;
        }

        let t_m_prev = self.state.t_m;
        let (phi_ia, phi_st, phi_m) = self.calc_heat_flows(
            internal_gains,
            solar_gains,
            energy_demand,
            solar_gains_opaque,
            include_sky_radiation,
        );
        let phi_m_tot = self.calc_phi_m_tot(t_out, phi_ia, phi_st, phi_m, hour_of_year);
        let t_m_next = self.calc_t_m_next(t_m_prev, phi_m_tot, dt_hours);
        self.state.t_m = (t_m_next + t_m_prev) / 2.0;
        self.state.t_s = self.calc_t_s(self.state.t_m, t_out, phi_ia, phi_st);
        self.state.t_air = self.calc_t_air(self.state.t_s, t_out, phi_ia);

        if enable_window_opening && self.ventilation_behaviour > 0.0 && self.windows_open {
            self.update_ventilation_coefficient(None)?;
        }

        self.state.t_m = t_m_next;
        Ok((self.state.t_air, self.state.t_s, self.state.t_m))
    }

    /// [`Self::step_with_rng`] with deterministic window opening.
    #[inline]
    pub fn step_deterministic(
        &mut self,
        inputs: StepInputs,
        energy_demand: f64,
    ) -> Result<(f64, f64, f64), &'static str> {
        self.step_with_rng(inputs, energy_demand, true, || Ok(1.0))
    }

    /// Window opening from the current air temperature; mirrors `_handle_window_opening`.
    #[inline(always)]
    fn decide_window<F>(&self, t_out: f64, rng_fn: &mut F) -> Result<bool, &'static str>
    where
        F: FnMut() -> Result<f64, &'static str>,
    {
        if self.state.t_air <= self.t_set_cooling || t_out >= self.state.t_air {
            return Ok(false);
        }
        if self.ventilation_behaviour >= 1.0 {
            return Ok(true);
        }
        Ok(rng_fn()? < self.ventilation_behaviour)
    }

    /// Thermostat power for one hour (Eq. C.13): + heating, − cooling, in W.
    ///
    /// Runs the free-floating hour; if the control temperature lies outside the
    /// setpoints, the power that reaches the setpoint is found from the unit
    /// response (linear hours) or from a 10 W/m² trial (window opening), then
    /// clipped to the available heating or cooling capacity. The state ends at
    /// the controlled hour.
    #[inline(always)]
    pub fn calc_heating_demand_with_rng<F>(
        &mut self,
        inputs: StepInputs,
        controls: Thermostat,
        mut rng_fn: F,
    ) -> Result<f64, &'static str>
    where
        F: FnMut() -> Result<f64, &'static str>,
    {
        // Eq. (C.13) interpolates between trials of one network, so the window
        // state is decided once from the start-of-hour air temperature.
        if self.ventilation_behaviour > 0.0 {
            self.window_decision = Some(self.decide_window(inputs.t_out, &mut rng_fn)?);
        }
        let demand = self.demand_trials(inputs, controls, &mut rng_fn);
        self.window_decision = None;
        demand
    }

    #[inline(always)]
    fn demand_trials<F>(
        &mut self,
        inputs: StepInputs,
        controls: Thermostat,
        mut rng_fn: F,
    ) -> Result<f64, &'static str>
    where
        F: FnMut() -> Result<f64, &'static str>,
    {
        let Thermostat {
            max_heating_power,
            max_cooling_power,
            t_set_heat,
            t_set_cool,
        } = controls;
        let dt_hours = inputs.dt_hours;
        if self.prepared_hourly {
            self.hour_heat_flows = None;
        }
        let t_m_prev = self.state.t_m;

        self.step_with_rng(inputs, 0.0, true, &mut rng_fn)?;

        let free_state = (self.state.t_air, self.state.t_s, self.state.t_m);
        let t_ctrl_0 = if self.use_operative_temperature {
            self.t_operative()
        } else {
            self.state.t_air
        };

        if self.ventilation_behaviour <= 0.0 {
            if t_ctrl_0 < t_set_heat {
                if max_heating_power == 0.0 {
                    return Ok(0.0);
                }
                let (k_m, k_s, k_air) = self.sensitivities(dt_hours);
                let k_ctrl = if self.use_operative_temperature {
                    0.3 * k_air + 0.7 * k_s
                } else {
                    k_air
                };
                let q_ref = 10.0 * self.floor_area_m2;
                let mut q = if (k_ctrl * q_ref).abs() > 1e-6 {
                    (t_set_heat - t_ctrl_0) / k_ctrl
                } else {
                    0.0
                };
                if q > max_heating_power {
                    q = max_heating_power;
                }
                if q < 0.0 {
                    q = 0.0;
                }
                self.state.t_air = free_state.0 + k_air * q;
                self.state.t_s = free_state.1 + k_s * q;
                self.state.t_m = free_state.2 + k_m * q;
                return Ok(q);
            } else if t_ctrl_0 > t_set_cool {
                if max_cooling_power == 0.0 {
                    return Ok(0.0);
                }
                let (k_m, k_s, k_air) = self.sensitivities(dt_hours);
                let k_ctrl = if self.use_operative_temperature {
                    0.3 * k_air + 0.7 * k_s
                } else {
                    k_air
                };
                let q_ref = -10.0 * self.floor_area_m2;
                let mut q = if (k_ctrl * q_ref).abs() > 1e-6 {
                    (t_set_cool - t_ctrl_0) / k_ctrl
                } else {
                    0.0
                };
                let min_cool = -max_cooling_power.abs();
                if q < min_cool {
                    q = min_cool;
                }
                if q > 0.0 {
                    q = 0.0;
                }
                self.state.t_air = free_state.0 + k_air * q;
                self.state.t_s = free_state.1 + k_s * q;
                self.state.t_m = free_state.2 + k_m * q;
                return Ok(q);
            } else {
                return Ok(0.0);
            }
        }

        self.state.t_m = t_m_prev;

        let q_demand = if t_ctrl_0 < t_set_heat {
            if self.prepared_hourly && self.ventilation_behaviour <= 0.0 && max_heating_power == 0.0
            {
                self.state.t_air = free_state.0;
                self.state.t_s = free_state.1;
                self.state.t_m = free_state.2;
                return Ok(0.0);
            }
            let t_ctrl_set = t_set_heat;
            let q_ref = 10.0 * self.floor_area_m2;
            self.step_with_rng(inputs, q_ref, true, &mut rng_fn)?;
            let t_ctrl_10 = if self.use_operative_temperature {
                self.t_operative()
            } else {
                self.state.t_air
            };
            self.state.t_m = t_m_prev;

            let mut q = if (t_ctrl_10 - t_ctrl_0).abs() > 1e-6 {
                q_ref * (t_ctrl_set - t_ctrl_0) / (t_ctrl_10 - t_ctrl_0)
            } else {
                0.0
            };
            if q > max_heating_power {
                q = max_heating_power;
            }
            if q < 0.0 {
                q = 0.0;
            }
            q
        } else if t_ctrl_0 > t_set_cool {
            if self.prepared_hourly && self.ventilation_behaviour <= 0.0 && max_cooling_power == 0.0
            {
                self.state.t_air = free_state.0;
                self.state.t_s = free_state.1;
                self.state.t_m = free_state.2;
                return Ok(0.0);
            }
            let t_ctrl_set = t_set_cool;
            let q_ref = -10.0 * self.floor_area_m2;
            self.step_with_rng(inputs, q_ref, true, &mut rng_fn)?;
            let t_ctrl_10 = if self.use_operative_temperature {
                self.t_operative()
            } else {
                self.state.t_air
            };
            self.state.t_m = t_m_prev;

            let mut q = if (t_ctrl_10 - t_ctrl_0).abs() > 1e-6 {
                q_ref * (t_ctrl_set - t_ctrl_0) / (t_ctrl_10 - t_ctrl_0)
            } else {
                0.0
            };
            let min_cool = -max_cooling_power.abs();
            if q < min_cool {
                q = min_cool;
            }
            if q > 0.0 {
                q = 0.0;
            }
            q
        } else {
            0.0
        };

        if self.prepared_hourly && q_demand == 0.0 {
            self.state.t_air = free_state.0;
            self.state.t_s = free_state.1;
            self.state.t_m = free_state.2;
            return Ok(q_demand);
        }

        self.step_with_rng(inputs, q_demand, true, &mut rng_fn)?;

        Ok(q_demand)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(prepared: bool, airflow: f64) -> Core5R1C {
        let mut core = Core5R1C {
            floor_area_m2: 100.0,
            building_vol: 250.0,
            a_tot: 450.0,
            mass_area: 250.0,
            c_m: 16_000_000.0,
            h_tr_op: 250.0,
            h_tr_op_walls_roof: 250.0,
            h_tr_w: 50.0,
            h_tr_ms: 910.0,
            h_tr_is: 1725.0,
            h_tr_em: 1.0 / (1.0 / 250.0 - 1.0 / 910.0),
            h_ms: 9.1,
            air_volumetric_heat_capacity_j_m3_k: 1200.0,
            ach_infiltration: airflow,
            convective_fraction_internal: 0.5,
            t_set_heating: 20.0,
            t_set_cooling: 26.0,
            state: ThermalState {
                t_air: 20.0,
                t_s: 20.0,
                t_m: 20.0,
            },
            prepared_hourly: prepared,
            prepared_mass_ratio: 250.0 / 450.0,
            prepared_window_ratio: 50.0 / (9.1 * 450.0),
            ..Core5R1C::default()
        };
        core.update_ventilation_coefficient(None).unwrap();
        core
    }

    fn inputs(dt_hours: f64) -> StepInputs {
        StepInputs {
            t_out: -5.0,
            solar_gains: 250.0,
            internal_gains: 400.0,
            solar_gains_opaque: 20.0,
            hour_of_year: Some(744),
            dt_hours,
            include_sky_radiation: true,
        }
    }

    #[test]
    fn analytical_sensitivities_match_independent_trial_steps() {
        for prepared in [false, true] {
            for airflow in [0.0, 0.2, 1.0] {
                for dt in [0.25, 1.0, 2.0] {
                    let mut free = model(prepared, airflow);
                    let mut trial = free.clone();
                    let (k_m, k_s, k_air) = free.sensitivities(dt);
                    free.step_deterministic(inputs(dt), 0.0).unwrap();
                    trial.step_deterministic(inputs(dt), 500.0).unwrap();
                    for (actual, expected) in [
                        (trial.state.t_m, free.state.t_m + 500.0 * k_m),
                        (trial.state.t_s, free.state.t_s + 500.0 * k_s),
                        (trial.state.t_air, free.state.t_air + 500.0 * k_air),
                    ] {
                        assert!((actual - expected).abs() < 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn zero_ventilation_is_the_continuous_limit_of_small_airflow() {
        let controls = Thermostat {
            max_heating_power: f64::INFINITY,
            max_cooling_power: 0.0,
            t_set_heat: 20.0,
            t_set_cool: 26.0,
        };
        for prepared in [false, true] {
            let mut demand = [0.0; 2];
            for (index, airflow) in [0.0, 1e-9].into_iter().enumerate() {
                let mut core = model(prepared, airflow);
                for _ in 0..2000 {
                    demand[index] = core
                        .calc_heating_demand_with_rng(inputs(1.0), controls, || Ok(1.0))
                        .unwrap();
                }
                assert!((core.state.t_air - 20.0).abs() < 1e-9);
            }
            assert!((demand[0] - demand[1]).abs() < 1e-4 * demand[1]);
        }
    }

    #[test]
    fn window_decision_is_held_across_thermostat_trials() {
        // A warm start opens the windows; the cold free trial must not close
        // them again before the reference trial, or Eq. (C.13) misses its target.
        let mut core = model(false, 0.2);
        core.ventilation_behaviour = 1.0;
        core.ach_window_open = 6.0;
        core.state.t_air = 28.0;
        let controls = Thermostat {
            max_heating_power: f64::INFINITY,
            max_cooling_power: 0.0,
            t_set_heat: 20.0,
            t_set_cool: 26.0,
        };
        let power = core
            .calc_heating_demand_with_rng(inputs(1.0), controls, || Ok(1.0))
            .unwrap();
        assert!(power > 0.0);
        assert!(core.windows_open);
        assert!(core.window_decision.is_none());
        assert!((core.state.t_air - 20.0).abs() < 1e-9);
    }

    #[test]
    fn control_respects_capacity_and_reaches_unlimited_setpoint() {
        for capacity in [0.0, 100.0, f64::INFINITY] {
            let mut core = model(true, 0.5);
            let controls = Thermostat {
                max_heating_power: capacity,
                max_cooling_power: 0.0,
                t_set_heat: 20.0,
                t_set_cool: 26.0,
            };
            let power = core
                .calc_heating_demand_with_rng(inputs(1.0), controls, || Ok(1.0))
                .unwrap();
            assert!(power >= 0.0 && power <= capacity);
            if capacity.is_finite() {
                assert_eq!(power, capacity);
                assert!(core.state.t_air < 20.0);
            } else {
                assert!((core.state.t_air - 20.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn ground_month_boundaries_wrap_in_both_directions() {
        for (hour, month) in [
            (-1, 11),
            (0, 0),
            (743, 0),
            (744, 1),
            (1415, 1),
            (1416, 2),
            (8759, 11),
            (8760, 0),
        ] {
            assert_eq!(Core5R1C::month_index_from_hour_of_year(hour), month);
        }
    }
}
