use crate::types::{
    KernelResult, SimulationInputs, SimulationOutput, StepInputs, ThermalState, Thermostat,
};

// Forced inlining is limited to measured control/airflow hot paths. Keep
// benchmarks/native.py comparisons when changing these boundaries.

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

const MONTH_START_HOURS: [i64; 12] = [
    0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016,
];

#[derive(Clone, Debug)]
#[cfg_attr(test, derive(Default))]
pub struct Core5R1C {
    pub floor_area_m2: f64,
    pub building_vol: f64,
    pub a_tot: f64,
    pub mass_area: f64,
    pub c_m: f64,
    pub h_tr_op: f64,
    pub h_tr_op_walls_roof: f64,
    pub h_tr_ground: f64,
    pub h_tr_tb: f64,
    pub h_tr_w: f64,
    pub h_tr_ms: f64,
    pub h_tr_is: f64,
    pub h_tr_em: f64,
    pub h_ms: f64,
    pub air_volumetric_heat_capacity_j_m3_k: f64,
    pub ach_infiltration: f64,
    pub ach_ventilation: f64,
    pub heat_recovery_efficiency: f64,
    pub convective_fraction_internal: f64,
    pub internal_gains_w_per_m2: f64,
    pub t_set_heating: f64,
    pub t_set_cooling: f64,
    pub t_set_heating_setback: Option<f64>,
    pub t_set_cooling_setback: Option<f64>,
    pub use_operative_temperature: bool,
    pub ventilation_behaviour: f64,
    pub ach_window_open: f64,
    pub delta_theta_er: f64,
    pub sky_radiation_loss_w: f64,
    pub ground_temps_monthly: Option<[f64; 12]>,
    pub ground_temp_annual_mean: f64,
    pub prepared_hourly: bool,
    pub prepared_mass_ratio: f64,
    pub prepared_window_ratio: f64,
    pub state: ThermalState,
    pub t_supply: f64,
    pub windows_open: bool,
    pub h_ve: f64,
    pub prepared_airflow: Option<(f64, f64, f64)>,
    pub prepared_network: (f64, f64, f64),
    pub prepared_cn: (f64, f64),
    pub prepared_sens: (f64, f64, f64),
    pub hour_heat_flows: Option<(f64, f64, f64)>,
}

impl Core5R1C {
    #[inline]
    pub fn month_index_from_hour_of_year(hour_of_year: i64) -> usize {
        let hour_wrapped = hour_of_year.rem_euclid(8760);
        MONTH_START_HOURS.partition_point(|&h| h <= hour_wrapped) - 1
    }

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

    #[inline]
    pub fn h_tr_2(&self) -> f64 {
        if self.prepared_hourly {
            return self.prepared_network.1;
        }
        self.h_tr_1() + self.h_tr_w
    }

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

    #[inline]
    pub fn t_operative(&self) -> f64 {
        0.3 * self.state.t_air + 0.7 * self.state.t_s
    }

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
        let d_phi_m_tot = if self.h_ve > 0.0 && h2 > 0.0 {
            (h3 * h1) / (h2 * self.h_ve)
        } else {
            0.0
        };
        let k_m = if b > 0.0 { d_phi_m_tot / b } else { 0.0 };
        let k_s = if self.h_ve <= 0.0 {
            let denom = self.h_tr_ms + self.h_tr_w + self.h_tr_is;
            if denom > 0.0 {
                (self.h_tr_ms * 0.5 * k_m) / denom
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

    #[inline]
    pub fn sensitivities(&self, dt_hours: f64) -> (f64, f64, f64) {
        if self.prepared_hourly && dt_hours == 1.0 {
            self.prepared_sens
        } else {
            self.compute_sensitivities(dt_hours)
        }
    }

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
            return phi_m + self.h_tr_em * t_e_opaque;
        }

        let h1 = self.h_tr_1();
        let h2 = self.h_tr_2();
        let h3 = self.h_tr_3();
        phi_m
            + self.h_tr_em * t_e_opaque
            + h3 * (phi_st + self.h_tr_w * t_out + h1 * ((phi_ia / self.h_ve) + t_supply)) / h2
    }

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

    #[inline]
    pub fn calc_t_s(&self, t_m: f64, t_out: f64, phi_ia: f64, phi_st: f64) -> f64 {
        let t_supply = t_out;
        if self.h_ve <= 0.0 {
            let h_sum = self.h_tr_ms + self.h_tr_w + self.h_tr_is;
            if h_sum > 0.0 {
                return (self.h_tr_ms * t_m + phi_st + self.h_tr_w * t_out) / h_sum;
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
            self.windows_open = false;
            let ach_window = if self.state.t_air > self.t_set_cooling && t_out < self.state.t_air {
                let open = if self.ventilation_behaviour >= 1.0 {
                    true
                } else {
                    rng_fn()? < self.ventilation_behaviour
                };
                if open {
                    self.windows_open = true;
                    self.ach_window_open
                } else {
                    0.0
                }
            } else {
                0.0
            };
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

    #[inline]
    pub fn step_deterministic(
        &mut self,
        inputs: StepInputs,
        energy_demand: f64,
    ) -> Result<(f64, f64, f64), &'static str> {
        self.step_with_rng(inputs, energy_demand, true, || Ok(1.0))
    }

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
