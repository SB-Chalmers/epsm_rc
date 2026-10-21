use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const MONTH_START_HOURS: [i64; 12] = [
    0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016,
];

#[derive(Clone, Debug)]
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
    pub t_m: f64,
    pub t_s: f64,
    pub t_air: f64,
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
    pub fn from_py(model: &Bound<'_, PyAny>) -> PyResult<Self> {
        let floor_area_m2: f64 = model.getattr("floor_area_m2")?.extract()?;
        let building_vol: f64 = model.getattr("_building_vol")?.extract()?;
        let a_tot: f64 = model.getattr("_a_tot")?.extract()?;
        let mass_area: f64 = model.getattr("_mass_area")?.extract()?;
        let c_m: f64 = model.getattr("_c_m")?.extract()?;
        let h_tr_op: f64 = model.getattr("_h_tr_op")?.extract()?;
        let h_tr_op_walls_roof: f64 = model.getattr("_h_tr_op_walls_roof")?.extract()?;
        let h_tr_ground: f64 = model.getattr("_h_tr_ground")?.extract()?;
        let h_tr_tb: f64 = model.getattr("_h_tr_tb")?.extract()?;
        let h_tr_w: f64 = model.getattr("_h_tr_w")?.extract()?;
        let h_tr_ms: f64 = model.getattr("_h_tr_ms")?.extract()?;
        let h_tr_is: f64 = model.getattr("_h_tr_is")?.extract()?;
        let h_tr_em: f64 = model.getattr("_h_tr_em")?.extract()?;
        let h_ms: f64 = model.getattr("h_ms")?.extract()?;
        let air_volumetric_heat_capacity_j_m3_k: f64 = model
            .getattr("air_volumetric_heat_capacity_j_m3_k")?
            .extract()?;
        let ach_infiltration: f64 = model.getattr("ach_infiltration")?.extract()?;
        let ach_ventilation: f64 = model.getattr("ach_ventilation")?.extract()?;
        let heat_recovery_efficiency: f64 = model.getattr("heat_recovery_efficiency")?.extract()?;
        let convective_fraction_internal: f64 =
            model.getattr("convective_fraction_internal")?.extract()?;
        let internal_gains_w_per_m2: f64 = model.getattr("internal_gains_w_per_m2")?.extract()?;
        let t_set_heating: f64 = model.getattr("t_set_heating")?.extract()?;
        let t_set_cooling: f64 = model.getattr("t_set_cooling")?.extract()?;
        let t_set_heating_setback: Option<f64> =
            model.getattr("t_set_heating_setback")?.extract()?;
        let t_set_cooling_setback: Option<f64> =
            model.getattr("t_set_cooling_setback")?.extract()?;
        let use_operative_temperature: bool =
            model.getattr("use_operative_temperature")?.extract()?;
        let ventilation_behaviour: f64 = model.getattr("ventilation_behaviour")?.extract()?;
        let ach_window_open: f64 = model.getattr("ach_window_open")?.extract()?;
        let delta_theta_er: f64 = model.getattr("delta_theta_er")?.extract()?;
        let prepared_hourly: bool = model.getattr("_prepared_hourly")?.extract()?;

        let sky_radiation_loss_w: f64 = if prepared_hourly {
            model.getattr("_prepared_sky")?.extract()?
        } else {
            let r_se_roof: f64 = model.getattr("r_se_roof")?.extract()?;
            let u_roof: f64 = model.getattr("u_roof")?.extract()?;
            let roof_area_m2: f64 = model.getattr("roof_area_m2")?.extract()?;
            let h_r: f64 = model.getattr("h_r")?.extract()?;
            let form_factor_roof: f64 = model.getattr("form_factor_roof")?.extract()?;
            let r_se_wall: f64 = model.getattr("r_se_wall")?.extract()?;
            let u_walls: f64 = model.getattr("u_walls")?.extract()?;
            let wall_area_m2: f64 = model.getattr("wall_area_m2")?.extract()?;
            let form_factor_wall: f64 = model.getattr("form_factor_wall")?.extract()?;
            let phi_r_roof =
                r_se_roof * u_roof * roof_area_m2 * h_r * delta_theta_er * form_factor_roof;
            let phi_r_wall =
                r_se_wall * u_walls * wall_area_m2 * h_r * delta_theta_er * form_factor_wall;
            phi_r_roof + phi_r_wall
        };

        let ground_vec: Option<Vec<f64>> = model.getattr("ground_temps_monthly")?.extract()?;
        let (ground_temps_monthly, ground_temp_annual_mean) = match ground_vec {
            Some(v) if v.len() == 12 => {
                let mut arr = [0.0f64; 12];
                let mut sum = 0.0f64;
                for (i, &val) in v.iter().enumerate() {
                    arr[i] = val;
                    sum += val;
                }
                (Some(arr), sum / 12.0)
            }
            _ => (None, 0.0),
        };

        let (prepared_mass_ratio, prepared_window_ratio, prepared_airflow, prepared_network, prepared_cn) =
            if prepared_hourly {
                let pmr: f64 = model.getattr("_prepared_mass_ratio")?.extract()?;
                let pwr: f64 = model.getattr("_prepared_window_ratio")?.extract()?;
                let paf: Option<(f64, f64, f64)> = model.getattr("_prepared_airflow")?.extract()?;
                let pnet: (f64, f64, f64) = model.getattr("_prepared_network")?.extract()?;
                let pcn: (f64, f64) = model.getattr("_prepared_cn")?.extract()?;
                (pmr, pwr, paf, pnet, pcn)
            } else {
                (0.0, 0.0, None, (0.0, 0.0, 0.0), (0.0, 0.0))
            };

        let t_m: f64 = model.getattr("t_m")?.extract()?;
        let t_s: f64 = model.getattr("t_s")?.extract()?;
        let t_air: f64 = model.getattr("t_air")?.extract()?;
        let t_supply: f64 = model.getattr("_t_supply")?.extract()?;
        let windows_open: bool = model.getattr("_windows_open")?.extract()?;
        let h_ve: f64 = model.getattr("_h_ve")?.extract()?;

        let mut core = Self {
            floor_area_m2,
            building_vol,
            a_tot,
            mass_area,
            c_m,
            h_tr_op,
            h_tr_op_walls_roof,
            h_tr_ground,
            h_tr_tb,
            h_tr_w,
            h_tr_ms,
            h_tr_is,
            h_tr_em,
            h_ms,
            air_volumetric_heat_capacity_j_m3_k,
            ach_infiltration,
            ach_ventilation,
            heat_recovery_efficiency,
            convective_fraction_internal,
            internal_gains_w_per_m2,
            t_set_heating,
            t_set_cooling,
            t_set_heating_setback,
            t_set_cooling_setback,
            use_operative_temperature,
            ventilation_behaviour,
            ach_window_open,
            delta_theta_er,
            sky_radiation_loss_w,
            ground_temps_monthly,
            ground_temp_annual_mean,
            prepared_hourly,
            prepared_mass_ratio,
            prepared_window_ratio,
            t_m,
            t_s,
            t_air,
            t_supply,
            windows_open,
            h_ve,
            prepared_airflow,
            prepared_network,
            prepared_cn,
            prepared_sens: (0.0, 0.0, 0.0),
            hour_heat_flows: None,
        };
        core.prepared_sens = core.compute_sensitivities(1.0);
        Ok(core)
    }

    pub fn sync_to_py(&self, model: &Bound<'_, PyAny>) -> PyResult<()> {
        model.setattr("t_m", self.t_m)?;
        model.setattr("t_s", self.t_s)?;
        model.setattr("t_air", self.t_air)?;
        model.setattr("_t_supply", self.t_supply)?;
        model.setattr("_windows_open", self.windows_open)?;
        model.setattr("t_set_heating", self.t_set_heating)?;
        model.setattr("t_set_cooling", self.t_set_cooling)?;
        model.setattr("ach_infiltration", self.ach_infiltration)?;
        model.setattr("ach_ventilation", self.ach_ventilation)?;
        model.setattr(
            "convective_fraction_internal",
            self.convective_fraction_internal,
        )?;
        model.setattr("_h_ve", self.h_ve)?;
        if self.prepared_hourly {
            model.setattr("_prepared_airflow", self.prepared_airflow)?;
            model.setattr("_prepared_network", self.prepared_network)?;
            model.setattr("_prepared_cn", self.prepared_cn)?;
            model.setattr("_hour_heat_flows", self.hour_heat_flows)?;
        }
        Ok(())
    }

    #[inline(always)]
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
            self.h_ve = self.air_volumetric_heat_capacity_j_m3_k
                * self.building_vol
                * ach_effective
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

    #[inline(always)]
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

    #[inline(always)]
    pub fn h_tr_2(&self) -> f64 {
        if self.prepared_hourly {
            return self.prepared_network.1;
        }
        self.h_tr_1() + self.h_tr_w
    }

    #[inline(always)]
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

    #[inline(always)]
    pub fn t_operative(&self) -> f64 {
        0.3 * self.t_air + 0.7 * self.t_s
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

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
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
        t_out: f64,
        solar_gains: f64,
        internal_gains: f64,
        energy_demand: f64,
        dt_hours: f64,
        solar_gains_opaque: f64,
        include_sky_radiation: bool,
        enable_window_opening: bool,
        hour_of_year: Option<i64>,
        mut rng_fn: F,
    ) -> Result<(f64, f64, f64), &'static str>
    where
        F: FnMut() -> Result<f64, &'static str>,
    {
        if enable_window_opening && self.ventilation_behaviour > 0.0 {
            self.windows_open = false;
            let ach_window = if self.t_air > self.t_set_cooling && t_out < self.t_air {
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

        let t_m_prev = self.t_m;
        let (phi_ia, phi_st, phi_m) = self.calc_heat_flows(
            internal_gains,
            solar_gains,
            energy_demand,
            solar_gains_opaque,
            include_sky_radiation,
        );
        let phi_m_tot = self.calc_phi_m_tot(t_out, phi_ia, phi_st, phi_m, hour_of_year);
        let t_m_next = self.calc_t_m_next(t_m_prev, phi_m_tot, dt_hours);
        self.t_m = (t_m_next + t_m_prev) / 2.0;
        self.t_s = self.calc_t_s(self.t_m, t_out, phi_ia, phi_st);
        self.t_air = self.calc_t_air(self.t_s, t_out, phi_ia);

        if enable_window_opening && self.ventilation_behaviour > 0.0 && self.windows_open {
            self.update_ventilation_coefficient(None)?;
        }

        self.t_m = t_m_next;
        Ok((self.t_air, self.t_s, self.t_m))
    }

    #[inline(always)]
    pub fn step_deterministic(
        &mut self,
        t_out: f64,
        solar_gains: f64,
        internal_gains: f64,
        energy_demand: f64,
        dt_hours: f64,
        solar_gains_opaque: f64,
        include_sky_radiation: bool,
        hour_of_year: Option<i64>,
    ) -> Result<(f64, f64, f64), &'static str> {
        self.step_with_rng(
            t_out,
            solar_gains,
            internal_gains,
            energy_demand,
            dt_hours,
            solar_gains_opaque,
            include_sky_radiation,
            true,
            hour_of_year,
            || Ok(1.0),
        )
    }

    #[inline(always)]
    pub fn calc_heating_demand_with_rng<F>(
        &mut self,
        t_out: f64,
        solar_gains: f64,
        internal_gains: f64,
        dt_hours: f64,
        max_heating_power: f64,
        max_cooling_power: f64,
        solar_gains_opaque: f64,
        hour_of_year: Option<i64>,
        include_sky_radiation: bool,
        t_set_heat: f64,
        t_set_cool: f64,
        mut rng_fn: F,
    ) -> Result<f64, &'static str>
    where
        F: FnMut() -> Result<f64, &'static str>,
    {
        if self.prepared_hourly {
            self.hour_heat_flows = None;
        }
        let t_m_prev = self.t_m;

        self.step_with_rng(
            t_out,
            solar_gains,
            internal_gains,
            0.0,
            dt_hours,
            solar_gains_opaque,
            include_sky_radiation,
            true,
            hour_of_year,
            &mut rng_fn,
        )?;

        let free_state = (self.t_air, self.t_s, self.t_m);
        let t_ctrl_0 = if self.use_operative_temperature {
            self.t_operative()
        } else {
            self.t_air
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
                self.t_air = free_state.0 + k_air * q;
                self.t_s = free_state.1 + k_s * q;
                self.t_m = free_state.2 + k_m * q;
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
                self.t_air = free_state.0 + k_air * q;
                self.t_s = free_state.1 + k_s * q;
                self.t_m = free_state.2 + k_m * q;
                return Ok(q);
            } else {
                return Ok(0.0);
            }
        }

        self.t_m = t_m_prev;

        let q_demand = if t_ctrl_0 < t_set_heat {
            if self.prepared_hourly && self.ventilation_behaviour <= 0.0 && max_heating_power == 0.0
            {
                self.t_air = free_state.0;
                self.t_s = free_state.1;
                self.t_m = free_state.2;
                return Ok(0.0);
            }
            let t_ctrl_set = t_set_heat;
            let q_ref = 10.0 * self.floor_area_m2;
            self.step_with_rng(
                t_out,
                solar_gains,
                internal_gains,
                q_ref,
                dt_hours,
                solar_gains_opaque,
                include_sky_radiation,
                true,
                hour_of_year,
                &mut rng_fn,
            )?;
            let t_ctrl_10 = if self.use_operative_temperature {
                self.t_operative()
            } else {
                self.t_air
            };
            self.t_m = t_m_prev;

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
                self.t_air = free_state.0;
                self.t_s = free_state.1;
                self.t_m = free_state.2;
                return Ok(0.0);
            }
            let t_ctrl_set = t_set_cool;
            let q_ref = -10.0 * self.floor_area_m2;
            self.step_with_rng(
                t_out,
                solar_gains,
                internal_gains,
                q_ref,
                dt_hours,
                solar_gains_opaque,
                include_sky_radiation,
                true,
                hour_of_year,
                &mut rng_fn,
            )?;
            let t_ctrl_10 = if self.use_operative_temperature {
                self.t_operative()
            } else {
                self.t_air
            };
            self.t_m = t_m_prev;

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
            self.t_air = free_state.0;
            self.t_s = free_state.1;
            self.t_m = free_state.2;
            return Ok(q_demand);
        }

        self.step_with_rng(
            t_out,
            solar_gains,
            internal_gains,
            q_demand,
            dt_hours,
            solar_gains_opaque,
            include_sky_radiation,
            true,
            hour_of_year,
            &mut rng_fn,
        )?;

        Ok(q_demand)
    }
}

#[pyfunction]
#[pyo3(signature = (
    model,
    t_outdoor,
    solar_gains,
    internal_gains,
    solar_gains_opaque,
    occupancy_schedule,
    dt_hours,
    max_heating_power,
    max_cooling_power,
    include_sky_radiation,
    random_fn=None,
    use_hour_of_year=true
))]
#[allow(clippy::too_many_arguments)]
pub fn simulate_5r1c_hourly<'py>(
    py: Python<'py>,
    model: &Bound<'py, PyAny>,
    t_outdoor: PyReadonlyArray1<'py, f64>,
    solar_gains: PyReadonlyArray1<'py, f64>,
    internal_gains: PyReadonlyArray1<'py, f64>,
    solar_gains_opaque: PyReadonlyArray1<'py, f64>,
    occupancy_schedule: Option<PyReadonlyArray1<'py, bool>>,
    dt_hours: f64,
    max_heating_power: f64,
    max_cooling_power: f64,
    include_sky_radiation: bool,
    random_fn: Option<&Bound<'py, PyAny>>,
    use_hour_of_year: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mut core = Core5R1C::from_py(model)?;

    let t_out_slice = t_outdoor.as_slice()?;
    let sol_slice = solar_gains.as_slice()?;
    let int_slice = internal_gains.as_slice()?;
    let opq_slice = solar_gains_opaque.as_slice()?;
    let occ_slice = match &occupancy_schedule {
        Some(arr) => Some(arr.as_slice()?),
        None => None,
    };

    let n = t_out_slice.len();
    let mut q_heat = Vec::with_capacity(n);
    let mut t_air = Vec::with_capacity(n);
    let mut t_m = Vec::with_capacity(n);
    let mut t_s = Vec::with_capacity(n);
    let mut t_op = Vec::with_capacity(n);
    let mut windows_open = Vec::with_capacity(n);

    // Reset to heating setpoint
    core.t_m = core.t_set_heating;
    core.t_s = core.t_set_heating;
    core.t_air = core.t_set_heating;
    core.windows_open = false;

    for i in 0..n {
        let hoy_opt = if use_hour_of_year {
            Some((i as f64 * dt_hours) as i64)
        } else {
            None
        };
        let (t_set_heat, t_set_cool) = match (occ_slice, hoy_opt) {
            (Some(occ), Some(hour_of_year))
                if hour_of_year >= 0 && (hour_of_year as usize) < occ.len() =>
            {
                if !occ[hour_of_year as usize] {
                    (
                        core.t_set_heating_setback.unwrap_or(core.t_set_heating),
                        core.t_set_cooling_setback.unwrap_or(core.t_set_cooling),
                    )
                } else {
                    (core.t_set_heating, core.t_set_cooling)
                }
            }
            _ => (core.t_set_heating, core.t_set_cooling),
        };

        let q = core
            .calc_heating_demand_with_rng(
                t_out_slice[i],
                sol_slice[i],
                int_slice[i],
                dt_hours,
                max_heating_power,
                max_cooling_power,
                opq_slice[i],
                hoy_opt,
                include_sky_radiation,
                t_set_heat,
                t_set_cool,
                || match random_fn {
                    Some(f) => f
                        .call0()
                        .and_then(|v| v.extract::<f64>())
                        .map_err(|_| "Failed to call random.random()"),
                    None => Ok(1.0),
                },
            )
            .map_err(PyValueError::new_err)?;

        q_heat.push(q);
        t_air.push(core.t_air);
        t_m.push(core.t_m);
        t_s.push(core.t_s);
        t_op.push(core.t_operative());
        windows_open.push(core.windows_open);
    }

    core.sync_to_py(model)?;

    let result = PyDict::new(py);
    result.set_item("heating_demand", PyArray1::from_vec(py, q_heat))?;
    result.set_item("t_air", PyArray1::from_vec(py, t_air))?;
    result.set_item("t_m", PyArray1::from_vec(py, t_m))?;
    result.set_item("t_s", PyArray1::from_vec(py, t_s))?;
    result.set_item("t_operative", PyArray1::from_vec(py, t_op))?;
    result.set_item("windows_open", PyArray1::from_vec(py, windows_open))?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (
    model,
    t_out,
    solar_gains=0.0,
    internal_gains=None,
    dt_hours=1.0,
    max_heating_power=f64::INFINITY,
    max_cooling_power=0.0,
    solar_gains_opaque=0.0,
    hour_of_year=None,
    include_sky_radiation=true,
    t_set_heat=None,
    t_set_cool=None,
    random_fn=None
))]
pub fn calc_heating_demand_5r1c<'py>(
    model: &Bound<'py, PyAny>,
    t_out: f64,
    solar_gains: f64,
    internal_gains: Option<f64>,
    dt_hours: f64,
    max_heating_power: f64,
    max_cooling_power: f64,
    solar_gains_opaque: f64,
    hour_of_year: Option<i64>,
    include_sky_radiation: bool,
    t_set_heat: Option<f64>,
    t_set_cool: Option<f64>,
    random_fn: Option<&Bound<'py, PyAny>>,
) -> PyResult<f64> {
    let mut core = Core5R1C::from_py(model)?;
    let int_gains =
        internal_gains.unwrap_or(core.internal_gains_w_per_m2 * core.floor_area_m2);
    let heat_set = t_set_heat.unwrap_or(core.t_set_heating);
    let cool_set = t_set_cool.unwrap_or(core.t_set_cooling);

    let q = core
        .calc_heating_demand_with_rng(
            t_out,
            solar_gains,
            int_gains,
            dt_hours,
            max_heating_power,
            max_cooling_power,
            solar_gains_opaque,
            hour_of_year,
            include_sky_radiation,
            heat_set,
            cool_set,
            || match random_fn {
                Some(f) => f
                    .call0()
                    .and_then(|v| v.extract::<f64>())
                    .map_err(|_| "Failed to call random.random()"),
                None => Ok(1.0),
            },
        )
        .map_err(PyValueError::new_err)?;

    core.sync_to_py(model)?;
    Ok(q)
}
