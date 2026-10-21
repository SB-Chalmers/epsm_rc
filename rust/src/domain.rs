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
    pub coefficients: (f64, f64, f64, f64),
    pub aggregate_split: bool,
    pub heating_limits_w: &'a [f64],
    pub cooling_limits_w: &'a [f64],
}

impl<'a> DomainArrays<'a> {
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

    #[inline]
    pub fn apply_at(&self, core: &mut Core5R1C, hour: usize) -> KernelResult<StepInputs> {
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

pub fn run_single_domain(
    core: &mut Core5R1C,
    arrays: &DomainArrays<'_>,
) -> KernelResult<DomainOutput> {
    let count = arrays.temperatures.len();
    let mut power = Vec::with_capacity(count);
    let mut indoor = Vec::with_capacity(count);
    for hour in 0..count {
        let inputs = arrays.apply_at(core, hour)?;
        let controls = Thermostat {
            max_heating_power: arrays.heating_limits_w[hour],
            max_cooling_power: arrays.cooling_limits_w[hour],
            t_set_heat: core.t_set_heating,
            t_set_cool: core.t_set_cooling,
        };
        power.push(core.calc_heating_demand_with_rng(inputs, controls, || Ok(1.0))?);
        indoor.push(core.state.t_air);
    }
    Ok(DomainOutput { power, indoor })
}
