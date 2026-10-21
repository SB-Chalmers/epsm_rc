use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::iso13790::Core5R1C;

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
    #[inline(always)]
    pub fn apply_at(
        &self,
        core: &mut Core5R1C,
        hour: usize,
    ) -> Result<(f64, f64, f64, f64, Option<i64>), &'static str> {
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
            + self.coefficients.1 * (core.t_air - temperature).abs()
            + self.coefficients.2 * w
            + self.coefficients.3 * w.powi(2);
        core.ach_infiltration = self.infiltration[hour] * correction + self.natural[hour];
        core.ach_ventilation = self.ventilation[hour];
        if !core.ach_infiltration.is_finite() || core.ach_infiltration < 0.0 {
            return Err("Hourly infiltration must be finite and nonnegative");
        }
        core.update_ventilation_coefficient(None)?;
        let hour_of_year = self.month_hours.map(|mh| mh[hour]);
        Ok((
            temperature,
            self.solar[hour],
            gain,
            self.opaque[hour],
            hour_of_year,
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (
    model,
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
    month_hours,
    coefficients,
    aggregate_split,
    heating_limits_w,
    cooling_limits_w
))]
#[allow(clippy::too_many_arguments)]
pub fn run_single_domain_hourly<'py>(
    py: Python<'py>,
    model: &Bound<'py, PyAny>,
    temperatures: PyReadonlyArray1<'py, f64>,
    gains: PyReadonlyArray1<'py, f64>,
    convective: PyReadonlyArray1<'py, f64>,
    heating_setpoints: PyReadonlyArray1<'py, f64>,
    cooling_setpoints: PyReadonlyArray1<'py, f64>,
    wind: PyReadonlyArray1<'py, f64>,
    infiltration: PyReadonlyArray1<'py, f64>,
    natural: PyReadonlyArray1<'py, f64>,
    ventilation: PyReadonlyArray1<'py, f64>,
    solar: PyReadonlyArray1<'py, f64>,
    opaque: PyReadonlyArray1<'py, f64>,
    month_hours: Option<PyReadonlyArray1<'py, i64>>,
    coefficients: (f64, f64, f64, f64),
    aggregate_split: bool,
    heating_limits_w: PyReadonlyArray1<'py, f64>,
    cooling_limits_w: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let mut core = Core5R1C::from_py(model)?;

    let mh_slice = match &month_hours {
        Some(arr) => Some(arr.as_slice()?),
        None => None,
    };

    let arrays = DomainArrays {
        temperatures: temperatures.as_slice()?,
        gains: gains.as_slice()?,
        convective: convective.as_slice()?,
        heating_setpoints: heating_setpoints.as_slice()?,
        cooling_setpoints: cooling_setpoints.as_slice()?,
        wind: wind.as_slice()?,
        infiltration: infiltration.as_slice()?,
        natural: natural.as_slice()?,
        ventilation: ventilation.as_slice()?,
        solar: solar.as_slice()?,
        opaque: opaque.as_slice()?,
        month_hours: mh_slice,
        coefficients,
        aggregate_split,
        heating_limits_w: heating_limits_w.as_slice()?,
        cooling_limits_w: cooling_limits_w.as_slice()?,
    };

    let count = arrays.temperatures.len();
    let mut power = Vec::with_capacity(count);
    let mut indoor = Vec::with_capacity(count);

    for hour in 0..count {
        let (t_out, sol, int_gains, opq, hoy) =
            match arrays.apply_at(&mut core, hour) {
                Ok(v) => v,
                Err(msg) => {
                    core.sync_to_py(model)?;
                    return Err(PyValueError::new_err(msg));
                }
            };

        let t_set_heat = core.t_set_heating;
        let t_set_cool = core.t_set_cooling;
        let q = match core.calc_heating_demand_with_rng(
            t_out,
            sol,
            int_gains,
            1.0,
            arrays.heating_limits_w[hour],
            arrays.cooling_limits_w[hour],
            opq,
            hoy,
            true,
            t_set_heat,
            t_set_cool,
            || Ok(1.0),
        ) {
            Ok(val) => val,
            Err(msg) => {
                core.sync_to_py(model)?;
                return Err(PyValueError::new_err(msg));
            }
        };

        power.push(q);
        indoor.push(core.t_air);
    }

    core.sync_to_py(model)?;
    Ok((
        PyArray1::from_vec(py, power),
        PyArray1::from_vec(py, indoor),
    ))
}
