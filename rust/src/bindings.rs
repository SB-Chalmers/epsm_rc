//! Python/NumPy conversion and model synchronization, outside numerical loops.
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::coupled::{run_coupled_domains, solve_controls_inner};
use crate::domain::{run_single_domain, DomainArrays, DomainSnapshot};
use crate::iso13790::{simulate, Core5R1C};
use crate::types::{ControlSolution, SimulationInputs, StepInputs, ThermalState, Thermostat};

// Bindings retain their existing Python signatures; numerical kernels use named
// input/output structs instead of these Python-facing tuples and argument lists.
type PySeries<'py> = Bound<'py, PyArray1<f64>>;
type PyDomainPair<'py> = Bound<'py, PyArray2<f64>>;
type PyControlSolution = ((f64, f64), (f64, f64), (f64, f64), f64);
type PyCoupledOutput<'py> = (
    PyDomainPair<'py>,
    PyDomainPair<'py>,
    PyDomainPair<'py>,
    PySeries<'py>,
);

// Borrow only storage backed by immutable Python bytes, retaining those bytes
// until after detach. A NumPy WRITEABLE=false flag alone is insufficient: a
// caller may still own a writable alias. Subclasses may override .base, so they
// always take the owned-snapshot path.
fn immutable_backing<'py>(array: &Bound<'py, PyUntypedArray>) -> Option<Bound<'py, PyBytes>> {
    let mut current = array.as_any().clone();
    for _ in 0..4 {
        if current.is_instance_of::<PyBytes>() {
            return current.cast_into::<PyBytes>().ok();
        }
        if !current.is_exact_instance_of::<PyUntypedArray>() {
            return None;
        }
        current = current.getattr("base").ok()?;
    }
    None
}

fn immutable_backings<'py>(
    columns: [&Bound<'py, PyUntypedArray>; 13],
    month_hours: Option<&Bound<'py, PyUntypedArray>>,
) -> Option<Vec<Bound<'py, PyBytes>>> {
    columns
        .into_iter()
        .chain(month_hours)
        .map(immutable_backing)
        .collect()
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

        let (
            prepared_mass_ratio,
            prepared_window_ratio,
            prepared_airflow,
            prepared_network,
            prepared_cn,
        ) = if prepared_hourly {
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
            state: ThermalState { t_m, t_s, t_air },
            t_supply,
            windows_open,
            window_decision: None,
            h_ve,
            prepared_airflow,
            prepared_network,
            prepared_cn,
            prepared_sens: (0.0, 0.0, 0.0),
            hour_heat_flows: None,
        };
        // Standalone models calculate sensitivities for their actual timestep.
        // Only the prepared hourly path consumes this cached one-hour response.
        if core.prepared_hourly {
            core.prepared_sens = core.compute_sensitivities(1.0);
        }
        Ok(core)
    }

    pub fn sync_to_py(&self, model: &Bound<'_, PyAny>) -> PyResult<()> {
        model.setattr("t_m", self.state.t_m)?;
        model.setattr("t_s", self.state.t_s)?;
        model.setattr("t_air", self.state.t_air)?;
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

    let forcing = SimulationInputs {
        outdoor: t_outdoor.as_slice()?.to_vec(),
        solar: solar_gains.as_slice()?.to_vec(),
        internal: internal_gains.as_slice()?.to_vec(),
        opaque: solar_gains_opaque.as_slice()?.to_vec(),
        occupancy: occupancy_schedule
            .as_ref()
            .map(|array| array.as_slice().map(<[bool]>::to_vec))
            .transpose()?,
        dt_hours,
        include_sky_radiation,
        use_hour_of_year,
    };
    let controls = Thermostat {
        max_heating_power,
        max_cooling_power,
        t_set_heat: core.t_set_heating,
        t_set_cool: core.t_set_cooling,
    };
    forcing.validate(controls).map_err(PyValueError::new_err)?;
    let mut random_error = None;
    let outcome = if let Some(callback) = random_fn {
        simulate(&mut core, &forcing, controls, || {
            callback
                .call0()
                .and_then(|value| value.extract())
                .map_err(|error| {
                    random_error = Some(error);
                    "Window opening random callback failed"
                })
        })
    } else {
        py.detach(|| simulate(&mut core, &forcing, controls, || Ok(1.0)))
    };
    core.sync_to_py(model)?;
    if let Some(error) = random_error {
        return Err(error);
    }
    let output = outcome.map_err(PyValueError::new_err)?;

    let result = PyDict::new(py);
    result.set_item(
        "heating_demand",
        PyArray1::from_vec(py, output.heating_demand),
    )?;
    result.set_item("t_air", PyArray1::from_vec(py, output.t_air))?;
    result.set_item("t_m", PyArray1::from_vec(py, output.t_m))?;
    result.set_item("t_s", PyArray1::from_vec(py, output.t_s))?;
    result.set_item("t_operative", PyArray1::from_vec(py, output.t_operative))?;
    result.set_item("windows_open", PyArray1::from_vec(py, output.windows_open))?;
    Ok(result)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
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
    let int_gains = internal_gains.unwrap_or(core.internal_gains_w_per_m2 * core.floor_area_m2);
    let heat_set = t_set_heat.unwrap_or(core.t_set_heating);
    let cool_set = t_set_cool.unwrap_or(core.t_set_cooling);

    let inputs = StepInputs {
        t_out,
        solar_gains,
        internal_gains: int_gains,
        solar_gains_opaque,
        hour_of_year,
        dt_hours,
        include_sky_radiation,
    };
    let controls = Thermostat {
        max_heating_power,
        max_cooling_power,
        t_set_heat: heat_set,
        t_set_cool: cool_set,
    };
    controls.validate(dt_hours).map_err(PyValueError::new_err)?;
    let q = core
        .calc_heating_demand_with_rng(inputs, controls, || match random_fn {
            Some(f) => f
                .call0()
                .and_then(|v| v.extract::<f64>())
                .map_err(|_| "Failed to call random.random()"),
            None => Ok(1.0),
        })
        .map_err(PyValueError::new_err)?;

    core.sync_to_py(model)?;
    Ok(q)
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
) -> PyResult<(PySeries<'py>, PySeries<'py>)> {
    let mut core = Core5R1C::from_py(model)?;

    let mh_slice = match &month_hours {
        Some(arr) => Some(arr.as_slice()?),
        None => None,
    };

    let columns = [
        temperatures.as_slice()?,
        gains.as_slice()?,
        convective.as_slice()?,
        heating_setpoints.as_slice()?,
        cooling_setpoints.as_slice()?,
        wind.as_slice()?,
        infiltration.as_slice()?,
        natural.as_slice()?,
        ventilation.as_slice()?,
        solar.as_slice()?,
        opaque.as_slice()?,
        heating_limits_w.as_slice()?,
        cooling_limits_w.as_slice()?,
    ];
    let borrowed = DomainArrays::new(columns, mh_slice, coefficients, aggregate_split);
    borrowed
        .validate(borrowed.temperatures.len())
        .map_err(PyValueError::new_err)?;
    let keepalive = immutable_backings(
        [
            temperatures.as_untyped(),
            gains.as_untyped(),
            convective.as_untyped(),
            heating_setpoints.as_untyped(),
            cooling_setpoints.as_untyped(),
            wind.as_untyped(),
            infiltration.as_untyped(),
            natural.as_untyped(),
            ventilation.as_untyped(),
            solar.as_untyped(),
            opaque.as_untyped(),
            heating_limits_w.as_untyped(),
            cooling_limits_w.as_untyped(),
        ],
        month_hours.as_ref().map(|array| array.as_untyped()),
    );
    let snapshot = if keepalive.is_some() {
        None
    } else {
        Some(
            DomainSnapshot::new(columns, mh_slice, coefficients, aggregate_split)
                .map_err(PyValueError::new_err)?,
        )
    };
    let arrays = match &snapshot {
        Some(snapshot) => snapshot.view(),
        None => borrowed,
    };

    arrays
        .validate(arrays.temperatures.len())
        .map_err(PyValueError::new_err)?;
    let outcome = py.detach(|| run_single_domain(&mut core, &arrays));
    drop(keepalive);
    core.sync_to_py(model)?;
    let output = outcome.map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, output.power),
        PyArray1::from_vec(py, output.indoor),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    free_c,
    response_k_w,
    conductance_w_k,
    heat_c,
    cool_c,
    heating_limit_w=None,
    cooling_limit_w=None
))]
pub fn solve_controls_rs(
    free_c: Vec<f64>,
    response_k_w: Vec<f64>,
    conductance_w_k: f64,
    heat_c: Vec<f64>,
    cool_c: Vec<f64>,
    heating_limit_w: Option<Vec<f64>>,
    cooling_limit_w: Option<Vec<f64>>,
) -> PyResult<PyControlSolution> {
    let heating_limit_w = heating_limit_w.unwrap_or_else(|| vec![f64::INFINITY, f64::INFINITY]);
    let cooling_limit_w = cooling_limit_w.unwrap_or_else(|| vec![0.0, 0.0]);
    if free_c.len() != 2 || response_k_w.len() != 2 {
        return Err(PyValueError::new_err(
            "Two finite free temperatures and positive responses are required",
        ));
    }
    if heat_c.len() != 2 || cool_c.len() != 2 {
        return Err(PyValueError::new_err(
            "Finite ordered heating and cooling setpoints are required",
        ));
    }
    if heating_limit_w.len() != 2 || cooling_limit_w.len() != 2 {
        return Err(PyValueError::new_err(
            "Heating limits must be nonnegative; cooling limits nonpositive",
        ));
    }
    let ControlSolution {
        powers,
        temperatures: temps,
        exchanges: exch,
        residual: res,
    } = solve_controls_inner(
        [free_c[0], free_c[1]],
        [response_k_w[0], response_k_w[1]],
        conductance_w_k,
        [heat_c[0], heat_c[1]],
        [cool_c[0], cool_c[1]],
        [heating_limit_w[0], heating_limit_w[1]],
        [cooling_limit_w[0], cooling_limit_w[1]],
    )
    .map_err(PyValueError::new_err)?;

    Ok((
        (powers[0], powers[1]),
        (temps[0], temps[1]),
        (exch[0], exch[1]),
        res,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    model0,
    model1,
    temperatures0,
    gains0,
    convective0,
    heating_setpoints0,
    cooling_setpoints0,
    wind0,
    infiltration0,
    natural0,
    ventilation0,
    solar0,
    opaque0,
    month_hours0,
    coefficients0,
    aggregate_split0,
    heating_limits_w0,
    cooling_limits_w0,
    temperatures1,
    gains1,
    convective1,
    heating_setpoints1,
    cooling_setpoints1,
    wind1,
    infiltration1,
    natural1,
    ventilation1,
    solar1,
    opaque1,
    month_hours1,
    coefficients1,
    aggregate_split1,
    heating_limits_w1,
    cooling_limits_w1,
    conductance_w_k
))]
#[allow(clippy::too_many_arguments)]
pub fn run_coupled_domains_hourly<'py>(
    py: Python<'py>,
    model0: &Bound<'py, PyAny>,
    model1: &Bound<'py, PyAny>,
    temperatures0: PyReadonlyArray1<'py, f64>,
    gains0: PyReadonlyArray1<'py, f64>,
    convective0: PyReadonlyArray1<'py, f64>,
    heating_setpoints0: PyReadonlyArray1<'py, f64>,
    cooling_setpoints0: PyReadonlyArray1<'py, f64>,
    wind0: PyReadonlyArray1<'py, f64>,
    infiltration0: PyReadonlyArray1<'py, f64>,
    natural0: PyReadonlyArray1<'py, f64>,
    ventilation0: PyReadonlyArray1<'py, f64>,
    solar0: PyReadonlyArray1<'py, f64>,
    opaque0: PyReadonlyArray1<'py, f64>,
    month_hours0: Option<PyReadonlyArray1<'py, i64>>,
    coefficients0: (f64, f64, f64, f64),
    aggregate_split0: bool,
    heating_limits_w0: PyReadonlyArray1<'py, f64>,
    cooling_limits_w0: PyReadonlyArray1<'py, f64>,
    temperatures1: PyReadonlyArray1<'py, f64>,
    gains1: PyReadonlyArray1<'py, f64>,
    convective1: PyReadonlyArray1<'py, f64>,
    heating_setpoints1: PyReadonlyArray1<'py, f64>,
    cooling_setpoints1: PyReadonlyArray1<'py, f64>,
    wind1: PyReadonlyArray1<'py, f64>,
    infiltration1: PyReadonlyArray1<'py, f64>,
    natural1: PyReadonlyArray1<'py, f64>,
    ventilation1: PyReadonlyArray1<'py, f64>,
    solar1: PyReadonlyArray1<'py, f64>,
    opaque1: PyReadonlyArray1<'py, f64>,
    month_hours1: Option<PyReadonlyArray1<'py, i64>>,
    coefficients1: (f64, f64, f64, f64),
    aggregate_split1: bool,
    heating_limits_w1: PyReadonlyArray1<'py, f64>,
    cooling_limits_w1: PyReadonlyArray1<'py, f64>,
    conductance_w_k: f64,
) -> PyResult<PyCoupledOutput<'py>> {
    let mut core0 = Core5R1C::from_py(model0)?;
    let mut core1 = Core5R1C::from_py(model1)?;

    let mh0_slice = match &month_hours0 {
        Some(arr) => Some(arr.as_slice()?),
        None => None,
    };
    let mh1_slice = match &month_hours1 {
        Some(arr) => Some(arr.as_slice()?),
        None => None,
    };

    let columns0 = [
        temperatures0.as_slice()?,
        gains0.as_slice()?,
        convective0.as_slice()?,
        heating_setpoints0.as_slice()?,
        cooling_setpoints0.as_slice()?,
        wind0.as_slice()?,
        infiltration0.as_slice()?,
        natural0.as_slice()?,
        ventilation0.as_slice()?,
        solar0.as_slice()?,
        opaque0.as_slice()?,
        heating_limits_w0.as_slice()?,
        cooling_limits_w0.as_slice()?,
    ];
    let borrowed0 = DomainArrays::new(columns0, mh0_slice, coefficients0, aggregate_split0);
    borrowed0
        .validate(borrowed0.temperatures.len())
        .map_err(PyValueError::new_err)?;
    let keepalive0 = immutable_backings(
        [
            temperatures0.as_untyped(),
            gains0.as_untyped(),
            convective0.as_untyped(),
            heating_setpoints0.as_untyped(),
            cooling_setpoints0.as_untyped(),
            wind0.as_untyped(),
            infiltration0.as_untyped(),
            natural0.as_untyped(),
            ventilation0.as_untyped(),
            solar0.as_untyped(),
            opaque0.as_untyped(),
            heating_limits_w0.as_untyped(),
            cooling_limits_w0.as_untyped(),
        ],
        month_hours0.as_ref().map(|array| array.as_untyped()),
    );
    let snapshot0 = if keepalive0.is_some() {
        None
    } else {
        Some(
            DomainSnapshot::new(columns0, mh0_slice, coefficients0, aggregate_split0)
                .map_err(PyValueError::new_err)?,
        )
    };
    let arrays0 = match &snapshot0 {
        Some(snapshot) => snapshot.view(),
        None => borrowed0,
    };

    let columns1 = [
        temperatures1.as_slice()?,
        gains1.as_slice()?,
        convective1.as_slice()?,
        heating_setpoints1.as_slice()?,
        cooling_setpoints1.as_slice()?,
        wind1.as_slice()?,
        infiltration1.as_slice()?,
        natural1.as_slice()?,
        ventilation1.as_slice()?,
        solar1.as_slice()?,
        opaque1.as_slice()?,
        heating_limits_w1.as_slice()?,
        cooling_limits_w1.as_slice()?,
    ];
    let borrowed1 = DomainArrays::new(columns1, mh1_slice, coefficients1, aggregate_split1);
    borrowed1
        .validate(borrowed1.temperatures.len())
        .map_err(PyValueError::new_err)?;
    let keepalive1 = immutable_backings(
        [
            temperatures1.as_untyped(),
            gains1.as_untyped(),
            convective1.as_untyped(),
            heating_setpoints1.as_untyped(),
            cooling_setpoints1.as_untyped(),
            wind1.as_untyped(),
            infiltration1.as_untyped(),
            natural1.as_untyped(),
            ventilation1.as_untyped(),
            solar1.as_untyped(),
            opaque1.as_untyped(),
            heating_limits_w1.as_untyped(),
            cooling_limits_w1.as_untyped(),
        ],
        month_hours1.as_ref().map(|array| array.as_untyped()),
    );
    let snapshot1 = if keepalive1.is_some() {
        None
    } else {
        Some(
            DomainSnapshot::new(columns1, mh1_slice, coefficients1, aggregate_split1)
                .map_err(PyValueError::new_err)?,
        )
    };
    let arrays1 = match &snapshot1 {
        Some(snapshot) => snapshot.view(),
        None => borrowed1,
    };

    let count = arrays0.temperatures.len();
    arrays0.validate(count).map_err(PyValueError::new_err)?;
    arrays1.validate(count).map_err(PyValueError::new_err)?;
    let outcome = py.detach(|| {
        run_coupled_domains(&mut core0, &mut core1, &arrays0, &arrays1, conductance_w_k)
    });
    drop((keepalive0, keepalive1));
    core0.sync_to_py(model0)?;
    core1.sync_to_py(model1)?;
    let output = outcome.map_err(PyValueError::new_err)?;
    Ok((
        PyArray2::from_vec2(py, &output.powers)?,
        PyArray2::from_vec2(py, &output.temperatures)?,
        PyArray2::from_vec2(py, &output.exchanges)?,
        PyArray1::from_vec(py, output.residuals),
    ))
}
