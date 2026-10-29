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

/// Keys `Building5R1C._native_parameters()` always supplies, named as `Core5R1C` fields.
pub const BASE_PARAMETERS: [&str; 36] = [
    "floor_area_m2",
    "building_vol",
    "a_tot",
    "mass_area",
    "c_m",
    "h_tr_op",
    "h_tr_op_walls_roof",
    "h_tr_ground",
    "h_tr_tb",
    "h_tr_w",
    "h_tr_ms",
    "h_tr_is",
    "h_tr_em",
    "h_ms",
    "air_volumetric_heat_capacity_j_m3_k",
    "ach_infiltration",
    "ach_ventilation",
    "heat_recovery_efficiency",
    "convective_fraction_internal",
    "internal_gains_w_per_m2",
    "t_set_heating",
    "t_set_cooling",
    "t_set_heating_setback",
    "t_set_cooling_setback",
    "use_operative_temperature",
    "ventilation_behaviour",
    "ach_window_open",
    "delta_theta_er",
    "ground_temps_monthly",
    "prepared_hourly",
    "t_m",
    "t_s",
    "t_air",
    "t_supply",
    "windows_open",
    "h_ve",
];
/// Cached coefficients supplied only by prepared hourly models.
pub const PREPARED_PARAMETERS: [&str; 6] = [
    "prepared_sky",
    "prepared_mass_ratio",
    "prepared_window_ratio",
    "prepared_airflow",
    "prepared_network",
    "prepared_cn",
];
/// Sky-loss components supplied only by standalone models.
pub const STANDALONE_PARAMETERS: [&str; 9] = [
    "r_se_roof",
    "u_roof",
    "roof_area_m2",
    "h_r",
    "form_factor_roof",
    "r_se_wall",
    "u_walls",
    "wall_area_m2",
    "form_factor_wall",
];
/// `domains.HourlyInputs` columns, in `DomainArrays::new` order before the limits.
pub const FORCING_COLUMNS: [&str; 11] = [
    "temperatures",
    "gains",
    "convective",
    "heating_setpoints",
    "cooling_setpoints",
    "wind",
    "infiltration",
    "natural",
    "ventilation",
    "solar",
    "opaque",
];

/// The model's named native parameters, checked against the expected key set.
///
/// `from_py` reads every expected key, so an equal key count plus successful
/// reads proves an exact match; the full diagnosis is built only on failure.
struct Parameters<'py> {
    dict: Bound<'py, PyDict>,
    prepared: bool,
}

impl<'py> Parameters<'py> {
    fn read(model: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dict = model
            .call_method0("_native_parameters")?
            .cast_into::<PyDict>()?;
        let prepared = match dict.get_item("prepared_hourly")? {
            Some(value) => value.extract()?,
            None => false,
        };
        let parameters = Self { dict, prepared };
        if parameters.dict.len() != parameters.expected().count() {
            return Err(parameters.mismatch());
        }
        Ok(parameters)
    }

    fn expected(&self) -> impl Iterator<Item = &'static str> {
        let mode: &'static [&'static str] = if self.prepared {
            &PREPARED_PARAMETERS
        } else {
            &STANDALONE_PARAMETERS
        };
        BASE_PARAMETERS.iter().chain(mode).copied()
    }

    fn mismatch(&self) -> PyErr {
        let missing: Vec<&str> = self
            .expected()
            .filter(|key| !self.dict.contains(*key).unwrap_or(false))
            .collect();
        let unexpected: Vec<String> = self
            .dict
            .keys()
            .iter()
            .filter_map(|key| key.extract::<String>().ok())
            .filter(|key| !self.expected().any(|name| name == key))
            .collect();
        PyValueError::new_err(format!(
            "Native parameter contract mismatch: missing {missing:?}, unexpected {unexpected:?}"
        ))
    }

    fn get(&self, key: &str) -> PyResult<Bound<'py, PyAny>> {
        self.dict.get_item(key)?.ok_or_else(|| self.mismatch())
    }
}

impl Core5R1C {
    pub fn from_py(model: &Bound<'_, PyAny>) -> PyResult<Self> {
        let p = Parameters::read(model)?;
        let prepared_hourly: bool = p.get("prepared_hourly")?.extract()?;
        let delta_theta_er: f64 = p.get("delta_theta_er")?.extract()?;

        let sky_radiation_loss_w: f64 = if prepared_hourly {
            p.get("prepared_sky")?.extract()?
        } else {
            let r_se_roof: f64 = p.get("r_se_roof")?.extract()?;
            let u_roof: f64 = p.get("u_roof")?.extract()?;
            let roof_area_m2: f64 = p.get("roof_area_m2")?.extract()?;
            let h_r: f64 = p.get("h_r")?.extract()?;
            let form_factor_roof: f64 = p.get("form_factor_roof")?.extract()?;
            let r_se_wall: f64 = p.get("r_se_wall")?.extract()?;
            let u_walls: f64 = p.get("u_walls")?.extract()?;
            let wall_area_m2: f64 = p.get("wall_area_m2")?.extract()?;
            let form_factor_wall: f64 = p.get("form_factor_wall")?.extract()?;
            let phi_r_roof =
                r_se_roof * u_roof * roof_area_m2 * h_r * delta_theta_er * form_factor_roof;
            let phi_r_wall =
                r_se_wall * u_walls * wall_area_m2 * h_r * delta_theta_er * form_factor_wall;
            phi_r_roof + phi_r_wall
        };

        let ground_vec: Option<Vec<f64>> = p.get("ground_temps_monthly")?.extract()?;
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
            (
                p.get("prepared_mass_ratio")?.extract()?,
                p.get("prepared_window_ratio")?.extract()?,
                p.get("prepared_airflow")?.extract()?,
                p.get("prepared_network")?.extract()?,
                p.get("prepared_cn")?.extract()?,
            )
        } else {
            (0.0, 0.0, None, (0.0, 0.0, 0.0), (0.0, 0.0))
        };

        let mut core = Self {
            floor_area_m2: p.get("floor_area_m2")?.extract()?,
            building_vol: p.get("building_vol")?.extract()?,
            a_tot: p.get("a_tot")?.extract()?,
            mass_area: p.get("mass_area")?.extract()?,
            c_m: p.get("c_m")?.extract()?,
            h_tr_op: p.get("h_tr_op")?.extract()?,
            h_tr_op_walls_roof: p.get("h_tr_op_walls_roof")?.extract()?,
            h_tr_ground: p.get("h_tr_ground")?.extract()?,
            h_tr_tb: p.get("h_tr_tb")?.extract()?,
            h_tr_w: p.get("h_tr_w")?.extract()?,
            h_tr_ms: p.get("h_tr_ms")?.extract()?,
            h_tr_is: p.get("h_tr_is")?.extract()?,
            h_tr_em: p.get("h_tr_em")?.extract()?,
            h_ms: p.get("h_ms")?.extract()?,
            air_volumetric_heat_capacity_j_m3_k: p
                .get("air_volumetric_heat_capacity_j_m3_k")?
                .extract()?,
            ach_infiltration: p.get("ach_infiltration")?.extract()?,
            ach_ventilation: p.get("ach_ventilation")?.extract()?,
            heat_recovery_efficiency: p.get("heat_recovery_efficiency")?.extract()?,
            convective_fraction_internal: p.get("convective_fraction_internal")?.extract()?,
            internal_gains_w_per_m2: p.get("internal_gains_w_per_m2")?.extract()?,
            t_set_heating: p.get("t_set_heating")?.extract()?,
            t_set_cooling: p.get("t_set_cooling")?.extract()?,
            t_set_heating_setback: p.get("t_set_heating_setback")?.extract()?,
            t_set_cooling_setback: p.get("t_set_cooling_setback")?.extract()?,
            use_operative_temperature: p.get("use_operative_temperature")?.extract()?,
            ventilation_behaviour: p.get("ventilation_behaviour")?.extract()?,
            ach_window_open: p.get("ach_window_open")?.extract()?,
            delta_theta_er,
            sky_radiation_loss_w,
            ground_temps_monthly,
            ground_temp_annual_mean,
            prepared_hourly,
            prepared_mass_ratio,
            prepared_window_ratio,
            state: ThermalState {
                t_m: p.get("t_m")?.extract()?,
                t_s: p.get("t_s")?.extract()?,
                t_air: p.get("t_air")?.extract()?,
            },
            t_supply: p.get("t_supply")?.extract()?,
            windows_open: p.get("windows_open")?.extract()?,
            window_decision: None,
            h_ve: p.get("h_ve")?.extract()?,
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

    /// Return advanced state through `Building5R1C._apply_native_state`.
    pub fn sync_to_py(&self, model: &Bound<'_, PyAny>) -> PyResult<()> {
        let state = PyDict::new(model.py());
        state.set_item("t_m", self.state.t_m)?;
        state.set_item("t_s", self.state.t_s)?;
        state.set_item("t_air", self.state.t_air)?;
        state.set_item("t_supply", self.t_supply)?;
        state.set_item("windows_open", self.windows_open)?;
        state.set_item("t_set_heating", self.t_set_heating)?;
        state.set_item("t_set_cooling", self.t_set_cooling)?;
        state.set_item("ach_infiltration", self.ach_infiltration)?;
        state.set_item("ach_ventilation", self.ach_ventilation)?;
        state.set_item(
            "convective_fraction_internal",
            self.convective_fraction_internal,
        )?;
        state.set_item("h_ve", self.h_ve)?;
        if self.prepared_hourly {
            state.set_item("prepared_airflow", self.prepared_airflow)?;
            state.set_item("prepared_network", self.prepared_network)?;
            state.set_item("prepared_cn", self.prepared_cn)?;
            state.set_item("hour_heat_flows", self.hour_heat_flows)?;
        }
        model.call_method1("_apply_native_state", (state,))?;
        Ok(())
    }
}

/// Hourly forcing read by attribute name from a `domains.HourlyInputs`.
struct PyForcing<'py> {
    columns: Vec<PyReadonlyArray1<'py, f64>>,
    month_hours: Option<PyReadonlyArray1<'py, i64>>,
    coefficients: (f64, f64, f64, f64),
    aggregate_split: bool,
}

impl<'py> PyForcing<'py> {
    fn read(forcing: &Bound<'py, PyAny>) -> PyResult<Self> {
        let named =
            |name: &str, error: PyErr| PyValueError::new_err(format!("forcing.{name}: {error}"));
        let columns = FORCING_COLUMNS
            .iter()
            .map(|name| {
                forcing
                    .getattr(*name)
                    .and_then(|value| value.extract().map_err(PyErr::from))
                    .map_err(|error| named(name, error))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            columns,
            month_hours: forcing
                .getattr("month_hours")
                .and_then(|value| value.extract().map_err(PyErr::from))
                .map_err(|error| named("month_hours", error))?,
            coefficients: forcing
                .getattr("coefficients")
                .and_then(|value| value.extract())
                .map_err(|error| named("coefficients", error))?,
            aggregate_split: forcing
                .getattr("aggregate_split")
                .and_then(|value| value.extract())
                .map_err(|error| named("aggregate_split", error))?,
        })
    }
}

/// Validate one domain's forcing and limits, borrowing immutable storage or
/// snapshotting it, then run `kernel` while that storage is kept alive.
fn with_domain_arrays<'py, R>(
    forcing: &PyForcing<'py>,
    heating_limits_w: &PyReadonlyArray1<'py, f64>,
    cooling_limits_w: &PyReadonlyArray1<'py, f64>,
    kernel: impl FnOnce(&DomainArrays<'_>) -> R,
) -> PyResult<R> {
    let month_hours = forcing
        .month_hours
        .as_ref()
        .map(|array| array.as_slice())
        .transpose()?;
    let mut columns: [&[f64]; 13] = [&[]; 13];
    for (column, array) in columns.iter_mut().zip(&forcing.columns) {
        *column = array.as_slice()?;
    }
    columns[11] = heating_limits_w.as_slice()?;
    columns[12] = cooling_limits_w.as_slice()?;
    let borrowed = DomainArrays::new(
        columns,
        month_hours,
        forcing.coefficients,
        forcing.aggregate_split,
    );
    borrowed
        .validate(borrowed.temperatures.len())
        .map_err(PyValueError::new_err)?;
    let keepalive = immutable_backings(
        std::array::from_fn(|index| match index {
            11 => heating_limits_w.as_untyped(),
            12 => cooling_limits_w.as_untyped(),
            _ => forcing.columns[index].as_untyped(),
        }),
        forcing.month_hours.as_ref().map(|array| array.as_untyped()),
    );
    let snapshot = if keepalive.is_some() {
        None
    } else {
        Some(
            DomainSnapshot::new(
                columns,
                month_hours,
                forcing.coefficients,
                forcing.aggregate_split,
            )
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
    let result = kernel(&arrays);
    drop(keepalive);
    Ok(result)
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
#[pyo3(signature = (model, forcing, heating_limits_w, cooling_limits_w))]
pub fn run_single_domain_hourly<'py>(
    py: Python<'py>,
    model: &Bound<'py, PyAny>,
    forcing: &Bound<'py, PyAny>,
    heating_limits_w: PyReadonlyArray1<'py, f64>,
    cooling_limits_w: PyReadonlyArray1<'py, f64>,
) -> PyResult<(PySeries<'py>, PySeries<'py>)> {
    let mut core = Core5R1C::from_py(model)?;
    let forcing = PyForcing::read(forcing)?;
    let outcome = with_domain_arrays(&forcing, &heating_limits_w, &cooling_limits_w, |arrays| {
        py.detach(|| run_single_domain(&mut core, arrays))
    })?;
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
    forcing0,
    forcing1,
    heating_limits_w0,
    cooling_limits_w0,
    heating_limits_w1,
    cooling_limits_w1,
    conductance_w_k
))]
#[allow(clippy::too_many_arguments)]
pub fn run_coupled_domains_hourly<'py>(
    py: Python<'py>,
    model0: &Bound<'py, PyAny>,
    model1: &Bound<'py, PyAny>,
    forcing0: &Bound<'py, PyAny>,
    forcing1: &Bound<'py, PyAny>,
    heating_limits_w0: PyReadonlyArray1<'py, f64>,
    cooling_limits_w0: PyReadonlyArray1<'py, f64>,
    heating_limits_w1: PyReadonlyArray1<'py, f64>,
    cooling_limits_w1: PyReadonlyArray1<'py, f64>,
    conductance_w_k: f64,
) -> PyResult<PyCoupledOutput<'py>> {
    let mut core0 = Core5R1C::from_py(model0)?;
    let mut core1 = Core5R1C::from_py(model1)?;
    let forcing0 = PyForcing::read(forcing0)?;
    let forcing1 = PyForcing::read(forcing1)?;
    let outcome = with_domain_arrays(&forcing0, &heating_limits_w0, &cooling_limits_w0, |a0| {
        with_domain_arrays(&forcing1, &heating_limits_w1, &cooling_limits_w1, |a1| {
            let count = a0.temperatures.len();
            a0.validate(count)?;
            a1.validate(count)?;
            py.detach(|| run_coupled_domains(&mut core0, &mut core1, a0, a1, conductance_w_k))
        })
    })??;
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
