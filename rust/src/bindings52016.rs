//! Python binding for the ISO 52016-1 hourly recurrence.
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::iso52016::{run, Airflow52016, Hourly52016, Network52016};

type PySeries<'py> = Bound<'py, PyArray1<f64>>;

fn series<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<PyReadonlyArray1<'py, f64>> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing array {key}")))?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{key} must be a float64 1-D array")))
}

fn owned(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<f64>> {
    Ok(series(dict, key)?.as_slice()?.to_vec())
}

/// Run the hourly ISO 52016-1 zone; `network` and `hourly` hold contiguous arrays.
#[pyfunction]
#[pyo3(signature = (network, hourly, coefficients, volume_m3, air_capacity_j_m3_k,
                    recovery_effectiveness, recovery_bypass, warmup_passes))]
#[allow(clippy::too_many_arguments)]
pub fn run_iso52016_hourly<'py>(
    py: Python<'py>,
    network: &Bound<'py, PyDict>,
    hourly: &Bound<'py, PyDict>,
    coefficients: [f64; 4],
    volume_m3: f64,
    air_capacity_j_m3_k: f64,
    recovery_effectiveness: f64,
    recovery_bypass: bool,
    warmup_passes: usize,
) -> PyResult<(PySeries<'py>, PySeries<'py>, PySeries<'py>)> {
    let c_dt = owned(network, "c_dt")?;
    let n = c_dt.len();
    let m = owned(network, "m")?;
    let mp = owned(network, "mp")?;
    if n == 0 || m.len() != n * n || mp.len() % n != 0 {
        return Err(PyValueError::new_err(
            "Inconsistent ISO 52016-1 network shapes",
        ));
    }
    let k = mp.len() / n;
    let inner: Vec<usize> = owned(network, "inner")?
        .iter()
        .map(|v| *v as usize)
        .collect();
    let net = Network52016 {
        n,
        k,
        m,
        mp,
        m_air: owned(network, "m_air")?,
        m_hvac: owned(network, "m_hvac")?,
        c_dt,
        control: owned(network, "control")?,
        areas: owned(network, "areas")?,
        inner,
    };
    if net.m_air.len() != n
        || net.m_hvac.len() != n
        || net.control.len() != n
        || net.inner.len() != net.areas.len()
        || net.inner.iter().any(|i| *i >= n)
        || k < 5
    {
        return Err(PyValueError::new_err(
            "Inconsistent ISO 52016-1 network vectors",
        ));
    }
    let names = [
        "theta_e",
        "theta_gr",
        "phi_int",
        "phi_sol",
        "set_heating",
        "set_cooling",
        "heat_limit",
        "cool_limit",
        "wind",
        "infiltration",
        "natural",
        "ventilation",
    ];
    let arrays: Vec<PyReadonlyArray1<f64>> = names
        .iter()
        .map(|name| series(hourly, name))
        .collect::<PyResult<_>>()?;
    let absorbed = series(hourly, "absorbed")?;
    let slices: Vec<&[f64]> = arrays
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<_, _>>()?;
    let count = slices[0].len();
    if count == 0 || slices.iter().any(|s| s.len() != count) {
        return Err(PyValueError::new_err(
            "Hourly ISO 52016-1 inputs must share one length",
        ));
    }
    let absorbed = absorbed.as_slice()?;
    if absorbed.len() != count * (k - 5) {
        return Err(PyValueError::new_err(
            "Absorbed solar must be hours x elements",
        ));
    }
    let hours = Hourly52016 {
        theta_e: slices[0],
        theta_gr: slices[1],
        phi_int: slices[2],
        phi_sol: slices[3],
        absorbed,
        set_heating: slices[4],
        set_cooling: slices[5],
        heat_limit: slices[6],
        cool_limit: slices[7],
        wind: slices[8],
        infiltration: slices[9],
        natural: slices[10],
        ventilation: slices[11],
    };
    let airflow = Airflow52016 {
        coefficients,
        volume_m3,
        air_capacity_j_m3_k,
        recovery_effectiveness,
        recovery_bypass,
    };
    let output = py
        .detach(|| run(&net, &hours, &airflow, warmup_passes))
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, output.power),
        PyArray1::from_vec(py, output.air),
        PyArray1::from_vec(py, output.operative),
    ))
}
