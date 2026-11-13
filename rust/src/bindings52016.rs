// File: bindings52016.rs
// Purpose: Python binding for the ISO 52016-1 hourly multi-zone recurrence.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sanjay Somanath
// License: MIT; see the project-root LICENSE for the full permission and warranty terms.
// Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
// Project PI: Alexander Hollberg
// Computational Sustainable Design Group
// Research Area of Sustainable Built Environments
// Chalmers University of Technology, Gothenburg
// Date created: 2026-11-12

//! Python binding for the ISO 52016-1 hourly multi-zone recurrence.
//!
//! Arrays arrive as contiguous float64 vectors from
//! `epsm_rc._iso52016_platform._run_native`; shapes are checked here so the
//! kernel can index without bounds surprises.
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::iso52016::{run, Airflow, Network52016, ZoneHourly};

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

fn value<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing value {key}")))
}

fn scalar(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    value(dict, key)?.extract::<f64>()
}

/// Owned per-zone arrays, so the kernel can run without the GIL.
struct ZoneArrays {
    thermostat: [Vec<f64>; 4],
    fixed: Option<Vec<f64>>,
    flows: [Vec<f64>; 4],
    coefficients: [f64; 4],
    volume_m3: f64,
    air_capacity_j_m3_k: f64,
    recovery_effectiveness: f64,
    recovery_bypass: bool,
}

fn zone_arrays(zone: &Bound<'_, PyDict>, count: usize) -> PyResult<ZoneArrays> {
    let thermostat = [
        owned(zone, "set_heating")?,
        owned(zone, "set_cooling")?,
        owned(zone, "heat_limit")?,
        owned(zone, "cool_limit")?,
    ];
    let mut lengths: Vec<usize> = thermostat.iter().map(Vec::len).collect();
    let arrays = if zone.contains("h_ve")? {
        let fixed = owned(zone, "h_ve")?;
        lengths.push(fixed.len());
        ZoneArrays {
            thermostat,
            fixed: Some(fixed),
            flows: Default::default(),
            coefficients: [0.0; 4],
            volume_m3: 0.0,
            air_capacity_j_m3_k: 0.0,
            recovery_effectiveness: 0.0,
            recovery_bypass: false,
        }
    } else {
        let flows = [
            owned(zone, "wind")?,
            owned(zone, "infiltration")?,
            owned(zone, "natural")?,
            owned(zone, "ventilation")?,
        ];
        lengths.extend(flows.iter().map(Vec::len));
        let coefficients = owned(zone, "coefficients")?;
        if coefficients.len() != 4 {
            return Err(PyValueError::new_err(
                "Four infiltration coefficients are required",
            ));
        }
        ZoneArrays {
            thermostat,
            fixed: None,
            flows,
            coefficients: [
                coefficients[0],
                coefficients[1],
                coefficients[2],
                coefficients[3],
            ],
            volume_m3: scalar(zone, "volume_m3")?,
            air_capacity_j_m3_k: scalar(zone, "air_capacity_j_m3_k")?,
            recovery_effectiveness: scalar(zone, "recovery_effectiveness")?,
            recovery_bypass: value(zone, "recovery_bypass")?.extract::<bool>()?,
        }
    };
    if lengths.iter().any(|&length| length != count) {
        return Err(PyValueError::new_err(
            "Hourly ISO 52016-1 zone inputs must match the forcing hours",
        ));
    }
    Ok(arrays)
}

/// Run the hourly ISO 52016-1 network.
///
/// `network` holds the row-major matrices of `Network52016._build`, the
/// hours × columns `forcing` and flattened `links` triples; `zones` holds one
/// dict per zone (thermostat arrays plus either `h_ve` or airflow arrays).
/// Returns power, air and operative (hours × zones, flattened) and the probe.
#[pyfunction]
#[pyo3(signature = (network, zones, warmup_passes))]
#[allow(clippy::type_complexity)]
pub fn run_iso52016_network<'py>(
    py: Python<'py>,
    network: &Bound<'py, PyDict>,
    zones: &Bound<'py, PyList>,
    warmup_passes: usize,
) -> PyResult<(PySeries<'py>, PySeries<'py>, PySeries<'py>, PySeries<'py>)> {
    let c_dt = owned(network, "c_dt")?;
    let n = c_dt.len();
    let z = zones.len();
    let m = owned(network, "m")?;
    let mp = owned(network, "mp")?;
    if n == 0 || z == 0 || z > n || m.len() != n * n || mp.len() % n != 0 {
        return Err(PyValueError::new_err(
            "Inconsistent ISO 52016-1 network shapes",
        ));
    }
    let k = mp.len() / n;
    let net = Network52016 {
        n,
        z,
        k,
        m,
        mp,
        m_air: owned(network, "m_air")?,
        m_hvac: owned(network, "m_hvac")?,
        c_dt,
        control: owned(network, "control")?,
        operative: owned(network, "operative")?,
    };
    if net.m_air.len() != n * z
        || net.m_hvac.len() != n * z
        || net.control.len() != z * n
        || net.operative.len() != z * n
        || k < 2 + 3 * z
    {
        return Err(PyValueError::new_err(
            "Inconsistent ISO 52016-1 network vectors",
        ));
    }
    let forcing = owned(network, "forcing")?;
    if forcing.is_empty() || forcing.len() % k != 0 {
        return Err(PyValueError::new_err("Forcing must be hours x columns"));
    }
    let count = forcing.len() / k;
    let flat_links = owned(network, "links")?;
    if flat_links.len() % 3 != 0 {
        return Err(PyValueError::new_err(
            "Links must be (node, node, conductance) triples",
        ));
    }
    let links: Vec<(usize, usize, f64)> = flat_links
        .chunks(3)
        .map(|link| (link[0] as usize, link[1] as usize, link[2]))
        .collect();
    if links.iter().any(|&(a, b, _)| a >= n || b >= n) {
        return Err(PyValueError::new_err("Link nodes out of range"));
    }
    let arrays: Vec<ZoneArrays> = zones
        .iter()
        .map(|zone| zone_arrays(zone.cast::<PyDict>()?, count))
        .collect::<PyResult<_>>()?;
    let output = py
        .detach(|| {
            let hourly: Vec<ZoneHourly> = arrays
                .iter()
                .map(|zone| ZoneHourly {
                    set_heating: &zone.thermostat[0],
                    set_cooling: &zone.thermostat[1],
                    heat_limit: &zone.thermostat[2],
                    cool_limit: &zone.thermostat[3],
                    airflow: match &zone.fixed {
                        Some(series) => Airflow::Fixed(series),
                        None => Airflow::Forcing {
                            wind: &zone.flows[0],
                            infiltration: &zone.flows[1],
                            natural: &zone.flows[2],
                            ventilation: &zone.flows[3],
                            coefficients: zone.coefficients,
                            volume_m3: zone.volume_m3,
                            air_capacity_j_m3_k: zone.air_capacity_j_m3_k,
                            recovery_effectiveness: zone.recovery_effectiveness,
                            recovery_bypass: zone.recovery_bypass,
                        },
                    },
                })
                .collect();
            run(&net, &forcing, &hourly, &links, warmup_passes)
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, output.power),
        PyArray1::from_vec(py, output.air),
        PyArray1::from_vec(py, output.operative),
        PyArray1::from_vec(py, output.probe),
    ))
}
