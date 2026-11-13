// File: lib.rs
// Purpose: Python extension module registering the native RC kernels.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Sanjay Somanath
// License: MIT; see the project-root LICENSE for the full permission and warranty terms.
// Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
// Project PI: Alexander Hollberg
// Computational Sustainable Design Group
// Research Area of Sustainable Built Environments
// Chalmers University of Technology, Gothenburg
// Date created: 2026-10-29

//! Native kernels of `epsm_rc`, built as the Python extension `epsm_rc._core_rs`.
//!
//! - `iso13790`: the ISO 13790 5R1C zone and thermostat.
//! - `domain` and `coupled`: the platform hourly loops for one domain and for
//!   a main zone coupled to a conditioned basement.
//! - `iso52016`: the ISO 52016-1 multi-zone, multi-node recurrence.
//! - `self_shading`: horizon profiles and hourly sunlit fractions of a
//!   building's own geometry.
//! - `bindings` and `bindings52016`: NumPy and model conversion at the boundary.
//!
//! Every kernel mirrors a Python implementation and is tested against it; the
//! Python path remains the reference and runs when the extension is absent.

use pyo3::prelude::*;

mod bindings;
mod bindings52016;
mod coupled;
mod domain;
mod iso13790;
mod iso52016;
mod self_shading;
mod types;

/// Register the kernels and the parameter-name contracts with Python.
#[pymodule]
fn _core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bindings::simulate_5r1c_hourly, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::calc_heating_demand_5r1c, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::run_single_domain_hourly, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::solve_controls_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::run_coupled_domains_hourly, m)?)?;
    m.add_function(wrap_pyfunction!(bindings52016::run_iso52016_network, m)?)?;
    m.add_function(wrap_pyfunction!(
        self_shading::self_shading_horizon_profiles,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        self_shading::self_shading_sunlit_fraction,
        m
    )?)?;
    m.add("NATIVE_BASE_PARAMETERS", bindings::BASE_PARAMETERS.to_vec())?;
    m.add(
        "NATIVE_PREPARED_PARAMETERS",
        bindings::PREPARED_PARAMETERS.to_vec(),
    )?;
    m.add(
        "NATIVE_STANDALONE_PARAMETERS",
        bindings::STANDALONE_PARAMETERS.to_vec(),
    )?;
    m.add("NATIVE_FORCING_COLUMNS", bindings::FORCING_COLUMNS.to_vec())?;
    Ok(())
}
