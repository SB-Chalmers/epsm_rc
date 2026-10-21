use pyo3::prelude::*;

mod bindings;
mod coupled;
mod domain;
mod iso13790;
mod types;

#[pymodule]
fn _core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bindings::simulate_5r1c_hourly, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::calc_heating_demand_5r1c, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::run_single_domain_hourly, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::solve_controls_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::run_coupled_domains_hourly, m)?)?;
    Ok(())
}
