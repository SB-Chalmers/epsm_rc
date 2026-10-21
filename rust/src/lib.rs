use pyo3::prelude::*;

pub mod coupled;
pub mod domain;
pub mod iso13790;

#[pymodule]
fn _core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(iso13790::simulate_5r1c_hourly, m)?)?;
    m.add_function(wrap_pyfunction!(iso13790::calc_heating_demand_5r1c, m)?)?;
    m.add_function(wrap_pyfunction!(domain::run_single_domain_hourly, m)?)?;
    m.add_function(wrap_pyfunction!(coupled::solve_controls_rs, m)?)?;
    m.add_function(wrap_pyfunction!(coupled::run_coupled_domains_hourly, m)?)?;
    Ok(())
}
