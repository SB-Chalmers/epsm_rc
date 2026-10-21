use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::domain::DomainArrays;
use crate::iso13790::Core5R1C;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ControlMode {
    Free,
    Heat,
    HeatLimit,
    Cool,
    CoolLimit,
}

#[inline(always)]
fn solve_2x2(mut a: [[f64; 2]; 2], mut b: [f64; 2]) -> [f64; 2] {
    if a[1][0].abs() > a[0][0].abs() {
        a.swap(0, 1);
        b.swap(0, 1);
    }
    let l21 = a[1][0] / a[0][0];
    let u11 = a[1][1] - l21 * a[0][1];
    let y0 = b[0];
    let y1 = b[1] - y0 * l21;
    let x1 = y1 / u11;
    let x0 = (y0 - x1 * a[0][1]) / a[0][0];
    [x0, x1]
}

#[inline(always)]
pub fn solve_controls_inner(
    free: [f64; 2],
    response: [f64; 2],
    conductance_w_k: f64,
    heat: [f64; 2],
    cool: [f64; 2],
    maximum: [f64; 2],
    minimum: [f64; 2],
) -> Result<([f64; 2], [f64; 2], [f64; 2], f64), &'static str> {
    if !free[0].is_finite()
        || !free[1].is_finite()
        || !response[0].is_finite()
        || !response[1].is_finite()
        || response[0] <= 0.0
        || response[1] <= 0.0
    {
        return Err("Two finite free temperatures and positive responses are required");
    }
    if !conductance_w_k.is_finite() || conductance_w_k < 0.0 {
        return Err("Inter-domain conductance must be finite and nonnegative");
    }
    if !heat[0].is_finite()
        || !heat[1].is_finite()
        || !cool[0].is_finite()
        || !cool[1].is_finite()
        || heat[0] > cool[0]
        || heat[1] > cool[1]
    {
        return Err("Finite ordered heating and cooling setpoints are required");
    }
    if maximum[0].is_nan()
        || maximum[1].is_nan()
        || minimum[0].is_nan()
        || minimum[1].is_nan()
        || maximum[0] < 0.0
        || maximum[1] < 0.0
        || minimum[0] > 0.0
        || minimum[1] > 0.0
    {
        return Err("Heating limits must be nonnegative; cooling limits nonpositive");
    }

    let h = conductance_w_k;
    let matrix = [
        [1.0 + response[0] * h, -response[0] * h],
        [-response[1] * h, 1.0 + response[1] * h],
    ];

    let mut choices = [[ControlMode::Free; 5]; 2];
    let mut lens = [1usize; 2];
    for index in 0..2 {
        let mut len = 1;
        if maximum[index] > 0.0 {
            choices[index][len] = ControlMode::Heat;
            len += 1;
            if maximum[index].is_finite() {
                choices[index][len] = ControlMode::HeatLimit;
                len += 1;
            }
        }
        if minimum[index] < 0.0 {
            choices[index][len] = ControlMode::Cool;
            len += 1;
            if minimum[index].is_finite() {
                choices[index][len] = ControlMode::CoolLimit;
                len += 1;
            }
        }
        lens[index] = len;
    }

    let tolerance = 1e-7;
    for i0 in 0..lens[0] {
        for i1 in 0..lens[1] {
            let modes = [choices[0][i0], choices[1][i1]];
            let mut equations = matrix;
            let mut rhs = free;

            for index in 0..2 {
                match modes[index] {
                    ControlMode::Heat => {
                        equations[index] = [0.0, 0.0];
                        equations[index][index] = 1.0;
                        rhs[index] = heat[index];
                    }
                    ControlMode::Cool => {
                        equations[index] = [0.0, 0.0];
                        equations[index][index] = 1.0;
                        rhs[index] = cool[index];
                    }
                    ControlMode::HeatLimit => {
                        rhs[index] += response[index] * maximum[index];
                    }
                    ControlMode::CoolLimit => {
                        rhs[index] += response[index] * minimum[index];
                    }
                    ControlMode::Free => {}
                }
            }

            let temperatures = solve_2x2(equations, rhs);
            let mat_t = [
                matrix[0][0] * temperatures[0] + matrix[0][1] * temperatures[1],
                matrix[1][0] * temperatures[0] + matrix[1][1] * temperatures[1],
            ];
            let mut powers = [
                (mat_t[0] - free[0]) / response[0],
                (mat_t[1] - free[1]) / response[1],
            ];

            let mut valid = true;
            for index in 0..2 {
                let t = temperatures[index];
                let q = powers[index];
                let mode_ok = match modes[index] {
                    ControlMode::Free => {
                        (maximum[index] == 0.0 || t >= heat[index] - tolerance)
                            && (minimum[index] == 0.0 || t <= cool[index] + tolerance)
                    }
                    ControlMode::Heat => -tolerance <= q && q <= maximum[index] + tolerance,
                    ControlMode::Cool => minimum[index] - tolerance <= q && q <= tolerance,
                    ControlMode::HeatLimit => t <= heat[index] + tolerance,
                    ControlMode::CoolLimit => t >= cool[index] - tolerance,
                };
                valid = valid && mode_ok;
            }

            if valid {
                for index in 0..2 {
                    match modes[index] {
                        ControlMode::Free => powers[index] = 0.0,
                        ControlMode::HeatLimit => powers[index] = maximum[index],
                        ControlMode::CoolLimit => powers[index] = minimum[index],
                        ControlMode::Heat | ControlMode::Cool => {}
                    }
                    if powers[index] < minimum[index] {
                        powers[index] = minimum[index];
                    }
                    if powers[index] > maximum[index] {
                        powers[index] = maximum[index];
                    }
                }
                let exchange = h * (temperatures[1] - temperatures[0]);
                let res0 = ((mat_t[0] - free[0] - response[0] * powers[0]) / response[0]).abs();
                let res1 = ((mat_t[1] - free[1] - response[1] * powers[1]) / response[1]).abs();
                let max_res = if res0 >= res1 { res0 } else { res1 };
                return Ok((powers, temperatures, [exchange, -exchange], max_res));
            }
        }
    }

    Err("No admissible coupled heating/cooling control state")
}

#[inline(always)]
pub fn step_coupled_pair(
    core0: &mut Core5R1C,
    core1: &mut Core5R1C,
    in0: (f64, f64, f64, f64, Option<i64>, f64, bool),
    in1: (f64, f64, f64, f64, Option<i64>, f64, bool),
    conductance_w_k: f64,
    heating_limit_w: [f64; 2],
    cooling_limit_w: [f64; 2],
) -> Result<([f64; 2], [f64; 2], [f64; 2], f64), &'static str> {
    if core0.ventilation_behaviour <= 0.0 && core1.ventilation_behaviour <= 0.0 {
        core0.hour_heat_flows = None;
        core0.step_deterministic(in0.0, in0.1, in0.2, 0.0, in0.5, in0.3, in0.6, in0.4)?;
        let free0 = (core0.t_air, core0.t_s, core0.t_m);
        let (k_m0, k_s0, resp0) = core0.sensitivities(in0.5);

        core1.hour_heat_flows = None;
        core1.step_deterministic(in1.0, in1.1, in1.2, 0.0, in1.5, in1.3, in1.6, in1.4)?;
        let free1 = (core1.t_air, core1.t_s, core1.t_m);
        let (k_m1, k_s1, resp1) = core1.sensitivities(in1.5);

        let (powers, temperatures, exchanges, residual) = solve_controls_inner(
            [free0.0, free1.0],
            [resp0, resp1],
            conductance_w_k,
            [core0.t_set_heating, core1.t_set_heating],
            [core0.t_set_cooling, core1.t_set_cooling],
            heating_limit_w,
            cooling_limit_w,
        )?;

        let q0 = powers[0] + exchanges[0];
        core0.t_air = free0.0 + resp0 * q0;
        core0.t_s = free0.1 + k_s0 * q0;
        core0.t_m = free0.2 + k_m0 * q0;
        if !(core0.t_air.is_finite() && temperatures[0].is_finite())
            || (core0.t_air - temperatures[0]).abs() > 1e-7
        {
            return Err("Coupled control failed its committed-state temperature balance");
        }

        let q1 = powers[1] + exchanges[1];
        core1.t_air = free1.0 + resp1 * q1;
        core1.t_s = free1.1 + k_s1 * q1;
        core1.t_m = free1.2 + k_m1 * q1;
        if !(core1.t_air.is_finite() && temperatures[1].is_finite())
            || (core1.t_air - temperatures[1]).abs() > 1e-7
        {
            return Err("Coupled control failed its committed-state temperature balance");
        }

        return Ok((powers, temperatures, exchanges, residual));
    }

    let state0 = (core0.t_air, core0.t_s, core0.t_m);
    let state1 = (core1.t_air, core1.t_s, core1.t_m);

    // Domain 0 trial steps
    core0.hour_heat_flows = None;
    core0.step_deterministic(in0.0, in0.1, in0.2, 0.0, in0.5, in0.3, in0.6, in0.4)?;
    let base0 = core0.t_air;
    core0.t_air = state0.0;
    core0.t_s = state0.1;
    core0.t_m = state0.2;
    let ref0 = 10.0 * core0.floor_area_m2;
    core0.step_deterministic(in0.0, in0.1, in0.2, ref0, in0.5, in0.3, in0.6, in0.4)?;
    let resp0 = (core0.t_air - base0) / ref0;
    core0.t_air = state0.0;
    core0.t_s = state0.1;
    core0.t_m = state0.2;

    // Domain 1 trial steps
    core1.hour_heat_flows = None;
    core1.step_deterministic(in1.0, in1.1, in1.2, 0.0, in1.5, in1.3, in1.6, in1.4)?;
    let base1 = core1.t_air;
    core1.t_air = state1.0;
    core1.t_s = state1.1;
    core1.t_m = state1.2;
    let ref1 = 10.0 * core1.floor_area_m2;
    core1.step_deterministic(in1.0, in1.1, in1.2, ref1, in1.5, in1.3, in1.6, in1.4)?;
    let resp1 = (core1.t_air - base1) / ref1;
    core1.t_air = state1.0;
    core1.t_s = state1.1;
    core1.t_m = state1.2;

    let (powers, temperatures, exchanges, residual) = solve_controls_inner(
        [base0, base1],
        [resp0, resp1],
        conductance_w_k,
        [core0.t_set_heating, core1.t_set_heating],
        [core0.t_set_cooling, core1.t_set_cooling],
        heating_limit_w,
        cooling_limit_w,
    )?;

    // Commit step for both domains
    core0.step_deterministic(
        in0.0,
        in0.1,
        in0.2,
        powers[0] + exchanges[0],
        in0.5,
        in0.3,
        in0.6,
        in0.4,
    )?;
    if !(core0.t_air.is_finite() && temperatures[0].is_finite())
        || (core0.t_air - temperatures[0]).abs() > 1e-7
    {
        return Err("Coupled control failed its committed-state temperature balance");
    }

    core1.step_deterministic(
        in1.0,
        in1.1,
        in1.2,
        powers[1] + exchanges[1],
        in1.5,
        in1.3,
        in1.6,
        in1.4,
    )?;
    if !(core1.t_air.is_finite() && temperatures[1].is_finite())
        || (core1.t_air - temperatures[1]).abs() > 1e-7
    {
        return Err("Coupled control failed its committed-state temperature balance");
    }

    Ok((powers, temperatures, exchanges, residual))
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
) -> PyResult<((f64, f64), (f64, f64), (f64, f64), f64)> {
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
    let (powers, temps, exch, res) = solve_controls_inner(
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
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
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

    let arrays0 = DomainArrays {
        temperatures: temperatures0.as_slice()?,
        gains: gains0.as_slice()?,
        convective: convective0.as_slice()?,
        heating_setpoints: heating_setpoints0.as_slice()?,
        cooling_setpoints: cooling_setpoints0.as_slice()?,
        wind: wind0.as_slice()?,
        infiltration: infiltration0.as_slice()?,
        natural: natural0.as_slice()?,
        ventilation: ventilation0.as_slice()?,
        solar: solar0.as_slice()?,
        opaque: opaque0.as_slice()?,
        month_hours: mh0_slice,
        coefficients: coefficients0,
        aggregate_split: aggregate_split0,
        heating_limits_w: heating_limits_w0.as_slice()?,
        cooling_limits_w: cooling_limits_w0.as_slice()?,
    };

    let arrays1 = DomainArrays {
        temperatures: temperatures1.as_slice()?,
        gains: gains1.as_slice()?,
        convective: convective1.as_slice()?,
        heating_setpoints: heating_setpoints1.as_slice()?,
        cooling_setpoints: cooling_setpoints1.as_slice()?,
        wind: wind1.as_slice()?,
        infiltration: infiltration1.as_slice()?,
        natural: natural1.as_slice()?,
        ventilation: ventilation1.as_slice()?,
        solar: solar1.as_slice()?,
        opaque: opaque1.as_slice()?,
        month_hours: mh1_slice,
        coefficients: coefficients1,
        aggregate_split: aggregate_split1,
        heating_limits_w: heating_limits_w1.as_slice()?,
        cooling_limits_w: cooling_limits_w1.as_slice()?,
    };

    let count = arrays0.temperatures.len();
    let mut powers0 = Vec::with_capacity(count);
    let mut powers1 = Vec::with_capacity(count);
    let mut temps0 = Vec::with_capacity(count);
    let mut temps1 = Vec::with_capacity(count);
    let mut exch0 = Vec::with_capacity(count);
    let mut exch1 = Vec::with_capacity(count);
    let mut residuals = Vec::with_capacity(count);

    for hour in 0..count {
        let f0 = match arrays0.apply_at(&mut core0, hour) {
            Ok(v) => v,
            Err(msg) => {
                core0.sync_to_py(model0)?;
                core1.sync_to_py(model1)?;
                return Err(PyValueError::new_err(msg));
            }
        };
        let f1 = match arrays1.apply_at(&mut core1, hour) {
            Ok(v) => v,
            Err(msg) => {
                core0.sync_to_py(model0)?;
                core1.sync_to_py(model1)?;
                return Err(PyValueError::new_err(msg));
            }
        };

        let (p, t, ex, res) = match step_coupled_pair(
            &mut core0,
            &mut core1,
            (f0.0, f0.1, f0.2, f0.3, f0.4, 1.0, true),
            (f1.0, f1.1, f1.2, f1.3, f1.4, 1.0, true),
            conductance_w_k,
            [arrays0.heating_limits_w[hour], arrays1.heating_limits_w[hour]],
            [arrays0.cooling_limits_w[hour], arrays1.cooling_limits_w[hour]],
        ) {
            Ok(v) => v,
            Err(msg) => {
                core0.sync_to_py(model0)?;
                core1.sync_to_py(model1)?;
                return Err(PyValueError::new_err(msg));
            }
        };

        powers0.push(p[0]);
        powers1.push(p[1]);
        temps0.push(t[0]);
        temps1.push(t[1]);
        exch0.push(ex[0]);
        exch1.push(ex[1]);
        residuals.push(res);
    }

    core0.sync_to_py(model0)?;
    core1.sync_to_py(model1)?;

    let powers_arr = PyArray2::from_vec2(py, &[powers0, powers1])?;
    let temps_arr = PyArray2::from_vec2(py, &[temps0, temps1])?;
    let exch_arr = PyArray2::from_vec2(py, &[exch0, exch1])?;
    let res_arr = PyArray1::from_vec(py, residuals);

    Ok((powers_arr, temps_arr, exch_arr, res_arr))
}
