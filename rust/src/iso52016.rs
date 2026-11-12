//! ISO 52016-1 hourly zone recurrence on a pre-inverted network matrix.
//!
//! Python assembles the network (`epsm_rc.iso52016.Zone52016`) and inverts the
//! constant matrix once. This loop evaluates hourly airflow exactly as
//! `HourlyInputs.at` does, applies the rank-one ventilation update and the exact
//! linear thermostat, and returns power, air and operative temperatures.

pub struct Network52016 {
    pub n: usize,
    pub k: usize,
    /// Row-major n x n inverse of the constant matrix.
    pub m: Vec<f64>,
    /// Row-major n x k product of the inverse and the forcing map.
    pub mp: Vec<f64>,
    pub m_air: Vec<f64>,
    pub m_hvac: Vec<f64>,
    pub c_dt: Vec<f64>,
    pub control: Vec<f64>,
    pub inner: Vec<usize>,
    pub areas: Vec<f64>,
}

pub struct Hourly52016<'a> {
    pub theta_e: &'a [f64],
    pub theta_gr: &'a [f64],
    pub phi_int: &'a [f64],
    pub phi_sol: &'a [f64],
    /// Row-major hours x elements absorbed solar (W/m²).
    pub absorbed: &'a [f64],
    pub set_heating: &'a [f64],
    pub set_cooling: &'a [f64],
    pub heat_limit: &'a [f64],
    pub cool_limit: &'a [f64],
    pub wind: &'a [f64],
    pub infiltration: &'a [f64],
    pub natural: &'a [f64],
    pub ventilation: &'a [f64],
}

pub struct Airflow52016 {
    pub coefficients: [f64; 4],
    pub volume_m3: f64,
    pub air_capacity_j_m3_k: f64,
    pub recovery_effectiveness: f64,
    pub recovery_bypass: bool,
}

pub struct Output52016 {
    pub power: Vec<f64>,
    pub air: Vec<f64>,
    pub operative: Vec<f64>,
}

fn matvec_add(matrix: &[f64], rows: usize, cols: usize, x: &[f64], out: &mut [f64]) {
    for (row, value) in out.iter_mut().enumerate().take(rows) {
        let line = &matrix[row * cols..(row + 1) * cols];
        let mut sum = 0.0;
        for (a, b) in line.iter().zip(x) {
            sum += a * b;
        }
        *value += sum;
    }
}

pub fn run(
    net: &Network52016,
    hours: &Hourly52016,
    air: &Airflow52016,
    warmup_passes: usize,
) -> Result<Output52016, String> {
    let count = hours.theta_e.len();
    let n = net.n;
    let k = net.k;
    let elements = k - 5;
    let area_total: f64 = net.areas.iter().sum();
    let mut state = vec![hours.set_heating[0]; n];
    let mut weighted = vec![0.0; n];
    let mut forcing = vec![0.0; k];
    let mut y = vec![0.0; n];
    let mut response = vec![0.0; n];
    let mut last = 0.0;
    let mut out = Output52016 {
        power: vec![0.0; count],
        air: vec![0.0; count],
        operative: vec![0.0; count],
    };
    let [c0, c1, c2, c3] = air.coefficients;
    for _ in 0..=warmup_passes {
        for hour in 0..count {
            let theta_e = hours.theta_e[hour];
            let t_air = state[0];
            let w = hours.wind[hour];
            let correction = c0 + c1 * (t_air - theta_e).abs() + c2 * w + c3 * w * w;
            let ach_infiltration = hours.infiltration[hour] * correction + hours.natural[hour];
            if !ach_infiltration.is_finite() || ach_infiltration < 0.0 {
                return Err("Hourly infiltration must be finite and nonnegative".into());
            }
            let eta = if air.recovery_bypass {
                let heating = last > 0.0 || t_air <= hours.set_heating[hour] + 1e-9;
                let cooling = last < 0.0 || t_air >= hours.set_cooling[hour] - 1e-9;
                let active = (heating && theta_e < t_air) || (cooling && theta_e > t_air);
                if active {
                    air.recovery_effectiveness
                } else {
                    0.0
                }
            } else {
                air.recovery_effectiveness
            };
            let ach_effective = ach_infiltration + hours.ventilation[hour] * (1.0 - eta);
            let h_ve = air.air_capacity_j_m3_k * air.volume_m3 * ach_effective / 3600.0;

            forcing[0] = theta_e;
            forcing[1] = hours.theta_gr[hour];
            forcing[2] = hours.phi_int[hour];
            forcing[3] = hours.phi_sol[hour];
            forcing[4] = 1.0;
            forcing[5..].copy_from_slice(&hours.absorbed[hour * elements..(hour + 1) * elements]);
            for i in 0..n {
                weighted[i] = net.c_dt[i] * state[i];
                y[i] = net.m_air[i] * h_ve * theta_e;
            }
            matvec_add(&net.m, n, n, &weighted, &mut y);
            matvec_add(&net.mp, n, k, &forcing, &mut y);

            let denominator = 1.0 + h_ve * net.m_air[0];
            let scale_free = h_ve * y[0] / denominator;
            let scale_hvac = h_ve * net.m_hvac[0] / denominator;
            let mut free_control = 0.0;
            let mut gain = 0.0;
            for i in 0..n {
                y[i] -= net.m_air[i] * scale_free;
                response[i] = net.m_hvac[i] - net.m_air[i] * scale_hvac;
                free_control += net.control[i] * y[i];
                gain += net.control[i] * response[i];
            }
            let (heating, cooling) = (hours.set_heating[hour], hours.set_cooling[hour]);
            let (heat_limit, cool_limit) = (hours.heat_limit[hour], hours.cool_limit[hour]);
            let mut power = 0.0;
            if free_control < heating && heat_limit > 0.0 {
                power = ((heating - free_control) / gain).min(heat_limit);
            } else if free_control > cooling && cool_limit < 0.0 {
                power = ((cooling - free_control) / gain).max(cool_limit);
            }
            for i in 0..n {
                state[i] = y[i] + power * response[i];
            }
            let mut radiant = 0.0;
            for (index, area) in net.inner.iter().zip(&net.areas) {
                radiant += area * state[*index];
            }
            if !state[0].is_finite() || !power.is_finite() {
                return Err("ISO 52016-1 recurrence produced a non-finite value".into());
            }
            out.power[hour] = power;
            out.air[hour] = state[0];
            out.operative[hour] = 0.5 * state[0] + 0.5 * radiant / area_total;
            last = power;
        }
    }
    Ok(out)
}
