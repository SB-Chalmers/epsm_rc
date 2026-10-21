use crate::domain::DomainArrays;
use crate::iso13790::Core5R1C;
use crate::types::{ControlSolution, CoupledOutput, KernelResult, StepInputs};

pub fn run_coupled_domains(
    core0: &mut Core5R1C,
    core1: &mut Core5R1C,
    arrays0: &DomainArrays<'_>,
    arrays1: &DomainArrays<'_>,
    conductance_w_k: f64,
) -> KernelResult<CoupledOutput> {
    let count = arrays0.temperatures.len();
    let mut result = CoupledOutput {
        powers: std::array::from_fn(|_| Vec::with_capacity(count)),
        temperatures: std::array::from_fn(|_| Vec::with_capacity(count)),
        exchanges: std::array::from_fn(|_| Vec::with_capacity(count)),
        residuals: Vec::with_capacity(count),
    };
    for hour in 0..count {
        let inputs0 = arrays0.apply_at(core0, hour)?;
        let inputs1 = arrays1.apply_at(core1, hour)?;
        let solution = step_coupled_pair(
            core0,
            core1,
            inputs0,
            inputs1,
            conductance_w_k,
            [
                arrays0.heating_limits_w[hour],
                arrays1.heating_limits_w[hour],
            ],
            [
                arrays0.cooling_limits_w[hour],
                arrays1.cooling_limits_w[hour],
            ],
        )?;
        for index in 0..2 {
            result.powers[index].push(solution.powers[index]);
            result.temperatures[index].push(solution.temperatures[index]);
            result.exchanges[index].push(solution.exchanges[index]);
        }
        result.residuals.push(solution.residual);
    }
    Ok(result)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ControlMode {
    Free,
    Heat,
    HeatLimit,
    Cool,
    CoolLimit,
}

#[inline]
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
) -> KernelResult<ControlSolution> {
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
                return Ok(ControlSolution {
                    powers,
                    temperatures,
                    exchanges: [exchange, -exchange],
                    residual: max_res,
                });
            }
        }
    }

    Err("No admissible coupled heating/cooling control state")
}

#[inline(always)]
pub fn step_coupled_pair(
    core0: &mut Core5R1C,
    core1: &mut Core5R1C,
    in0: StepInputs,
    in1: StepInputs,
    conductance_w_k: f64,
    heating_limit_w: [f64; 2],
    cooling_limit_w: [f64; 2],
) -> KernelResult<ControlSolution> {
    if core0.ventilation_behaviour <= 0.0 && core1.ventilation_behaviour <= 0.0 {
        core0.hour_heat_flows = None;
        core0.step_deterministic(in0, 0.0)?;
        let free0 = (core0.state.t_air, core0.state.t_s, core0.state.t_m);
        let (k_m0, k_s0, resp0) = core0.sensitivities(in0.dt_hours);

        core1.hour_heat_flows = None;
        core1.step_deterministic(in1, 0.0)?;
        let free1 = (core1.state.t_air, core1.state.t_s, core1.state.t_m);
        let (k_m1, k_s1, resp1) = core1.sensitivities(in1.dt_hours);

        let ControlSolution {
            powers,
            temperatures,
            exchanges,
            residual,
        } = solve_controls_inner(
            [free0.0, free1.0],
            [resp0, resp1],
            conductance_w_k,
            [core0.t_set_heating, core1.t_set_heating],
            [core0.t_set_cooling, core1.t_set_cooling],
            heating_limit_w,
            cooling_limit_w,
        )?;

        let q0 = powers[0] + exchanges[0];
        core0.state.t_air = free0.0 + resp0 * q0;
        core0.state.t_s = free0.1 + k_s0 * q0;
        core0.state.t_m = free0.2 + k_m0 * q0;
        if !(core0.state.t_air.is_finite() && temperatures[0].is_finite())
            || (core0.state.t_air - temperatures[0]).abs() > 1e-7
        {
            return Err("Coupled control failed its committed-state temperature balance");
        }

        let q1 = powers[1] + exchanges[1];
        core1.state.t_air = free1.0 + resp1 * q1;
        core1.state.t_s = free1.1 + k_s1 * q1;
        core1.state.t_m = free1.2 + k_m1 * q1;
        if !(core1.state.t_air.is_finite() && temperatures[1].is_finite())
            || (core1.state.t_air - temperatures[1]).abs() > 1e-7
        {
            return Err("Coupled control failed its committed-state temperature balance");
        }

        return Ok(ControlSolution {
            powers,
            temperatures,
            exchanges,
            residual,
        });
    }

    let state0 = (core0.state.t_air, core0.state.t_s, core0.state.t_m);
    let state1 = (core1.state.t_air, core1.state.t_s, core1.state.t_m);

    // Domain 0 trial steps
    core0.hour_heat_flows = None;
    core0.step_deterministic(in0, 0.0)?;
    let base0 = core0.state.t_air;
    core0.state.t_air = state0.0;
    core0.state.t_s = state0.1;
    core0.state.t_m = state0.2;
    let ref0 = 10.0 * core0.floor_area_m2;
    core0.step_deterministic(in0, ref0)?;
    let resp0 = (core0.state.t_air - base0) / ref0;
    core0.state.t_air = state0.0;
    core0.state.t_s = state0.1;
    core0.state.t_m = state0.2;

    // Domain 1 trial steps
    core1.hour_heat_flows = None;
    core1.step_deterministic(in1, 0.0)?;
    let base1 = core1.state.t_air;
    core1.state.t_air = state1.0;
    core1.state.t_s = state1.1;
    core1.state.t_m = state1.2;
    let ref1 = 10.0 * core1.floor_area_m2;
    core1.step_deterministic(in1, ref1)?;
    let resp1 = (core1.state.t_air - base1) / ref1;
    core1.state.t_air = state1.0;
    core1.state.t_s = state1.1;
    core1.state.t_m = state1.2;

    let ControlSolution {
        powers,
        temperatures,
        exchanges,
        residual,
    } = solve_controls_inner(
        [base0, base1],
        [resp0, resp1],
        conductance_w_k,
        [core0.t_set_heating, core1.t_set_heating],
        [core0.t_set_cooling, core1.t_set_cooling],
        heating_limit_w,
        cooling_limit_w,
    )?;

    // Commit step for both domains
    core0.step_deterministic(in0, powers[0] + exchanges[0])?;
    if !(core0.state.t_air.is_finite() && temperatures[0].is_finite())
        || (core0.state.t_air - temperatures[0]).abs() > 1e-7
    {
        return Err("Coupled control failed its committed-state temperature balance");
    }

    core1.step_deterministic(in1, powers[1] + exchanges[1])?;
    if !(core1.state.t_air.is_finite() && temperatures[1].is_finite())
        || (core1.state.t_air - temperatures[1]).abs() > 1e-7
    {
        return Err("Coupled control failed its committed-state temperature balance");
    }

    Ok(ControlSolution {
        powers,
        temperatures,
        exchanges,
        residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pivoted_solver_handles_zero_first_diagonal() {
        assert_eq!(solve_2x2([[0.0, 2.0], [1.0, 3.0]], [4.0, 7.0]), [1.0, 2.0]);
    }

    #[test]
    fn mixed_active_sets_conserve_exchange_and_obey_limits() {
        for h in [0.0, 10.0, 1000.0] {
            for free in [[10.0, 35.0], [30.0, 10.0], [22.0, 24.0]] {
                for maximum in [[100.0, 0.0], [f64::INFINITY; 2]] {
                    let response = [0.01, 0.02];
                    let minimum = [0.0, -100.0];
                    let result = solve_controls_inner(
                        free,
                        response,
                        h,
                        [20.0, 18.0],
                        [26.0; 2],
                        maximum,
                        minimum,
                    )
                    .unwrap();
                    assert_eq!(result.exchanges[0] + result.exchanges[1], 0.0);
                    assert!(result.residual < 1e-8);
                    for i in 0..2 {
                        assert!(result.powers[i] >= minimum[i] && result.powers[i] <= maximum[i]);
                        let balanced =
                            free[i] + response[i] * (result.powers[i] + result.exchanges[i]);
                        assert!((balanced - result.temperatures[i]).abs() < 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_conductance_and_response_fail_before_solving() {
        for conductance in [-1.0, f64::INFINITY, f64::NAN] {
            assert!(solve_controls_inner(
                [10.0; 2],
                [0.01; 2],
                conductance,
                [20.0; 2],
                [26.0; 2],
                [f64::INFINITY; 2],
                [0.0; 2]
            )
            .is_err());
        }
        assert!(solve_controls_inner(
            [10.0; 2],
            [0.0; 2],
            10.0,
            [20.0; 2],
            [26.0; 2],
            [f64::INFINITY; 2],
            [0.0; 2]
        )
        .is_err());
    }
}
