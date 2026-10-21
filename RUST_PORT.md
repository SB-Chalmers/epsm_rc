# Rust + PyO3 Acceleration Plan (`RUST_PORT.md`)

## 1. Overview & Goals

Accelerate the performance-critical thermal simulation loops in `epsm_rc` by porting them to Rust with **PyO3** bindings, built as a mixed Python/Rust package via **`maturin`**.

### Core Requirements
- **Unchanged Public Python API**: All public functions, classes, signatures, and module exports (`epsm_rc`, `epsm_rc.platform`, `epsm_rc.domains`, `epsm_rc.coupled`, `epsm_rc.iso13790`, `epsm_rc.heating`, etc.) remain 100% compatible.
- **Numerical Accuracy Target (< 0.1% relative tolerance)**: Bitwise-exact IEEE-754 parity is **not** required. Results within **0.1%** of the current Python implementation are acceptable where they enable significant performance gains (e.g., replacing the 3-step trial/reset Thales interpolation in `calc_heating_demand` and the 6-step + `np.linalg.solve` active-set loop in `coupled.py` with direct 1-pass analytical linear sensitivity updates, as well as enabling full Rust/LLVM vectorization and FMA optimizations).
- **Test & Monkeypatch Compatibility**: Unit tests that monkeypatch `Building5R1C` construction in `platform.py` or `model.step` / `model.calc_heating_demand` on `Building5R1C` instances must continue to pass without modification.

---

## 2. Baseline Profiling Summary

Measured on an 8,760-hour annual simulation (Python 3.14.2 in `/Users/dagwastberg/repos/epsm/.venv`):

| Execution Path | Baseline Time | Breakdown | Primary Bottleneck in `epsm_rc` |
| :--- | ---: | :--- | :--- |
| **Warm Single-Domain** (`simulate_building` + `PreparationCache`) | **~65.3 ms** | • `_prepare_building`: 7.9 ms (12%)<br>• `_prepare_domain`: 8.8 ms (14%)<br>• **`_run_domain`: 48.5 ms (74%)** | 8,760-hour Python loop in `_run_domain` calling `HourlyInputs.at` and `Building5R1C.calc_heating_demand` (47.5 ms) + `.tolist()` array conversions in `_prepare_domain`. |
| **Cold Single-Domain** (`simulate_building`, no cache) | **~136.4 ms** | • `_prepare_building`: 59.3 ms (43%)<br>• `_prepare_domain`: 29.0 ms (21%)<br>• **`_run_domain`: 48.1 ms (35%)** | `_run_domain` (48.1 ms) and `_prepare_radiation` (~10 ms after `pvlib` SPA). |
| **Warm Coupled 2-Domain** (`simulate_conditioned_domains`) | **~515.1 ms** | • `_prepare_building`: 12.8 ms (2.5%)<br>• `2× _prepare_domain`: 14.1 ms (2.7%)<br>• **Coupled loop: 488.2 ms (94.8%)** | 8,760-hour Python loop in `simulate_conditioned_domains` calling `step_domains` & `solve_controls` (6 `model.step` calls + multiple `np.linalg.solve` 2×2 allocations per hour). |
| **Standalone `Building5R1C.simulate`** | **~87.4 ms** | • **8,760-hour loop: 87.4 ms (100%)** | Per-hour Python loop in `Building5R1C.simulate`, `bisect` month lookup, setpoint checks, and trial steps. |

---

## 3. Package & Build Architecture (`maturin`)

- **Build Backend**: `maturin` (`requires = ["maturin>=1.7,<2.0"]`, `build-backend = "maturin"` in `pyproject.toml`).
- **Layout**:
  - Python package source remains in `src/epsm_rc/` (`[tool.maturin] python-source = "src"`, `module-name = "epsm_rc._core_rs"`).
  - Rust crate lives at the repository root (`Cargo.toml`) with source files in `rust/` (`src/lib.rs`, `src/iso13790.rs`, `src/domain.rs`, `src/coupled.rs`).
- **Dependencies**:
  - `pyo3` (with `extension-module` and `abi3-py310` for broad Python 3.10–3.14 compatibility)
  - `numpy` (Rust PyO3-NumPy bindings for zero-copy array reads and allocations)

---

## 4. Implementation Phases

### Phase 1: Core 5R1C Thermal Kernel & Platform Single-Domain Solver
1. **Rust 5R1C Kernel (`rust/src/iso13790.rs`)**:
   - Port `Building5R1C` thermal parameters, precomputed hourly factors (`_h_tr_em_base`, `_one_minus_hr`, `_vent_factor_base`, `_ach_to_h_ve`), and mutable state (`t_air`, `t_s`, `t_m`, `t_supply`, `windows_open`) into a Rust struct `Core5R1C`.
   - Implement exact IEEE-754 `f64` equivalents of:
     - `calc_h_ve`
     - `calc_heat_flows`
     - `calc_phi_m_tot`
     - `calc_t_m_next`
     - `calc_t_s`
     - `calc_t_air`
     - `calc_t_operative`
     - `step` (both general `dt_hours` and prepared hourly fast path)
     - `calc_heating_demand` (including 1-step free-float shortcut and 2-step linear interpolation with `unit_power = floor_area * 10.0`)
2. **Single-Domain Batch Runner (`rust/src/domain.rs`)**:
   - Expose `run_single_domain_hourly` to Python and invoke it from `_run_domain` in `src/epsm_rc/domains.py`.
   - Pass NumPy `f64` / `bool` arrays from `HourlyInputs` directly into Rust (avoiding eager `.tolist()` conversions in `_prepare_domain`, while keeping lazy/on-demand sequence access on `HourlyInputs` if accessed directly).
   - Compute hourly `power`, `indoor`, and `operative` arrays in Rust and update `domain.model` final state (`t_air`, `t_s`, `t_m`, `_t_supply`, `_windows_open`).

### Phase 2: Coupled Multi-Domain Solver (`coupled.py` & `domains.py`)
1. **Rust Coupled Active-Set Solver (`rust/src/coupled.rs`)**:
   - Port `step_domains`, `solve_controls`, `_solve_for_active_modes`, `_within_active_power_limits`, `_unmet_limits_satisfied`, `_violation_score`, and `_commit_step` from `src/epsm_rc/coupled.py`.
   - Implement exact 1×1 and 2×2 linear system solves matching `np.linalg.solve` (LAPACK `dgesv` partial pivoting) so coupled active-set transitions and power allocations match Python bit-for-bit.
   - Expose both:
     - `run_coupled_domains_hourly`: runs the full 8,760-hour coupled simulation loop for `simulate_conditioned_domains` in `src/epsm_rc/domains.py`.
     - `step_coupled_domains`: accelerates single-step `step_domains` in `src/epsm_rc/coupled.py`.

### Phase 3: Standalone `Building5R1C` & Heating System Loops (`iso13790.py` & `heating.py`)
1. **Standalone `Building5R1C.simulate` (`src/epsm_rc/iso13790.py`)**:
   - Delegate `Building5R1C.simulate` to Rust (`simulate_5r1c_hourly`) whenever `step` and `calc_heating_demand` are not monkeypatched on the `Building5R1C` instance (`"step" not in self.__dict__ and "calc_heating_demand" not in self.__dict__`).
   - Support arbitrary `dt_hours`, `occupancy_schedule` setback deltas, scalar or array `ach_schedule` / `ventilation_schedule`, scalar or 12-month `t_ground` (using exact `bisect_right` month boundaries), operative temperature control, and deterministic/stochastic window opening (preserving `np.random.default_rng(window_opening_seed)` draw values when stochastic window opening is enabled).
2. **Single-Step `Building5R1C.calc_heating_demand` & `step` Acceleration**:
   - Accelerate single-step `calc_heating_demand` (used by `heating.py` models: `HeatPumpHPLib`, `HeatPumpCarnot`, `DistrictHeating`) via Rust when `self.step` is not monkeypatched in `self.__dict__`.

### Phase 4 (Future Work): Radiation, Angular Optics & Gain Preparation
- Port `SimpleGlazingOptics.beam_properties`, `calc_g_value_angle_modifier`, `irradiance.aoi`, `plane_irradiance` (Perez & isotropic sky models), and `_gain_inputs` schedule broadcasting to Rust to further reduce cold-cache preparation time.

---

## 5. Verification & Achieved Results (Phases 1–3)

### Micro-Benchmark & End-to-End Timings (8,760h Annual Simulation)

| Execution Path | Baseline (Python) | Rust + PyO3 (`_core_rs`) | Speedup |
| :--- | ---: | ---: | ---: |
| **Single-Domain `_run_domain` (8,760h)** | 48.50 ms | **1.32 ms** *(~0.35 ms in Rust loop)* | **36.7×** *(~135× kernel)* |
| **Warm Single-Domain Total** (`simulate_building` + cache) | 65.30 ms | **12.56–15.02 ms** | **4.3–5.2×** |
| **Coupled 2-Domain Loop (8,760h)** | 488.20 ms | **0.67 ms** | **728×** |
| **Warm Coupled 2-Domain Total** (`simulate_conditioned_domains`) | 515.10 ms | **21.54 ms** | **23.9×** |
| **Standalone `Building5R1C.simulate` (8,760h)** | 87.40 ms | **0.35 ms** | **250×** |

### Benchmark Suite Summary (`benchmarks/run.py`)

| Fixture | Cold Baseline | Cold Rust | Cold Speedup | Warm Batch Baseline | Warm Batch Rust | Warm Batch Speedup | Max Scalar Rel. Error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `controlled_sensible_0` | 133.22 ms | 78.57 ms | **1.70×** | 65.90 ms | 12.56 ms | **5.25×** | $3.33 \times 10^{-13}\%$ |
| `controlled_sensible_1` | 134.66 ms | 81.41 ms | **1.65×** | 67.57 ms | 13.35 ms | **5.06×** | $2.85 \times 10^{-13}\%$ |
| `dynamic_airflow_cooling_ground` | 168.87 ms | 95.69 ms | **1.76×** | 72.33 ms | 14.01 ms | **5.16×** | $1.20 \times 10^{-13}\%$ |
| `weather_concave` | 166.75 ms | 100.31 ms | **1.66×** | 58.83 ms | 15.02 ms | **3.92×** | $2.62 \times 10^{-14}\%$ |
| `weather_rectangle` | 143.45 ms | 101.90 ms | **1.41×** | 57.10 ms | 13.73 ms | **4.16×** | $7.76 \times 10^{-14}\%$ |
| `zero_airflow` | 146.10 ms | 100.33 ms | **1.46×** | 65.14 ms | 13.75 ms | **4.74×** | $0.00\%$ |

- **Overall Maximum Scalar Relative Error**: $3.33 \times 10^{-13}\% \ll 0.1\%$
- **Overall Maximum Hourly Array Relative Error**: $7.08 \times 10^{-9}\% \ll 0.1\%$
- **Unit Test Suite**: 526 passed, 0 failed.
