# Rust + PyO3 Acceleration Plan (`RUST_PORT.md`)

The port originated from RC 0.8.0. It is now integrated with the RC 0.9.0 source
and its Core `>=0.8.2,<0.9` requirement. The profiling and timing tables in
sections 2 and 5 are historical measurements from the original 0.8.0 port. Section 6 records the
separate verification of the current 0.9.0 integration against current main.

## 1. Overview & Goals

Accelerate the performance-critical thermal simulation loops in `epsm_rc` by porting them to Rust with **PyO3** bindings, built as a mixed Python/Rust package via **`maturin`**.

### Core Requirements
- **Unchanged Public Python API**: All public functions, classes, signatures, and module exports (`epsm_rc`, `epsm_rc.platform`, `epsm_rc.domains`, `epsm_rc.coupled`, `epsm_rc.iso13790`, `epsm_rc.heating`, etc.) remain 100% compatible.
- **Numerical Accuracy Target (< 0.1% relative tolerance)**: Bitwise-exact IEEE-754 parity is **not** required. Results within **0.1%** of the current Python implementation are acceptable where they enable significant performance gains (e.g., replacing the 3-step trial/reset Thales interpolation in `calc_heating_demand` and the 6-step + `np.linalg.solve` active-set loop in `coupled.py` with direct 1-pass analytical linear sensitivity updates, as well as enabling full Rust/LLVM vectorization and FMA optimizations).
- **Test & Monkeypatch Compatibility**: Unit tests that monkeypatch `Building5R1C` construction in `platform.py` or `model.step` / `model.calc_heating_demand` on `Building5R1C` instances must continue to pass without modification.

---

## 2. Historical 0.8.0 Baseline Profiling Summary

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
- **Version**: maturin reads the distribution version from `Cargo.toml`. Keep it
  synchronized with `src/epsm_rc/_version.py` and the root package entry in
  `Cargo.lock`; all three currently specify `0.9.0`.
- **Layout**:
  - Python package source remains in `src/epsm_rc/` (`[tool.maturin] python-source = "src"`, `module-name = "epsm_rc._core_rs"`).
  - Rust crate lives at the repository root (`Cargo.toml`) with source files in `rust/` (`src/lib.rs`, `src/iso13790.rs`, `src/domain.rs`, `src/coupled.rs`).
- **Dependencies**:
  - `pyo3` (the `extension-module` feature is enabled by maturin; wheels target each supported CPython version, starting at 3.10)
  - `numpy` (Rust PyO3-NumPy bindings for zero-copy array reads and allocations)

Install a matching wheel with `python -m pip install /path/to/epsm_rc-...whl`.
Source installations require a stable Rust toolchain and a native linker:

```sh
python -m pip install ".[dev]"
python -m pip install build
python -m build
```

Run these commands inside an activated virtual environment. `python -m build`
creates an sdist and builds the wheel from it. After changing Rust sources,
rebuild the installed extension with `python -m pip install --force-reinstall --no-deps .`
or use `maturin develop --release --locked` after installing maturin in that
environment.

The [native wheel workflow](.github/workflows/native-wheels.yml) builds
manylinux2014 x86_64, macOS arm64/x86_64 and Windows x86_64 wheels and smoke-tests
their installation. Pull requests cover CPython 3.10–3.14 on Linux and 3.14 on
the other platforms; manual and tagged builds cover all platform/Python pairs.
Artifacts are uploaded for inspection; the workflow does not publish a release.
The [release workflow](.github/workflows/tests.yml) retains the coordinated
Core/Data compatibility matrix and installed-package validation.

---

## 4. Implementation Phases

### Phase 1: Core 5R1C Thermal Kernel & Platform Single-Domain Solver
1. **Rust 5R1C Kernel (`rust/src/iso13790.rs`)**:
   - Port `Building5R1C` thermal parameters, precomputed hourly factors (`_h_tr_em_base`, `_one_minus_hr`, `_vent_factor_base`, `_ach_to_h_ve`), and mutable state (`t_air`, `t_s`, `t_m`, `t_supply`, `windows_open`) into a Rust struct `Core5R1C`.
   - Implement `f64` equivalents, checked against Python trial steps, of:
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
   - Expose `run_single_domain_hourly` to Python and invoke it from `_run_domain` in `src/epsm_rc/platform.py`.
   - Pass NumPy `f64` / `bool` arrays from `HourlyInputs` directly into Rust (avoiding eager `.tolist()` conversions in `_prepare_domain`, while keeping lazy/on-demand sequence access on `HourlyInputs` if accessed directly).
   - Compute hourly `power` and `indoor` arrays in Rust and update `domain.model` final state (`t_air`, `t_s`, `t_m`, `_t_supply`, `_windows_open`).

### Phase 2: Coupled Multi-Domain Solver (`coupled.py` & `domains.py`)
1. **Rust Coupled Active-Set Solver (`rust/src/coupled.rs`)**:
   - Port `step_domains`, `solve_controls`, `_solve_for_active_modes`, `_within_active_power_limits`, `_unmet_limits_satisfied`, `_violation_score`, and `_commit_step` from `src/epsm_rc/coupled.py`.
   - Implement pivoted 2×2 linear system solves with the same active-set ordering as Python and numerical agreement within the stated tolerance.
   - Expose both:
     - `run_coupled_domains_hourly`: runs the full 8,760-hour coupled simulation loop for `simulate_conditioned_domains` in `src/epsm_rc/domains.py`.
     - `solve_controls_rs`: accelerates the active-set solve while retaining Python `step_domains` for reference calculations and method overrides.

### Phase 3: Standalone `Building5R1C` & Heating System Loops (`iso13790.py` & `heating.py`)
1. **Standalone `Building5R1C.simulate` (`src/epsm_rc/iso13790.py`)**:
   - Delegate `Building5R1C.simulate` to Rust (`simulate_5r1c_hourly`) only when `_can_use_native` confirms that the public methods and numerical helpers replaced by Rust retain their original behavior. Instance, class and subclass overrides use the Python path.
   - Support arbitrary `dt_hours`, hourly occupancy setbacks, 12-month ground temperatures with the same month boundaries, operative temperature control, and deterministic/stochastic window opening with Python's existing `random.random` draws.
2. **Single-Step `Building5R1C.calc_heating_demand` Acceleration**:
   - Accelerate single-step `calc_heating_demand` (used by `heating.py` models: `HeatPumpHPLib`, `HeatPumpCarnot`, `DistrictHeating`) through the same numerical-helper guard. Public standalone calls retain validation, physical-input refresh and public method overrides.
   - Owned prepared platform engines retain their existing private equation path: public `step` and demand wrappers are intentionally bypassed. Overrides to private equations or their numerical helpers still select the Python path.

### Phase 4 (Future Work): Radiation, Angular Optics & Gain Preparation
- Port `SimpleGlazingOptics.beam_properties`, `calc_g_value_angle_modifier`, `irradiance.aoi`, `plane_irradiance` (Perez & isotropic sky models), and `_gain_inputs` schedule broadcasting to Rust to further reduce cold-cache preparation time.

---

## 5. Historical 0.8.0 Results and Current Verification (Phases 1–3)

### Historical Micro-Benchmark & End-to-End Timings (8,760h Annual Simulation)

| Execution Path | Baseline (Python) | Rust + PyO3 (`_core_rs`) | Speedup |
| :--- | ---: | ---: | ---: |
| **Single-Domain `_run_domain` (8,760h)** | 48.50 ms | **1.32 ms** *(~0.35 ms in Rust loop)* | **36.7×** *(~135× kernel)* |
| **Warm Single-Domain Total** (`simulate_building` + cache) | 65.30 ms | **12.56–15.02 ms** | **4.3–5.2×** |
| **Coupled 2-Domain Loop (8,760h)** | 488.20 ms | **0.67 ms** | **728×** |
| **Warm Coupled 2-Domain Total** (`simulate_conditioned_domains`) | 515.10 ms | **21.54 ms** | **23.9×** |
| **Standalone `Building5R1C.simulate` (8,760h)** | 87.40 ms | **0.35 ms** | **250×** |

### Historical Benchmark Suite Summary (`benchmarks/run.py`)

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
- **Original port's unit-test snapshot**: 526 passed, 0 failed.

### Hardening and repeatable verification

`rust/src/bindings.rs` owns Python conversion and synchronization. The numerical
kernels use named forcing, control and state structs in `rust/src/types.rs`.
Shapes are checked before state changes. Deterministic batches release the GIL;
prepared immutable byte buffers are borrowed, while mutable inputs are copied
into owned snapshots. A NumPy read-only flag alone does not authorize borrowing.
Original method identities preserve overrides to the behavior replaced by Rust,
with the standalone/prepared distinction described above.

Native binary hashes now participate in calculation provenance. Release
comparisons use the stated 0.1% numerical tolerance and retain exact structural
and identity checks. Rust unit tests, differential Python/native tests and
concurrent-writer tests cover the accelerated paths. Run:

```sh
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
python -m pytest tests
```

See `benchmarks/README.md` for before/after timing commands. Keep forced inlining
limited to measured hot paths, and compare end-to-end preparation as well as the
native loops when changing input ownership or dispatch.

## 6. Local 0.9.0 integration verification — 2026-10-22

The integration was reconciled against main `c4c143a` and checked on macOS arm64
with CPython 3.12.3, NumPy 2.5.4, Core 0.8.2 and Data 0.6.0. The installed wheel
passed all 721 Python tests with no skips, including required EnergyPlus 24.2,
hplib and GIS checks. Rust formatting, Clippy and six Rust tests passed, as did
Ruff, Sphinx HTML/doctests and the three release notebooks. An sdist was built
and used to build a second wheel; its Python sources, native extension, packaged
weather and notices were checked.
The sdist-built wheel also passed 104 native, dispatch and benchmark-safety tests.

Installed main and candidate wheels were compared using `benchmarks/run.py`
for all seven fixtures, including the conditioned basement, under both policies.
Both environments used the same numerical dependencies. Each case ran seven
times with `--no-output`; repeated timings use the median after the first call.
Warm runs used `--warm --batch`. Every hourly/scalar comparison passed the stated
0.1% tolerance, structural checks passed, and standalone radiation outputs
matched exactly.

| Annual path | Cold speedup | Warm speedup |
| --- | ---: | ---: |
| Six single-domain fixtures, both policies | 1.36–1.72× | 3.35–4.56× |
| Conditioned basement, both policies | 4.87–5.50× | 21.28–24.16× |

These local timings cover repeated illustrative fixtures and exclude compressed
output writes. They do not measure stock throughput or scientific prediction
accuracy. A separate differential check of 20,000 randomized coupled-control
cases matched main's independent Python solver at `rtol=1e-9`, `atol=1e-7`;
the maximum power difference was below `1e-9 W`.

Linux, Windows, Intel macOS and other Python combinations are configured in
the native wheel workflow but have not been executed for this local integration.
Run those checks before promotion through `dev`; no remote branch, PR, tag or
package was published during this verification.
