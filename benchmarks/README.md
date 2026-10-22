# Installed-package annual regression and timing

The seven explicit snapshots exercise rectangle/concave geometry, 0%/100%
occupant sensible heat, scheduled airflow with cooling and monthly ground
temperatures, zero airflow, and a heated basement with coupled controls.
The six ordinary fixtures come from the coordinated 2026-10-15 comparison;
the basement fixture extends coverage to two thermal domains. These are
illustrative software fixtures, not measured-building validation.
The ordinary Building remains empty; the snapshots retain their selected
illustrative construction and operation provenance. Goteborg weather is the
existing packaged EPW with its original source terms; controlled weather is
generated explicitly by this runner.

`heated_basement.json` has a 10 × 8 m footprint, two 3 m above-ground
storeys, a 2.5 m conditioned basement, and no glazing or internal sensible
gains. The main and basement heating setpoints are 20 °C and 18 °C,
respectively; cooling is disabled. It uses Goteborg weather and an explicitly
authored constant 18 °C ground boundary. The separating floor exchanges
heat between the two domains. The runner compares both domain outputs
and their combined demand using the same exact checks as the ordinary cases.

Build and install released Core 0.7.0 and RC 0.7.1 wheels in a reference environment and
candidate wheels in another. This is the retained legacy correctness comparison.
For the 0.8.1 optimisation measurement, use RC 0.8.0 and Core 0.7.1 as the
reference, with the candidate also using Core 0.7.1. Both need the same pinned
Data 0.6.0 and numerical dependencies. Run this checkout's script with the candidate interpreter:

```sh
python benchmarks/run.py --directory outputs/benchmark-independent --reference-python /path/to/reference/bin/python
python benchmarks/run.py --directory outputs/benchmark-prepared --warm --reference-directory outputs/benchmark-independent/reference
python benchmarks/run.py --directory outputs/benchmark-batch --warm --batch --reference-directory outputs/benchmark-independent/reference
```

Each directory must be new. Floating hourly channels and annual totals use
`rtol=0.001` (0.1%) and `atol=1e-8`, with matching shapes/dtypes and no NaNs.
Calendars, reduced properties, snapshots and weather identities remain exact.
Installed implementation identities are retained and intentionally
differ between reference and candidate. First-call and repeated timings are
reported separately. Preparation reuse is not a cached final simulation result.
No absolute timing threshold is imposed on shared CI runners.

`--warm --batch` retains the compatibility batch context. Calculation identity
is captured lazily once per process; restart Python after changing code or dependencies.
The first call includes initial preparation; repeated timings exclude that fill.
Warm repetitions reuse the same physical fixture, so these are not measured
throughput for a stock of different buildings or parallel worker processes.

Use `--no-output` to measure in-memory results separately from compressed output.
`--policy 2` exercises angular optics against a reference that supports v2
(RC 0.7.1 or newer). Both modes use the same numerical tolerances. Radiation-only
comparisons remain exact because the Rust port does not change those calculations.

For tighter native-kernel comparisons and annual loop timings, run:

```sh
python benchmarks/native.py --directory outputs/native-before
# Rebuild the native extension after making changes, then compare:
python benchmarks/native.py --directory outputs/native-after --reference-directory outputs/native-before
```

This measures standalone, heating-system, single-domain and coupled-domain paths
with repeated batches. It checks outputs at `rtol=1e-10`, `atol=1e-9` and reports
timing ratios without a machine-dependent pass/fail timing threshold. Compare
isolated runs on the same machine and repeat close timing differences.

The Rust hardening comparison on macOS arm64, Python 3.14.2 and NumPy 2.5.4 used
35 batches of 30 annual calls against a saved copy of the original branch:

| Annual path | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Standalone simulation | 0.372 | 0.316 |
| Heating-system demand | 0.367 | 0.322 |
| Prepared single-domain kernel | 0.473 | 0.438 |
| Prepared coupled-domain kernel | 0.848 | 0.779 |

All native channels passed the tighter comparison above. Alternating 50 warm
before/after pairs for each of the six fixtures and a conditioned basement kept
end-to-end median paired timings within 0.3% of baseline. Scalar heating-demand
calls stayed within 1%. These local measurements are evidence for the change,
not portable timing thresholds.
