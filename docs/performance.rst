Compute efficiency
==================

RC 0.8.1 reduces Python dispatch and scalar-comparison overhead in prepared
annual calculations. The physical equations, timestep, control trials and
floating-point operation order are unchanged. The measurements below use Core
0.7.1 and Data 0.6.x. RC 0.8.2 additionally accepts Core 0.8.x; it introduces no
solver changes or new performance measurements.

Calculation boundaries
----------------------

The Core Building route captures and validates physical inputs once per building,
then prepares each conditioned domain. During execution, airflow, gains and
setpoints can change each hour; fixed physical coefficients remain prepared.
The ordinary loop calls ``Building5R1C._calculate_demand`` with ``_step`` as its
advance operation. Coupled controls also use ``_step`` on their owned prepared
models. These private methods contain the shared thermostat and 5R1C equations.
They require inputs supplied by a validated public operation or prepared adapter.

Public ``step`` and ``calc_heating_demand`` retain their signatures and refresh
editable standalone inputs at the outer calculation boundary. The public
thermostat passes ``self.step`` to the shared calculation so public method
overrides, nested calls and stochastic window behaviour keep their existing
meaning. Do not call the private methods as a replacement for public validation.

The final coupled temperature check uses scalar ``math.isclose`` with zero
relative tolerance and an absolute tolerance of 1e-7 degrees Celsius. This is the
same tolerance as the previous scalar NumPy comparison. The check remains in
place; neither comparison is a separate finiteness validator.

Why a heated basement takes longer
----------------------------------

A conditioned basement introduces a second thermal state and independent
controls. The separating floor exchanges equal and opposite heat between the
domains. Each coupled hour uses free, reference-power and final calculations for
both models, followed by control-state checks. Ordinary calculations can skip
reference and final trials when conditioning is unnecessary. Coupled control
also evaluates small two-by-two systems for candidate heating and cooling states.

With one or two fixed domains, calculation time is O(T) for T hours. Preparation
also depends on physical element count, geometry complexity and unique schedules.
The hourly recurrence follows the previous thermal state, so hours cannot simply
be computed independently. The 0.8.1 changes reduce per-hour overhead.

Measured release comparison
---------------------------

Measured on 20 October 2026 on an Apple M4 Pro with Python 3.12.3. Both installed
packages used Core 0.7.1, Data 0.6.0, NumPy 2.5.4, pandas 3.0.6 and pvlib 0.16.1.
Each value is a median of 27 calls across three sequential process blocks, with
release/candidate order alternated. In-memory timings include preparation and
exclude imports, Building reconstruction and result writing.

.. list-table:: RC 0.8.0 compared with RC 0.8.1
   :header-rows: 1

   * - Case
     - Preparation
     - 0.8.0 ms
     - 0.8.1 ms
     - Faster
   * - Rectangle v1
     - Warm
     - 59.4
     - 53.5
     - 10.0%
   * - Rectangle v1
     - Uncached
     - 147.4
     - 137.9
     - 6.5%
   * - Heated basement v1
     - Warm
     - 557.6
     - 411.5
     - 26.2%
   * - Heated basement v1
     - Uncached
     - 614.0
     - 500.0
     - 18.6%
   * - Rectangle v2
     - Warm
     - 61.8
     - 55.9
     - 9.6%
   * - Heated basement v2
     - Warm
     - 539.2
     - 416.4
     - 22.8%

All 84 fixture/policy/preparation/block comparisons matched exactly. Warm
ordinary fixtures improved by 9.6–12.1%; the two basement policies improved by
22.8–26.2%. Every pooled preparation/fixture median improved, while individual
blocks showed normal timing variation. These figures describe the seven
authored software fixtures, not arbitrary building stocks or parallel throughput.
The basement fixture uses its explicit 18 °C ground boundary and distinct 20/18 °C
controls; it differs from the earlier exploratory profiling fixture.

Raw samples, environment metadata and validation results accompany the
`0.8.1 release <https://github.com/SB-Chalmers/epsm_rc/releases/tag/v0.8.1>`_.

Verification and measurement
----------------------------

``benchmarks/run.py`` compares installed reference and candidate packages using
seven explicit physical snapshots, including a heated basement with distinct
main/basement controls. Every hourly channel, time axis, annual scalar, reduced
property and input identity must match exactly. Both reduction policies run with
and without preparation reuse. Implementation identities may differ.

Use the same Python and numerical dependencies in both environments. From the
candidate checkout, run the candidate interpreter with a released reference::

    python benchmarks/run.py --directory outputs/performance-v1 \
      --reference-python /path/to/release/bin/python --repeats 10 --no-output
    python benchmarks/run.py --directory outputs/performance-v1-warm \
      --reference-directory outputs/performance-v1/reference \
      --repeats 10 --no-output --warm

Repeat with ``--policy 2``. Each output directory must be new. The first annual
call is reported separately; ten calls leave nine measured repetitions. For
performance claims, use three sequential process blocks and alternate which
interpreter runs first. Compare medians and individual block results. Do not run
timing workers concurrently. In-memory measurements include preparation and
exclude imports and Building reconstruction. Uncached preparation is not a fresh
process or a cold filesystem cache. Measure compressed result writing separately
by omitting ``--no-output``.

CI keeps its existing three jobs. Exact comparisons run once on Python 3.13 with
two repetitions; these are correctness checks, not statistically useful timing
samples. No wall-clock threshold is imposed on shared hosted runners. Standalone
edit/validation, method-override, window-opening and coupled-conservation tests
remain required alongside the annual comparisons.

Further optimisation
--------------------

Profile again before changing the two-by-two control solver, hourly forcing
handoff or radiation accumulation. Those changes have no established speedup in
this release; solver arithmetic and radiation grouping can change floating-point
results. Keep one implementation of each equation and forcing rule. Weather-byte
verification and physical-input ownership protect correctness and remain part of
the preparation boundary.
