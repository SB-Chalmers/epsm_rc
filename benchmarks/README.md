# Installed-package annual regression and timing

The six explicit snapshots exercise rectangle/concave geometry, 0%/100%
occupant sensible heat, scheduled airflow with cooling and monthly ground
temperatures, and zero airflow. They are illustrative software fixtures from
the coordinated 2026-10-15 comparison, not measured-building validation.
The ordinary Building remains empty; the snapshots retain their selected
illustrative construction and operation provenance. Goteborg weather is the
existing packaged EPW with its original source terms; controlled weather is
generated explicitly by this runner.

Build and install released Core/RC 0.4.0 wheels in a reference environment and
candidate wheels in another. Both need the same pinned Data 0.6.0 and numerical
dependencies. Run this checkout's script with the candidate interpreter:

```sh
python benchmarks/run.py --directory outputs/benchmark-independent --reference-python /path/to/reference/bin/python
python benchmarks/run.py --directory outputs/benchmark-prepared --warm --reference-directory outputs/benchmark-independent/reference
```

Each directory must be new. Every hourly channel, calendar, annual total,
reduced property, snapshot and weather identity is compared exactly; no tolerance
is used. Installed implementation identities are retained and intentionally
differ between reference and candidate. First-call and repeated timings are
reported separately. Preparation reuse is not a cached final simulation result.
No absolute timing threshold is imposed on shared CI runners.
