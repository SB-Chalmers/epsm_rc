# Release 0.6.0 — 18 October 2026

Shared Building calculations respect Core's physical element boundaries. Shared
contact patches contribute no outdoor transmission, opaque solar absorption or
sky loss; their installed opaque constructions retain thermal capacity. Internal
walls within the aggregated conditioned domain likewise retain mass without
outdoor exchange. Core owns partial contact geometry, window placement and one
physical material owner per internal wall. Neighbour temperature and heating
evidence are not required for the explicit adiabatic approximation.

Boundary-area diagnostics make the reduction inspectable. Unsupported geometry
and basement readiness checks still apply. Detached calculations preserve their
hourly thermal results; input and implementation identities include the changed
boundary contract. See [boundary reduction](docs/boundaries.md).

The platform extra requires Core >=0.6.0,<0.7 and Data >=0.6.0,<0.7 because these
versions supply the tested public boundary and construction contracts. Numerical
standalone imports remain independent of Core. Research runners must additionally
pin the exact tested package releases or commits; these compatibility ranges are
not a substitute for study input identities.

# Release 0.5.0 — 18 October 2026

Deterministic Core reductions reuse fixed thermal properties, hourly network
coefficients and zero-load states. Calendar preparation, NumPy radiation and
orientation reuse reduce repeated work. Optional caller-owned `PreparationCache`
reuses weather and radiation preparation while rechecking source/weather contents
and creating fresh thermal state on every run. The installed-wheel benchmark
compares every hourly channel and annual total with released v1 behavior.

The explicitly selected `epsm_rc_5r1c_v2` policy adds midpoint solar sampling and
documented angular simple-glazing optics with nominal inward absorbed solar.
`epsm_rc_5r1c_v1` preserves its numerical behavior. New radiation components,
method assumptions and limitations are inspectable. Dynamic longwave and geometric
shading are excluded. See [radiation and migration](docs/radiation.md).

The platform extra now requires Core >=0.5.0 and Data >=0.6.0. CI records approved
dependency commits and also tests development dependencies. Third-party optical
correlation notices ship with the distribution. No measured-building accuracy or
identical EnergyPlus/RC demand is claimed.

# Release 0.4.0 — 15 October 2026

Physical-model reduction, annual calculations and GIS operations. Completion records retain the calculated run identity.

Libraries emit through Python's standard logging without configuring output.
Instrumented operations accept optional keyword-only `log_context`; identifiers
stay separate from scientific inputs, saved models and cache keys. INFO shows
major stages; DEBUG adds decisions and timing. Standalone commands support
`--log-level`, including `OFF`, where the package provides a command.

The EPSM application configures the combined stderr and optional DEBUG JSONL
output with `configure_logging`; `enabled=False` disables both destinations.
Existing callers and supported older collaborator signatures remain supported.
No new mandatory dependency is introduced by the logging implementation.

The coordinated release was requested for manual review and local validation
because GitHub refused to start several hosted jobs due to account billing.
Hosted checks that did not run are not counted as passes. See the
[coordinated verification record](https://github.com/SB-Chalmers/epsm_development/blob/main/docs/logging-verification.md)
for exact tested revisions, executed checks and remaining limitations.
