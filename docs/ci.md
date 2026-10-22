# Continuous integration

Pull requests into `dev` or `main` run the Python 3.13 `ci` job and a separate
Rust kernel job. The Python job exercises standalone thermal balances, physics,
heat pumps and native differential, dispatch and concurrency regressions, then
builds a wheel. The Rust job runs `cargo fmt --all --check`,
`cargo clippy --locked --all-targets -- -D warnings` and `cargo test --locked`.
The executable commands are in [.github/workflows/ci.yml](../.github/workflows/ci.yml).
These development checks do not certify the entire EPSM ecosystem.

[Native wheels](../.github/workflows/native-wheels.yml) also builds and installs
release wheels on pull requests. Its eight combinations cover CPython 3.10–3.14
on Linux x86_64 and CPython 3.14 on macOS arm64, macOS x86_64 and Windows x86_64.
Linux builds use manylinux2014 containers and select the Python interpreter
inside the container; installation tests use the matching host Python.
Each combination checks package metadata, runs the offline example and exercises
native and standalone APIs without Core or Data. Wheels are retained as workflow
artifacts. Manual runs and `v*` tags build all twenty platform/Python combinations;
this workflow does not publish packages.

Run **Release verification** explicitly before tagging or publishing a release,
and when a change needs the full engine, database, browser or installed-package
checks. Select one exact dependency cohort and keep that selection through the
release; do not use labels to change the tested dependencies. A merge or branch
synchronization does not publish a release or dispatch another full suite.

Do not retry dependency conflicts or missing credentials unchanged. Fix their
cause, then rerun only the affected checks. Preserve failing evidence.

See the [shared branch and CI policy](https://github.com/SB-Chalmers/epsm_development/blob/main/docs/branches-and-releases.md).

The validation suite that loads strict EPW weather through Core/Data remains in
the full [release workflow](../.github/workflows/tests.yml), using the shared
compatibility manifest to select coordinated dependencies. It builds and tests
an installed wheel with required EnergyPlus and hplib, and retains documentation,
notebook and released-output comparisons. The source build uses maturin and
requires a Rust toolchain; native wheel checks complement this release gate.
