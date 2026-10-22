# EPSM RC

<!-- EPSM introduction: Pocket CAD -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/welcome/epsm_rc-dark.png">
  <img src="docs/assets/welcome/epsm_rc.png" width="160" height="160" alt="EPSM RC — Thermal RC network: input, resistor, active junction and output; a capacitor branches from the junction to ground.">
</picture>

Try an hourly building energy calculation with a compact thermal model. EPSM RC
uses an ISO 13790 5R1C model for standalone experiments or a reduced EPSM Core
Building. Explicit settings and diagnostics keep the modelling assumptions
available alongside the results.

**[Explore the EPSM platform →](https://github.com/SB-Chalmers/epsm/blob/main/docs/platform.md)**
<!-- /EPSM introduction -->

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/SB-Chalmers/epsm_rc/actions/workflows/ci.yml) [![Releases](https://img.shields.io/badge/releases-GitHub-181717?logo=github&logoColor=white)](https://github.com/SB-Chalmers/epsm_rc/releases) [![Issues](https://img.shields.io/badge/issues-GitHub-181717?logo=github&logoColor=white)](https://github.com/SB-Chalmers/epsm_rc/issues)

## Getting started

Use Python 3.10 or newer for standalone calculations. From a terminal on macOS
or Linux:

```sh
git clone https://github.com/SB-Chalmers/epsm_rc.git
cd epsm_rc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python examples/00_standalone.py
```

The example runs a synthetic 24-hour calculation offline. Its inputs are
illustrative assumptions; it does not validate a real building.

Thermal solver loops run in a Rust extension while the public interface remains
Python. Installing a matching wheel needs no Rust toolchain. Building from source,
including an editable checkout, requires Rust and Cargo; the build backend is
`maturin`. See [native builds and verification](RUST_PORT.md) for details.

For the shared Building interface and EPW loading, use Python 3.11 or newer,
install compatible Core/Data packages, then run:

```sh
python -m pip install -e ".[platform]"
```

Use the [supported EPSM package combination](https://github.com/SB-Chalmers/epsm/blob/main/compatibility.json)
and the [Building quick start](docs/quickstart.rst) with explicit weather and
reduction settings. Private dependencies need an accessible index or local checkouts.

## Documentation

- [Shared Building quick start](docs/quickstart.rst)
- [Standalone numerical API](docs/standalone.rst)
- [Radiation methods and limits](docs/radiation.md)
- [Performance and benchmarks](benchmarks/README.md)
- [Project guide and development](docs/guide.md)

## License

EPSM additions use [MIT](LICENSE). Retain the DIBS/IWU and
RC_BuildingSimulator/ETH attributions in the source, the packaged
[third-party notices](src/epsm_rc/notices), and bundled weather source headers.

## People and support

Developed at Chalmers University of Technology as part of the EPSM research platform.

**Funding**: Swedish Energy Agency (Project ID: P2024-04053)

**Partners**:

- Lindholmen Science Park AB
- Sinom AB
- Stiftelsen Chalmers Industriteknik

**Team**:

- **Lead Developer**: Sanjay Somanath
- **Principal Investigator**: Alexander Hollberg
- **Contributors**: Yinan Yu, Dag Wästberg

<p align="center">
  <img src="docs/assets/welcome/logos.png" width="100%" alt="Chalmers University of Technology, Chalmers Industriteknik, Sinom, Lindholmen Science Park and Swedish Energy Agency logos">
</p>
