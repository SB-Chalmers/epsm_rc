# epsm_rc

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/SB-Chalmers/epsm_rc/actions/workflows/ci.yml) [![Releases](https://img.shields.io/badge/releases-GitHub-181717?logo=github&logoColor=white)](https://github.com/SB-Chalmers/epsm_rc/releases) [![Issues](https://img.shields.io/badge/issues-GitHub-181717?logo=github&logoColor=white)](https://github.com/SB-Chalmers/epsm_rc/issues)

Estimate hourly building heating and cooling with an ISO 13790 5R1C thermal
model. Use explicit numerical inputs for standalone experiments or reduce a
shared EPSM Core Building with recorded modelling assumptions and diagnostics.

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
