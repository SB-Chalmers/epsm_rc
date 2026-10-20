# RC examples

The recommended physical Building route uses RC 0.8.0, Core 0.7.1 and Data 0.6.x
on Python 3.11 or newer. Install `epsm_rc[platform,gis,notebook]` in the notebook
kernel. Install EnergyPlus 24.2 for engine comparison. Environment installation
belongs outside notebook cells. Base numerical examples also support Python 3.10.

| Example | Purpose and prerequisites |
| --- | --- |
| `00_standalone.py` | Synthetic 24-hour thermal calculation and explicit Carnot COP; no Core, weather file or network. Run `python examples/00_standalone.py`. |
| `09_5r1c_heating_demo.py` | Standalone illustrative 5R1C calculation using the explicitly selected bundled Göteborg EPW; needs `platform,dev` for weather checks and plotting. |
| `10_osm_neighbourhood_simulation.py` | RC-only GIS example. Inspect its explicit external input/service prerequisites; needs `platform,gis,dev` for reading and plotting. |
| `11_core_engine_comparison.py` | A captured physical Core Building through both compilers; explicit snapshot, EPW, RC settings and EnergyPlus required. |
| `12_prepared_radiation.ipynb` | Independent thermal runs, exact preparation-cache equivalence and angular radiation using a named illustrative Building and bundled EPW. |
| `13_shared_wall_boundaries.ipynb` | Shared wall geometry, adiabatic inventory, outdoor-only windows and boundary diagnostics using bundled weather. |
| `14_roofs_and_basements.ipynb` | Separate main/basement controls and a finite cold attic with declared ventilation using bundled weather. |

Run every notebook in a fresh installed-package kernel. Clear outputs and execution
counts in source notebooks; keep executed evidence in a separate output directory.
The assertions establish the stated software behavior, calendar checks and energy
accounting. They do not certify prediction accuracy against measured buildings.
Optional empirical heat-pump operations require `epsm_rc[heatpump]`; select the
concrete heat-pump class explicitly.

The two plotting scripts write PNG files under `examples/output` when their main
entry points run. Existing files with the same names are replaced. The OSM example
explicitly contacts the public Overpass service; it is an optional network recipe,
not part of the offline example or notebook release checks.
