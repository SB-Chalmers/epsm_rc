# RC examples

Use Python 3.11 or 3.13 for the shared Core interface. The coordinated release
uses RC 0.7.0, Core 0.7.0 and Data 0.6.0. Install `epsm_rc[platform,gis,notebook]`
in the notebook's kernel; install EnergyPlus 24.2 only for the engine comparison.
Package installation belongs in the environment, not in notebook cells.

| Example | Purpose and prerequisites |
| --- | --- |
| `09_5r1c_heating_demo.py` | Standalone 5R1C calculation with explicit example construction inputs. |
| `10_osm_neighbourhood_simulation.py` | GIS ingestion; inspect its declared external input/service prerequisites. |
| `11_core_engine_comparison.py` | A captured physical Core Building through both engines; requires explicit snapshot, weather, RC settings and EnergyPlus. |
| `12_prepared_radiation.ipynb` | Fresh independent thermal runs with optional preparation reuse, legacy equivalence and explicitly selected angular radiation. Uses one named illustrative Building and the packaged Goteborg EPW; writes no result directory. |
| `13_shared_wall_boundaries.ipynb` | Partial shared wall from explicit metric neighbour geometry, adiabatic opaque inventory, outdoor-only windows and RC boundary diagnostics. Uses the packaged EPW and writes no result directory. |
| `14_roofs_and_basements.ipynb` | Separate main/basement controls, conservative floor exchange and an explicitly ventilated finite cold-attic buffer. Uses packaged weather and writes no result directory. |

Run the notebook from a fresh installed-package kernel. Its assertions check exact
cache equivalence, calendar preservation and optical heat-gain accounting. Clear
execution counts/outputs before committing; retain executed evidence separately.
The illustrative inputs and accepted simulations do not establish prediction
accuracy against measured buildings. Additional optional heat-pump examples need
the advertised `heatpump` extra.
