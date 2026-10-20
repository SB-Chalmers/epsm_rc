# EPSM RC

Version **0.8.0**. RC reduces the same physical Building used by Core to an hourly
ISO 13790 5R1C thermal network. Core owns physical inputs and weather inspection;
RC owns reduction assumptions, the thermal calculation and its result meaning.
The shared Building route uses **Core 0.7.1** and **Data 0.6.x**, on Python 3.11+.
The standalone numerical API continues to support Python 3.10 without Core.

Install the shared Building interface with `python -m pip install "epsm_rc[platform]"`.
Supply a complete local hourly EPW and explicitly select the reduction assumptions:

```python
from epsm_core import Building, Weather
from epsm_rc import RCSettings, simulate_building

building = Building(preset="illustrative-reference")
weather = Weather.from_epw("weather.epw")
settings = RCSettings.from_policy(
    "epsm_rc_5r1c_v2",
    airflow_policy="sensible_only_unlimited_supply",
    ground_policy="conductance_multiplier",
)
result = simulate_building(building, weather=weather, settings=settings)
print(result.scalars["space_heating_demand_kwh"])
```

The named preset is an illustrative assumption set. Each evaluation captures its
inputs, creates fresh thermal state and returns hourly values, units, diagnostics
and identities. Result acceptance and engine comparisons establish specified
software behavior; they do not certify prediction accuracy for measured buildings.

Use `python examples/00_standalone.py` for an offline numerical example with the
base package. More [examples](examples/README.md), [API documentation](docs/index.rst),
[radiation policies](docs/radiation.md), [roof and basement policies](docs/roofs-and-basements.md)
and [installed-wheel benchmarks](benchmarks/README.md) describe their prerequisites
and limits. Historical experiment records are retained with their source revision
under [docs/history](docs/history/README.md).

Build the API documentation after installing `epsm_rc[platform,gis,docs]`:

```sh
python -m sphinx -W --keep-going -b html docs outputs/docs
python -m sphinx -W --keep-going -b doctest docs outputs/doctest
```

Run tests with `python -m pytest tests`. The CI base lane verifies Python 3.10
without Core; platform lanes cover Python 3.11 and 3.13 with pinned and development
dependencies. The optional heat-pump lane requires hplib and executes its tests.

The package was previously `SweThermalModel` / `swethermalmodel`. The supported
standalone solver remains available. Deprecated interfaces will be removed no
earlier than **0.9.0**; see [migration details](docs/migration.rst). In particular,
choose `HeatPumpCarnot` or `HeatPumpHPLib` explicitly: the historical `HeatPump()`
factory now warns with its selected model, rejects unknown keywords, and explicit
hplib failures are reported rather than replaced by another calculation.

EPW loading and annual EPW simulation require coordinated `epsm_rc[platform]`
(Python 3.11+) for Core's shared checksum and calendar validation. Ordinary RC
imports and explicitly supplied numerical arrays remain usable without Core.
Weather selection is explicit: `load_epw("weather.epw")`, `load_epw(weather)`
with a captured Core Weather, or `load_epw(preset="goteborg")`. The temperature,
irradiance and monthly helpers accept the same explicit input/preset. Calls with
neither fail; `default_weather` is retired. `DEFAULT_EPW` remains an explicit
path constant. `ParametricBuilding` starts with no selected weather.

Standalone loading and `SolarGainsCalculator.from_epw()` reject incomplete,
duplicated, malformed or missing physical records. They never pad, repeat or
truncate annual data. Captured Weather checksums are verified; cached data is
keyed by content identity and calendar, and arrays are read-only. Valid ordinary
years retain every value and their declared calendar; complete leap calendars
currently raise an explicit unsupported-calendar error. Monthly statistics use
that calendar's month lengths. Date-range helpers reuse selected annual weather
by month/day (including requests across year boundaries); February 29 fails for
ordinary-year input instead of shifting later dates. These stricter input and
dependency requirements replace the previous standalone weather defaults.

`BuildingGeometry` now delegates footprint validation, extrusion and facade
identity to Core. Install the `platform` extra with coordinated Core/Data versions
for authoring; ordinary RC imports still work without Core. Supply explicit
`num_floors` and `floor_height`; WWR starts missing and partial assignments remain
partial. Numeric segment IDs enumerate Core's facade catalogue, whose stable
IDs are available on each segment. Core `window_rectangles` supplies actual
per-storey window areas using the inspectable `window_layout` (scaled by default).
No second window-placement or facade-orientation algorithm remains in RC.

Direct Shapely, GeoJSON and shapefile inputs preserve holes and disconnected
components in `core_geometry`. GIS transforms every ring/component once in one
common coordinate frame. A 400 m² exterior with a 100 m² courtyard retains its
300 m² net area; 100 m² and 50 m² components retain their 150 m² total and relative
positions. Invalid or 3-D footprints fail clearly. OSM relation assembly still
supports one closed way explicitly marked `outer`; inner members, multiple outer
ways or ambiguous roles require a complete assembled polygon supplied through
another entry point. Batches warn and skip unsupported or incomplete features.

`to_core_building()` creates a Core empty draft with only supplied physical
inputs. Missing constructions, glazing, HVAC and operation stay missing.
Transferring selected WWR requires explicit glazing rather than a guessed U-value
or SHGC. `to_core_facade(glazing)` creates the shared facade specification.
Selected geometry values and GIS `height_evidence` transfer to Core. Source
records distinguish inferred heights from caller assumptions and survive snapshot
roundtrips. Geometry edits invalidate the associated source records; the original
wrapper also remains available for source inspection.

```python
from epsm_rc import BuildingGeometry
geometry = BuildingGeometry.from_rectangle(20, 10, num_floors=2, floor_height=3)
draft = geometry.to_core_building()  # Geometry only; inspect missing physical inputs.
geometry.set_wwr_uniform(0.2)  # Explicit illustrative opening assumption.
print(geometry.get_5r1c_params())
```

Legacy `get_eplus_surfaces()`, `generate_idf()` and `generate_polygon_idf()` are
retired with actionable Core migration errors. No independent IDF geometry,
material or schedule emitter remains. Historical `ParametricBuilding` and
`get_5r1c_params()` retain standalone RC conveniences and their documented
assumptions; they are not converted into shared physical inputs.
`examples/10_osm_neighbourhood_simulation.py` is an RC-only neighbourhood demo.

Use `prepare_engine_comparison()` to compile EnergyPlus inputs and calculate RC
demand from one captured, fully authored Core Building:

```python
from epsm_rc import prepare_engine_comparison, RCSettings
comparison = prepare_engine_comparison(
    building, weather="weather.epw", directory="comparison",
    energyplus="/path/to/EnergyPlus-24.2/energyplus",
    rc_settings=RCSettings.from_policy(
        "epsm_rc_5r1c_v1", airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier", self_shadowing_policy="ignore",
    ),
)
```

Here `building` is the caller's complete Core Building with supplied geometry,
fabric, glazing, operation and HVAC; the helper infers none of these inputs.
Core's `EnergyPlusCompiler` discovers the executable version, loads its adjacent
`Energy+.schema.epJSON` and validates emitted epJSON. If the executable argument
is omitted, Core checks `ENERGYPLUS_EXE` then PATH. The caller's Building remains
unchanged. The compilation report and RC result expose the actual consumed
physical specification hash, and `comparison.json` ties these outputs to the
captured snapshot and explicit RC reductions. Output files overwrite sequentially;
a later failure can leave earlier artifacts.

This helper compiles EnergyPlus inputs and runs an annual RC calculation. It does
not run an EnergyPlus annual simulation or establish scientific equivalence or
calibration. `examples/11_core_engine_comparison.py` accepts a Core snapshot,
local EPW, explicit settings JSON and output directory. The historical module
entry point uses the same path:

```shell
python -m epsm_rc.parametric --building building-snapshot.json --weather weather.epw \
  --rc-settings rc-settings.json --directory comparison --energyplus /path/to/energyplus
```

Local integration tests require EnergyPlus 24.2 and exercise rectangle and concave
polygon compilation, supplied materials/glazing/schedules, boundary geometry and
output identity. These two tests explicitly skip if EnergyPlus is absent in CI.

GIS height inputs have no implicit defaults. Omitted `num_floors` is resolved from
`building:levels`, `levels`, `floors` or `num_floors`; omitted `floor_height` is
resolved from `floor_height_m`, `storey_height_m` or `floor_height`, or from total
`height` / `building:height` / `height_m` divided by the selected floor count.
Heights use metres; numeric strings optionally ending in `m` are accepted.
Explicit overrides take precedence, including `num_floors=2`. Missing or invalid
required evidence raises `ValueError`; batches warn and skip the feature. Returned
geometry retains the original source fields and selections in `height_evidence`.
For a footprint without height evidence, select assumptions explicitly:

```python
from epsm_rc.gis_footprint import FootprintImporter
geometry = FootprintImporter.from_geojson(
    "building.geojson", num_floors=2, floor_height=2.5,
)
print(geometry.height_evidence)
```

## Shared Building interface

Core physical boundaries control which elements exchange heat with outdoors or
ground. Shared facade contact is adiabatic under an explicit modelling assumption;
neighbour temperatures are not required. Only the contacting part is removed
from outdoor transmission, opaque solar absorption and sky radiation. Core places
windows on the remaining outdoor area. All installed opaque constructions,
including shared walls and internal elements, retain their depth-limited thermal
storage. See [boundary reduction](docs/boundaries.md) for the input contract and
limitations.

Explicit Core gain partitions now drive the RC sensible heat budget and hourly
convective/radiant split. Equipment latent heat is excluded from that sensible
network; visible lighting energy is deposited as radiant surface heat. Active
return-air heat and engine-calculated occupant sensible heat are rejected before
simulation. Supply an occupant sensible fraction to run those explicit inputs.
The selected legacy RC policy retains its older gain treatment for legacy inputs.
With explicit partitions, the aggregate `convective_internal_gains_fraction` is
reported as inactive, and the effective channels are retained in diagnostics.

GIS `height_evidence` is transferred into Core `SourceRecord.evidence` by
`to_core_building()`. Derived source heights are inferred, caller assumptions are
assumed, and optional caller-supplied provenance takes precedence. Copying and JSON
roundtrips retain that evidence; editing geometry invalidates its source records.

The platform extra requires Python 3.11+ and compatible Data/Core packages.
Install the tested dependency wheels first, then `python -m pip install ".[platform]"`.
Use the tested package combination in the EPSM README for the tested versions and source revisions;
private dependencies need access through your configured index or local checkouts.

```python
from epsm_core import Building, Weather
from epsm_rc import RCSettings, simulate_building

building = Building(preset="illustrative-reference")  # Explicit illustrative inputs.
weather = Weather.from_epw("weather.epw")
settings = RCSettings.from_policy(
    "epsm_rc_5r1c_v1", airflow_policy="sensible_only_unlimited_supply",
    ground_policy="conductance_multiplier",
)
result = simulate_building(building, weather=weather, settings=settings)
```

For automatic LCA and retained combined results, use
`epsm.Building.simulate(engine="epsm_rc", weather=weather, settings=settings)`.
The adapter reads the same physical constructions, schedules and weather as Core.
It uses Core's explicit carrier and fan accounting; thermal demand and delivered
energy remain separate. No construction defaults from the legacy factories are
injected into this path. Detailed material quantities remain in the Building.

### Sequential preparation reuse

For sequential batches, reuse preparation explicitly:

```python
from epsm_rc import PreparationCache, simulate_building

cache = PreparationCache()
with cache.batch():
    results = [
        simulate_building(building, weather=weather, settings=settings, preparation_cache=cache)
        for building in buildings
    ]
```

Each call checks weather bytes, source-file and directory metadata, and numerical
dependency versions. Unchanged source hashes are reused only within the batch;
edits, additions, removals and replacements trigger fresh hashes. This requires a
filesystem that reports edits normally; install packages outside active runs.
Use a separate cache in each worker. Preparation arrays have immutable storage;
thermal states and returned results remain independent. The public
`cache.get_or_compute()` continues returning detached writable copies.
`cache.clear()` releases preparation and source reuse. Batch exit releases source
reuse even on failure, retaining bounded weather/geometry preparation.

### Reduction assumptions

The ordinary 5R1C calculation uses one common temperature domain. A conditioned
basement uses two coupled domains with separate controls; an explicitly selected
cold-attic buffer adds a finite quasi-steady boundary. Each conditioned domain
uses area-weighted thermal conductance and the first 0.1 m of construction from
the room side for effective mass. Explicit `settings` are required; missing
settings fail before input I/O.
`RCSettings.from_policy("epsm_rc_5r1c_v1", ...)` selects the versioned baseline
profile. `RCSettings(...)` also explicitly selects that baseline. Both retain
caller overrides as the effective configuration. The profile identifies model
assumptions, including internal-area and mass-area ratios, capacity thresholds,
internal gain split, air heat capacity, node transfer coefficients, outside film
resistances and sky-exchange coefficients. These assumptions are separate from
physical Building evidence and all adjustable coefficients configure the actual
reduced properties or forcing. Fixed numerical methods and their constants are
also reported. Results expose the complete `effective_configuration` and derived
areas, capacitance, conductances and sky loss in `reduced_properties` diagnostics.
The inherited mass-to-exterior coupling uses a series inverse-conductance rule
and falls back to opaque conductance when its denominator is nonpositive;
configuration declares this method and diagnostics report when it applies.
Missing construction mass cannot receive a fallback building class.
The hourly calendar and EPW checksum are validated. Results retain these settings
and the reduction assumptions. Calculation identity includes inputs, weather,
settings, installed source hashes and numerical dependency versions. Daylight with zero diffuse radiation uses the
exact beam-plus-ground limit of the irradiance calculation.

Airflow support is sensible-only and does not implement heating/cooling supply
temperature limits. The default airflow policy rejects these inputs, including
Core's default 50 °C / 13 °C limits. Select
`airflow_policy="sensible_only_unlimited_supply"` explicitly to acknowledge
unlimited supply temperature and omitted latent recovery. Diagnostics retain
the requested unsupported field names and values; run identity includes this
policy. Sensible recovery, infiltration and availability remain applied.

The adapter does not implement Kiva soil/foundation physics. A Kiva request
requires an explicit approximation: `ground_policy="conductance_multiplier"`
uses `ground_coupling_factor` once, while
`ground_policy="prescribed_monthly_temperature"` requires 12 finite
`monthly_ground_temperatures_c` and a `monthly_ground_temperature_source`.
An already prescribed monthly boundary in the input specification is retained
exactly. Monthly boundaries use full ground conductance and bypass the static
multiplier. Diagnostics record the requested foundation method and assumptions,
effective method, selected boundary/source and effective multiplier. The shared
Core Building currently compiles a Kiva specification; the RC settings route
provides an explicit monthly approximation for that authoring interface.

RC does not calculate geometry-derived self-shadowing. Courtyards, concave
footprints and multiple building parts require explicit
`self_shadowing_policy="ignore"`; the default rejects these geometries before
the solver. Diagnostics identify affected parts and the selected omission, and
run identity includes the policy. One convex part remains supported with the
default policy. Explicit neighbour and facade shading geometry remain unsupported
even when self-shadowing is ignored.

See [roofs and basements](docs/roofs-and-basements.md) for explicit cold-attic buffers and coupled conditioned-basement controls.

Readiness explicitly rejects unsupported geometry, site shading, unconditioned
parts, FVP/PV delivery and detailed zone controls. This is a reduced calculation,
not an equivalence claim with EnergyPlus. The common delivered-energy boundary
currently covers space heating and ventilation fans; DHW/cooling remain separate.

Both the standalone `Building5R1C` and shared Building adapter now apply sensible
recovery once to mechanical conductance, leaving infiltration and window/natural
flow unrecovered. `calc_supply_air_temperature(t_out)` returns the outdoor
**driving temperature of the reduced node**, not the physical post-exchanger
mechanical supply temperature. It no longer warms that driving temperature.
This deliberately changes historical standalone numerical results: recovery no
longer reduces infiltration or applies a second mechanical reduction. For 300 m³,
0.2 ACH infiltration, 0.5 ACH mechanical ventilation, 80% recovery and 20 K,
the loss is 400 W + 200 W = 600 W. With no mechanical flow, changing recovery
leaves infiltration loss and calculated heating demand unchanged. Additional
window flow uses the same coefficient calculation and receives no recovery;
known mechanical recovery remains applied. A total-flow override must include
the declared mechanical share; added flow is unrecovered. The sensible-only reduction
treats occupant activity as sensible gain, uses the ISO 50% convective gain split
and internal area of 4.5 times floor area, and uses a sky-air temperature
difference of 11 K in the baseline reduction profile. These values may be
explicitly overridden in `RCSettings`. They are recorded in diagnostics;
they are not inferred properties of the physical Building.

The standalone factories retain historical defaults and should not be used to
assign evidence in the harmonised Core Building model.

Code includes adaptations credited in the source to DIBS/IWU and
RC_BuildingSimulator/ETH. Retain their original notices. EPSM-owned additions use
MIT; see LICENSE. Local ISO reference extracts are excluded from publication.
Bundled weather inputs retain their source headers.

## Migrating to 0.3.0

Select explicit `RCSettings.from_policy(...)`, weather and physical authoring
inputs. Courtyard, concave and multipart geometry requires an explicit
self-shadowing omission; unsupported airflow and ground behavior need the
corresponding reduction policies. The Core-based geometry and shared gain inputs
require Core 0.3.0+. Historical standalone recovery results must be recalculated.
No scientific equivalence with EnergyPlus is implied by this version change.

## Operation logging

Library calls use Python's standard `logging` module and install no console or
file output. Instrumented operations accept optional keyword-only `log_context`
with small identifier mappings; nested events carry operation and parent IDs.
INFO describes major stages; DEBUG exposes selections, conversions and caches.
Logging context does not change physical models or scientific cache inputs.

For an application using EPSM's coordinated logging setup:

```python
from epsm import configure_logging, reset_logging

configure_logging(directory="outputs/example")  # INFO stderr, DEBUG JSONL
# Run operations here.
configure_logging(enabled=False)  # Stop console and file logging.
reset_logging()  # Restore the host application's settings.
```

EPSM is needed only for this combined configuration, not for ordinary library
logging. Existing scripts can keep using `logging.basicConfig(level=logging.INFO)`.
