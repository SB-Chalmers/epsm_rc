# EPSM RC

Reduced-order thermal calculation using the existing ISO 13790 5R1C implementation.
The package and import are now `epsm_rc` (previously `SweThermalModel` /
`swethermalmodel`). Numerical solver behavior was preserved during the rename.

```python
from epsm_rc import create_swedish_villa
villa = create_swedish_villa(floor_area_m2=150, era="1976_1990")
result = villa.simulate_with_epw("weather.epw")
print(result["heating_demand_kwh"])
```

Install with `python -m pip install .`; tests with `python -m pytest tests`.

Legacy `BuildingGeometry.from_shapely` and GIS footprint imports support polygons
without holes. Interior rings (courtyards) raise `ValueError`, because this geometry
cannot represent their floor, roof and wall surfaces correctly. Batch GIS imports
skip these unsupported features with a warning. For example, a 400 m² exterior
with a 100 m² courtyard is rejected rather than reported as a 400 m² footprint.
OSM relations containing members with an `inner` role are also rejected.
Disconnected `MultiPolygon` inputs are rejected before coordinate transformation;
no component is selected or discarded. Batch imports warn and skip unsupported
features. OSM relations support one closed way explicitly marked `outer`;
multiple outer ways or ambiguous member roles require an unsupported assembly
step and are rejected.

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

The platform extra requires Python 3.11+ and the coordinated Data/Core versions
on `codex/harmonisation`. Install these repositories first, then this package.

```python
from epsm_core import Building, Weather
from epsm_rc import RCSettings, simulate_building

building = Building()
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

### Reduction assumptions

The 5R1C calculation uses one common temperature domain, area-weighted thermal
conductance and the first 0.1 m of construction from the room side for effective
mass. Explicit `settings` are required; missing settings fail before input I/O.
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

Readiness explicitly rejects unsupported geometry, site shading, unconditioned
parts, FVP/PV delivery and detailed zone controls. This is a reduced calculation,
not an equivalence claim with EnergyPlus. The common delivered-energy boundary
currently covers space heating and ventilation fans; DHW/cooling remain separate.

The platform adapter corrects a legacy heat-recovery double count: mechanical
conductance is reduced once, with infiltration left unrecovered. Standalone
`Building5R1C` behavior is retained for compatibility. The sensible-only reduction
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
