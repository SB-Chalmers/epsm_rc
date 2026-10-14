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

## Shared Building interface

The platform extra requires Python 3.11+ and the coordinated Data/Core versions
on `codex/harmonisation`. Install these repositories first, then this package.

```python
from epsm_core import Building, Weather
from epsm_rc import RCSettings, simulate_building

building = Building()
weather = Weather.from_epw("weather.epw")
result = simulate_building(building, weather=weather, settings=RCSettings())
```

For automatic LCA and retained combined results, use
`epsm.Building.simulate(engine="epsm_rc", weather=weather)`.
The adapter reads the same physical constructions, schedules and weather as Core.
It uses Core's explicit carrier and fan accounting; thermal demand and delivered
energy remain separate. No construction defaults from the legacy factories are
injected into this path. Detailed material quantities remain in the Building.

### Reduction assumptions

The 5R1C calculation uses one common temperature domain, area-weighted thermal
conductance and the first 0.1 m of construction from the room side for effective
mass. `RCSettings` exposes this depth, ground coupling and ground reflectance.
The hourly calendar and EPW checksum are validated. Results retain these settings
and the reduction assumptions. Calculation identity includes inputs, weather,
settings, installed source hashes and numerical dependency versions. Daylight with zero diffuse radiation uses the
exact beam-plus-ground limit of the irradiance calculation.

Readiness explicitly rejects unsupported geometry, site shading, unconditioned
parts, FVP/PV delivery and detailed zone controls. This is a reduced calculation,
not an equivalence claim with EnergyPlus. The common delivered-energy boundary
currently covers space heating and ventilation fans; DHW/cooling remain separate.

The platform adapter corrects a legacy heat-recovery double count: mechanical
conductance is reduced once, with infiltration left unrecovered. Standalone
`Building5R1C` behavior is retained for compatibility. The sensible-only reduction
treats occupant activity as sensible gain, uses the ISO 50% convective gain split
and internal area of 4.5 times floor area, and retains the solver's fixed sky-air
temperature difference of 11 K. These assumptions are recorded in diagnostics;
they are not inferred properties of the physical Building.

The standalone factories retain historical defaults and should not be used to
assign evidence in the harmonised Core Building model.

Code includes adaptations credited in the source to DIBS/IWU and
RC_BuildingSimulator/ETH. Retain their original notices. EPSM-owned additions use
MIT; see LICENSE. Local ISO reference extracts are excluded from publication.
Bundled weather inputs retain their source headers.
