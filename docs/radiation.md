# Explicit radiation policies and preparation reuse

RC 0.5.0 requires Core >=0.5.0 and Data >=0.6.0 for its `platform` extra.
Ordinary `Building()` remains empty. Supply physical constructions, glazing,
operating inputs and validated Weather, or explicitly select a documented
illustrative fixture. The following example assumes a complete `building` and
an explicit `weather` already exist:

```python
from epsm_rc import PreparationCache, RCSettings, simulate_building

cache = PreparationCache(max_entries=64)
legacy = RCSettings.from_policy(
    "epsm_rc_5r1c_v1",
    airflow_policy="sensible_only_unlimited_supply",
    ground_policy="conductance_multiplier",
)
angular = RCSettings.from_policy(
    "epsm_rc_5r1c_v2",
    airflow_policy="sensible_only_unlimited_supply",
    ground_policy="conductance_multiplier",
)
old_result = simulate_building(building, weather=weather, settings=legacy,
                              preparation_cache=cache)
new_result = simulate_building(building, weather=weather, settings=angular,
                              preparation_cache=cache)
cache.clear()
```

The selected airflow and ground policies are explicit approximations in this
example, not measurements. A supplied monthly ground boundary is preserved.
Self-shadowing is rejected by default; selecting `ignore` records its omission.
Explicit context, overhang and reveal shading remain unsupported.

## Reproducibility and migration

`epsm_rc_5r1c_v1` retains hour-start solar sampling, constant supplied SHGC and
all existing thermal results. Select it when reproducing saved calculations.
Optimization does not change timestep, control interpolation, constructions,
load reference area or physical assumptions. Standalone solar entry points keep
their prior midpoint convention, cardinal interface and legacy sum ordering.
They delegate their common irradiance operations to the same radiation kernel.

`epsm_rc_5r1c_v2` explicitly selects:

- `solar_time_reference="interval_midpoint"`: solar position at interval start
  plus 30 minutes, retaining the original hour-start result labels.
- `window_optics_method="simple_glazing_angular_v1"`: separate beam and diffuse
  optical response, including the nominal inward-flowing fraction of absorbed
  solar heat. A constant SHGC is not a solar transmittance.

The fields may also be overridden explicitly on either baseline policy. Effective
settings and source identities enter diagnostics and calculation identity.
Old settings dictionaries without these fields keep v1 behavior; selecting a
new policy never edits a retained snapshot. The physical Building's supplied
U-value and SHGC remain unchanged. This release does not add hidden physical
defaults, infer measured optical data or tune coefficients to match annual demand.

## Angular model and its limits

The equivalent-layer normal transmittance, reflectance and nominal inward fraction
follow EnergyPlus version 24.2's simple glazing implementation. Angular properties
use its 28-cell U/SHGC map and A-J polynomial curves, sampled at 0–90 degrees in
10-degree increments. Between samples RC uses linear interpolation. At normal
incidence, transmitted plus inward absorbed heat reproduces the supplied SHGC.

The implementation intentionally follows the **actual v24.2.0 summer-film
interpolation** in `Material.cc`, including its reversed difference between
3.4 and 4.5 W/(m² K). That source behavior is recorded rather than silently
replacing it with a different correlation. Source reference tests use independently
compiled original C++ routines, not the new Python implementation.

Diffuse transmittance and reflectance are exact hemispherical integrals of RC's
piecewise-linear angular model, weighted by `sin(2 theta)`. Absorptance is the
remaining fraction. This conserves optical energy; it differs from EnergyPlus's
coarse diffuse quadrature and subsequent curve fits. Both Perez sky diffuse and
ground-reflected diffuse use this isotropic optical average; detailed directional
sky radiance is not resolved.

Nominal summer inward absorption stays fixed as weather changes. RC does not
solve the temperature-dependent glazing heat balance, detailed frames or interior
solar distribution. The approximation applies to specular simple glazing, with
finite `0 < U <= 7` and `0 <= SHGC <= 1`, positive nominal resistances and physical
correlated optical fractions. Unphysical combinations fail before the thermal
simulation; no alternative coefficients or model are silently substituted.

New-policy hourly diagnostics expose incident window beam/sky/ground radiation,
transmitted solar, inward absorbed solar, total window gains, opaque solar gains
and the unchanged fixed sky loss, all in W. Annual component integrals are retained
in diagnostics in kWh. These channels identify different physical quantities:
EnergyPlus transmitted solar alone must not be compared to RC total SHGC heat gain.
Dynamic sky longwave radiation and geometric shading are separate future work.

## Cache ownership

`PreparationCache` is optional and owned by one sequential caller. Geometry,
validated weather, solar position and orientation irradiance have bounded LRU
storage. Each run has a new thermal model. Returns are detached from retained
preparation; exceptions and failed calculations are not cached.

Every call rechecks weather bytes and installed implementation contents. Keys
include weather, calendar, climate, solar timing, albedo and surface orientation.
Core's geometry cache independently verifies its source and geometry controls.
Material and schedule assignments remain current. Cache use does not change
scientific identity or results. Source fingerprinting therefore remains a cost
even in warm runs. `clear()` releases entries and resets counters.

Use `preparation_cache` for this operation: the umbrella `epsm.Building.simulate`
already uses `cache` for retained evaluation results. Worker scheduling and durable
result caching remain outside RC. See [benchmarks](../benchmarks/README.md) for
installed-wheel regressions and separate cold/repeated/warm measurements.

## Sources and terms

- [EnergyPlus 24.2 Simple Window Model](https://bigladdersoftware.com/epx/docs/24-2/engineering-reference/window-calculation-module.html#simple-window-model)
- [EnergyPlus v24.2.0 Material.cc](https://github.com/NatLabRockies/EnergyPlus/blob/v24.2.0/src/EnergyPlus/Material.cc)
- [EnergyPlus v24.2.0 WindowManager.cc](https://github.com/NatLabRockies/EnergyPlus/blob/v24.2.0/src/EnergyPlus/WindowManager.cc)
- [pvlib interval-average modeling](https://pvlib-python.readthedocs.io/en/stable/gallery/irradiance-transposition/plot_interval_transposition_error.html)

Correlation source copyright and the complete EnergyPlus terms are included in
`epsm_rc/notices/EnergyPlus.txt` in both wheel and source distributions. EPSM-owned
code remains MIT. Using these correlations does not establish measured-building
validation or imply endorsement by the EnergyPlus contributors.
