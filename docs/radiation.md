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
Context buildings use the ISO 13790 hourly horizon approximation described below.
Overhang, reveal and balcony shading remain unsupported.

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
Dynamic sky longwave radiation and detailed shadow projection remain future work.

## Context-building shading

Core `building.context.buildings` is accepted by the RC adapter. The default
`context_shading_method="iso13790_hourly_horizon"` implements ISO 13790:2008
11.4.4 and the informative Annex G.5.4.2 hourly horizon method, equation G.3:
below the average obstruction angle, remove incident beam radiation; otherwise
retain it. Sky-diffuse and ground-reflected radiation remain unchanged. This is
the ISO horizon simplification, not a calculation of obstructed sky radiation or
reflections between neighbours. The rule is applied before either constant-SHGC
or angular window optics, and before opaque solar absorption, including cold
attic weather shells. Fixed longwave sky exchange is unchanged.

The average obstruction angle is calculated at one area-weighted representative
point per physical outdoor element, using Core's resolved faces and panes.
Facade and window identities retain storey elevation. Neighbour footprints,
including holes and disconnected components, are translated and rotated into
the same frame as the target; roof heights include the supplied base elevation.
Overlapping angular obstructions use the maximum elevation in each direction,
so duplicate obstacles do not double-count shading. Neighbours do not become
thermal zones and their temperatures are not required. A taller neighbour whose
outline passes through a receiver's representative point is rejected: that point
would see a 90-degree horizon in every direction and remove all beam radiation
from the whole receiver, however small the contact. Model such contact as a shared
boundary or separate the footprints. Contact that does not touch the point, such
as an aligned terrace neighbour, is retained.

The numerical integration uses equally spaced azimuth midpoints across the
180-degree horizon facing a facade. The default maximum spacing is 10 degrees;
`horizon_azimuth_step_deg=5` refines it. The actual spacing divides the integration
span evenly and does not exceed the requested spacing (1 to 30 degrees).
The sampling grid rotates with the scene. Representative-point
sampling and quadrature spacing are implementation approximations, not values
prescribed by ISO. Horizontal roof receivers use a full 360-degree average;
inclined attic faces use their front azimuth half-plane. These roof extensions
are reported separately from the standard's facade horizon rule. Partial-window
shadows, narrow obstacles between sampled directions, self-shadowing and facade
shading details are not resolved by this method. National provisions may require
different inputs or methods; implementing this clause does not establish
whole-standard or regulatory certification.

Use the existing explicit settings alongside the context method:

```python
settings = RCSettings.from_policy(
    "epsm_rc_5r1c_v2",
    airflow_policy="sensible_only_unlimited_supply",
    ground_policy="conductance_multiplier",
    context_shading_method="iso13790_hourly_horizon",
    horizon_azimuth_step_deg=10,
)
result = simulate_building(building, weather=weather, settings=settings,
                           preparation_cache=cache)
print(result.diagnostics["context_shading"])
```

Diagnostics retain the standard reference, obstacle count, sampling assumptions
and each receiver's horizon angle. Context, receiving geometry, numerical settings
and installed implementation identify the cached angles; weather, constructions
and operation do not. Geometry preparation is reused across those variants.
Only vectorized hourly comparisons are needed after preparation. Changes to
context heights/position or receiving geometry invalidate the angles. Settings
and physical context also enter run identity. Empty context bypasses shading
without changing the existing solar arithmetic. Select
`context_shading_method="reject"` to explicitly refuse context inputs.

Core 0.8.1 removes its former blanket RC-context readiness rejection. With older
supported Core releases, the RC adapter removes only that exact obsolete
restriction before invoking the generic physical handoff. All other missing and
unsupported readiness checks remain required. This keeps the central pinned
Core combination usable without silently selecting a different dependency.
`self_shadowing_policy` remains a separate decision for the target's concave,
courtyard or multipart geometry.

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

Use `preparation_cache` on RC's direct `simulate_building` API. The existing
umbrella `epsm.Building.simulate` serializes all calculation options and cannot
accept this transient cache object; its `cache` argument is for retained evaluation
results. Ordinary umbrella calls with explicit v1/v2 RC settings remain supported.
Exposing preparation reuse there requires a separate EPSM wrapper change; it is
outside this Core/RC release. Worker scheduling and durable result caching remain
outside RC. See [benchmarks](../benchmarks/README.md) for
installed-wheel regressions and separate cold/repeated/warm measurements.

## Sources and terms

- [EnergyPlus 24.2 Simple Window Model](https://bigladdersoftware.com/epx/docs/24-2/engineering-reference/window-calculation-module.html#simple-window-model)
- [EnergyPlus v24.2.0 Material.cc](https://github.com/NatLabRockies/EnergyPlus/blob/v24.2.0/src/EnergyPlus/Material.cc)
- [EnergyPlus v24.2.0 WindowManager.cc](https://github.com/NatLabRockies/EnergyPlus/blob/v24.2.0/src/EnergyPlus/WindowManager.cc)
- [pvlib interval-average modeling](https://pvlib-python.readthedocs.io/en/stable/gallery/irradiance-transposition/plot_interval_transposition_error.html)
- [ISO 13790:2008](https://www.iso.org/standard/41974.html), 11.4.4 and Annex G.5.4.2

Correlation source copyright and the complete EnergyPlus terms are included in
`epsm_rc/notices/EnergyPlus.txt` in both wheel and source distributions. EPSM-owned
code remains MIT. Using these correlations does not establish measured-building
validation or imply endorsement by the EnergyPlus contributors.
