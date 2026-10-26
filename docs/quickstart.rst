Calculate a physical Building
=============================

Install ``epsm_rc[platform]`` with Core ``>=0.8.2,<0.9`` and Data 0.6.x. Supply a complete
local hourly EPW explicitly. The named Building preset below is an illustrative
set of assumptions; it is not evidence about a real building.

.. code-block:: python

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

Each call captures its physical inputs and creates independent thermal state.
The result retains hourly values, units, diagnostics and input identities. Supply
``directory=...`` only when an hourly artifact should be written; the documented
``hourly.npz`` in that directory is replaced.

For a slab or fully heated basement with explicit foundation geometry and soil
properties, an alternative to the fixed multiplier is available:

.. code-block:: python

   settings = RCSettings.from_policy(
       "epsm_rc_5r1c_v2",
       airflow_policy="sensible_only_unlimited_supply",
       ground_policy="iso13370_periodic_approximation",
       iso13370_wall_thickness_m=0.30,
       iso13370_wall_thickness_source="Explicit illustrative wall-width assumption",
   )

The wall width above is an illustrative assumption, not a universal default.
This policy uses ISO 13370 ground coefficients and monthly annual-periodic
forcing; it remains an approximation when coupled to the hourly 5R1C network.
See :doc:`ground` for supported foundations, boundary definitions and limits.

Repeated independent runs
-------------------------

``PreparationCache`` belongs to the sequential caller. It reuses verified
preparation while each run gets new thermal state. Use one cache per worker.

.. code-block:: python

   from epsm_rc import PreparationCache

   cache = PreparationCache()
   with cache.batch():
       results = [
           simulate_building(
               building, weather=weather, settings=settings, preparation_cache=cache
           )
           for _ in range(2)
       ]

Keep the chosen settings and assumptions with study results. The v1 policy
preserves the historical hour-start solar sampling; v2 explicitly selects
midpoint sampling and angular glazing. Both support context buildings through the
ISO 13790 hourly average-horizon approximation. Diffuse radiation is unchanged;
facade shading details remain unsupported. See the repository's
``docs/radiation.md`` for scope.
