Standalone numerical experiments
================================

The base installation supports explicit numerical arrays without Core. It performs
no catalogue lookup or weather acquisition. ``examples/00_standalone.py`` runs a
small synthetic calculation offline; it is a software example, not validation
against a real building.

.. doctest::

   >>> import numpy as np
   >>> from epsm_rc import Building5R1C
   >>> building = Building5R1C(floor_area_m2=100)
   >>> result = building.simulate(np.full(24, 5.0))
   >>> len(result["heating_demand"])
   24

.. autoclass:: epsm_rc.iso13790.Building5R1C
   :members: simulate, reset, summary

.. autoclass:: epsm_rc.iso13790.ConstructionLayer

.. autoclass:: epsm_rc.iso13790.Construction

Solar and weather
-----------------

EPW reading requires the coordinated ``platform`` extra for Core's weather checks.
A local path, captured Weather or explicitly selected ``goteborg`` preset is
required. No weather is selected at import.

.. autoclass:: epsm_rc.solar.SolarGainsCalculator

.. autofunction:: epsm_rc.weather.load_epw

.. autofunction:: epsm_rc.weather.get_temperature

.. autofunction:: epsm_rc.weather.get_irradiance

Explicit heating models
-----------------------

Choose the model explicitly for repeatable calculations. ``HeatPumpHPLib`` requires
``epsm_rc[heatpump]`` and reports failed model loading or design calculations.
It does not replace a failed requested model with another heat pump.

.. autoclass:: epsm_rc.heating.HeatPumpCarnot
   :members: cop, calculate_load

.. autoclass:: epsm_rc.heating.HeatPumpHPLib
   :members: simulate_hplib, calculate_load

.. autoclass:: epsm_rc.heating.DistrictHeating

.. autoclass:: epsm_rc.heating.DirectElectric

.. autoclass:: epsm_rc.heating.WoodPelletHybrid

Seasonal and geometry conveniences
----------------------------------

TABULA is a separate seasonal method with its own explicit assumptions; it is not
an hourly Core Building adapter. The geometry wrapper delegates physical geometry
to Core. GIS file/service readers require their corresponding extras.

.. autoclass:: epsm_rc.tabula.BuildingTABULA
   :members: calc_annual_heating_demand

.. autoclass:: epsm_rc.tabula.ClimateData

.. autoclass:: epsm_rc.building_geometry.BuildingGeometry
   :members: from_polygon, from_rectangle, from_shapely, to_core_building

.. autoclass:: epsm_rc.gis_footprint.FootprintImporter
   :members: from_geojson, from_osm, from_shapefile
