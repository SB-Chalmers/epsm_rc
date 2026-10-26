Physical Building API
=====================

These operations receive the same physical Core Building used by the full engine.
Read the assumptions and failure conditions before selecting a reduction policy.

.. autoclass:: epsm_rc.platform.RCSettings
   :members: from_policy, effective_configuration

Check inputs before simulating
------------------------------

``preflight`` reports every unmet requirement at once, as structured rows,
without starting the thermal solver. ``simulate_building`` stops at the first one.

.. autofunction:: epsm_rc.platform.preflight

.. autofunction:: epsm_rc.platform.simulate_building

Outputs
-------

``output_capabilities`` lists the hourly and annual signals, with units, that a
Building and settings will produce; conditional optics and basement outputs
appear only when they apply.

.. autofunction:: epsm_rc.outputs.output_capabilities

.. autoclass:: epsm_rc.preparation.PreparationCache
   :members: batch, clear, get_or_compute

Engine comparison
-----------------

This advanced operation needs an explicit local output directory, complete weather
and EnergyPlus 24.2. It compiles shared physical inputs and runs RC; it does not
run an EnergyPlus year.

.. autoclass:: epsm_rc.comparison.EngineComparison

.. autofunction:: epsm_rc.comparison.prepare_engine_comparison
