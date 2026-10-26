Compatibility and migration
===========================

RC retains supported standalone and advanced public APIs. The physical
Building route is recommended for new building studies; it preserves input
ownership and the common full/reduced engine boundary.

The following interfaces remain during migration. Their removal will be no
earlier than 0.11.0, with a release note describing the replacement:

* ``create_swedish_apartment`` retains historical illustrative era assumptions.
  Supply Data constructions and explicit Core geometry for new building studies.
* ``HeatPump`` retains installation-dependent selection and now warns with the
  selected concrete model. Choose ``HeatPumpCarnot`` or ``HeatPumpHPLib`` explicitly.
* ``use_config`` in the Swedish factories and ``thermal_mass_class`` in historical
  parametric authoring are compatibility inputs. Prefer explicit numerical inputs.
* ``generate_idf``, ``generate_polygon_idf`` and ``get_eplus_surfaces`` are already
  retired migration errors. Use Core's ``EnergyPlusCompiler`` or
  ``prepare_engine_comparison`` with fully authored physical inputs.
* ``default_weather`` is a retired migration error. Select a local EPW, captured
  Core Weather or ``load_epw(preset="goteborg")`` explicitly.

``ParametricBuilding`` and ``get_5r1c_params`` remain historical standalone
conveniences. They do not infer a physical Core Building. Existing import names
continue to resolve; advanced imports are loaded only when requested.

``HeatPump`` rejects unknown keyword arguments. Failed explicit hplib model or
design calculations now raise actionable errors instead of choosing a generic
model or estimating half the reference capacity. Treat that behavior as a
scientific failure correction, not numerical equivalence with the fallback.

The first ``Building5R1C.simulate_with_epw`` call now rebuilds ground conductance
when it selects the existing monthly defaults. Its demand matches repeated runs
and runs with those same defaults supplied explicitly. Earlier first runs could
combine monthly ground forcing with conductance prepared for the static boundary;
affected standalone results intentionally change.

Historical calculation reports and the BESTEST script are retained under
``docs/history`` with their source revision. Those records are not current release
validation or certification. Bundled weather files remain distributed resources;
no station is removed in this release.

Preparation and source identity
-------------------------------

Simulation creates one owned preparation per Building and independent state for
each conditioned domain. Source and dependency identity is captured lazily once
per Python process. Restart after editing installed source or dependencies.
``PreparationCache.batch()`` remains accepted as a compatibility context manager;
it does not monitor source files or change cache lifetime.

Weather cache ownership
-----------------------

``load_epw`` returns a detached wrapper on every call, sharing only immutable
hourly buffers. Code must compare recorded content fingerprints rather than
Python object identity. Reassigning an attribute on one wrapper cannot affect
another load. Calendar validation and parsing use the same captured weather bytes.
