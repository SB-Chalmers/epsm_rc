Ground heat transfer
====================

RC keeps ground reduction explicit. The default ``ground_policy="reject"``
does not silently replace a physical Kiva foundation with a fixed multiplier.
The existing ``conductance_multiplier`` and ``prescribed_monthly_temperature``
policies retain their original behavior.

ISO 13370 coefficient and monthly approximation
-----------------------------------------------

Select ``ground_policy="iso13370_periodic_approximation"`` with a positive
``iso13370_wall_thickness_m`` and an evidence or assumption string in
``iso13370_wall_thickness_source``. The full wall width at grade is a separate
input from a footing-block width. A value such as 0.30 m must be justified for
the study; RC does not choose it automatically.

The policy derives the characteristic floor dimension from physical floor area
and exposed perimeter. ISO 13370 slab and fully heated basement equations use
material resistance, wall width, burial depth and the declared soil conductivity.
Annual-periodic coefficients also use the declared soil density and specific
heat. RC retains these supplied properties rather than using stock TABULA
U-values or fitting a ground multiplier to another engine's heating demand.

Core's thermal-bridge convention is applied once to construction resistance
before calculating soil heat transfer. Layer depths and heat capacity remain
unchanged. The generic material uplift does not represent a resolved wall/floor
junction; no additional linear junction correction is assumed.

The monthly reference heat loss is calculated as

.. math::

   \Phi_m = H_g(\bar\theta_i-\bar\theta_e)
            + H_{pi}(\theta_{i,m}-\bar\theta_i)
            - H_{pe}(\theta_{e,m}-\bar\theta_e).

Monthly outdoor means come from the captured EPW. Indoor reference means come
from the domain's heating-setpoint schedule. Actual calendar-month hours weight
both annual means, including February in leap years. Phase offsets are omitted.
An equivalent monthly environment is then obtained from
``theta_equivalent = theta_indoor_reference - Phi / H_g``.

This environment is not an observed soil or outside-slab temperature. At the
declared indoor reference it reproduces the monthly heat-flow equation. The
hourly 5R1C network still uses its equivalent central node and instantaneous
coupling to that boundary. Indoor changes, free-floating summer temperatures,
intermittent controls, surface exchange and ground thermal history are therefore
approximated. This policy does not implement ISO 13370's detailed dynamic
simulation provisions, and matching annual totals does not validate hourly loads.

Existing explicit Core monthly ground temperatures take precedence. They keep
raw assembly conductance and bypass this coefficient reduction, so ground
resistance is not counted twice. Results report the effective method and inputs
under ``diagnostics["ground_reduction"]``. Coupled main/basement results include
separate domain ground reports; only the domain in contact with soil receives
the foundation conductance and forcing.

The adapter restricts this policy to the foundation topologies it can resolve
explicitly. Unsupported edge, footing, mixed-foundation or unconditioned-space
conditions raise a diagnostic rather than silently selecting another method.
The standalone coefficient functions in ``epsm_rc.ground`` accept explicit
dimensions and resistances independently of Core.

Verification
------------

Tests compare slab and basement coefficients with published government reference
tables, check limiting cases and the weighted annual energy identity, verify
Core resistance/geometry handoff and unchanged thermal storage, and compare
native with Python execution. Legacy prescribed-temperature and fixed-multiplier
tests remain separate. Inter-engine studies retain their EnergyPlus baseline
and declare the new method's assumptions.

References
----------

* `ISO 13370:2017 <https://www.iso.org/standard/65716.html>`_, slab and heated
  basement coefficients, annual-periodic and monthly methods.
* `UK Home Energy Model technical documentation
  <https://www.gov.uk/government/publications/home-energy-model-technical-documentation>`_.
* `ISO 13790:2008 <https://www.iso.org/standard/41974.html>`_, the reduced zone
  network; the ground coefficient alone does not establish its prediction accuracy.
