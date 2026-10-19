# Cold attics and conditioned basements

Core owns physical geometry, construction assignments and zone operation. RC
reduces those explicit inputs; it does not infer a basement depth, heating
setpoint, attic ventilation rate, roof construction or ground temperature.

A conditioned basement is represented by a second 5R1C domain coupled to the main
space. This preserves declared basement operation, including an 18 °C basement
below a 20 °C main programme. An unconditioned basement remains unsupported.

```python
from epsm_core import (
    Building, Footprint, GeometrySpec, RoofSpec, ScheduleSpec,
    UnconditionedExchangeSpec, ZoneUseSpec,
)
from epsm_rc import RCSettings

building = Building(
    preset="illustrative-reference",
    geometry=GeometrySpec.from_footprint(
        Footprint.rectangle(10, 8), storeys=3, storey_height_m=3,
        roof=RoofSpec(shape="gable", pitch_deg=30),
        basement_depth_m=2.5, basement_conditioned=True,
    ),
    unconditioned_exchange=(UnconditionedExchangeSpec(
        zone_kind="attic", outdoor_air_ach=0.5,
        source="declared illustrative attic exchange",
    ),),
)
building.operation = building.operation.updated(
    heating_setpoint=ScheduleSpec.constant(20),
)
building.zone_uses = (ZoneUseSpec(
    zone_kind="basement",
    use=building.operation.updated(heating_setpoint=ScheduleSpec.constant(18)),
    source="declared illustrative basement control",
),)
settings = RCSettings.from_policy(
    "epsm_rc_5r1c_v2",
    airflow_policy="sensible_only_unlimited_supply",
    ground_policy="conductance_multiplier",
    cold_attic_policy="quasi_steady_buffer",
)
```

The named preset and numerical values are illustrative assumptions. The Core
snapshot retains the same explicit controls and attic exchange when compiled for
EnergyPlus. `examples/14_roofs_and_basements.ipynb` runs this public interface with
packaged weather and inspects the domain diagnostics.

## Conditioned basement reduction

Core's resolved zone inputs select the use, airflow, conditioned volume and load
reference area for each thermal zone. RC aggregates equal main-space programmes
into one domain and equal basement programmes into another. Different controls
within either group, apartment/common-space programmes and a whole-building zone
that merges both physical regions are rejected explicitly. Per-storey or reviewed
perimeter/core zoning can retain both regions. Exact-zone or kind assignments
remain supported when their selected programmes agree within each group.

Each domain has independent thermal storage, schedules, sensible internal gains,
airflow, ground exchange and ideal unlimited heating/cooling control. The initial
mass/surface/air temperatures equal that domain's first-hour heating setpoint;
there is no additional warm-up year. Full hourly calendars, including leap days,
are retained. Ideal-load availability is applied separately to each domain.
A finite shared HVAC capacity is not supplied by the supported Core interface;
FVP, PV and detailed HVAC remain unsupported.

The separating floor provides conductance
`H12 = area / (sum(layer R) + surface R)`. The declared combined room-side
surface resistance is `interzone_floor_surface_resistance_m2_k_w`, initially
0.27 m² K/W (0.17 + 0.10). This is an explicit reduction assumption. Internal
floor transmission receives no exterior thermal-bridge uplift. Heat entering
one air node is exactly the heat leaving the other.

For each hour, independent zero-load and reference-load trials determine each
5R1C model's linear response. The coupled air-node equations and heating/cooling
active sets are solved together. Trials share the previous state; only the final
coupled solution advances the two storage states. The existing single-domain
calculation remains in place for buildings without basements.

The separating floor remains one physical installed element. Its existing
one-room-side, depth-limited active capacity is divided equally between the two
models, conserving the prior active-capacity total. This is a lumped storage
approximation, not a detailed transient two-face floor model. Basement slab and
walls belong to the basement domain. Both slab and buried walls receive the
selected ground approximation once; the internal ceiling does not become another
ground-exposed slab.

Delivered heat and fan electricity use each domain's selected reference area and
airflow. Their hourly energies are summed. The top-level indoor temperature is a
volume-weighted reporting value; separate `main_indoor_temperature_c` and
`basement_indoor_temperature_c` channels retain the controls. Interzone exchange
is reported in watts and is not added to purchased energy or heating demand.

## Finite cold-attic reduction

Cold attics require explicit `cold_attic_policy="quasi_steady_buffer"` and a
sourced `UnconditionedExchangeSpec(zone_kind="attic", ...)`. An explicit zero rate
means a sealed scenario. A missing declaration stays missing. The current buffer
supports constant exchange fractions; varying exchange schedules are rejected.

Core's public resolved surfaces supply the true roof/gable areas, orientations,
construction owners and attic volume. The attic's steady air balance is

```
Hc (Tc - Ta) + (Hs + Hv) (To - Ta) + Qshell = 0
Heq = Hc (Hs + Hv) / (Hc + Hs + Hv)
f = Hc / (Hc + Hs + Hv)
```

`Hc` is ceiling conductance, `Hs` is outdoor shell conductance, and `Hv` is the
explicit attic air-exchange conductance. Ceiling loss becomes
`Heq (Tc - To) - f Qshell`. A sealed attic retains finite shell resistance;
very large ventilation tends towards the ceiling's outdoor-temperature limit.
No infinite ventilation is assumed.

Actual sloped shell orientations receive the selected Perez solar and sky
exchange. Their sol-air forcing reaches the conditioned domain through `f`;
the internal ceiling does not also receive fictitious flat-roof solar/sky forcing.
Adiabatic shell portions receive no outdoor forcing. Ceiling storage is retained
on the conditioned side. Attic air, roof and gable storage are explicitly omitted
by this quasi-steady buffer. This approximation can differ from EnergyPlus's
transient attic, especially in summer or for heavy roof construction.

The result records `attic_reduction`, `domain_reduction`, per-domain diagnostics,
physical input hashes and the selected policy. Software balance tests and real
EnergyPlus acceptance cases check conservation and shared input handling. They do
not establish accuracy against a measured building's EPC.
