# Physical boundaries in the RC reduction

`simulate_building()` consumes the boundary of each Core `BuildingElement`.
Construction roles say what is installed; boundaries say where heat can flow.
An exterior-wall construction may therefore remain installed on a shared facade
without contributing outdoor heat loss.

| Core boundary | RC transmission and radiation | RC thermal storage |
| --- | --- | --- |
| `Outdoors` | Wall, roof and window areas contribute their corresponding outdoor paths; walls and roofs receive solar and sky exchange | Installed opaque construction contributes |
| `Ground` | Ground floor and buried basement walls contribute the explicitly selected ground approximation | Installed opaque construction contributes |
| `Adiabatic` | No outdoor or ground transmission, solar absorption or sky exchange | Installed opaque construction contributes |
| `Internal` | No transmission out of the aggregated conditioned domain | Installed opaque construction contributes once per owned physical element |

Core owns contact geometry, area splitting, material ownership and window
placement. A partial shared wall becomes separate outdoor and adiabatic physical
areas. RC does not infer contacts from proximity or use neighbouring temperatures.
For example, a 15 m² adiabatic patch with U-value 0.5 W/(m² K) removes 7.5 W/K
from outdoor wall transmission before the declared thermal-bridge uplift. The
patch remains in construction thermal capacity. The uplift applies only to the
remaining transmission areas, so it cannot restore heat loss through the shared
patch.

The result's `boundary_reduction` diagnostic records outdoor opaque, outdoor
window, ground, adiabatic opaque and internal opaque areas in square metres. The
effective configuration identifies Core element boundaries as the exposure
method. Physical snapshots and implementation hashes identify the inputs to
ordinary and prepared runs; preparation never retains a thermal state.

The model still uses one conditioned temperature domain and the selected
internal-surface-area factor. Thermal storage keeps the existing single room-side
effective-depth traversal per installed opaque element, including internal walls
and intermediate floors. It does not infer an additional active depth from the
second conditioned face. This reduction assumption is separate from retaining
all installed material in Core quantities. Cold attics and separately controlled conditioned basements use the explicit
[roof and basement reductions](roofs-and-basements.md). It does not calculate
heat exchange with an unconditioned basement or a neighbouring building. An adiabatic neighbour
is an explicit single-building approximation, not evidence that its temperature
equals the simulated building's. Unsupported physical geometry and unconditioned basements
remain subject to Core readiness checks. Shared-wall support does not bypass
those checks or establish agreement with measured energy certificates.
