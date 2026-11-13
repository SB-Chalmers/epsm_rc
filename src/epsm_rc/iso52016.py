# File: iso52016.py
# Purpose: ISO 52016-1:2017 hourly zone network (Clause 6.5) with element node chains.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-12

"""ISO 52016-1 hourly multi-zone model with element node chains.

Notes
-----
Each construction element is a chain of nodes: five for opaque elements and two
for windows (Clause 6.5.7). Every thermal zone has one air node holding air and
furniture capacity. Inside surfaces exchange long-wave radiation through an
area-weighted mean radiant temperature of their own zone, and exterior nodes
receive absorbed solar and the sky term.

Several zones form one network (Clause 6.5.5.1 multi-zone method). An element
between two zones, such as the floor over a conditioned basement or the
ceiling under a cold attic, has its inner surface node in one zone and its
outer surface node in the other. Unconditioned zones have no HVAC and float.

Each hour is solved implicitly (backward Euler). The network is linear, so the
free-floating state and the response to unit HVAC power in each zone give the
exact thermostat powers directly; for one zone this is the linear
interpolation of Clause 6.5.5.2. Coupled thermostats are solved by finding the
consistent active set (free, at a setpoint, or at a capacity limit per zone).
The matrix changes hourly only through the ventilation conductance on each air
node. It is factorised once and updated with the Woodbury identity, which is
Sherman-Morrison for a single zone.

Conventional values follow Table 25 (surface coefficients) and Annex B (gain
fractions, internal capacity). Interpretations of ambiguous text are listed in
``docs/iso52016.md``. This module performs no I/O.

Examples
--------
>>> import numpy as np
>>> wall = ZoneElement("wall", 10.0, "opaque", "exterior", "horizontal", 2.0, 50_000.0, "D")
>>> zone = Zone52016((wall,), floor_area_m2=10.0, air_volume_m3=25.0)
>>> zone.steady_conductance_w_k() > 0
True

Two zones joined by a floor; the lower zone is unconditioned:

>>> floor = ZoneElement("floor", 50.0, "opaque", "zone", "downwards", 1.0, 1e5, "D",
...                     zone=0, adjacent_zone=1, adjacent_direction="downwards")
>>> roof = ZoneElement("roof", 50.0, "opaque", "exterior", "upwards", 4.0, 1e5, "D")
>>> wall = ZoneElement("wall", 40.0, "opaque", "exterior", "horizontal", 2.0, 1e5, "D", zone=1)
>>> network = Network52016((floor, roof, wall),
...     (ThermalZone("room", 50.0, 125.0), ThermalZone("crawl", 50.0, 50.0, conditioned=False)))
>>> network.zone_count
2
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Table 25 conventional surface heat-transfer coefficients, W/(m² K).
H_CI = {"upwards": 5.0, "horizontal": 2.5, "downwards": 0.7}
H_RI = 5.13
H_CE = 20.0
H_RE = 4.14
# Surface resistances consistent with Table 25 (ISO 6946 conventions), m² K/W.
R_SI = {"upwards": 0.10, "horizontal": 0.13, "downwards": 0.17}
R_SE = 0.04
# Annex B informative defaults.
F_INT_C = 0.4
F_SOL_C = 0.1
F_HC_C = 1.0
DELTA_THETA_ER_K = 11.0
INTERNAL_CAPACITY_J_M2_K = 10_000.0
# Clause 6.5.8 soil layer below ground floors: 0.5 m, λ = 2.0 W/(m K), ρc = 2.0 MJ/(m³ K).
SOIL_THICKNESS_M = 0.5
SOIL_CONDUCTIVITY_W_M_K = 2.0
SOIL_VOLUMETRIC_CAPACITY_J_M3_K = 2.0e6
MASS_CLASSES = frozenset({"I", "E", "IE", "D", "M"})
DIRECTIONS = frozenset(H_CI)
BOUNDARIES = frozenset({"exterior", "ground", "adiabatic", "zone"})
TIMESTEP_S = 3600.0


def mass_distribution(mass_class: str, kappa_m: float) -> np.ndarray:
    """Clause 6.5.7.2 nodal areal capacities (outside node first) for five-node elements."""
    if mass_class == "I":
        return np.array([0, 0, 0, 0, kappa_m], float)
    if mass_class == "E":
        return np.array([kappa_m, 0, 0, 0, 0], float)
    if mass_class == "IE":
        return np.array([kappa_m / 2, 0, 0, 0, kappa_m / 2], float)
    if mass_class == "D":
        return np.array([kappa_m / 8, kappa_m / 4, kappa_m / 4, kappa_m / 4, kappa_m / 8], float)
    if mass_class == "M":
        return np.array([0, 0, kappa_m, 0, 0], float)
    raise ValueError(f"Unknown ISO 52016-1 mass class {mass_class!r}")


def classify_mass(layers) -> str:
    """Select a Clause 6.5.7.2 mass class from (resistance, capacity) layers, outside first.

    The standard leaves the class to the user or a national annex. EPSM selects it
    from the layer sequence: the layer with the largest resistance is treated as
    the insulation. If no layer holds more than half the total resistance, mass is
    taken as distributed (D). Otherwise the share of capacity on the room side
    of that layer selects I (>= 80 %), E (<= 20 %) or IE.
    """
    layers = [(float(r), float(k)) for r, k in layers]
    total_r = sum(r for r, _ in layers)
    total_k = sum(k for _, k in layers)
    if not layers or total_k <= 0 or total_r <= 0:
        return "D"
    index = max(range(len(layers)), key=lambda i: layers[i][0])
    if layers[index][0] <= 0.5 * total_r:
        return "D"
    inside = sum(k for _, k in layers[index + 1 :]) + 0.5 * layers[index][1]
    share = inside / total_k
    return "I" if share >= 0.8 else "E" if share <= 0.2 else "IE"


@dataclass(frozen=True)
class ZoneElement:
    """One construction element facing a zone.

    Parameters
    ----------
    name : str
        Identifier for diagnostics.
    area_m2 : float
        Positive area.
    kind : {'opaque', 'window'}
        Opaque elements use five nodes; windows use two nodes without capacity.
    boundary : {'exterior', 'ground', 'adiabatic', 'zone'}
        Exterior faces outdoor air and sky; ground faces a monthly boundary through
        ``boundary_resistance_m2_k_w``; adiabatic has no heat flow on the outer face;
        zone puts the outer surface node in ``adjacent_zone``.
    direction : {'upwards', 'horizontal', 'downwards'}
        Heat-flow direction at the inner surface selecting Table 25 ``h_ci``.
    resistance_m2_k_w : float
        Thermal resistance between the two surface nodes, R_c.
    capacity_j_m2_k : float
        Areal heat capacity κ_m of the element (ignored for windows).
    mass_class : str
        Clause 6.5.7.2 class I, E, IE, D or M. The outer node is the first node.
    solar_absorptance : float
        Exterior solar absorptance of opaque elements.
    sky_view_factor : float
        View factor to the sky for the exterior sky term.
    boundary_resistance_m2_k_w : float or None
        Ground: virtual-layer resistance to the monthly boundary temperature.
    outdoor_resistance_m2_k_w : float
        Exterior: additional resistance in series with the exterior films, used
        only by caller-selected conductance approximations; default 0.
    zone : int
        Index of the zone that the inner surface node faces.
    adjacent_zone : int or None
        Zone boundary only: index of the zone that the outer surface node faces.
    adjacent_direction : str or None
        Zone boundary only: heat-flow direction at the outer surface.
    """

    name: str
    area_m2: float
    kind: str
    boundary: str
    direction: str
    resistance_m2_k_w: float
    capacity_j_m2_k: float
    mass_class: str
    solar_absorptance: float = 0.0
    sky_view_factor: float = 0.0
    boundary_resistance_m2_k_w: float | None = None
    outdoor_resistance_m2_k_w: float = 0.0
    zone: int = 0
    adjacent_zone: int | None = None
    adjacent_direction: str | None = None

    def __post_init__(self):
        if not np.isfinite(self.area_m2) or self.area_m2 <= 0:
            raise ValueError(f"{self.name}: area must be positive")
        if self.kind not in {"opaque", "window"}:
            raise ValueError(f"{self.name}: kind must be 'opaque' or 'window'")
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"{self.name}: unknown boundary {self.boundary!r}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"{self.name}: unknown direction {self.direction!r}")
        if not np.isfinite(self.resistance_m2_k_w) or self.resistance_m2_k_w <= 0:
            raise ValueError(f"{self.name}: construction resistance must be positive")
        if not np.isfinite(self.capacity_j_m2_k) or self.capacity_j_m2_k < 0:
            raise ValueError(f"{self.name}: capacity must be nonnegative")
        if self.mass_class not in MASS_CLASSES:
            raise ValueError(f"{self.name}: unknown mass class {self.mass_class!r}")
        if self.boundary == "ground" and not (
            self.boundary_resistance_m2_k_w is not None and self.boundary_resistance_m2_k_w > 0
        ):
            raise ValueError(f"{self.name}: ground elements need a positive boundary resistance")
        if self.kind == "window" and self.boundary != "exterior":
            raise ValueError(f"{self.name}: windows must face the exterior")
        if self.boundary == "zone":
            if self.adjacent_zone is None or self.adjacent_zone == self.zone:
                raise ValueError(f"{self.name}: a zone boundary needs a different adjacent zone")
            if self.adjacent_direction not in DIRECTIONS:
                raise ValueError(f"{self.name}: unknown adjacent direction")
        elif self.adjacent_zone is not None:
            raise ValueError(f"{self.name}: only zone boundaries have an adjacent zone")

    @property
    def node_count(self) -> int:
        return 2 if self.kind == "window" else 5

    def internal_conductances(self) -> np.ndarray:
        """Clause 6.5.7 node-to-node conductances per area, outside first."""
        r = self.resistance_m2_k_w
        if self.kind == "window":
            return np.array([1 / r])
        return np.array([6 / r, 3 / r, 3 / r, 6 / r])

    def nodal_capacities(self) -> np.ndarray:
        if self.kind == "window":
            return np.zeros(2)
        return mass_distribution(self.mass_class, self.capacity_j_m2_k)

    def outer_conductance(self) -> float:
        """Conductance per area from the outer node to its boundary temperature.

        Zone boundaries return 0: their outer node couples to a zone's air and
        surfaces inside the network instead.
        """
        if self.boundary == "exterior":
            return 1.0 / (1.0 / (H_CE + H_RE) + self.outdoor_resistance_m2_k_w)
        if self.boundary == "ground":
            return 1.0 / self.boundary_resistance_m2_k_w
        return 0.0

    def u_value(self) -> float:
        """Steady transmittance to outdoors or ground including Table 25 films."""
        outer = self.outer_conductance()
        if outer == 0:
            return 0.0
        h_si = H_CI[self.direction] + H_RI
        return 1.0 / (1.0 / h_si + self.resistance_m2_k_w + 1.0 / outer)


@dataclass(frozen=True)
class ThermalZone:
    """Air volume and operating role of one zone in a :class:`Network52016`.

    Parameters
    ----------
    name : str
        Identifier for diagnostics and results.
    floor_area_m2, air_volume_m3 : float
        Used for internal capacity and air capacity.
    conditioned : bool
        Unconditioned zones receive no HVAC power and float freely.
    thermal_bridge_w_k : float
        Linear/point bridge conductance H_tb from outdoor air to this air node.
    internal_capacity_j_m2_k : float
        Furniture and internal capacity per floor area on the air node.
    control : {'operative', 'air'}
        Controlled temperature; the standard controls operative temperature.
    """

    name: str
    floor_area_m2: float
    air_volume_m3: float
    conditioned: bool = True
    thermal_bridge_w_k: float = 0.0
    internal_capacity_j_m2_k: float = INTERNAL_CAPACITY_J_M2_K
    control: str = "operative"

    def __post_init__(self):
        if self.control not in {"operative", "air"}:
            raise ValueError("control must be 'operative' or 'air'")
        for name in ("floor_area_m2", "air_volume_m3"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(self.thermal_bridge_w_k) or self.thermal_bridge_w_k < 0:
            raise ValueError("thermal_bridge_w_k must be nonnegative")
        if not np.isfinite(self.internal_capacity_j_m2_k) or self.internal_capacity_j_m2_k < 0:
            raise ValueError("internal_capacity_j_m2_k must be nonnegative")


# Order of the thermostat modes tried per zone. The first consistent combination
# is used, so a zone heats or cools only when floating would leave its band.
_MODES = ("free", "heat", "heat_limit", "cool", "cool_limit")


def solve_thermostats(free, gain, set_heating, set_cooling, heat_limit, cool_limit, tol=1e-7):
    """Return HVAC powers (W) that make coupled zone thermostats consistent.

    The control temperatures are ``free + gain @ power``. Each zone is free
    (zero power, temperature inside its band), at a setpoint (power within its
    capacity) or at a capacity limit (temperature on the far side of the
    setpoint). Zones whose limits are both zero stay free. Combinations are
    tried in a fixed order and the first consistent one is returned; for one
    zone this is the linear interpolation of ISO 52016-1 Clause 6.5.5.2.
    """
    free = np.asarray(free, float)
    count = len(free)
    options = []
    for i in range(count):
        modes = ["free"]
        if heat_limit[i] > 0:
            modes += ["heat"] + (["heat_limit"] if np.isfinite(heat_limit[i]) else [])
        if cool_limit[i] < 0:
            modes += ["cool"] + (["cool_limit"] if np.isfinite(cool_limit[i]) else [])
        options.append(modes)
    for combination in _combinations(options):
        power = np.zeros(count)
        target = [i for i, mode in enumerate(combination) if mode in {"heat", "cool"}]
        for i, mode in enumerate(combination):
            if mode == "heat_limit":
                power[i] = heat_limit[i]
            elif mode == "cool_limit":
                power[i] = cool_limit[i]
        if target:
            rest = free + gain @ power
            setpoint = np.array(
                [set_heating[i] if combination[i] == "heat" else set_cooling[i] for i in target]
            )
            power[target] = np.linalg.solve(gain[np.ix_(target, target)], setpoint - rest[target])
        temperature = free + gain @ power
        if all(
            _consistent(
                combination[i],
                temperature[i],
                power[i],
                set_heating[i],
                set_cooling[i],
                heat_limit[i],
                cool_limit[i],
                tol,
            )
            for i in range(count)
        ):
            # Within tolerance a held setpoint can overshoot a limit by rounding only.
            power = np.minimum(np.maximum(power, cool_limit), heat_limit)
            return power, temperature
    raise ValueError("No consistent ISO 52016-1 thermostat state")


def _combinations(options):
    if not options:
        yield ()
        return
    for first in options[0]:
        for rest in _combinations(options[1:]):
            yield (first, *rest)


def _consistent(mode, temperature, power, heating, cooling, heat_limit, cool_limit, tol):
    if mode == "free":
        return not (heat_limit > 0 and temperature < heating) and not (
            cool_limit < 0 and temperature > cooling
        )
    if mode == "heat":
        return -tol <= power <= heat_limit * (1 + tol) + tol
    if mode == "cool":
        return cool_limit * (1 + tol) - tol <= power <= tol
    if mode == "heat_limit":
        return temperature <= heating + tol
    return temperature >= cooling - tol


class Network52016:
    """Zones and elements assembled into one ISO 52016-1 node network.

    Parameters
    ----------
    elements : sequence of ZoneElement
        Each element's ``zone`` (and ``adjacent_zone``) index into ``zones``.
    zones : sequence of ThermalZone
    zone_bridges : sequence of (int, int, float)
        Thermal-bridge conductances (W/K) between two zone air nodes.
    air_volumetric_heat_capacity_j_m3_k : float
    f_int_c, f_sol_c, f_hc_c : float
        Convective fractions of internal gains, window solar and HVAC power.
    delta_theta_er_k : float
        Mean difference between outdoor air and apparent sky temperature.

    Notes
    -----
    Nodes: one air node per zone (in zone order), then each element's chain,
    outer node first. Forcing columns: outdoor air, sky constant, then per zone
    ground boundary temperature, internal gains and window solar, then absorbed
    solar per element (W/m²). See :meth:`forcing_vector`.
    """

    def __init__(
        self,
        elements,
        zones,
        *,
        zone_bridges=(),
        air_volumetric_heat_capacity_j_m3_k=1200.0,
        f_int_c=F_INT_C,
        f_sol_c=F_SOL_C,
        f_hc_c=F_HC_C,
        delta_theta_er_k=DELTA_THETA_ER_K,
    ):
        self.elements = tuple(elements)
        self.zones = tuple(zones)
        self.zone_bridges = tuple((int(i), int(j), float(h)) for i, j, h in zone_bridges)
        self.air_volumetric_heat_capacity_j_m3_k = air_volumetric_heat_capacity_j_m3_k
        self.f_int_c, self.f_sol_c, self.f_hc_c = f_int_c, f_sol_c, f_hc_c
        self.delta_theta_er_k = delta_theta_er_k
        if not self.zones:
            raise ValueError("An ISO 52016-1 network needs at least one zone")
        for name in ("f_int_c", "f_sol_c", "f_hc_c"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        z = len(self.zones)
        for element in self.elements:
            if not 0 <= element.zone < z or (
                element.adjacent_zone is not None and not 0 <= element.adjacent_zone < z
            ):
                raise ValueError(f"{element.name}: zone index out of range")
        for i, j, h in self.zone_bridges:
            if not (0 <= i < z and 0 <= j < z and i != j) or not np.isfinite(h) or h < 0:
                raise ValueError("Zone bridges need two distinct zones and nonnegative W/K")
        for index in range(z):
            if not any(e.zone == index or e.adjacent_zone == index for e in self.elements):
                raise ValueError(f"Zone {self.zones[index].name!r} has no elements")
        self._build()

    @property
    def zone_count(self) -> int:
        return len(self.zones)

    @property
    def node_count(self) -> int:
        return self._built["n"]

    @property
    def column_count(self) -> int:
        return 2 + 3 * self.zone_count + len(self.elements)

    # ------------------------------------------------------------------ assembly
    def _build(self):
        """Assemble the constant matrix A0 (W/K) and forcing maps; all equations in W."""
        elements, zones = self.elements, self.zones
        z = len(zones)
        offsets, n = [], z
        for element in elements:
            offsets.append(n)
            n += element.node_count
        inner = np.array([o + e.node_count - 1 for o, e in zip(offsets, elements)], dtype=int)
        outer = np.array(offsets, dtype=int)
        # Surfaces facing each zone: (node, area, direction).
        faces = [[] for _ in zones]
        for element, i_node, o_node in zip(elements, inner, outer):
            faces[element.zone].append((i_node, element.area_m2, element.direction))
            if element.boundary == "zone":
                faces[element.adjacent_zone].append(
                    (o_node, element.area_m2, element.adjacent_direction)
                )
        capacity = np.zeros(n)
        a = np.zeros((n, n))
        p = np.zeros((n, self.column_count))
        hvac = np.zeros((n, z))
        control = np.zeros((z, n))
        operative = np.zeros((z, n))
        for index, zone in enumerate(zones):
            capacity[index] = (
                self.air_volumetric_heat_capacity_j_m3_k * zone.air_volume_m3
                + zone.internal_capacity_j_m2_k * zone.floor_area_m2
            )
            a[index, index] += zone.thermal_bridge_w_k
            p[index, 0] += zone.thermal_bridge_w_k
            p[index, 3 + 3 * index] += self.f_int_c
            p[index, 4 + 3 * index] += self.f_sol_c
            if zone.conditioned:
                hvac[index, index] = self.f_hc_c
            nodes = np.array([node for node, _, _ in faces[index]], dtype=int)
            areas = np.array([area for _, area, _ in faces[index]])
            a_tot = areas.sum()
            for node, area, direction in faces[index]:
                # Convection to the zone air and radiation to its mean radiant temperature.
                h_ci = H_CI[direction]
                a[node, node] += area * (h_ci + H_RI)
                a[index, index] += area * h_ci
                a[node, index] -= area * h_ci
                a[index, node] -= area * h_ci
                a[node, nodes] -= H_RI * area * areas / a_tot
                share = area / a_tot
                p[node, 3 + 3 * index] += (1 - self.f_int_c) * share
                p[node, 4 + 3 * index] += (1 - self.f_sol_c) * share
                if zone.conditioned:
                    hvac[node, index] += (1 - self.f_hc_c) * share
            operative[index, index] = 0.5
            operative[index, nodes] += 0.5 * areas / a_tot
            if zone.control == "air":
                control[index, index] = 1.0
            else:
                control[index] = operative[index]
        for i, j, h in self.zone_bridges:
            a[i, i] += h
            a[j, j] += h
            a[i, j] -= h
            a[j, i] -= h
        for index, (element, start) in enumerate(zip(elements, offsets)):
            area = element.area_m2
            capacity[start : start + element.node_count] += area * element.nodal_capacities()
            for k, h in enumerate(element.internal_conductances()):
                i, j = start + k, start + k + 1
                a[i, i] += area * h
                a[j, j] += area * h
                a[i, j] -= area * h
                a[j, i] -= area * h
            h_out = element.outer_conductance()
            a[start, start] += area * h_out
            if element.boundary == "exterior":
                p[start, 0] += area * h_out
                p[start, 1] -= area * element.sky_view_factor * H_RE * self.delta_theta_er_k
                if element.kind == "opaque":
                    p[start, 2 + 3 * z + index] = area * element.solar_absorptance
            elif element.boundary == "ground":
                p[start, 2 + 3 * element.zone] += area * h_out
        c_dt = capacity / TIMESTEP_S
        a0 = a + np.diag(c_dt)
        m = np.linalg.inv(a0)
        air = np.arange(z)
        self._built = {
            "n": n,
            "z": z,
            "a": a,
            "a0": a0,
            "m": m,
            "p": p,
            "mp": m @ p,
            "m_air": m[:, air].copy(),
            "hvac": hvac,
            "m_hvac": m @ hvac,
            "c_dt": c_dt,
            "control": control,
            "operative": operative,
            "inner": inner,
            "outer": outer,
            "faces": faces,
        }

    def steady_conductance_w_k(self) -> float:
        """Steady transmission conductance to outdoor air and ground (no ventilation)."""
        return float(
            sum(e.area_m2 * e.u_value() for e in self.elements)
            + sum(zone.thermal_bridge_w_k for zone in self.zones)
        )

    # ------------------------------------------------------------------ stepping
    def forcing_vector(self, theta_e, theta_gr, phi_int, phi_sol, absorbed_w_m2):
        """Pack one hour of forcing; ``theta_gr``, ``phi_int``, ``phi_sol`` are per zone."""
        per_zone = np.column_stack(
            [
                np.broadcast_to(np.asarray(v, float), (self.zone_count,))
                for v in (theta_gr, phi_int, phi_sol)
            ]
        ).ravel()
        return np.concatenate(([theta_e, 1.0], per_zone, absorbed_w_m2))

    def step(self, state, forcing, h_ve, set_heating, set_cooling, heat_limit, cool_limit):
        """Advance one hour; returns (new state, HVAC power, air, operative), per zone in W/°C.

        ``h_ve`` holds each zone's ventilation conductance (W/K) to outdoor air at
        ``forcing[0]``. Cooling limits are nonpositive; heating is positive power.
        """
        built = self._built
        h_ve = np.asarray(h_ve, float)
        theta_e = forcing[0]
        m_air = built["m_air"]
        y = built["m"] @ (built["c_dt"] * state) + built["mp"] @ forcing + m_air @ (h_ve * theta_e)
        z = built["z"]
        # Woodbury: (A0 + U D Uᵀ)⁻¹ = M − M U (I + D Uᵀ M U)⁻¹ D Uᵀ M, U = air selectors.
        s = np.eye(z) + h_ve[:, None] * m_air[:z]
        free = y - m_air @ np.linalg.solve(s, h_ve * y[:z])
        hvac = built["m_hvac"]
        response = hvac - m_air @ np.linalg.solve(s, h_ve[:, None] * hvac[:z])
        control = built["control"]
        power, _ = solve_thermostats(
            control @ free,
            control @ response,
            np.asarray(set_heating, float),
            np.asarray(set_cooling, float),
            np.asarray(heat_limit, float),
            np.asarray(cool_limit, float),
        )
        new = free + response @ power if power.any() else free
        return new, power, new[:z].copy(), built["operative"] @ new

    def initial_state(self, temperature: float) -> np.ndarray:
        return np.full(self.node_count, float(temperature))

    def interface_heat_w(self, state, element_index):
        """Heat (W) conducted through an element's middle link towards its inner face."""
        element = self.elements[element_index]
        start = self._built["outer"][element_index]
        middle = element.node_count // 2
        h = element.internal_conductances()[middle - 1] * element.area_m2
        return float(h * (state[start + middle - 1] - state[start + middle]))

    def summary(self) -> dict:
        """Detached description of the assembled network."""
        return {
            "method": "iso52016_1",
            "node_count": self.node_count,
            "element_count": len(self.elements),
            "zones": [
                {
                    "name": zone.name,
                    "conditioned": zone.conditioned,
                    "control": zone.control,
                    "air_node_capacity_j_k": float(self._built["c_dt"][index] * TIMESTEP_S),
                    "floor_area_m2": zone.floor_area_m2,
                    "air_volume_m3": zone.air_volume_m3,
                }
                for index, zone in enumerate(self.zones)
            ],
            "convective_fractions": {
                "internal": self.f_int_c,
                "solar": self.f_sol_c,
                "hvac": self.f_hc_c,
            },
            "steady_transmission_w_k": self.steady_conductance_w_k(),
            "elements": [
                {
                    "name": e.name,
                    "zone": self.zones[e.zone].name,
                    "adjacent_zone": (
                        None if e.adjacent_zone is None else self.zones[e.adjacent_zone].name
                    ),
                    "area_m2": e.area_m2,
                    "kind": e.kind,
                    "boundary": e.boundary,
                    "direction": e.direction,
                    "resistance_m2_k_w": e.resistance_m2_k_w,
                    "capacity_j_m2_k": e.capacity_j_m2_k,
                    "mass_class": e.mass_class,
                    "u_value_w_m2_k": e.u_value(),
                }
                for e in self.elements
            ],
        }


@dataclass
class Zone52016:
    """Single conditioned zone assembled from elements (ISO 52016-1 Clause 6.5).

    A one-zone :class:`Network52016` with scalar forcing and results.

    Parameters
    ----------
    elements : tuple of ZoneElement
    floor_area_m2, air_volume_m3 : float
        Used for internal capacity and air capacity.
    thermal_bridge_w_k : float
        Linear/point bridge conductance H_tb connected from outdoor air to the air node.
    air_volumetric_heat_capacity_j_m3_k : float
    internal_capacity_j_m2_k : float
        Furniture and internal capacity per floor area on the air node.
    f_int_c, f_sol_c, f_hc_c : float
        Convective fractions of internal gains, window solar and HVAC power.
    control : {'operative', 'air'}
        Controlled temperature; the standard controls operative temperature.
    delta_theta_er_k : float
        Mean difference between outdoor air and apparent sky temperature.
    """

    elements: tuple
    floor_area_m2: float
    air_volume_m3: float
    thermal_bridge_w_k: float = 0.0
    air_volumetric_heat_capacity_j_m3_k: float = 1200.0
    internal_capacity_j_m2_k: float = INTERNAL_CAPACITY_J_M2_K
    f_int_c: float = F_INT_C
    f_sol_c: float = F_SOL_C
    f_hc_c: float = F_HC_C
    control: str = "operative"
    delta_theta_er_k: float = DELTA_THETA_ER_K
    network: Network52016 = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.elements = tuple(self.elements)
        if not self.elements:
            raise ValueError("An ISO 52016-1 zone needs at least one element")
        if any(e.boundary == "zone" or e.zone != 0 for e in self.elements):
            raise ValueError("A single zone cannot hold zone-boundary elements; use Network52016")
        self.network = Network52016(
            self.elements,
            (
                ThermalZone(
                    "zone",
                    self.floor_area_m2,
                    self.air_volume_m3,
                    thermal_bridge_w_k=self.thermal_bridge_w_k,
                    internal_capacity_j_m2_k=self.internal_capacity_j_m2_k,
                    control=self.control,
                ),
            ),
            air_volumetric_heat_capacity_j_m3_k=self.air_volumetric_heat_capacity_j_m3_k,
            f_int_c=self.f_int_c,
            f_sol_c=self.f_sol_c,
            f_hc_c=self.f_hc_c,
            delta_theta_er_k=self.delta_theta_er_k,
        )

    @property
    def _built(self):
        return self.network._built

    @property
    def node_count(self) -> int:
        return self.network.node_count

    def steady_conductance_w_k(self) -> float:
        return self.network.steady_conductance_w_k()

    def forcing_vector(self, theta_e, theta_gr, phi_int, phi_sol, absorbed_w_m2):
        """Pack one hour of forcing in the network column order."""
        return self.network.forcing_vector(theta_e, theta_gr, phi_int, phi_sol, absorbed_w_m2)

    def step(self, state, forcing, h_ve, set_heating, set_cooling, heat_limit, cool_limit):
        """Advance one hour; returns (new state, HVAC power W, air, operative)."""
        new, power, air, operative = self.network.step(
            state, forcing, [h_ve], [set_heating], [set_cooling], [heat_limit], [cool_limit]
        )
        return new, float(power[0]), float(air[0]), float(operative[0])

    def initial_state(self, temperature: float) -> np.ndarray:
        return self.network.initial_state(temperature)

    def surface_temperatures(self, state) -> np.ndarray:
        return np.asarray(state)[self._built["inner"]]

    def summary(self) -> dict:
        summary = self.network.summary()
        summary["control"] = self.control
        return summary


def simulate_network(network: Network52016, hours, *, warmup_passes: int = 1):
    """Run hourly inputs through a network, optionally repeating the year to warm up.

    ``hours`` is a callable ``hours(index, air_temperatures, last_powers)`` returning
    ``(forcing, h_ve, set_heating, set_cooling, heat_limit, cool_limit)`` with
    per-zone arrays, and supports ``len``. ``air_temperatures`` is None before the
    first hour. The state starts at the first zone's first heating setpoint.
    Returns per-zone power, air and operative arrays (hours × zones) and the state.
    """
    count, z = len(hours), network.zone_count
    state, last = None, np.zeros(z)
    for _ in range(warmup_passes + 1):
        power, air, operative = (np.zeros((count, z)) for _ in range(3))
        for hour in range(count):
            inputs = hours(hour, None if state is None else state[:z].copy(), last)
            if state is None:
                state = network.initial_state(np.asarray(inputs[2], float)[0])
            state, power[hour], air[hour], operative[hour] = network.step(state, *inputs)
            last = power[hour]
    return power, air, operative, state


def simulate_zone(zone: Zone52016, hours, *, warmup_passes: int = 1):
    """Single-zone form of :func:`simulate_network` with scalar inputs and outputs.

    ``hours(index, air_temperature, last_power)`` returns
    ``(forcing, h_ve, set_heating, set_cooling, heat_limit, cool_limit)``.
    """

    class _Wrapped:
        def __len__(self):
            return len(hours)

        def __call__(self, hour, air, last):
            forcing, *rest = hours(hour, None if air is None else float(air[0]), float(last[0]))
            return (forcing, *([value] for value in rest))

    power, air, operative, state = simulate_network(
        zone.network, _Wrapped(), warmup_passes=warmup_passes
    )
    return power[:, 0], air[:, 0], operative[:, 0], state
