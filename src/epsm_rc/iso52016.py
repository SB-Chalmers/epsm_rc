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

"""ISO 52016-1 hourly zone model with element node chains.

Each construction element is a chain of nodes: five for opaque elements and two
for windows (Clause 6.5.7). The air node holds air and furniture capacity. Inside
surfaces exchange long-wave radiation through an area-weighted mean radiant
temperature, and exterior nodes receive absorbed solar and the sky term.

Each hour is solved implicitly (backward Euler). The network is linear, so the
free-floating state and the response to unit HVAC power give the exact
thermostat power directly; this is the same linear interpolation as Clause 6.5.5.2.
The matrix changes hourly only through the ventilation conductance on the air
node. It is factorised once and updated with a rank-one (Sherman-Morrison) correction.

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
BOUNDARIES = frozenset({"exterior", "ground", "adiabatic"})
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
    """One construction element facing the zone.

    Parameters
    ----------
    name : str
        Identifier for diagnostics.
    area_m2 : float
        Positive area.
    kind : {'opaque', 'window'}
        Opaque elements use five nodes; windows use two nodes without capacity.
    boundary : {'exterior', 'ground', 'adiabatic'}
        Exterior faces outdoor air and sky; ground faces a monthly boundary through
        ``boundary_resistance_m2_k_w``; adiabatic has no heat flow on the outer face.
    direction : {'upwards', 'horizontal', 'downwards'}
        Internal heat-flow direction selecting Table 25 ``h_ci``.
    resistance_m2_k_w : float
        Thermal resistance between the two surface nodes, R_c.
    capacity_j_m2_k : float
        Areal heat capacity κ_m of the element (ignored for windows).
    mass_class : str
        Clause 6.5.7.2 class I, E, IE, D or M.
    solar_absorptance : float
        Exterior solar absorptance of opaque elements.
    sky_view_factor : float
        View factor to the sky for the exterior sky term.
    boundary_resistance_m2_k_w : float or None
        Ground: virtual-layer resistance to the monthly boundary temperature.
    outdoor_resistance_m2_k_w : float
        Exterior: additional resistance in series with the exterior films, used
        only by caller-selected conductance approximations; default 0.
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
        """Conductance per area from the outer node to its boundary temperature."""
        if self.boundary == "exterior":
            return 1.0 / (1.0 / (H_CE + H_RE) + self.outdoor_resistance_m2_k_w)
        if self.boundary == "ground":
            return 1.0 / self.boundary_resistance_m2_k_w
        return 0.0

    def u_value(self) -> float:
        """Steady transmittance including Table 25 films (and ground layers)."""
        outer = self.outer_conductance()
        if outer == 0:
            return 0.0
        h_si = H_CI[self.direction] + H_RI
        return 1.0 / (1.0 / h_si + self.resistance_m2_k_w + 1.0 / outer)


@dataclass
class Zone52016:
    """Single conditioned zone assembled from elements (ISO 52016-1 Clause 6.5).

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
    _built: dict = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.elements = tuple(self.elements)
        if not self.elements:
            raise ValueError("An ISO 52016-1 zone needs at least one element")
        if self.control not in {"operative", "air"}:
            raise ValueError("control must be 'operative' or 'air'")
        for name in ("f_int_c", "f_sol_c", "f_hc_c"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("floor_area_m2", "air_volume_m3"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(self.thermal_bridge_w_k) or self.thermal_bridge_w_k < 0:
            raise ValueError("thermal_bridge_w_k must be nonnegative")
        self._build()

    # ------------------------------------------------------------------ assembly
    def _build(self):
        """Assemble the constant matrix A0 (W/K) and forcing maps; all equations in W."""
        elements = self.elements
        offsets, n = [], 1
        for element in elements:
            offsets.append(n)
            n += element.node_count
        inner = np.array([o + e.node_count - 1 for o, e in zip(offsets, elements)])
        outer = np.array(offsets)
        areas = np.array([e.area_m2 for e in elements])
        a_tot = areas.sum()
        capacity = np.zeros(n)
        capacity[0] = (
            self.air_volumetric_heat_capacity_j_m3_k * self.air_volume_m3
            + self.internal_capacity_j_m2_k * self.floor_area_m2
        )
        a = np.zeros((n, n))
        # Forcing columns: outdoor air, ground boundary, internal gains, window
        # solar, HVAC, sky constant (per unit 1) and absorbed solar per element.
        columns = ["theta_e", "theta_gr", "phi_int", "phi_sol", "sky"]
        p = np.zeros((n, len(columns) + len(elements)))
        a[0, 0] += self.thermal_bridge_w_k
        p[0, 0] += self.thermal_bridge_w_k
        p[0, 2] += self.f_int_c
        p[0, 3] += self.f_sol_c
        hvac = np.zeros(n)
        hvac[0] = self.f_hc_c
        for index, (element, start) in enumerate(zip(elements, offsets)):
            area = element.area_m2
            capacity[start : start + element.node_count] += area * element.nodal_capacities()
            for k, h in enumerate(element.internal_conductances()):
                i, j = start + k, start + k + 1
                a[i, i] += area * h
                a[j, j] += area * h
                a[i, j] -= area * h
                a[j, i] -= area * h
            # Outer face.
            h_out = element.outer_conductance()
            a[start, start] += area * h_out
            if element.boundary == "exterior":
                p[start, 0] += area * h_out
                p[start, 4] -= area * element.sky_view_factor * H_RE * self.delta_theta_er_k
                if element.kind == "opaque":
                    p[start, len(columns) + index] = area * element.solar_absorptance
            elif element.boundary == "ground":
                p[start, 1] += area * h_out
            # Inner face: convection to air and radiation to the mean radiant temperature.
            s = start + element.node_count - 1
            h_ci = H_CI[element.direction]
            a[s, s] += area * h_ci
            a[0, 0] += area * h_ci
            a[s, 0] -= area * h_ci
            a[0, s] -= area * h_ci
            a[s, s] += area * H_RI
            a[s, inner] -= H_RI * area * areas / a_tot
            share = area / a_tot
            p[s, 2] += (1 - self.f_int_c) * share
            p[s, 3] += (1 - self.f_sol_c) * share
            hvac[s] += (1 - self.f_hc_c) * share
        c_dt = capacity / TIMESTEP_S
        a0 = a + np.diag(c_dt)
        m = np.linalg.inv(a0)
        control = np.zeros(n)
        if self.control == "air":
            control[0] = 1.0
        else:
            control[0] = 0.5
            control[inner] += 0.5 * areas / a_tot
        self._built = {
            "n": n,
            "a": a,
            "a0": a0,
            "m": m,
            "mp": m @ p,
            "m_air": m[:, 0].copy(),
            "m_hvac": m @ hvac,
            "c_dt": c_dt,
            "control": control,
            "inner": inner,
            "outer": outer,
            "areas": areas,
            "n_columns": len(columns),
        }

    @property
    def node_count(self) -> int:
        return self._built["n"]

    def steady_conductance_w_k(self) -> float:
        """Steady transmission conductance to outdoor air and ground (no ventilation)."""
        return float(sum(e.area_m2 * e.u_value() for e in self.elements) + self.thermal_bridge_w_k)

    # ------------------------------------------------------------------ stepping
    def forcing_vector(self, theta_e, theta_gr, phi_int, phi_sol, absorbed_w_m2):
        """Pack one hour of forcing in the column order used by ``_build``."""
        return np.concatenate(([theta_e, theta_gr, phi_int, phi_sol, 1.0], absorbed_w_m2))

    def step(self, state, forcing, h_ve, set_heating, set_cooling, heat_limit, cool_limit):
        """Advance one hour; returns (new state, HVAC power W, air, operative).

        ``h_ve`` is the hour's ventilation conductance (W/K) to outdoor air at
        ``theta_e``. ``cool_limit`` is nonpositive. Heating is positive power.
        """
        built = self._built
        theta_e = forcing[0]
        y = built["m"] @ (built["c_dt"] * state) + built["mp"] @ forcing
        y += built["m_air"] * (h_ve * theta_e)
        m_air = built["m_air"]
        denominator = 1.0 + h_ve * m_air[0]
        free = y - m_air * (h_ve * y[0] / denominator)
        hvac = built["m_hvac"]
        response = hvac - m_air * (h_ve * hvac[0] / denominator)
        control = built["control"]
        free_control = control @ free
        gain = control @ response
        power = 0.0
        if free_control < set_heating and heat_limit > 0:
            power = min((set_heating - free_control) / gain, heat_limit)
        elif free_control > set_cooling and cool_limit < 0:
            power = max((set_cooling - free_control) / gain, cool_limit)
        new = free + power * response if power else free
        operative = 0.5 * new[0] + 0.5 * float(
            built["areas"] @ new[built["inner"]] / built["areas"].sum()
        )
        return new, power, float(new[0]), operative

    def initial_state(self, temperature: float) -> np.ndarray:
        return np.full(self.node_count, float(temperature))

    def surface_temperatures(self, state) -> np.ndarray:
        return np.asarray(state)[self._built["inner"]]

    def summary(self) -> dict:
        """Detached description of the assembled network."""
        return {
            "method": "iso52016_1",
            "node_count": self.node_count,
            "element_count": len(self.elements),
            "control": self.control,
            "convective_fractions": {
                "internal": self.f_int_c,
                "solar": self.f_sol_c,
                "hvac": self.f_hc_c,
            },
            "air_node_capacity_j_k": float(self._built["c_dt"][0] * TIMESTEP_S),
            "steady_transmission_w_k": self.steady_conductance_w_k(),
            "elements": [
                {
                    "name": e.name,
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


def simulate_zone(zone: Zone52016, hours, *, warmup_passes: int = 1):
    """Run a sequence of hourly inputs, optionally repeating the year to warm up.

    ``hours`` is a callable ``hours(index, air_temperature, last_power)`` returning
    ``(forcing, h_ve, set_heating, set_cooling, heat_limit, cool_limit)``
    and a count attribute ``len``. Returns power, air, operative and the final state.
    """
    count = len(hours)
    state, last = None, 0.0
    for _ in range(warmup_passes + 1):
        power = np.zeros(count)
        air = np.zeros(count)
        operative = np.zeros(count)
        for hour in range(count):
            if state is None:
                inputs = hours(hour, None, last)
                state = zone.initial_state(inputs[2])
            else:
                inputs = hours(hour, float(state[0]), last)
            state, power[hour], air[hour], operative[hour] = zone.step(state, *inputs)
            last = power[hour]
    return power, air, operative, state
