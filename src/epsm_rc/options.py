# File: options.py
# Purpose: Named, discoverable RCSettings choices with EnergyPlus-matching defaults.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# License: MIT; see the project-root LICENSE for the full permission and warranty terms.
# Author: Sanjay Somanath <sanjay.somanath@chalmers.se>
# Project PI: Alexander Hollberg
# Computational Sustainable Design Group
# Research Area of Sustainable Built Environments
# Chalmers University of Technology, Gothenburg
# Date created: 2026-11-13

"""Named choices for ``RCSettings``.

Notes
-----
Each choice is a string enum, so ``RCSettings(window_optics="angular")`` and
``RCSettings(window_optics=WindowOptics.ANGULAR)`` are equivalent, and
``list(WindowOptics)`` lists the options. Defaults are the options that agreed
best with EnergyPlus in the comparisons in ``benchmarks/iso52016``.

Examples
--------
>>> from epsm_rc.options import HeatRecovery
>>> [option.value for option in HeatRecovery]
['when_beneficial', 'always']
>>> HeatRecovery.parse("load_dependent_bypass")  # deprecated spelling
<HeatRecovery.WHEN_BENEFICIAL: 'when_beneficial'>
"""

from __future__ import annotations

import warnings
from enum import Enum


class _Choice(str, Enum):
    """String enum that accepts its values and documented deprecated spellings."""

    def __str__(self):
        return self.value

    @classmethod
    def _aliases(cls) -> dict[str, str]:
        return {}

    @classmethod
    def parse(cls, value, field: str | None = None):
        """Return the member for ``value``, listing every option when it is unknown."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            if value in cls._value2member_map_:
                return cls(value)
            if value in cls._aliases():
                new = cls(cls._aliases()[value])
                warnings.warn(
                    f"{field or cls.__name__}={value!r} is deprecated; use {new.value!r}",
                    DeprecationWarning,
                    stacklevel=4,
                )
                return new
        valid = ", ".join(repr(option.value) for option in cls)
        raise ValueError(f"Unknown {field or cls.__name__} {value!r}; choose one of {valid}")


class ThermalModel(_Choice):
    """Zone thermal network."""

    ISO52016_1 = "iso52016_1"
    """ISO 52016-1:2017 Clause 6.5 hourly method with node chains per element."""
    ISO13790_5R1C = "iso13790_5r1c"
    """ISO 13790 simple hourly 5R1C network."""

    @classmethod
    def _aliases(cls):
        return {"iso52016_1_hourly": "iso52016_1"}


class ControlTemperature(_Choice):
    """Temperature held by the thermostat in the ISO 52016-1 model."""

    AIR = "air"
    """Air temperature, as EnergyPlus ideal loads control."""
    OPERATIVE = "operative"
    """Operative temperature, as ISO 52016-1 specifies."""


class WindowOptics(_Choice):
    """Window solar transmission."""

    ANGULAR = "angular"
    """Simple-glazing equivalent layer with angle-dependent beam and diffuse optics."""
    CONSTANT_SHGC = "constant_shgc"
    """Normal-incidence SHGC for every hour and angle."""

    @classmethod
    def _aliases(cls):
        return {"simple_glazing_angular_v1": "angular"}


class SolarTime(_Choice):
    """Instant within each weather hour at which the sun position is evaluated."""

    MIDPOINT = "midpoint"
    """Half an hour into the interval, matching EPW hourly averages."""
    HOUR_START = "hour_start"
    """Start of the interval."""

    @classmethod
    def _aliases(cls):
        return {"interval_midpoint": "midpoint"}


class HeatRecovery(_Choice):
    """Operation of the sensible ventilation heat exchanger."""

    WHEN_BENEFICIAL = "when_beneficial"
    """Recover only when it reduces the heating or cooling load, else bypass."""
    ALWAYS = "always"
    """Recover in every hour, including when it adds cooling load."""

    @classmethod
    def _aliases(cls):
        return {"load_dependent_bypass": "when_beneficial", "continuous": "always"}


class AirflowPolicy(_Choice):
    """Treatment of airflow inputs the RC models cannot represent."""

    REJECT = "reject"
    SENSIBLE_ONLY_UNLIMITED_SUPPLY = "sensible_only_unlimited_supply"


class GroundPolicy(_Choice):
    """Ground heat transfer approximation."""

    REJECT = "reject"
    CONDUCTANCE_MULTIPLIER = "conductance_multiplier"
    PRESCRIBED_MONTHLY_TEMPERATURE = "prescribed_monthly_temperature"
    ISO13370_PERIODIC_APPROXIMATION = "iso13370_periodic_approximation"


class SelfShadowingPolicy(_Choice):
    """Treatment of building self-shadowing the RC models do not calculate."""

    REJECT = "reject"
    IGNORE = "ignore"


class ColdAtticPolicy(_Choice):
    """Treatment of unconditioned attics.

    ``unconditioned_zone`` (default, ISO 52016-1 only) keeps the attic as an
    unconditioned zone of the multi-zone network, as EnergyPlus does.
    ``quasi_steady_buffer`` (ISO 13790 5R1C only) eliminates the attic air node
    in steady state. ``reject`` refuses buildings with cold attics.
    """

    UNCONDITIONED_ZONE = "unconditioned_zone"
    QUASI_STEADY_BUFFER = "quasi_steady_buffer"
    REJECT = "reject"


class ContextShading(_Choice):
    """Shading from neighbouring buildings."""

    ISO13790_HOURLY_HORIZON = "iso13790_hourly_horizon"
    REJECT = "reject"


class OverhangShading(_Choice):
    """Shading from horizontal overhangs above windows."""

    ISO13790_HOURLY_OVERHANG = "iso13790_hourly_overhang"
    REJECT = "reject"


# Former named profiles, kept only so older scripts still run.
LEGACY_PROFILES = {
    "epsm_rc_5r1c_v1": dict(
        thermal_model=ThermalModel.ISO13790_5R1C,
        window_optics=WindowOptics.CONSTANT_SHGC,
        solar_time=SolarTime.HOUR_START,
        heat_recovery=HeatRecovery.ALWAYS,
    ),
    "epsm_rc_5r1c_v2": dict(
        thermal_model=ThermalModel.ISO13790_5R1C,
        window_optics=WindowOptics.ANGULAR,
        solar_time=SolarTime.MIDPOINT,
        heat_recovery=HeatRecovery.ALWAYS,
    ),
}

__all__ = [
    "AirflowPolicy",
    "ColdAtticPolicy",
    "ContextShading",
    "ControlTemperature",
    "GroundPolicy",
    "HeatRecovery",
    "OverhangShading",
    "SelfShadowingPolicy",
    "SolarTime",
    "ThermalModel",
    "WindowOptics",
]
