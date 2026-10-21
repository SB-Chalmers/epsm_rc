# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
"""Discover the shared-building RC output contract without running its solver."""

from __future__ import annotations

from ._results import ANGULAR_DERIVED_OUTPUTS, ANGULAR_WINDOW_OUTPUTS, BASE_OUTPUT_UNITS


def output_capabilities(building=None, *, settings=None):
    """Return exact output names, units and selected-policy availability.

    Parameters
    ----------
    building : epsm_core.Building, optional
        Loaded inputs used to describe delivery and conditioned-basement fields.
    settings : RCSettings or dict, optional
        Explicit settings. Angular radiation channels remain unavailable unless
        the corresponding optics method is selected; no default is chosen here.

    Returns
    -------
    dict
        Versioned descriptors for hourly fields and annual energy sums, with
        ``available`` and a reason for inactive fields. Calendar support describes
        ``simulate_building``; the standalone weather API has its own ordinary-
        year restriction and is never selected by this discovery call.
    """
    from epsm_core import delivery_output_capabilities

    from .platform import RCSettings

    if isinstance(settings, dict):
        settings = RCSettings(**settings)
    if settings is not None and not isinstance(settings, RCSettings):
        raise TypeError("settings must be RCSettings or an explicit settings dictionary")
    outputs = []

    def add(name, unit, *, available=True, reason=None):
        for resolution in ("hourly", "annual") if unit == "kWh" else ("hourly",):
            outputs.append(
                dict(
                    name=name,
                    resolution=resolution,
                    level="building",
                    unit=unit,
                    available=bool(available),
                    **({"reason": reason} if reason else {}),
                )
            )

    for name, unit in BASE_OUTPUT_UNITS.items():
        add(name, unit)
    outputs.extend(delivery_output_capabilities(building))
    angular = settings is not None and settings.window_optics_method == "simple_glazing_angular_v1"
    for name in (*ANGULAR_WINDOW_OUTPUTS, *ANGULAR_DERIVED_OUTPUTS):
        add(
            name,
            "W",
            available=angular,
            reason=None if angular else "Requires simple_glazing_angular_v1 optics",
        )
    geometry = getattr(building, "geometry", None)
    basement = geometry is not None and geometry.basement_conditioned
    for domain in ("main", "basement"):
        for suffix, unit in (("indoor_temperature_c", "C"), ("space_heating_demand_kwh", "kWh")):
            add(
                f"{domain}_{suffix}",
                unit,
                available=basement,
                reason=None if basement else "Requires coupled conditioned-basement domains",
            )
        add(
            f"interzone_heat_to_{domain}_w",
            "W",
            available=basement,
            reason=None if basement else "Requires coupled conditioned-basement domains",
        )
    return dict(
        format="epsm-output-capabilities/1",
        engine="epsm_rc",
        outputs=outputs,
        calendar={
            "supports_leap": True,
            "interface": "simulate_building",
            "standalone_weather_supports_leap": False,
        },
        unverified_requests=[],
    )
