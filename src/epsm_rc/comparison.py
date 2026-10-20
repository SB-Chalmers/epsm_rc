# File: comparison.py
# Purpose: Compile shared Core inputs and calculate RC from one captured Building.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sanjay Somanath
# Author: Sanjay Somanath
# Research attribution: Numerical solver credits remain in iso13790.py.
# Date created: 2026-10-14
# Date modified: 2026-10-20
# Changelog:
#   2026-10-14 - Replace independent IDF emitters with shared compilation and RC input identity.
"""Compile Core inputs and calculate RC demand from one captured physical Building.

This helper performs no EnergyPlus annual simulation and no calibration. Core
owns the validated epJSON model; the RC adapter records its selected reductions.
No materials, constructions, glazing or schedules are inferred here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .platform import (
    RCSettings,
    _airflow_capabilities,
    _ground_reduction,
    _rc_readiness_engine,
    _self_shadowing_capabilities,
    simulate_building,
)

if TYPE_CHECKING:
    from epsm_core import Building, Weather
    from epsm_core.compiler import CompilationReport, CompilerSettings
    from epsm_core.results import SimulationResult


@dataclass(frozen=True)
class EngineComparison:
    """Compilation and RC results tied to one physical snapshot; outputs remain inspectable."""

    building_snapshot_sha256: str
    compilation_report: CompilationReport = field(repr=False)
    rc_result: SimulationResult = field(repr=False)
    energyplus_model_path: str
    manifest_path: str


def prepare_engine_comparison(
    building: Building,
    *,
    weather: Weather | str | Path,
    directory: str | Path,
    rc_settings: RCSettings | dict,
    energyplus: str | Path | None = None,
    compiler_settings: CompilerSettings | None = None,
) -> EngineComparison:
    """Compile EnergyPlus epJSON and simulate RC from the same captured authoring state.

    Parameters
    ----------
    building : epsm_core.Building
        Complete caller-authored physical Building. It is captured once and is
        not mutated. Both engines consume the same independent captured copy.
    weather : epsm_core.Weather or path-like
        Explicit complete local EPW, validated by Core before compilation.
    directory : path-like
        Create ``energyplus`` and ``rc`` artifact directories plus a comparison
        manifest. Existing Core JSON and RC hourly files are overwritten.
        Writes are sequential; a later failure may leave earlier artifacts.
    rc_settings : RCSettings or dict
        Required explicit reduction/omission policies, checked before compilation.
    energyplus : path-like or None
        EnergyPlus 24.2 executable with adjacent ``Energy+.schema.epJSON``.
        Core uses ``ENERGYPLUS_EXE`` then PATH if omitted. Initialization runs
        ``--version`` and reads the executable/schema; no annual engine run occurs.
    compiler_settings : epsm_core.CompilerSettings or None
        Explicit compiler controls, or Core's recorded baseline defaults.

    Returns
    -------
    EngineComparison
        Core compilation report, RC result, snapshot identity and artifact paths.
        Both output reports retain the same physical specification hash.

    Raises
    ------
    ValueError
        Physical readiness, weather, selected reduction policies or compiler
        validation fail, or the engine input identities differ.
    ImportError
        Coordinated Core/Data platform dependencies are unavailable.
    OSError
        The explicit engine, weather, schema or output location is unavailable.

    Notes
    -----
    This prepares a comparison; it does not establish equivalence between RC
    and EnergyPlus. The manifest distinguishes compiled EnergyPlus inputs from
    simulated RC output and records the RC approximations separately.
    """
    if rc_settings is None:
        raise ValueError("Supply explicit rc_settings for the comparison")
    settings = RCSettings(**rc_settings) if isinstance(rc_settings, dict) else rc_settings
    if not isinstance(settings, RCSettings):
        raise TypeError("rc_settings must be RCSettings or a settings dictionary")
    try:
        from epsm_core import Building, EnergyPlusCompiler, Weather
    except ImportError as error:
        raise ImportError(
            "Install epsm_rc[platform] with coordinated Core/Data versions"
        ) from error
    if not isinstance(building, Building):
        raise TypeError(
            "Author an explicit Core Building; legacy ParametricBuilding conversion is retired"
        )
    snapshot = building.snapshot()
    captured = snapshot.to_building()
    weather = weather if isinstance(weather, Weather) else Weather.from_epw(weather)
    weather.validate()
    captured.readiness(engine=_rc_readiness_engine(captured, weather), weather=weather).require()
    spec = captured.to_spec(weather)
    if spec.context.buildings and settings.context_shading_method == "reject":
        raise ValueError(
            "Context buildings require context_shading_method='iso13790_hourly_horizon'"
        )
    _airflow_capabilities(spec.airflow, settings)
    _ground_reduction(spec.envelope, settings)
    _self_shadowing_capabilities(spec.geometry, settings)
    compiler = EnergyPlusCompiler(energyplus=energyplus, settings=compiler_settings)
    directory = Path(directory)
    report = compiler.write(spec, directory / "energyplus")
    result = simulate_building(
        captured,
        weather=weather,
        settings=settings,
        directory=directory / "rc",
    )
    if report.spec_sha256 != result.diagnostics["physical_inputs_sha256"]:
        raise ValueError(
            "EnergyPlus compilation and RC calculation consumed different physical inputs"
        )
    if captured.snapshot().sha256 != snapshot.sha256:
        raise ValueError("An engine changed the captured physical Building during comparison")
    result.diagnostics["building_snapshot_sha256"] = snapshot.sha256
    model_path = directory / "energyplus" / "model.epJSON"
    manifest_path = directory / "comparison.json"
    manifest_path.write_text(
        json.dumps(
            {
                "building_snapshot_sha256": snapshot.sha256,
                "physical_inputs_sha256": report.spec_sha256,
                "weather_sha256": weather.sha256,
                "energyplus_status": "compiled_only",
                "energyplus_run_id": report.run_id,
                "energyplus_model_path": str(model_path),
                "rc_status": "simulated",
                "rc_run_id": result.run_id,
                "rc_effective_configuration": result.diagnostics["effective_configuration"],
                "rc_airflow_capabilities": result.diagnostics["airflow_capabilities"],
                "rc_ground_reduction": result.diagnostics["ground_reduction"],
                "rc_self_shadowing": result.diagnostics["self_shadowing"],
                "rc_context_shading": result.diagnostics["context_shading"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return EngineComparison(snapshot.sha256, report, result, str(model_path), str(manifest_path))
