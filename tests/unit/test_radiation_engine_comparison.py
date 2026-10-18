"""Required real annual comparison of v1/v2 radiation from the same captured inputs."""

import json
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from epsm_core import Building, CompilerSettings, HourlyOutputRequest, Weather, run_building

from epsm_rc import RCSettings, simulate_building


@pytest.mark.energyplus
@pytest.mark.parametrize("case", ["weather_rectangle", "weather_concave"])
def test_annual_radiation_policies_against_real_energyplus(case, tmp_path):
    engine = os.environ.get("ENERGYPLUS_EXE")
    if not engine or not Path(engine).is_file():
        if os.environ.get("EPSM_REQUIRE_ENERGYPLUS") == "1":
            pytest.fail("Required radiation release check needs EnergyPlus 24.2")
        pytest.skip("EnergyPlus 24.2 is required")
    source = Path(__file__).parents[2] / "benchmarks/fixtures" / (case + ".json")
    captured = json.loads(source.read_text())
    building = Building.from_dict(captured["snapshot"])
    snapshot = building.snapshot()
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    names = (
        "Surface Outside Face Incident Beam Solar Radiation Rate per Area",
        "Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area",
        "Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area",
        "Surface Window Transmitted Solar Radiation Rate",
    )
    detailed = run_building(
        snapshot,
        weather=weather,
        directory=tmp_path / "energyplus",
        energyplus=engine,
        settings=CompilerSettings(
            hourly_outputs=tuple(HourlyOutputRequest(name=name) for name in names)
        ),
    )
    results = {"energyplus": detailed}
    for version in (1, 2):
        results[f"rc_v{version}"] = simulate_building(
            snapshot.to_building(),
            weather=weather,
            directory=tmp_path / f"rc-v{version}",
            settings=RCSettings.from_policy(
                f"epsm_rc_5r1c_v{version}",
                airflow_policy="sensible_only_unlimited_supply",
                self_shadowing_policy="ignore",
            ),
        )
    for result in results.values():
        assert result.diagnostics["accepted"]
        np.testing.assert_array_equal(result.time_axis, detailed.time_axis)
        assert len(result.time_axis) == 8760
        assert all(np.isfinite(values).all() for values in result.hourly.values())
        for name, values in result.hourly.items():
            if name.endswith("_kwh"):
                assert result.scalars[name] == pytest.approx(
                    float(values.sum()), rel=0.001, abs=0.001
                )
    assert building.snapshot().sha256 == snapshot.sha256
    component_integrals = {}
    for requested in names:
        signals = {
            key: float(values.sum())
            for key, values in detailed.hourly.items()
            if key.split("::")[0].upper() == requested.upper()
        }
        assert signals
        component_integrals[requested] = signals
    # Raw EnergyPlus irradiance signals retain W/m² or W, not RC effective SHGC
    # heat gain. Keep their units and identities rather than asserting false parity.
    receipt = {
        "snapshot_sha256": snapshot.sha256,
        "weather_sha256": weather.sha256,
        "energyplus_radiation_native_hour_sums": component_integrals,
        "native_units": {
            key: detailed.units[key]
            for key in detailed.hourly
            if any(key.split("::")[0].upper() == name.upper() for name in names)
        },
        "results": {
            key: {
                "run_id": result.run_id,
                "scalars": result.scalars,
                "diagnostics": result.diagnostics,
            }
            for key, result in results.items()
        },
    }
    (tmp_path / "radiation-comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")
