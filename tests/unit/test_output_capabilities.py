"""Public output catalogue includes conditional channels and calendar evidence."""

from importlib.resources import files

import pytest


def test_rc_output_discovery_tracks_selected_optics_and_basement(reference_building):
    from epsm_rc import RCSettings, output_capabilities

    ordinary = output_capabilities(
        reference_building, settings=RCSettings.from_policy("epsm_rc_5r1c_v1")
    )
    angular = output_capabilities(
        reference_building, settings=RCSettings.from_policy("epsm_rc_5r1c_v2")
    )

    def row(result, name):
        return next(item for item in result["outputs"] if item["name"] == name)

    assert row(ordinary, "indoor_temperature_c")["unit"] == "C"
    assert not row(ordinary, "window_incident_beam_w")["available"]
    assert row(angular, "window_incident_beam_w")["available"]
    assert not row(ordinary, "basement_indoor_temperature_c")["available"]
    assert ordinary["calendar"]["supports_leap"]


def test_platform_preflight_and_execution_retain_leap_calendar(tmp_path, reference_building):
    from datetime import datetime, timedelta

    from epsm_core import Weather

    from epsm_rc import RCSettings, preflight, simulate_building
    from epsm_rc.weather import load_epw

    lines = (files("epsm_rc") / "data/epw/goteborg.epw").read_text().splitlines()
    template = lines[8].split(",")
    rows = []
    for hour in range(8784):
        stamp = datetime(2024, 1, 1) + timedelta(hours=hour)
        row = template.copy()
        row[:5] = ["2024", str(stamp.month), str(stamp.day), str(stamp.hour + 1), "60"]
        rows.append(",".join(row))
    path = tmp_path / "leap.epw"
    path.write_text("\n".join(lines[:8] + rows) + "\n")
    weather = Weather.from_epw(path, calendar_year=2024)
    settings = RCSettings(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    assert preflight(reference_building, weather=weather, settings=settings)["ready"]
    result = simulate_building(reference_building, weather=weather, settings=settings)
    assert len(result.time_axis) == 8784
    assert sum("02-29" in str(value) for value in result.time_axis) == 24
    with pytest.raises(ValueError, match="Leap calendar"):
        load_epw(weather)
