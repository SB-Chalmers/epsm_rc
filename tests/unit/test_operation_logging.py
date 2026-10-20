# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
# License: See LICENSE at the project root.
"""Public operation logging keeps calculation inputs and host configuration separate."""

import inspect
import logging

from epsm_rc import simulate_building


def test_public_operation_accepts_explicit_context():
    parameter = inspect.signature(simulate_building).parameters["log_context"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_import_does_not_install_output_handlers():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import epsm_rc; import logging; assert not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger('epsm_rc').handlers)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout and not result.stderr


def test_rejected_operation_keeps_context_and_reports_failure(caplog):
    import pytest

    with caplog.at_level(logging.DEBUG):
        with pytest.raises((ValueError, TypeError, AttributeError)):
            simulate_building(
                None,
                weather=None,
                log_context={"study_id": "logging-test", "operation_id": "parent"},
            )
    records = [record for record in caplog.records if hasattr(record, "epsm_event")]
    assert records[0].epsm_event.endswith(".started")
    assert records[-1].epsm_event.endswith(".failed")
    assert records[0].epsm_context["parent_operation_id"] == "parent"
    assert records[-1].epsm_context["study_id"] == "logging-test"
    assert records[-1].epsm_details["duration_s"] >= 0
    assert not any(record.exc_info for record in records)


def test_completed_calculation_identifies_run(caplog, reference_building):
    from importlib.resources import files

    from epsm_core import Weather

    from epsm_rc import RCSettings

    weather = Weather.from_epw(files("epsm_rc") / "data/epw/goteborg.epw")
    settings = RCSettings(
        airflow_policy="sensible_only_unlimited_supply", ground_policy="conductance_multiplier"
    )
    with caplog.at_level(logging.INFO):
        result = simulate_building(reference_building, weather=weather, settings=settings)
    completed = next(
        record
        for record in caplog.records
        if getattr(record, "epsm_event", "") == "energy.rc.completed"
    )
    assert completed.epsm_context["run_id"] == result.run_id
    assert completed.epsm_details["hours"] == len(result.time_axis)
