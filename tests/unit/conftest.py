"""Explicit coordinated Core reference inputs for RC integration regressions."""

from datetime import datetime, timedelta

import pytest


@pytest.fixture
def reference_building():
    from epsm_core import Building

    return Building(preset="illustrative-reference")


@pytest.fixture(scope="session")
def synthetic_weather(tmp_path_factory):
    """Complete controlled EPW for software balances, not a climate validation input."""
    path = tmp_path_factory.mktemp("controlled-weather") / "synthetic.epw"
    headers = [
        "LOCATION,Unit test,Test,SWE,SYNTHETIC,000000,57,12,1,20",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,Controlled integration weather",
        "COMMENTS 2,Not measured weather",
        "DATA PERIODS,1,1,Data,Sunday,1/1,12/31",
    ]
    with path.open("w") as stream:
        stream.write("\n".join(headers) + "\n")
        for hour in range(8760):
            stamp = datetime(2023, 1, 1) + timedelta(hours=hour)
            row = ["0"] * 35
            row[:10] = [
                "2023",
                str(stamp.month),
                str(stamp.day),
                str(stamp.hour + 1),
                "60",
                "?",
                "10",
                "5",
                "70",
                "101325",
            ]
            row[12] = "300"
            row[20:22] = ["180", "2"]
            stream.write(",".join(row) + "\n")
    return path
