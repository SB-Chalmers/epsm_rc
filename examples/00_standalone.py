# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
# License: See LICENSE at the project root.
"""Run a synthetic 24-hour numerical example offline, without Core or files."""

import numpy as np

from epsm_rc import Building5R1C
from epsm_rc.heating import HeatPumpCarnot


def main():
    """Return an independently initialized thermal run and explicit Carnot COPs."""
    # These are illustrative numerical inputs, not a model of a measured building.
    building = Building5R1C(floor_area_m2=100, t_set_heating=20)
    outdoor_temperature_c = np.full(24, 5.0)
    result = building.simulate(outdoor_temperature_c)
    assert len(result["heating_demand"]) == 24
    assert np.isfinite(result["t_air"]).all()
    heatpump = HeatPumpCarnot(supply_temp_c=45, carnot_efficiency=0.4)
    cop = heatpump.cop(np.array([-7.0, 7.0]))
    assert cop[1] > cop[0] > 1
    return result, cop


if __name__ == "__main__":
    main()
