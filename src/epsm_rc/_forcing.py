# SPDX-License-Identifier: MIT
"""Owned hourly forcing and prepared-domain records shared by RC execution paths.

This leaf module has no package imports, so the single-domain adapter and the
coupled-domain runner can both depend on it without importing each other.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np


def _immutable_series(values, dtype=np.float64):
    """Give native workers immutable storage that can be borrowed without the GIL."""
    array = np.ascontiguousarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError("Hourly inputs must be one-dimensional")
    if isinstance(array.base, bytes):
        return array
    return np.frombuffer(array.tobytes(), dtype=dtype)


def _immutable_columns(*values):
    """Pack prepared floating columns into one immutable allocation."""
    arrays = [np.ascontiguousarray(value, dtype=np.float64) for value in values]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Hourly inputs must be one-dimensional")
    storage = np.frombuffer(b"".join(arrays), dtype=np.float64)
    columns, offset = [], 0
    for array in arrays:
        columns.append(storage[offset : offset + len(array)])
        offset += len(array)
    return tuple(columns)


@dataclass(frozen=True)
class DomainSelection:
    name: str
    area_m2: float
    volume_m3: float
    reference_area_m2: float
    use: object
    airflow: object
    elements: tuple
    capacity_weights: dict

    def identity(self):
        return {
            "name": self.name,
            "area_m2": self.area_m2,
            "volume_m3": self.volume_m3,
            "reference_area_m2": self.reference_area_m2,
            "use": self.use.model_dump(mode="json"),
            "airflow": self.airflow.model_dump(mode="json"),
            "capacity_weights": self.capacity_weights,
        }


@dataclass
class PreparedDomain:
    name: str
    model: object
    input_at: Callable
    finish: Callable
    heating_limits_w: object
    cooling_limits_w: object
    time_axis: tuple
    reference_area_m2: float
    volume_m3: float
    forcing: object = None


@dataclass
class HourlyInputs:
    """Owned hourly forcing used by ordinary and coupled numerical execution."""

    model: object
    temperatures: np.ndarray
    gains: np.ndarray
    convective: np.ndarray
    heating_setpoints: np.ndarray
    cooling_setpoints: np.ndarray
    wind: np.ndarray
    infiltration: np.ndarray
    natural: np.ndarray
    ventilation: np.ndarray
    solar: np.ndarray
    opaque: np.ndarray
    month_hours: np.ndarray | None
    coefficients: tuple
    aggregate_split: bool

    def __post_init__(self):
        # The native kernels read these fields by name; fix their types once here.
        self.coefficients = tuple(float(value) for value in self.coefficients)
        self.aggregate_split = bool(self.aggregate_split)

    def at(self, hour):
        model = self.model
        temperature = float(self.temperatures[hour])
        gain = float(self.gains[hour])
        if not self.aggregate_split:
            model.convective_fraction_internal = (
                float(self.convective[hour]) / gain if gain > 0 else 0.0
            )
        model.t_set_heating = float(self.heating_setpoints[hour])
        model.t_set_cooling = float(self.cooling_setpoints[hour])
        w = float(self.wind[hour])
        correction = (
            self.coefficients[0]
            + self.coefficients[1] * abs(model.t_air - temperature)
            + self.coefficients[2] * w
            + self.coefficients[3] * w**2
        )
        model.ach_infiltration = float(self.infiltration[hour]) * correction + float(
            self.natural[hour]
        )
        model.ach_ventilation = float(self.ventilation[hour])
        if not isfinite(model.ach_infiltration) or model.ach_infiltration < 0:
            raise ValueError("Hourly infiltration must be finite and nonnegative")
        model._update_ventilation_coefficient()
        return {
            "t_out": temperature,
            "solar_gains": float(self.solar[hour]),
            "internal_gains": gain,
            "solar_gains_opaque": float(self.opaque[hour]),
            "hour_of_year": int(self.month_hours[hour]) if self.month_hours is not None else None,
        }
