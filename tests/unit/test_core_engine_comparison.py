"""Real Core compilation and RC calculation consume the same authored physical state."""

import json
import os
import shutil
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

import epsm_rc
from epsm_rc import RCSettings, parametric


@pytest.fixture
def energyplus():
    executable = os.environ.get("ENERGYPLUS_EXE") or shutil.which("energyplus")
    if executable is None:
        pytest.skip("Real compilation requires installed EnergyPlus 24.2 and its adjacent schema")
    return executable


def authored_fixture(case):
    core = pytest.importorskip("epsm_core")
    building = core.Building(preset="illustrative-reference")
    polygon = (
        Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])
        if case == "rectangle"
        else Polygon([(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)])
    )
    building.geometry = core.GeometrySpec.from_footprint(polygon, storeys=1, storey_height_m=3)
    # Supply unmistakable physical values by replacing immutable fixture leaves.
    selected = dict(building.construction_set.elements)
    wall_role = next(role for role in selected if role.value == "EXT_WALL")
    window_role = next(role for role in selected if role.value == "WINDOW")
    wall = selected[wall_role]
    component = wall.components[0]
    slot = component.layers[0]
    constituent = slot.layer.materials[0]
    material = replace(
        constituent.material,
        name="supplied comparison material",
        conductivity=0.73,
        density=1777.0,
        specific_heat=847.0,
    )
    layer = replace(
        slot.layer,
        thickness=0.09,
        materials=(replace(constituent, material=material, thickness=0.09, area_fraction=1.0),),
    )
    selected[wall_role] = replace(
        wall,
        components=(
            replace(component, layers=(replace(slot, layer=layer), *component.layers[1:])),
            *wall.components[1:],
        ),
    )
    selected[window_role] = replace(selected[window_role], u_value=1.37, shgc=0.44, vt=0.6)
    building.construction_set = replace(building.construction_set, elements=selected)
    building.operation = building.operation.updated(
        heating_setpoint=core.ScheduleSpec.constant(19.5),
        cooling_setpoint=core.ScheduleSpec.constant(29.0),
    )
    building.facade = building.facade.updated(default_wwr=0.12)
    return building


@pytest.mark.energyplus
@pytest.mark.parametrize("case,area", [("rectangle", 200.0), ("concave", 300.0)])
def test_real_compilation_and_rc_share_supplied_inputs(case, area, energyplus, tmp_path):
    core = pytest.importorskip("epsm_core")
    building = authored_fixture(case)
    before = building.snapshot().sha256
    weather = core.Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    comparison = epsm_rc.prepare_engine_comparison(
        building,
        weather=weather,
        directory=tmp_path,
        energyplus=energyplus,
        rc_settings=RCSettings.from_policy(
            "epsm_rc_5r1c_v1",
            airflow_policy="sensible_only_unlimited_supply",
            ground_policy="conductance_multiplier",
            self_shadowing_policy="ignore",
        ),
    )
    assert building.snapshot().sha256 == before == comparison.building_snapshot_sha256
    report = comparison.compilation_report
    result = comparison.rc_result
    assert report.spec_sha256 == result.diagnostics["physical_inputs_sha256"]
    assert result.diagnostics["building_snapshot_sha256"] == before
    assert report.data["conditioned_floor_area_m2"] == area
    assert report.data["conditioned_volume_m3"] == area * 3
    assert report.data["foundation"]["method"] == "kiva"
    assert result.diagnostics["ground_reduction"]["requested_foundation"]["method"] == "kiva"
    assert result.diagnostics["ground_reduction"]["effective_method"] == "conductance_multiplier"
    model = json.loads(Path(comparison.energyplus_model_path).read_text())
    # Core reduces layer resistance back to conductivity, introducing float roundoff.
    supplied_material = model["Material"][model["Construction"]["wall"]["outside_layer"]]
    assert {
        key: supplied_material[key]
        for key in (
            "conductivity",
            "density",
            "specific_heat",
            "thickness",
        )
    } == pytest.approx(
        {
            "conductivity": 0.73,
            "density": 1777.0,
            "specific_heat": 847.0,
            "thickness": 0.09,
        }
    )
    surfaces = model["BuildingSurface:Detailed"].values()
    for kind, boundary in (("Floor", "Foundation"), ("Roof", "Outdoors"), ("Wall", "Outdoors")):
        selected = [surface for surface in surfaces if surface["surface_type"] == kind]
        assert selected and all(
            surface["outside_boundary_condition"] == boundary for surface in selected
        )
        surface_area = 0.0
        for surface in selected:
            vertices = np.array(
                [
                    [vertex[f"vertex_{axis}_coordinate"] for axis in "xyz"]
                    for vertex in surface["vertices"]
                ]
            )
            surface_area += (
                np.linalg.norm(np.cross(vertices, np.roll(vertices, -1, axis=0)).sum(axis=0)) / 2
            )
        perimeter = sum(part.polygon.length for part in building.geometry.footprints)
        assert surface_area == pytest.approx(area if kind != "Wall" else perimeter * 3)
    assert all(
        glazing["u_factor"] == 1.37 and glazing["solar_heat_gain_coefficient"] == 0.44
        for glazing in model["WindowMaterial:SimpleGlazingSystem"].values()
    )
    for name, expected in (("heating_setpoint", 19.5), ("cooling_setpoint", 29.0)):
        emitted = model["Schedule:Compact"][name]["data"]
        assert {item["field"] for item in emitted if isinstance(item["field"], (float, int))} == {
            expected
        }
        assert any(
            setpoints[f"{name}_temperature_schedule_name"] == name
            for setpoints in model["ThermostatSetpoint:DualSetpoint"].values()
        )
    assert len(result.time_axis) == 8760
    assert result.scalars["space_heating_demand_kwh"] > 0
    assert Path(result.artifacts["hourly"]).is_file()
    manifest = json.loads(Path(comparison.manifest_path).read_text())
    assert manifest["energyplus_status"] == "compiled_only"
    assert manifest["rc_status"] == "simulated"
    assert manifest["physical_inputs_sha256"] == report.spec_sha256


def test_rectangle_idf_generator_is_retired_without_writing(tmp_path):
    target = tmp_path / "old.idf"
    with pytest.raises(ValueError, match="Core|EnergyPlusCompiler"):
        parametric.generate_idf(parametric.ParametricBuilding(), target)
    assert not target.exists()


def test_polygon_idf_generator_is_retired_without_inspecting_legacy_geometry(tmp_path):
    with pytest.raises(ValueError, match="Core|EnergyPlusCompiler"):
        parametric.generate_polygon_idf(parametric.ParametricBuilding(), None, tmp_path / "old.idf")
