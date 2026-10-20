"""Public imports and model selection report their real behavior."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from epsm_rc import heating


def test_heatpump_rejects_unknown_arguments():
    with pytest.raises(TypeError, match="unexpected keyword"):
        heating.HeatPump(accidental_typo=123)


def test_factory_reports_selected_model(monkeypatch):
    monkeypatch.setattr(heating, "HPLIB_AVAILABLE", False)
    with pytest.warns(DeprecationWarning, match="HeatPumpCarnot"):
        result = heating.HeatPump()
    assert isinstance(result, heating.HeatPumpCarnot)


def test_explicit_hplib_does_not_replace_failed_model(monkeypatch):
    requested = []

    def fail(**kwargs):
        requested.append(kwargs["model"])
        raise KeyError("unknown model")

    monkeypatch.setattr(heating, "HPLIB_AVAILABLE", True)
    monkeypatch.setattr(heating, "hpl", SimpleNamespace(get_parameters=fail), raising=False)
    with pytest.raises(ValueError, match="requested-model"):
        heating.HeatPumpHPLib(hp_model="requested-model")
    assert requested == ["requested-model"]


def test_explicit_hplib_does_not_guess_failed_design_output(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("design simulation failed")

    monkeypatch.setattr(heating, "HPLIB_AVAILABLE", True)
    monkeypatch.setattr(
        heating,
        "hpl",
        SimpleNamespace(
            get_parameters=lambda **kwargs: pd.DataFrame({"P_th_h_ref [W]": [5000.0]}),
            simulate=fail,
        ),
        raising=False,
    )
    with pytest.raises(ValueError, match="design"):
        heating.HeatPumpHPLib()


def test_parametric_cli_has_no_runpy_warning():
    result = subprocess.run(
        [sys.executable, "-m", "epsm_rc.parametric", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stderr


def test_base_import_works_with_core_unavailable():
    code = """
import importlib.abc
import sys
class NoCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'epsm_core', 'epsm_data'} or fullname.startswith(('epsm_core.', 'epsm_data.')):
            raise ImportError('Core intentionally unavailable')
sys.meta_path.insert(0, NoCore())
from epsm_rc import Building5R1C, ParametricBuilding, BuildingTABULA, FootprintImporter
assert Building5R1C(floor_area_m2=100).floor_area_m2 == 100
assert ParametricBuilding().epw_path is None
assert BuildingTABULA().floor_area_m2 > 0
assert FootprintImporter is not None
assert 'epsm_core' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout and not result.stderr


def test_version_has_one_source():
    import epsm_rc
    from epsm_rc._version import __version__

    assert epsm_rc.__version__ == __version__


@pytest.mark.skipif(not heating.HPLIB_AVAILABLE, reason="hplib extra not installed")
def test_hplib_extra_runs_real_model():
    model = heating.HeatPumpHPLib(hp_model="Generic", group_id=1)
    cop, electrical, thermal = model.simulate_hplib(np.array([-7.0, 0.0, 7.0]))
    assert all(values.shape == (3,) for values in (cop, electrical, thermal))
    assert all(np.isfinite(values).all() for values in (cop, electrical, thermal))
    assert (cop > 0).all() and (electrical > 0).all() and (thermal > 0).all()


def test_heating_representation_is_bounded_and_identifies_demand_source():
    from epsm_rc import Building5R1C

    for model_class in (
        heating.HeatPumpCarnot,
        heating.DistrictHeating,
        heating.DirectElectric,
        heating.WoodPelletHybrid,
    ):
        model = model_class(building_model=Building5R1C(floor_area_m2=100))
        assert len(repr(model)) <= 160
        assert model.summary()["thermal_model"] == "Building5R1C"


@pytest.mark.parametrize("dependency", ["pyproj", "requests", "fiona"])
def test_missing_gis_dependency_is_actionable(monkeypatch, dependency, tmp_path):
    from epsm_rc import FootprintImporter

    monkeypatch.setitem(sys.modules, dependency, None)
    with pytest.raises(ImportError, match=r"epsm_rc\[platform,gis\]"):
        if dependency == "requests":
            FootprintImporter.from_osm(way_id=1, num_floors=1, floor_height=3)
        elif dependency == "fiona":
            FootprintImporter.from_shapefile(tmp_path / "missing.shp", num_floors=1, floor_height=3)
        else:
            FootprintImporter.from_geojson(
                {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 5], [0, 5], [0, 0]]]},
                crs="EPSG:3006",
                num_floors=1,
                floor_height=3,
            )
