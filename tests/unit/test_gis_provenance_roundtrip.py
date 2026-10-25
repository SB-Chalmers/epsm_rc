"""Imported height evidence survives the physical authoring handoff."""

import json
from importlib.resources import files

import pytest
from epsm_core import Building, ProvenanceSpec, SourceRecord, Weather

from epsm_rc import FootprintImporter


def imported(**overrides):
    pytest.importorskip("pyproj")
    return FootprintImporter.from_geojson(
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [20, 0], [20, 12], [0, 12], [0, 0]]],
            },
            "properties": {"building:levels": 3, "height": "8.4 m"},
        },
        crs="EPSG:3006",
        **overrides,
    )


def complete(geometry, **options):
    reference = Building(preset="illustrative-reference")
    geometry.set_wwr_uniform(0.2)
    building = geometry.to_core_building(
        glazing=reference.facade.glazing,
        construction_set=reference.construction_set,
        hvac=reference.hvac,
        operation=reference.operation,
        **options,
    )
    building.foundation = reference.foundation
    building.thermal_bridge_uplift_fraction = 0
    building.load_reference_area_policy = "conditioned_floor_area"
    return building


def test_height_evidence_survives_all_handoffs_and_invalidates_on_geometry_edit():
    geometry = imported(num_floors=2)
    building = complete(geometry)
    for candidate in (
        building,
        building.copy(),
        Building.from_dict(json.loads(json.dumps(building.to_dict()))),
    ):
        record = next(
            r
            for r in candidate.provenance.records
            if r.field_path == "geometry.parts.0.storey_elevations_m"
        )
        assert record.status == "assumed"
        assert record.evidence == geometry.height_evidence
        assert record.evidence["source_properties"]["building:levels"] == 3
        assert record.evidence["selected_num_floors"] == 2
        weather = Weather.from_epw(files("epsm_rc") / "data/epw/goteborg.epw")
        assert candidate.to_spec(weather).provenance == candidate.provenance
    building.geometry = building.geometry.updated(orientation_deg=15)
    assert not any(r.field_path.startswith("geometry.") for r in building.provenance.records)


def test_source_derived_height_is_inferred_and_caller_provenance_wins():
    geometry = imported()
    draft = geometry.to_core_building()
    assert draft.geometry.parts[0].height_status == "inferred"
    assert draft.provenance.records[0].status == "inferred"
    supplied = SourceRecord(
        field_path="geometry.parts.0.storey_elevations_m",
        status="measured",
        source="survey",
        evidence={"survey_id": "S1"},
    )
    building = geometry.to_core_building(provenance=ProvenanceSpec(records=(supplied,)))
    assert building.provenance.records == (supplied,)
    assert building.geometry.parts[0].height_status == "measured"
