"""Independent G.3 balances and lightweight context-to-RC integration."""

from dataclasses import replace
from importlib.resources import files

import numpy as np
import pytest
from epsm_core import Weather
from epsm_core.specs import ContextBuilding, ContextSpec, Footprint

from epsm_rc import PreparationCache, RCSettings, simulate_building
from epsm_rc.shading import (
    average_horizon_angle,
    context_edges,
    prepare_horizons,
    shade_irradiance,
)


def neighbour(height=15, base=0, name="south"):
    return ContextBuilding(
        identifier=name,
        footprints=(Footprint(exterior=((-5, -20), (25, -20), (25, -10), (-5, -10))),),
        height_m=height,
        base_elevation_m=base,
    )


def settings(policy="epsm_rc_5r1c_v2", **overrides):
    return RCSettings.from_policy(
        policy,
        airflow_policy="sensible_only_unlimited_supply",
        ground_policy="conductance_multiplier",
        **overrides,
    )


def test_g3_blocks_only_direct_below_horizon_and_is_finite_at_zero_irradiance():
    # G.3 gives Fhor = 1 - Rdir/Rtot below alpha, otherwise 1.
    # 200 beam + 80 sky + 20 ground -> 100 when blocked, 300 otherwise.
    channels = {
        "poa_direct": np.array([200.0, 200.0, 200.0, 0.0]),
        "poa_sky_diffuse": np.array([80.0, 80.0, 80.0, 0.0]),
        "poa_ground_diffuse": np.array([20.0, 20.0, 20.0, 0.0]),
        "poa_diffuse": np.array([100.0, 100.0, 100.0, 0.0]),
        "poa_global": np.array([300.0, 300.0, 300.0, 0.0]),
    }
    originals = {key: value.copy() for key, value in channels.items()}
    result = shade_irradiance(channels, [19.9, 20, 20.1, -15], 20)
    np.testing.assert_array_equal(result["poa_global"], [100, 300, 300, 0])
    np.testing.assert_array_equal(result["poa_direct"], [0, 200, 200, 0])
    assert result["poa_global"][0] / 300 == pytest.approx(1 - 200 / 300)
    for name in ("poa_diffuse", "poa_sky_diffuse", "poa_ground_diffuse"):
        np.testing.assert_array_equal(result[name], originals[name])
    for name in channels:
        np.testing.assert_array_equal(channels[name], originals[name])
        assert np.isfinite(result[name]).all()


def test_horizon_is_front_average_including_open_directions_not_maximum_obstacle():
    context = ContextSpec(
        buildings=(
            ContextBuilding(
                identifier="east",
                height_m=11,
                footprints=(Footprint(exterior=((10, -10), (20, -10), (20, 10), (10, 10))),),
            ),
        )
    )
    edges = context_edges(context, (0, 0), 0)
    # Looking east, the near wall fills +/-45 degrees. At each sampled
    # direction its distance is 10/cos(offset), with a 10 m height difference.
    offsets = np.arange(-85, 90, 10)
    expected = np.where(
        np.abs(offsets) <= 45,
        np.degrees(np.arctan(np.cos(np.radians(offsets)))),
        0,
    ).mean()
    assert average_horizon_angle((0, 0, 1), 90, 90, edges) == pytest.approx(expected)
    assert 0 < expected < 45
    assert average_horizon_angle((0, 0, 1), 270, 90, edges) == 0
    assert average_horizon_angle((0, 0, 12), 90, 90, edges) == 0
    # Duplicate obstacles must not multiply or add overlapping reductions.
    doubled = tuple(np.concatenate((array, array)) for array in edges)
    assert average_horizon_angle((0, 0, 1), 90, 90, doubled) == pytest.approx(expected)


def test_translated_rotated_scene_and_vertical_datum_preserve_horizon():
    context = ContextSpec(buildings=(neighbour(height=20, base=-5),))
    plain = context_edges(context, (0, 0), 0)
    rotated = context_edges(context, (4, 7), 90)
    point = np.array((10.0, 0.0, 1.4))
    shifted_rotated = (point[1] - 7, -(point[0] - 4), point[2])
    assert average_horizon_angle(point, 180, 90, plain) == pytest.approx(
        average_horizon_angle(shifted_rotated, 270, 90, rotated)
    )
    below = context_edges(ContextSpec(buildings=(neighbour(height=4, base=-5),)), (0, 0), 0)
    assert average_horizon_angle(point, 180, 90, below) == 0


def test_courtyard_context_retains_inner_rings_and_touching_wall_is_finite():
    ring = Footprint(
        exterior=((-20, -20), (20, -20), (20, 20), (-20, 20)),
        holes=(((-10, -10), (-10, 10), (10, 10), (10, -10)),),
    )
    edges = context_edges(
        ContextSpec(
            buildings=(
                ContextBuilding(
                    identifier="ring",
                    footprints=(ring,),
                    height_m=11,
                ),
            )
        ),
        (0, 0),
        0,
    )
    assert len(edges[0]) == 8
    alpha = average_horizon_angle((0, 0, 1), 0, 90, edges)
    assert 35 < alpha < 45
    touching = average_horizon_angle((0, 10, 1), 0, 90, edges)
    assert touching == pytest.approx(90)


def test_partial_contact_at_representative_point_is_rejected(reference_building):
    # A 1 m neighbour flush with the middle of the 20 m south facade would
    # otherwise give the whole facade a 90 degree horizon.
    flush = ContextBuilding(
        identifier="flush",
        footprints=(Footprint(exterior=((9.5, -4), (10.5, -4), (10.5, 0), (9.5, 0))),),
        height_m=6,
    )
    reference_building.context = ContextSpec(buildings=(flush,))
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    with pytest.raises(ValueError, match="touch the representative point"):
        simulate_building(reference_building, weather=weather, settings=settings())


def test_contact_guard_ignores_collinear_and_lower_neighbours():
    receivers = {"wall": {"point_m": [5.0, 0.0, 1.5], "tilt_deg": 90, "azimuth_deg": 180}}
    terrace = ContextSpec(
        buildings=(
            ContextBuilding(
                identifier="terrace",
                footprints=(Footprint(exterior=((10, 0), (20, 0), (20, 12), (10, 12))),),
                height_m=9,
            ),
        )
    )
    assert prepare_horizons(receivers, context_edges(terrace, (0, 0), 0), 10)["wall"] == 0
    low = ContextSpec(
        buildings=(
            ContextBuilding(
                identifier="low",
                footprints=(Footprint(exterior=((4, -2), (6, -2), (6, 0), (4, 0))),),
                height_m=1,
            ),
        )
    )
    assert prepare_horizons(receivers, context_edges(low, (0, 0), 0), 10)["wall"] == 0


@pytest.mark.parametrize("policy", ["epsm_rc_5r1c_v1", "epsm_rc_5r1c_v2"])
def test_context_runs_both_optics_preserves_snapshot_and_increases_heating(
    reference_building, policy
):
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    baseline = simulate_building(reference_building, weather=weather, settings=settings(policy))
    reference_building.context = ContextSpec(buildings=(neighbour(),))
    before = reference_building.snapshot().sha256
    shaded = simulate_building(reference_building, weather=weather, settings=settings(policy))
    assert reference_building.snapshot().sha256 == before
    report = shaded.diagnostics["context_shading"]
    assert report["applied"] and report["obstacle_count"] == 1
    assert report["diffuse_and_ground_reflection"] == "unchanged_ISO_horizon_approximation"
    angles = report["horizon_angles_deg"]
    assert angles["main.r0.e0.storey.0.window"] > angles["main.r0.e0.storey.2.window"] > 0
    assert angles["main.r0.e2.storey.0.window"] == 0
    assert shaded.scalars["space_heating_demand_kwh"] > baseline.scalars["space_heating_demand_kwh"]
    assert shaded.run_id != baseline.run_id
    if policy.endswith("v2"):
        for name in (
            "window_incident_sky_diffuse_w",
            "window_incident_ground_diffuse_w",
            "sky_radiation_loss_w",
        ):
            np.testing.assert_array_equal(shaded.hourly[name], baseline.hourly[name])
        for name in (
            "window_incident_beam_w",
            "window_solar_heat_gain_w",
            "opaque_solar_heat_gain_w",
        ):
            assert shaded.hourly[name].sum() < baseline.hourly[name].sum()


def test_cache_reuses_geometry_across_operations_weather_and_invalidates_context(
    reference_building,
    synthetic_weather,
    monkeypatch,
):
    from epsm_core import ScheduleSpec

    from epsm_rc import shading

    calls = []
    original = shading.prepare_horizons

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(shading, "prepare_horizons", counted)
    reference_building.context = ContextSpec(buildings=(neighbour(),))
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    cache = PreparationCache()

    def run(selected_weather=weather, **overrides):
        return simulate_building(
            reference_building,
            weather=selected_weather,
            settings=settings(**overrides),
            preparation_cache=cache,
        )

    first, repeated = run(), run()
    assert len(calls) == 1
    assert first.diagnostics == repeated.diagnostics
    for key in first.hourly:
        np.testing.assert_array_equal(first.hourly[key], repeated.hourly[key])
    first.diagnostics["context_shading"]["horizon_angles_deg"].clear()
    assert run().diagnostics == repeated.diagnostics
    reference_building.operation = reference_building.operation.updated(
        heating_setpoint=ScheduleSpec.constant(24),
    )
    run()
    window = next(e for e in reference_building.elements if e.role.value == "WINDOW")
    reference_building.assign_construction(replace(window.construction, shgc=0.4))
    run()
    run(Weather.from_epw(synthetic_weather))
    assert len(calls) == 1
    run(horizon_azimuth_step_deg=5)
    assert len(calls) == 2
    reference_building.context = ContextSpec(buildings=(neighbour(height=25),))
    changed = run()
    assert len(calls) == 3
    assert changed.run_id != repeated.run_id
    key = "main.r0.e0.storey.0.window"
    assert (
        changed.diagnostics["context_shading"]["horizon_angles_deg"][key]
        > (repeated.diagnostics["context_shading"]["horizon_angles_deg"][key])
    )


def test_resolved_scene_rotation_preserves_relative_horizon_angles(
    reference_building, synthetic_weather
):
    reference_building.context = ContextSpec(buildings=(neighbour(),))
    weather = Weather.from_epw(synthetic_weather)
    plain = simulate_building(reference_building, weather=weather, settings=settings())
    reference_building.geometry = reference_building.geometry.updated(orientation_deg=37)
    rotated = simulate_building(reference_building, weather=weather, settings=settings())
    assert rotated.diagnostics["context_shading"]["horizon_angles_deg"] == pytest.approx(
        plain.diagnostics["context_shading"]["horizon_angles_deg"]
    )


@pytest.mark.parametrize("shape", ["flat", "gable", "mono_pitch"])
def test_cold_attic_shell_receives_horizon_shading(shape):
    from epsm_core import Building, GeometrySpec, RoofSpec, UnconditionedExchangeSpec

    building = Building(
        preset="illustrative-reference",
        geometry=GeometrySpec.from_footprint(
            Footprint.rectangle(10, 8),
            storeys=2,
            storey_height_m=3,
            roof=RoofSpec(shape=shape, topology="cold_attic"),
        ),
        unconditioned_exchange=(
            UnconditionedExchangeSpec(
                zone_kind="attic",
                outdoor_air_ach=0.5,
                source="explicit test assumption",
            ),
        ),
    )
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    policy = settings(cold_attic_policy="quasi_steady_buffer")
    plain = simulate_building(building, weather=weather, settings=policy)
    building.context = ContextSpec(buildings=(neighbour(),))
    shaded = simulate_building(building, weather=weather, settings=policy)
    angles = shaded.diagnostics["context_shading"]["horizon_angles_deg"]
    assert any("attic" in key and alpha > 0 for key, alpha in angles.items())
    assert (
        shaded.hourly["opaque_solar_heat_gain_w"].sum()
        < plain.hourly["opaque_solar_heat_gain_w"].sum()
    )
    np.testing.assert_array_equal(
        shaded.hourly["sky_radiation_loss_w"], plain.hourly["sky_radiation_loss_w"]
    )


def test_empty_context_keeps_existing_solar_arithmetic(reference_building):
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    plain = simulate_building(reference_building, weather=weather, settings=settings())
    rejected = simulate_building(
        reference_building, weather=weather, settings=settings(context_shading_method="reject")
    )
    assert not plain.diagnostics["context_shading"]["applied"]
    assert plain.diagnostics["context_shading"]["horizon_angles_deg"] == {}
    for name in plain.hourly:
        np.testing.assert_array_equal(plain.hourly[name], rejected.hourly[name])


def test_explicit_rejection_is_early_and_facade_details_remain_rejected(reference_building):
    reference_building.context = ContextSpec(buildings=(neighbour(),))
    with pytest.raises(ValueError, match="context_shading_method"):
        simulate_building(
            reference_building,
            weather="missing.epw",
            settings=settings(context_shading_method="reject"),
        )
    reference_building.facade = reference_building.facade.updated(reveal_depth_m=0.2)
    report = reference_building.readiness(engine="epsm_rc")
    assert any("facade shading" in item for item in report.unsupported)


@pytest.mark.parametrize("value", [0, 0.1, -1, 31, float("nan"), float("inf"), True])
def test_invalid_horizon_quadrature_rejected(value):
    with pytest.raises(ValueError, match="horizon_azimuth_step_deg"):
        settings(horizon_azimuth_step_deg=value)


def test_unknown_context_method_rejected():
    with pytest.raises(ValueError, match="context_shading_method"):
        settings(context_shading_method="unknown")


def test_comparison_reject_policy_is_checked_before_compiler(reference_building, tmp_path):
    from epsm_rc import prepare_engine_comparison

    reference_building.context = ContextSpec(buildings=(neighbour(),))
    weather = Weather.from_epw(files("epsm_rc").joinpath("data/epw/goteborg.epw"))
    with pytest.raises(ValueError, match="context_shading_method"):
        prepare_engine_comparison(
            reference_building,
            weather=weather,
            directory=tmp_path / "comparison",
            energyplus="missing-energyplus",
            rc_settings=settings(context_shading_method="reject"),
        )
    assert not (tmp_path / "comparison").exists()


def test_older_core_context_gate_is_adapted_without_bypassing_other_checks(
    reference_building,
    synthetic_weather,
    monkeypatch,
):
    from dataclasses import replace

    from epsm_core import Building

    original = Building.readiness

    def old_readiness(self, *args, **kwargs):
        report = original(self, *args, **kwargs)
        if kwargs.get("engine") == "epsm_rc" and self.context.buildings:
            return replace(
                report,
                unsupported=(
                    *report.unsupported,
                    "epsm_rc does not model explicit context shading",
                ),
            )
        return report

    monkeypatch.setattr(Building, "readiness", old_readiness)
    reference_building.context = ContextSpec(buildings=(neighbour(),))
    result = simulate_building(
        reference_building, weather=Weather.from_epw(synthetic_weather), settings=settings()
    )
    assert result.diagnostics["context_shading"]["applied"]
    reference_building.facade = reference_building.facade.updated(reveal_depth_m=0.2)
    with pytest.raises(ValueError, match="facade shading"):
        simulate_building(
            reference_building, weather=Weather.from_epw(synthetic_weather), settings=settings()
        )
    reference_building.hvac = None
    with pytest.raises(ValueError, match="hvac"):
        simulate_building(
            reference_building, weather=Weather.from_epw(synthetic_weather), settings=settings()
        )
