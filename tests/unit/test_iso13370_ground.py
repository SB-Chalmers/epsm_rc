"""Ground-method checks against published anchors, limits and energy identities."""

from dataclasses import asdict, replace
from math import fsum

import pytest

from epsm_rc.ground import (
    ground_coefficients,
    monthly_ground_heat_flows,
    monthly_ground_temperatures,
)


def coefficients(**overrides):
    inputs = dict(
        area_m2=80.0,
        exposed_perimeter_m=40.0,
        wall_thickness_m=0.3,
        floor_resistance_m2k_w=0.0,
        soil_conductivity_w_mk=1.5,
        soil_volumetric_heat_capacity_j_m3k=3.0e6,
    )
    inputs.update(overrides)
    return ground_coefficients(**inputs)


@pytest.mark.parametrize(
    "perimeter_area, floor_r, published_u",
    [(0.5, 0.0, 0.70), (0.5, 3.0, 0.22), (1.0, 0.5, 0.68), (0.05, 4.0, 0.07)],
)
def test_slab_agrees_with_scottish_government_table(perimeter_area, floor_r, published_u):
    # Primary source: Scottish Government, U-values: guidance, 6.C Table 1.
    # https://www.gov.scot/publications/tables-of-u-values-and-thermal-conductivity/
    # The table declares clay and insulation R; w=0.3 m is an explicit test
    # assumption (not specified in that table). Printed U precision is 0.01.
    actual = coefficients(
        exposed_perimeter_m=80 * perimeter_area,
        floor_resistance_m2k_w=floor_r,
    )
    assert actual.floor_u_w_m2k == pytest.approx(published_u, abs=0.005)


@pytest.mark.parametrize(
    "perimeter_area, depth, published_u",
    [(0.5, 0.5, 0.61), (0.5, 1.0, 0.55), (0.5, 2.5, 0.43),
     (1.0, 0.5, 0.89), (0.1, 2.5, 0.16)],
)
def test_basement_floor_agrees_with_scottish_government_table(
    perimeter_area, depth, published_u
):
    # Same primary source, 6.C Table 5, uninsulated basement floors.
    # Retain the declared w=0.3 m assumption and printed-rounding tolerance.
    actual = coefficients(
        exposed_perimeter_m=80 * perimeter_area,
        basement_depth_m=depth,
        basement_wall_resistance_m2k_w=0.5,
    )
    assert actual.floor_u_w_m2k == pytest.approx(published_u, abs=0.005)


def test_insulated_slab_irish_government_worked_example():
    # Irish TGD L 2022, Appendix A.4 worked example: Rf=0.1/0.02=5,
    # A=8.75*7.25, P=8.75+7.25+7.25 (one party wall excluded), w=.35,
    # soil lambda=2; published equivalent thickness 10.77 m and U=0.15.
    actual = coefficients(
        area_m2=8.75 * 7.25,
        exposed_perimeter_m=8.75 + 2 * 7.25,
        wall_thickness_m=0.35,
        floor_resistance_m2k_w=5.0,
        soil_conductivity_w_mk=2.0,
        soil_volumetric_heat_capacity_j_m3k=2.0e6,
    )
    assert actual.equivalent_floor_thickness_m == pytest.approx(10.77)
    assert actual.floor_u_w_m2k == pytest.approx(0.15, abs=0.005)


def test_zero_depth_and_tiny_depth_reduce_to_slab():
    slab = coefficients()
    zero = coefficients(basement_depth_m=0, basement_wall_resistance_m2k_w=0.5)
    tiny = coefficients(basement_depth_m=1e-9, basement_wall_resistance_m2k_w=0.5)
    assert zero == slab
    assert tiny.steady_conductance_w_k == pytest.approx(slab.steady_conductance_w_k, rel=1e-8)
    assert tiny.periodic_internal_w_k == pytest.approx(slab.periodic_internal_w_k, rel=1e-8)
    assert tiny.periodic_external_w_k == pytest.approx(slab.periodic_external_w_k, rel=1e-8)


def test_basement_wall_insulation_limits_and_whole_foundation_accounting():
    base = coefficients(basement_depth_m=2.5, basement_wall_resistance_m2k_w=0.5)
    # Insulated wall branch dw < df must limit df to dw in its correction.
    weak_wall = coefficients(
        basement_depth_m=2.5, basement_wall_resistance_m2k_w=0,
        floor_resistance_m2k_w=5,
    )
    stronger_wall = coefficients(
        basement_depth_m=2.5, basement_wall_resistance_m2k_w=50,
        floor_resistance_m2k_w=5,
    )
    still_weaker_floor = coefficients(
        basement_depth_m=2.5, basement_wall_resistance_m2k_w=0,
        floor_resistance_m2k_w=10,
    )
    assert stronger_wall.wall_u_w_m2k < weak_wall.wall_u_w_m2k
    assert weak_wall.wall_u_w_m2k == still_weaker_floor.wall_u_w_m2k
    assert stronger_wall.periodic_external_w_k < weak_wall.periodic_external_w_k
    assert base.steady_conductance_w_k == pytest.approx(
        80 * base.floor_u_w_m2k + 2.5 * 40 * base.wall_u_w_m2k
    )
    assert asdict(base)["equivalent_wall_thickness_m"] == pytest.approx(1.005)


def test_soil_heat_capacity_changes_periodic_response_without_changing_steady():
    ordinary = coefficients()
    high_capacity = coefficients(soil_volumetric_heat_capacity_j_m3k=1e8)
    # Higher capacity reduces penetration depth and increases the amplitude
    # of the indoor-driven heat flow; steady conductance is independent of it.
    assert high_capacity.penetration_depth_m < ordinary.penetration_depth_m
    assert high_capacity.periodic_internal_w_k > ordinary.periodic_internal_w_k
    assert high_capacity.steady_conductance_w_k == ordinary.steady_conductance_w_k


@pytest.mark.parametrize("conductivity, capacity, published_depth", [
    (1.5, 3e6, 2.2), (2.0, 2e6, 3.2), (3.5, 2e6, 4.2),
])
def test_annual_penetration_agrees_with_published_iso_soil_table(
    conductivity, capacity, published_depth
):
    # ISO 13370:2007 F.2 table F.1; tabulated depths have 0.1 m precision.
    model = coefficients(
        soil_conductivity_w_mk=conductivity,
        soil_volumetric_heat_capacity_j_m3k=capacity,
    )
    assert model.penetration_depth_m == pytest.approx(published_depth, abs=0.05)


def test_constant_temperatures_produce_steady_heat_flow_and_outdoor_boundary():
    for model in (coefficients(), coefficients(
        basement_depth_m=2.5, basement_wall_resistance_m2k_w=0.5
    )):
        flows = monthly_ground_heat_flows(model, [10] * 12, [20] * 12)
        temperatures = monthly_ground_temperatures(model, [10] * 12, [20] * 12)
        assert flows == pytest.approx([10 * model.steady_conductance_w_k] * 12)
        assert temperatures == pytest.approx([10] * 12)
        assert monthly_ground_heat_flows(model, [20] * 12, [20] * 12) == (0.0,) * 12


def test_monthly_heat_flow_signs_and_independent_numerical_example():
    # A hand-auditable coefficient fixture, independent of geometric formulas:
    # outdoor mean 10, indoor mean 20, Hg100,Hpi120,Hpe40.
    # January Ti19/Te0 => 1000-120+400=1280W; Tg=19-12.8=6.2C.
    model = replace(
        coefficients(), steady_conductance_w_k=100,
        periodic_internal_w_k=120, periodic_external_w_k=40,
    )
    outdoor = [0] * 6 + [20] * 6
    indoor = [19] * 6 + [21] * 6
    assert monthly_ground_heat_flows(model, outdoor, indoor) == (1280,) * 6 + (720,) * 6
    assert monthly_ground_temperatures(model, outdoor, indoor) == pytest.approx(
        (6.2,) * 6 + (13.8,) * 6
    )


def test_weighted_annual_energy_identity_and_equivalent_boundary():
    model = coefficients(basement_depth_m=2.5, basement_wall_resistance_m2k_w=0.5)
    hours = (744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744)
    outdoor = (-2, -1, 3, 7, 12, 16, 18, 17, 12, 7, 2, -1)
    indoor = (20, 20, 20, 20, 21, 22, 23, 23, 22, 21, 20, 20)
    flows = monthly_ground_heat_flows(model, outdoor, indoor, hours)
    boundary = monthly_ground_temperatures(model, outdoor, indoor, hours)
    expected_wh = model.steady_conductance_w_k * fsum(
        w * (ti - te) for w, ti, te in zip(hours, indoor, outdoor)
    )
    assert fsum(q * w for q, w in zip(flows, hours)) == pytest.approx(expected_wh)
    assert [model.steady_conductance_w_k * (ti - tg) for ti, tg in zip(indoor, boundary)] == (
        pytest.approx(flows)
    )


def test_junction_correction_once_has_instantaneous_monthly_temperature_difference():
    base = coefficients()
    bridged = coefficients(junction_psi_w_mk=0.2)
    outdoor = [0] * 6 + [20] * 6
    indoor = [19] * 6 + [21] * 6
    base_flows = monthly_ground_heat_flows(base, outdoor, indoor)
    bridge_flows = monthly_ground_heat_flows(bridged, outdoor, indoor)
    assert [b - a for a, b in zip(base_flows, bridge_flows)] == pytest.approx(
        [40 * 0.2 * (ti - te) for ti, te in zip(indoor, outdoor)]
    )


@pytest.mark.parametrize("field", [
    "area_m2", "exposed_perimeter_m", "wall_thickness_m", "soil_conductivity_w_mk",
    "soil_volumetric_heat_capacity_j_m3k",
])
@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf"), True])
def test_invalid_positive_inputs_rejected(field, invalid):
    with pytest.raises(ValueError, match=field):
        coefficients(**{field: invalid})


def test_missing_basement_wall_resistance_rejected():
    with pytest.raises(ValueError, match="basement_wall_resistance"):
        coefficients(basement_depth_m=2)


@pytest.mark.parametrize("invalid", [[1] * 11, [1] * 13, [float("nan")] * 12])
def test_invalid_monthly_temperatures_rejected(invalid):
    with pytest.raises(ValueError, match="12 finite"):
        monthly_ground_temperatures(coefficients(), invalid, [20] * 12)


def test_nonpositive_month_weights_rejected():
    with pytest.raises(ValueError, match="positive"):
        monthly_ground_temperatures(coefficients(), [10] * 12, [20] * 12, [0] * 12)
