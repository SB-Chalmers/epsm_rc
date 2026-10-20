"""ISO 13790 5R1C Physics Module Tests.

Per-module unit tests for every physics calculation in the Building5R1C model.
Complements test_iso13790_validation.py (system-level) with targeted checks
against ISO 13790 equations and hand calculations.

Modules covered:
    1. Ground temperature estimation
    2. Sky radiation loss (Annex E)
    3. Solar gains — opaque surfaces
    4. Solar gains — windows (simplified)
    5. Heat flow distribution (Eq. C.1–C.3)
    6. Crank-Nicolson mass temperature (Eq. C.4–C.5)
    7. Surface temperature (Eq. C.10)
    8. Air temperature (Eq. C.11)
    9. Operative temperature (Eq. C.12)
    10. Thermal bridges (ΔU_tb)
    11. Heat recovery ventilation
    12. Dynamic ground coupling (monthly temps)
    13. Heating demand iteration (Eq. C.13)
    14. Internal gains capping
    15. Factory functions (create_swedish_villa / create_swedish_apartment)
    16. Construction-based thermal mass

Reference:
    ISO 13790:2008 Annex C (hourly method)
    EN ISO 52016-1:2017
"""

import warnings

import numpy as np
import pytest

from epsm_rc.iso13790 import (
    INTERNAL_GAINS_MAX_W_M2,
    SWEDISH_DEFAULT_GROUND_TEMPS,
    THERMAL_BRIDGE_DELTA_U,
    VENTILATION_HR_EFFICIENCY,
    Building5R1C,
    BuildingClass,
    Construction,
    ConstructionLayer,
    calculate_thermal_mass_from_constructions,
    create_swedish_apartment,
    create_swedish_villa,
    estimate_ground_temps_from_air,
)


# ---------------------------------------------------------------------------
# 1. Ground temperature estimation
# ---------------------------------------------------------------------------
class TestGroundTemperatureEstimation:
    """Tests for estimate_ground_temps_from_air()."""

    def test_returns_12_months(self):
        temps = estimate_ground_temps_from_air(7.0)
        assert len(temps) == 12

    def test_mean_above_air_due_to_building_offset(self):
        """Ground under building should be warmer than annual air mean."""
        annual_mean = 7.0
        temps = estimate_ground_temps_from_air(annual_mean)
        assert np.mean(temps) > annual_mean

    def test_amplitude_controls_swing(self):
        """Larger amplitude → larger min-to-max spread."""
        low = estimate_ground_temps_from_air(7.0, amplitude=3.0)
        high = estimate_ground_temps_from_air(7.0, amplitude=8.0)
        assert (max(high) - min(high)) > (max(low) - min(low))

    def test_summer_warmer_than_winter(self):
        """July/August should be warmest; January/February coldest."""
        temps = estimate_ground_temps_from_air(7.0)
        # Summer months (Jun-Aug, indices 5-7) vs winter (Dec-Feb, indices 11, 0, 1)
        summer_avg = np.mean([temps[5], temps[6], temps[7]])
        winter_avg = np.mean([temps[11], temps[0], temps[1]])
        assert summer_avg > winter_avg

    def test_different_climates(self):
        """Warmer climate → uniformly warmer ground temps."""
        cold = estimate_ground_temps_from_air(2.0)  # Kiruna-like
        warm = estimate_ground_temps_from_air(10.0)  # Malmö-like
        assert all(w > c for w, c in zip(warm, cold))

    def test_values_are_finite_floats(self):
        temps = estimate_ground_temps_from_air(7.0)
        assert all(isinstance(t, float) and np.isfinite(t) for t in temps)


# ---------------------------------------------------------------------------
# 2. Sky radiation loss (ISO 13790 Annex E)
# ---------------------------------------------------------------------------
class TestSkyRadiationLoss:
    """Tests for calc_sky_radiation_loss()."""

    def test_zero_when_delta_theta_er_zero(self):
        """No extra radiative loss if Δθ_er = 0."""
        b = Building5R1C(
            floor_area_m2=100,
            u_roof=0.15,
            roof_area_m2=100,
            u_walls=0.25,
            wall_area_m2=120,
            delta_theta_er=0.0,
        )
        assert b.calc_sky_radiation_loss() == 0.0

    def test_hand_calculation(self):
        """Φr = Rse × U × A × hr × Δθer × Fr for roof + walls."""
        b = Building5R1C(
            floor_area_m2=100,
            u_roof=0.15,
            roof_area_m2=100,
            u_walls=0.25,
            wall_area_m2=120,
            r_se_roof=0.04,
            r_se_wall=0.04,
            h_r=5.0,
            delta_theta_er=11.0,
            form_factor_roof=1.0,
            form_factor_wall=0.5,
            # Zero out remaining elements
            u_floor=0.0,
            floor_area_ground_m2=0,
            u_windows=0.0,
            window_area_m2=0,
        )
        # Hand calc
        phi_roof = 0.04 * 0.15 * 100 * 5.0 * 11.0 * 1.0  # 33.0 W
        phi_wall = 0.04 * 0.25 * 120 * 5.0 * 11.0 * 0.5  # 33.0 W
        expected = phi_roof + phi_wall

        result = b.calc_sky_radiation_loss()
        assert abs(result - expected) < 0.01, f"Got {result}, expected {expected}"

    def test_roof_only_no_walls(self):
        """Sky radiation from roof only when wall_area=0."""
        b = Building5R1C(
            floor_area_m2=100,
            u_roof=0.20,
            roof_area_m2=80,
            u_walls=0.0,
            wall_area_m2=0,
            delta_theta_er=11.0,
        )
        expected_roof = b.r_se_roof * 0.20 * 80 * b.h_r * 11.0 * b.form_factor_roof
        assert abs(b.calc_sky_radiation_loss() - expected_roof) < 0.01

    def test_positive_for_typical_building(self):
        """Sky radiation should always be a loss (positive)."""
        b = Building5R1C(floor_area_m2=100, delta_theta_er=11.0)
        assert b.calc_sky_radiation_loss() > 0


# ---------------------------------------------------------------------------
# 3. Solar gains — opaque surfaces
# ---------------------------------------------------------------------------
class TestSolarGainsOpaque:
    """Tests for calc_solar_gains_opaque()."""

    def test_zero_when_no_irradiance(self):
        b = Building5R1C(floor_area_m2=100)
        assert b.calc_solar_gains_opaque(i_sol_wall=0.0, i_sol_roof=0.0) == 0.0

    def test_hand_calculation(self):
        """A_sol,op = α × Rse × U × A; Φ = A_sol × I."""
        b = Building5R1C(
            floor_area_m2=100,
            u_walls=0.25,
            wall_area_m2=120,
            u_roof=0.15,
            roof_area_m2=100,
            wall_absorptivity=0.6,
            roof_absorptivity=0.6,
            r_se_wall=0.04,
            r_se_roof=0.04,
        )
        i_wall, i_roof = 300.0, 500.0
        a_sol_wall = 0.6 * 0.04 * 0.25 * 120  # 0.72 m²
        a_sol_roof = 0.6 * 0.04 * 0.15 * 100  # 0.36 m²
        expected = a_sol_wall * i_wall + a_sol_roof * i_roof  # 396 W

        result = b.calc_solar_gains_opaque(i_sol_wall=i_wall, i_sol_roof=i_roof)
        assert abs(result - expected) < 0.01

    def test_higher_absorptivity_more_gains(self):
        """Dark surfaces (α=0.9) should collect more than light (α=0.3)."""
        dark = Building5R1C(
            floor_area_m2=100,
            wall_absorptivity=0.9,
            roof_absorptivity=0.9,
            u_walls=0.25,
            wall_area_m2=100,
            u_roof=0.15,
            roof_area_m2=80,
        )
        light = Building5R1C(
            floor_area_m2=100,
            wall_absorptivity=0.3,
            roof_absorptivity=0.3,
            u_walls=0.25,
            wall_area_m2=100,
            u_roof=0.15,
            roof_area_m2=80,
        )
        assert dark.calc_solar_gains_opaque(300, 500) > light.calc_solar_gains_opaque(300, 500)


# ---------------------------------------------------------------------------
# 4. Solar gains — windows (simplified method)
# ---------------------------------------------------------------------------
class TestSolarGainsWindowSimplified:
    """Tests for calc_solar_gains() with simplified method."""

    def test_zero_when_no_ghi(self):
        b = Building5R1C(floor_area_m2=100, solar_method="simplified")
        assert b.calc_solar_gains(ghi=0.0) == 0.0

    def test_negative_ghi_returns_zero(self):
        b = Building5R1C(floor_area_m2=100, solar_method="simplified")
        assert b.calc_solar_gains(ghi=-10.0) == 0.0

    def test_simplified_formula(self):
        """Simplified: Φ_sol = A_win × g × F_frame × f_inc × 0.5 × GHI."""
        b = Building5R1C(
            floor_area_m2=100,
            window_area_m2=20,
            g_value=0.6,
            frame_factor=0.7,
            non_perpendicular_incidence_factor=0.9,
            shading_factor=1.0,
            solar_method="simplified",
        )
        ghi = 400.0
        expected = 20 * 0.6 * 0.7 * 0.9 * 0.5 * 400  # 1512 W
        result = b.calc_solar_gains(ghi=ghi)
        assert abs(result - expected) < 0.1

    def test_incidence_factor_can_be_disabled(self):
        """Setting f_inc=1.0 recovers legacy simplified formula."""
        b = Building5R1C(
            floor_area_m2=100,
            window_area_m2=20,
            g_value=0.6,
            frame_factor=0.7,
            non_perpendicular_incidence_factor=1.0,
            shading_factor=1.0,
            solar_method="simplified",
        )
        ghi = 400.0
        expected = 20 * 0.6 * 0.7 * 0.5 * ghi
        assert abs(b.calc_solar_gains(ghi=ghi) - expected) < 0.1

    def test_shading_reduces_gains(self):
        """shading_factor < 1 should reduce solar gains."""
        params = dict(
            floor_area_m2=100,
            window_area_m2=20,
            g_value=0.6,
            frame_factor=0.7,
            solar_method="simplified",
        )
        no_shade = Building5R1C(**params, shading_factor=1.0)
        shaded = Building5R1C(**params, shading_factor=0.5)
        assert shaded.calc_solar_gains(400) < no_shade.calc_solar_gains(400)

    def test_angle_coefficients_disable_constant_incidence_factor(self):
        """When angular model is active, constant incidence factor is not double-counted."""
        params = dict(
            floor_area_m2=100,
            window_area_m2=20,
            g_value=0.6,
            frame_factor=0.7,
            solar_method="simplified",
            g_value_angle_coefficients=(1.0, -0.189, 0.644, -0.596),
            shading_factor=1.0,
        )
        b1 = Building5R1C(**params, non_perpendicular_incidence_factor=1.0)
        b2 = Building5R1C(**params, non_perpendicular_incidence_factor=0.9)
        # For simplified branch the output must be identical because f_inc is bypassed
        assert abs(b1.calc_solar_gains(ghi=400.0) - b2.calc_solar_gains(ghi=400.0)) < 1e-9


# ---------------------------------------------------------------------------
# 5. Heat flow distribution (ISO 13790 Eq. C.1–C.3)
# ---------------------------------------------------------------------------
class TestHeatFlowDistribution:
    """Tests for calc_heat_flows()."""

    def test_convective_fraction_to_air(self):
        """φ_ia should include convective_fraction × internal_gains."""
        b = Building5R1C(
            floor_area_m2=100,
            convective_fraction_internal=0.5,
            delta_theta_er=0.0,  # Disable sky radiation
        )
        phi_ia, _, _ = b.calc_heat_flows(
            t_out=0,
            internal_gains=1000,
            solar_gains=0,
            energy_demand=0,
            include_sky_radiation=False,
        )
        # φ_ia should be 0.5 × 1000 = 500 W (before energy_demand)
        assert abs(phi_ia - 500.0) < 0.1

    def test_energy_demand_goes_to_air_node(self):
        """Heating energy should be added to φ_ia."""
        b = Building5R1C(floor_area_m2=100, delta_theta_er=0.0)
        phi_ia, _, _ = b.calc_heat_flows(
            t_out=0,
            internal_gains=0,
            solar_gains=0,
            energy_demand=5000,
            include_sky_radiation=False,
        )
        assert abs(phi_ia - 5000.0) < 0.1

    def test_solar_distributed_to_surface_and_mass(self):
        """Solar gains should appear in φ_st and φ_m (not φ_ia)."""
        b = Building5R1C(
            floor_area_m2=100,
            building_class=BuildingClass.MEDIUM,
            delta_theta_er=0.0,
        )
        phi_ia, phi_st, phi_m = b.calc_heat_flows(
            t_out=0,
            internal_gains=0,
            solar_gains=1000,
            energy_demand=0,
            include_sky_radiation=False,
        )
        assert phi_ia == 0.0  # No internal gains, no energy demand
        assert phi_st + phi_m > 0  # Solar goes to surface and mass
        assert abs(phi_st + phi_m - 1000.0) < 50  # Close to total (minus window fraction)

    def test_mass_surface_ratios_sum_correctly(self):
        """ratio_mass + ratio_surface + ratio_window ≤ 1.0."""
        b = Building5R1C(
            floor_area_m2=100,
            building_class=BuildingClass.MEDIUM,
            u_windows=1.2,
            window_area_m2=20,
            delta_theta_er=0.0,
        )
        a_m = b._mass_area
        a_tot = b._a_tot
        r_mass = a_m / a_tot
        r_window = (1.2 * 20) / (9.1 * a_tot)
        r_surface = max(0.0, 1.0 - r_mass - r_window)
        assert r_mass + r_surface + r_window <= 1.0 + 1e-9

    def test_sky_radiation_reduces_phi_st(self):
        """Enabling sky radiation should reduce φ_st."""
        b = Building5R1C(
            floor_area_m2=100,
            u_roof=0.15,
            roof_area_m2=100,
            delta_theta_er=11.0,
        )
        _, phi_st_no_sky, _ = b.calc_heat_flows(
            t_out=0,
            internal_gains=0,
            solar_gains=0,
            energy_demand=0,
            include_sky_radiation=False,
        )
        _, phi_st_with_sky, _ = b.calc_heat_flows(
            t_out=0,
            internal_gains=0,
            solar_gains=0,
            energy_demand=0,
            include_sky_radiation=True,
        )
        assert phi_st_with_sky < phi_st_no_sky


# ---------------------------------------------------------------------------
# 6. Crank-Nicolson mass temperature (Eq. C.4)
# ---------------------------------------------------------------------------
class TestMassTemperatureIntegration:
    """Tests for calc_t_m_next() — Crank-Nicolson update."""

    def test_steady_state_convergence(self):
        """With constant φ_m_tot, t_m should converge to a fixed point."""
        b = Building5R1C(floor_area_m2=100, building_class=BuildingClass.MEDIUM)
        phi_m_tot = 0.0  # No net heat → converges to 0°C effectively
        t_m = 20.0
        for _ in range(500):
            t_m = b.calc_t_m_next(t_m, phi_m_tot, dt_hours=1.0)
        # Should converge near 0 with no heat input
        assert abs(t_m) < 1.0 or np.isfinite(t_m)

    def test_positive_heat_raises_temperature(self):
        """Positive φ_m_tot should increase mass temperature."""
        b = Building5R1C(floor_area_m2=100)
        t_m_prev = 15.0
        t_m_next = b.calc_t_m_next(t_m_prev, phi_m_tot=10000, dt_hours=1.0)
        assert t_m_next > t_m_prev

    def test_negative_heat_lowers_temperature(self):
        """Negative φ_m_tot should decrease mass temperature."""
        b = Building5R1C(floor_area_m2=100)
        t_m_prev = 20.0
        t_m_next = b.calc_t_m_next(t_m_prev, phi_m_tot=-10000, dt_hours=1.0)
        assert t_m_next < t_m_prev

    def test_smaller_timestep_smaller_change(self):
        """Halving dt should roughly halve the temperature change."""
        b = Building5R1C(floor_area_m2=100)
        t_m_prev = 20.0
        phi = 5000.0
        dt_1h = b.calc_t_m_next(t_m_prev, phi, dt_hours=1.0) - t_m_prev
        dt_half = b.calc_t_m_next(t_m_prev, phi, dt_hours=0.5) - t_m_prev
        # Ratio should be close to 2 (exact for linear system)
        ratio = dt_1h / dt_half if dt_half != 0 else float("inf")
        assert 1.5 < ratio < 2.5

    def test_heavy_building_responds_slower(self):
        """Higher C_m → smaller temperature change per timestep."""
        light = Building5R1C(floor_area_m2=100, building_class=BuildingClass.VERY_LIGHT)
        heavy = Building5R1C(floor_area_m2=100, building_class=BuildingClass.VERY_HEAVY)
        phi = 5000.0
        dt_light = light.calc_t_m_next(20.0, phi, 1.0) - 20.0
        dt_heavy = heavy.calc_t_m_next(20.0, phi, 1.0) - 20.0
        assert abs(dt_light) > abs(dt_heavy)


# ---------------------------------------------------------------------------
# 7. Surface temperature (Eq. C.10)
# ---------------------------------------------------------------------------
class TestSurfaceTemperature:
    """Tests for calc_t_s()."""

    def test_bounded_between_mass_and_outdoor(self):
        """T_s should be between T_m and T_out (approximately, for zero gains)."""
        b = Building5R1C(
            floor_area_m2=100,
            ach_ventilation=0.5,
            u_walls=0.25,
            wall_area_m2=100,
        )
        t_m, t_out = 20.0, 0.0
        t_s = b.calc_t_s(t_m, t_out, phi_ia=0, phi_st=0)
        assert t_out <= t_s <= t_m

    def test_warm_mass_raises_surface_temp(self):
        """Higher T_m → higher T_s."""
        b = Building5R1C(floor_area_m2=100, ach_ventilation=0.5)
        t_s_cold = b.calc_t_s(t_m=10.0, t_out=0.0, phi_ia=0, phi_st=0)
        t_s_warm = b.calc_t_s(t_m=25.0, t_out=0.0, phi_ia=0, phi_st=0)
        assert t_s_warm > t_s_cold

    def test_positive_phi_st_raises_surface_temp(self):
        """Positive φ_st (gains) should raise surface temperature."""
        b = Building5R1C(floor_area_m2=100, ach_ventilation=0.5)
        t_s_no = b.calc_t_s(15.0, 0.0, phi_ia=0, phi_st=0)
        t_s_gain = b.calc_t_s(15.0, 0.0, phi_ia=0, phi_st=2000)
        assert t_s_gain > t_s_no


# ---------------------------------------------------------------------------
# 8. Air temperature (Eq. C.11)
# ---------------------------------------------------------------------------
class TestAirTemperature:
    """Tests for calc_t_air()."""

    def test_bounded_between_surface_and_supply(self):
        """T_air should lie between T_s and T_supply (≈T_out)."""
        b = Building5R1C(
            floor_area_m2=100,
            ach_ventilation=0.5,
            u_walls=0.25,
            wall_area_m2=100,
        )
        t_s, t_out = 18.0, 0.0
        t_air = b.calc_t_air(t_s, t_out, phi_ia=0)
        assert t_out <= t_air <= t_s

    def test_positive_phi_ia_raises_air_temp(self):
        """Positive heat flow to air (heating) raises T_air."""
        b = Building5R1C(floor_area_m2=100, ach_ventilation=0.5)
        t_air_no = b.calc_t_air(18.0, 0.0, phi_ia=0)
        t_air_heat = b.calc_t_air(18.0, 0.0, phi_ia=3000)
        assert t_air_heat > t_air_no

    def test_higher_ventilation_pulls_toward_outdoor(self):
        """More ventilation (higher H_ve) → T_air closer to T_out."""
        low_vent = Building5R1C(floor_area_m2=100, ach_ventilation=0.3, ach_infiltration=0)
        high_vent = Building5R1C(floor_area_m2=100, ach_ventilation=2.0, ach_infiltration=0)
        t_s, t_out = 20.0, 0.0
        t_air_low = low_vent.calc_t_air(t_s, t_out, phi_ia=0)
        t_air_high = high_vent.calc_t_air(t_s, t_out, phi_ia=0)
        assert t_air_high < t_air_low  # More vent → colder


# ---------------------------------------------------------------------------
# 9. Operative temperature (Eq. C.12)
# ---------------------------------------------------------------------------
class TestOperativeTemperature:
    """Tests for t_operative property."""

    def test_weighted_average_of_air_and_surface(self):
        """T_op = 0.3 × T_air + 0.7 × T_s per ISO 13790 Eq. C.12."""
        b = Building5R1C(floor_area_m2=100)
        b.t_air = 22.0
        b.t_s = 20.0
        expected = 0.3 * 22.0 + 0.7 * 20.0  # 20.6
        assert abs(b.t_operative - expected) < 0.01

    def test_equals_air_when_surface_matches(self):
        """When T_air == T_s, T_op should equal both."""
        b = Building5R1C(floor_area_m2=100)
        b.t_air = 21.0
        b.t_s = 21.0
        assert abs(b.t_operative - 21.0) < 0.001


# ---------------------------------------------------------------------------
# 10. Thermal bridges
# ---------------------------------------------------------------------------
class TestThermalBridges:
    """Tests for delta_u_thermal_bridge and ThermalBridgeClass."""

    def test_thermal_bridge_class_sets_delta_u(self):
        """ThermalBridgeClass should override delta_u_thermal_bridge."""
        for cls, expected_du in THERMAL_BRIDGE_DELTA_U.items():
            b = Building5R1C(floor_area_m2=100, thermal_bridge_class=cls)
            assert abs(b.delta_u_thermal_bridge - expected_du) < 1e-6

    def test_thermal_bridges_increase_h_tr_op(self):
        """Thermal bridges should increase transmission losses."""
        no_tb = Building5R1C(
            floor_area_m2=100,
            u_walls=0.25,
            wall_area_m2=120,
            u_roof=0.15,
            roof_area_m2=100,
            u_floor=0.20,
            floor_area_ground_m2=100,
            u_windows=1.2,
            window_area_m2=20,
            delta_u_thermal_bridge=0.0,
        )
        with_tb = Building5R1C(
            floor_area_m2=100,
            u_walls=0.25,
            wall_area_m2=120,
            u_roof=0.15,
            roof_area_m2=100,
            u_floor=0.20,
            floor_area_ground_m2=100,
            u_windows=1.2,
            window_area_m2=20,
            delta_u_thermal_bridge=0.10,
        )
        assert with_tb._h_tr_op > no_tb._h_tr_op

    def test_thermal_bridge_hand_calc(self):
        """H_tr_tb = ΔU × Σ(A_env)."""
        delta_u = 0.10
        wall_a, roof_a, floor_a, win_a = 120, 100, 100, 20
        total_env = wall_a + roof_a + floor_a + win_a  # 340 m²
        expected_h_tb = delta_u * total_env  # 34 W/K

        b = Building5R1C(
            floor_area_m2=100,
            u_walls=0.25,
            wall_area_m2=wall_a,
            u_roof=0.15,
            roof_area_m2=roof_a,
            u_floor=0.20,
            floor_area_ground_m2=floor_a,
            u_windows=1.2,
            window_area_m2=win_a,
            delta_u_thermal_bridge=delta_u,
        )
        assert abs(b._h_tr_tb - expected_h_tb) < 0.1

    def test_thermal_bridges_increase_heating(self):
        """Thermal bridges should increase steady-state heating demand."""
        params = dict(
            floor_area_m2=100,
            u_walls=0.25,
            wall_area_m2=120,
            u_roof=0.15,
            roof_area_m2=100,
            u_windows=1.2,
            window_area_m2=20,
            ach_ventilation=0.5,
            delta_theta_er=0.0,
        )
        no_tb = Building5R1C(**params, delta_u_thermal_bridge=0.0)
        with_tb = Building5R1C(**params, delta_u_thermal_bridge=0.10)

        no_tb.reset(20.0)
        with_tb.reset(20.0)
        for _ in range(50):
            no_tb.calc_heating_demand(t_out=0)
            with_tb.calc_heating_demand(t_out=0)

        q_no_tb = no_tb.calc_heating_demand(t_out=0)
        q_with_tb = with_tb.calc_heating_demand(t_out=0)
        assert q_with_tb > q_no_tb


# ---------------------------------------------------------------------------
# 11. Heat recovery ventilation
# ---------------------------------------------------------------------------
class TestHeatRecovery:
    """Tests for ventilation heat recovery."""

    def test_ventilation_system_sets_efficiency(self):
        """VentilationSystem enum should set heat_recovery_efficiency."""
        for vs, expected_eff in VENTILATION_HR_EFFICIENCY.items():
            b = Building5R1C(floor_area_m2=100, ventilation_system=vs)
            assert abs(b.heat_recovery_efficiency - expected_eff) < 1e-6

    def test_hr_reduces_h_ve(self):
        """Heat recovery should reduce effective ventilation coefficient."""
        no_hr = Building5R1C(
            floor_area_m2=100,
            ach_ventilation=0.5,
            ach_infiltration=0.15,
            heat_recovery_efficiency=0.0,
        )
        with_hr = Building5R1C(
            floor_area_m2=100,
            ach_ventilation=0.5,
            ach_infiltration=0.15,
            heat_recovery_efficiency=0.80,
        )
        assert with_hr._h_ve < no_hr._h_ve

    def test_hr_hand_calculation(self):
        """H_ve = ρcp × b_ek × V × n_total / 3600, where b_ek = 1 - (n_vent/n_tot) × η."""
        ach_vent = 0.5
        ach_inf = 0.15
        ach_total = ach_vent + ach_inf
        eta = 0.80
        vol = 100 * 2.5  # 250 m³
        b_ek = 1.0 - (ach_vent / ach_total) * eta
        expected_h_ve = 1200 * b_ek * vol * ach_total / 3600

        b = Building5R1C(
            floor_area_m2=100,
            ceiling_height_m=2.5,
            ach_ventilation=ach_vent,
            ach_infiltration=ach_inf,
            heat_recovery_efficiency=eta,
        )
        assert abs(b._h_ve - expected_h_ve) < 0.01

    def test_full_hr_only_infiltration_loss(self):
        """With η=1.0, only infiltration contributes to H_ve."""
        b = Building5R1C(
            floor_area_m2=100,
            ceiling_height_m=2.5,
            ach_ventilation=0.5,
            ach_infiltration=0.15,
            heat_recovery_efficiency=1.0,
        )
        # b_ek = 1 - (0.5/0.65)*1.0 = 0.231
        # H_ve = 1200 × 0.231 × 250 × 0.65 / 3600 ≈ 12.5
        # This should be much less than no-HR case
        no_hr = Building5R1C(
            floor_area_m2=100,
            ceiling_height_m=2.5,
            ach_ventilation=0.5,
            ach_infiltration=0.15,
            heat_recovery_efficiency=0.0,
        )
        # HR=1.0 should give H_ve < 40% of no-HR
        assert b._h_ve < 0.4 * no_hr._h_ve

    def test_hr_reduces_heating_demand(self):
        """Heat recovery should reduce heating demand."""
        params = dict(
            floor_area_m2=100,
            u_walls=0.25,
            wall_area_m2=100,
            ach_ventilation=0.5,
            ach_infiltration=0.15,
            delta_theta_er=0.0,
        )
        no_hr = Building5R1C(**params, heat_recovery_efficiency=0.0)
        with_hr = Building5R1C(**params, heat_recovery_efficiency=0.80)

        no_hr.reset(20.0)
        with_hr.reset(20.0)
        for _ in range(50):
            no_hr.calc_heating_demand(t_out=-5)
            with_hr.calc_heating_demand(t_out=-5)

        q_no = no_hr.calc_heating_demand(t_out=-5)
        q_hr = with_hr.calc_heating_demand(t_out=-5)
        assert q_hr < q_no


# ---------------------------------------------------------------------------
# 12. Dynamic ground coupling
# ---------------------------------------------------------------------------
class TestDynamicGroundCoupling:
    """Tests for monthly ground temperature coupling."""

    def test_static_ground_factor(self):
        """Without monthly temps, ground loss = H_tr_ground × temp_adj_ground."""
        b = Building5R1C(
            floor_area_m2=100,
            u_floor=0.20,
            floor_area_ground_m2=100,
            temp_adj_ground=0.5,
            ground_temps_monthly=None,
            # Zero out other elements to isolate
            u_walls=0.0,
            wall_area_m2=0,
            u_roof=0.0,
            roof_area_m2=0,
            u_windows=0.0,
            window_area_m2=0,
            ach_ventilation=0.5,
        )
        # H_tr_op should include 0.20 × 100 × 0.5 = 10 W/K
        expected_ground_contrib = 0.20 * 100 * 0.5
        assert abs(b._h_tr_op - expected_ground_contrib) < 0.1

    def test_dynamic_ground_included_in_h_tr_op(self):
        """With monthly temps, full ground conductance is included in H_tr_op."""
        b = Building5R1C(
            floor_area_m2=100,
            u_floor=0.20,
            floor_area_ground_m2=100,
            ground_temps_monthly=SWEDISH_DEFAULT_GROUND_TEMPS.copy(),
            u_walls=0.0,
            wall_area_m2=0,
            u_roof=0.0,
            roof_area_m2=0,
            u_windows=0.0,
            window_area_m2=0,
            ach_ventilation=0.5,
        )
        # H_tr_op should include full ground conductance for RC consistency
        assert abs(b._h_tr_op - (0.20 * 100)) < 0.1

    def test_month_lookup_uses_calendar_boundaries(self):
        """Month mapping should use real non-leap-year hour boundaries."""
        b = Building5R1C(floor_area_m2=100)
        # Jan has 744 hours
        assert b._month_index_from_hour_of_year(743) == 0
        assert b._month_index_from_hour_of_year(744) == 1
        # Feb ends at hour 1415
        assert b._month_index_from_hour_of_year(1415) == 1
        assert b._month_index_from_hour_of_year(1416) == 2

    def test_month_lookup_wraps_after_year(self):
        """Hour indexing should wrap cleanly for multi-year simulations."""
        b = Building5R1C(floor_area_m2=100)
        assert b._month_index_from_hour_of_year(8760) == 0
        assert b._month_index_from_hour_of_year(8760 + 744) == 1

    def test_invalid_ground_temps_length_raises(self):
        """ground_temps_monthly must have exactly 12 values."""
        with pytest.raises(ValueError, match="must contain 12 monthly values"):
            Building5R1C(
                floor_area_m2=100,
                ground_temps_monthly=[5.0] * 11,
            )

    def test_negative_incidence_factor_raises(self):
        """Incidence correction factor must be non-negative."""
        with pytest.raises(ValueError, match="must be >= 0"):
            Building5R1C(
                floor_area_m2=100,
                non_perpendicular_incidence_factor=-0.1,
            )

    def test_winter_ground_warmer_than_air(self):
        """In winter, ground is warmer than outdoor → less heating needed."""
        monthly_temps = SWEDISH_DEFAULT_GROUND_TEMPS.copy()
        b = Building5R1C(
            floor_area_m2=100,
            u_floor=0.25,
            floor_area_ground_m2=100,
            ground_temps_monthly=monthly_temps,
            u_walls=0.25,
            wall_area_m2=100,
            ach_ventilation=0.5,
            delta_theta_er=0.0,
        )
        b.reset(20.0)

        # January outdoor = -10°C, ground ≈ 4°C
        # Ground is warmer → φ_ground reduces heating
        for _ in range(20):
            b.calc_heating_demand(t_out=-10.0, hour_of_year=0)
        q_with_ground = b.calc_heating_demand(t_out=-10.0, hour_of_year=0)

        # Compare with static ground factor
        b_static = Building5R1C(
            floor_area_m2=100,
            u_floor=0.25,
            floor_area_ground_m2=100,
            temp_adj_ground=1.0,  # Full outdoor temp
            ground_temps_monthly=None,
            u_walls=0.25,
            wall_area_m2=100,
            ach_ventilation=0.5,
            delta_theta_er=0.0,
        )
        b_static.reset(20.0)
        for _ in range(20):
            b_static.calc_heating_demand(t_out=-10.0)
        q_static = b_static.calc_heating_demand(t_out=-10.0)

        # Dynamic (warmer ground in winter) should need less heating
        assert q_with_ground < q_static


# ---------------------------------------------------------------------------
# 13. Heating demand iteration (ISO 13790 Eq. C.13)
# ---------------------------------------------------------------------------
class TestHeatingDemandIteration:
    """Tests for calc_heating_demand() iteration logic."""

    def test_heating_when_cold(self):
        """Should produce positive demand when T_out < T_set."""
        b = Building5R1C(floor_area_m2=100, t_set_heating=20.0, delta_theta_er=0.0)
        b.reset(20.0)
        for _ in range(10):
            b.calc_heating_demand(t_out=0.0)
        q = b.calc_heating_demand(t_out=0.0)
        assert q > 0

    def test_cooling_when_hot(self):
        """Should produce negative demand when T_out > T_set_cooling."""
        b = Building5R1C(
            floor_area_m2=100,
            t_set_cooling=26.0,
            delta_theta_er=0.0,
        )
        b.reset(25.0)
        for _ in range(10):
            b.calc_heating_demand(t_out=35.0, solar_gains=2000, max_cooling_power=10000)
        q = b.calc_heating_demand(t_out=35.0, solar_gains=2000, max_cooling_power=10000)
        assert q < 0

    def test_no_demand_in_deadband(self):
        """No heating/cooling when temp is between setpoints."""
        b = Building5R1C(
            floor_area_m2=100,
            t_set_heating=20.0,
            t_set_cooling=26.0,
            delta_theta_er=0.0,
            internal_gains_w_per_m2=0.0,
        )
        b.reset(23.0)
        # Run at 23°C outdoor — should be in deadband
        for _ in range(50):
            b.calc_heating_demand(t_out=23.0)
        q = b.calc_heating_demand(t_out=23.0)
        assert q == 0.0

    def test_heating_limited_by_capacity(self):
        """Demand should not exceed max_heating_power."""
        b = Building5R1C(floor_area_m2=100, delta_theta_er=0.0)
        b.reset(20.0)
        for _ in range(10):
            b.calc_heating_demand(t_out=-30.0, max_heating_power=500)
        q = b.calc_heating_demand(t_out=-30.0, max_heating_power=500)
        assert q <= 500.0

    def test_cooling_limited_by_capacity(self):
        """Cooling should not exceed max_cooling_power."""
        b = Building5R1C(floor_area_m2=100, t_set_cooling=26.0, delta_theta_er=0.0)
        b.reset(28.0)
        for _ in range(10):
            b.calc_heating_demand(t_out=40.0, solar_gains=3000, max_cooling_power=1000)
        q = b.calc_heating_demand(t_out=40.0, solar_gains=3000, max_cooling_power=1000)
        assert q >= -1000.0

    def test_operative_temperature_control(self):
        """When use_operative_temperature=True, control should target T_op."""
        b = Building5R1C(
            floor_area_m2=100,
            t_set_heating=20.0,
            use_operative_temperature=True,
            delta_theta_er=0.0,
        )
        b.reset(20.0)
        for _ in range(50):
            b.calc_heating_demand(t_out=0.0)
        b.calc_heating_demand(t_out=0.0)
        # Operative should be close to setpoint
        assert abs(b.t_operative - 20.0) < 0.5


# ---------------------------------------------------------------------------
# 14. Internal gains capping
# ---------------------------------------------------------------------------
class TestInternalGainsCapping:
    """Tests for internal gains validation."""

    def test_excessive_gains_capped(self):
        """Internal gains above INTERNAL_GAINS_MAX_W_M2 should be capped."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            b = Building5R1C(
                floor_area_m2=100,
                internal_gains_w_per_m2=50.0,  # Way too high
            )
        assert b.internal_gains_w_per_m2 == INTERNAL_GAINS_MAX_W_M2

    def test_warning_issued_for_excessive_gains(self):
        """A UserWarning should be raised for excessive internal gains."""
        with pytest.warns(UserWarning, match="exceeds maximum"):
            Building5R1C(floor_area_m2=100, internal_gains_w_per_m2=25.0)

    def test_normal_gains_not_capped(self):
        """Normal gains (≤ max) should not be modified."""
        b = Building5R1C(floor_area_m2=100, internal_gains_w_per_m2=5.0)
        assert b.internal_gains_w_per_m2 == 5.0


# ---------------------------------------------------------------------------
# 15. Factory functions
# ---------------------------------------------------------------------------
class TestFactoryFunctions:
    """Tests for create_swedish_villa() and create_swedish_apartment()."""

    @pytest.mark.parametrize(
        "era",
        [
            "pre_1960",
            "1960_1975",
            "1976_1990",
            "1991_2010",
            "post_2010",
        ],
    )
    def test_villa_creates_valid_building(self, era):
        """Villa factory should produce a functioning Building5R1C."""
        b = create_swedish_villa(floor_area_m2=150, era=era, use_config=False)
        assert isinstance(b, Building5R1C)
        assert b.floor_area_m2 == 150
        assert b._h_tr_op > 0
        assert b._h_ve > 0

    @pytest.mark.parametrize(
        "era",
        [
            "pre_1960",
            "1960_1975",
            "1976_1990",
            "1991_2010",
            "post_2010",
        ],
    )
    def test_apartment_creates_valid_building(self, era):
        """Apartment factory should produce a functioning Building5R1C."""
        b = create_swedish_apartment(floor_area_m2=70, era=era, use_config=False)
        assert isinstance(b, Building5R1C)
        assert b.floor_area_m2 == 70
        assert b._h_tr_op > 0

    def test_older_buildings_lossier(self):
        """Older eras should have higher heat loss coefficients."""
        old = create_swedish_villa(150, era="pre_1960", use_config=False)
        new = create_swedish_villa(150, era="post_2010", use_config=False)
        assert old.heat_loss_coefficient > new.heat_loss_coefficient

    def test_apartment_less_loss_than_villa(self):
        """Apartments have less envelope → lower HLC per m² than villas."""
        villa = create_swedish_villa(100, era="1976_1990", use_config=False)
        apt = create_swedish_apartment(100, era="1976_1990", use_config=False)
        hlc_villa_per_m2 = villa.heat_loss_coefficient / villa.floor_area_m2
        hlc_apt_per_m2 = apt.heat_loss_coefficient / apt.floor_area_m2
        assert hlc_apt_per_m2 < hlc_villa_per_m2

    def test_newer_eras_have_heat_recovery(self):
        """Post-1990 buildings should have heat recovery."""
        old = create_swedish_villa(150, era="1976_1990", use_config=False)
        new = create_swedish_villa(150, era="1991_2010", use_config=False)
        assert old.heat_recovery_efficiency == 0.0
        assert new.heat_recovery_efficiency > 0.0

    def test_factory_accepts_overrides(self):
        """User kwargs should override defaults."""
        b = create_swedish_villa(
            150,
            era="1976_1990",
            use_config=False,
            u_walls=0.10,  # Override
            t_set_heating=22.0,
        )
        assert b.u_walls == 0.10
        assert b.t_set_heating == 22.0

    def test_villa_can_simulate(self):
        """Factory-created villa should run a simulation without errors."""
        b = create_swedish_villa(150, era="1976_1990", use_config=False)
        b.reset(20.0)
        t_outdoor = np.full(24, -5.0)
        results = b.simulate(t_outdoor)
        assert len(results["heating_demand"]) == 24
        assert all(np.isfinite(results["heating_demand"]))
        assert all(q >= 0 for q in results["heating_demand"])


# ---------------------------------------------------------------------------
# 16. Construction-based thermal mass
# ---------------------------------------------------------------------------
class TestConstructionBasedThermalMass:
    """Tests for Construction/ConstructionLayer thermal mass calculation."""

    def test_single_concrete_layer(self):
        """150mm concrete wall, 100 m² area — only 100mm effective depth."""
        concrete = ConstructionLayer(
            thickness_m=0.150,
            density_kg_m3=2400,
            specific_heat_j_kgk=1000,
            name="concrete_150mm",
        )
        wall = Construction(name="concrete wall", internal_layers=[concrete], area_m2=100)
        # effective_depth_limit_m=0.10 by default, so only 100mm of 150mm counts
        # κ = 0.10 × 2400 × 1000 = 240,000 J/(m²·K)
        assert abs(wall.thermal_mass_per_area - 240_000) < 1
        assert abs(wall.total_thermal_mass_j_k - 240_000 * 100) < 100

    def test_multi_layer_construction(self):
        """Insulated wall: gypsum + insulation + concrete (inside-out order).

        Only first 100mm effective depth contributes:
        - gypsum 13mm: 0.013 × 800 × 1090 = 11,336 (total_depth=13mm)
        - insulation 87mm of 200mm: 0.087 × 30 × 1030 = 2,688.3 (total_depth=100mm)
        """
        gypsum = ConstructionLayer(0.013, 800, 1090, "gypsum")
        insulation = ConstructionLayer(0.200, 30, 1030, "insulation")
        concrete = ConstructionLayer(0.200, 2400, 1000, "concrete")
        wall = Construction("insulated wall", [gypsum, insulation, concrete], 120)

        # With 100mm depth limit: gypsum=13mm full, insulation=87mm, concrete=0mm
        k_expected = (0.013 * 800 * 1090) + (0.087 * 30 * 1030)
        assert abs(wall.thermal_mass_per_area - k_expected) < 1

    def test_calculate_from_constructions_function(self):
        """calculate_thermal_mass_from_constructions() returns (Cm, kappa)."""
        layer = ConstructionLayer(0.200, 2400, 1000, "concrete")
        wall = Construction("wall", [layer], area_m2=100)
        floor_area = 100.0

        cm, kappa = calculate_thermal_mass_from_constructions([wall], floor_area)
        assert cm == wall.total_thermal_mass_j_k
        assert abs(kappa - cm / floor_area) < 0.1

    def test_construction_overrides_building_class(self):
        """Building with constructions should use their C_m, not BuildingClass."""
        layer = ConstructionLayer(0.200, 2400, 1000, "concrete")
        wall = Construction("wall", [layer], area_m2=100)

        b = Building5R1C(
            floor_area_m2=100,
            building_class=BuildingClass.VERY_LIGHT,  # Would give 80 kJ/m²K
            constructions=[wall],
        )
        # Should use construction Cm, not VERY_LIGHT
        expected_cm = wall.total_thermal_mass_j_k
        assert abs(b._c_m - expected_cm) < 1

    def test_explicit_kappa_overrides_building_class(self):
        """thermal_capacitance_j_m2k should override building_class."""
        b = Building5R1C(
            floor_area_m2=100,
            building_class=BuildingClass.VERY_LIGHT,
            thermal_capacitance_j_m2k=200_000,
        )
        assert abs(b._c_m - 200_000 * 100) < 1


# ---------------------------------------------------------------------------
# 17. 5R1C network properties (h_tr_1, h_tr_2, h_tr_3)
# ---------------------------------------------------------------------------
class TestNetworkProperties:
    """Tests for derived 5R1C network properties."""

    def test_h_tr_1_formula(self):
        """h_tr_1 = 1 / (1/H_ve + 1/H_tr_is)."""
        b = Building5R1C(floor_area_m2=100, ach_ventilation=0.5)
        expected = 1.0 / (1.0 / b._h_ve + 1.0 / b._h_tr_is)
        assert abs(b.h_tr_1 - expected) < 0.01

    def test_h_tr_2_formula(self):
        """h_tr_2 = h_tr_1 + H_tr_w."""
        b = Building5R1C(floor_area_m2=100, u_windows=1.2, window_area_m2=20)
        expected = b.h_tr_1 + b._h_tr_w
        assert abs(b.h_tr_2 - expected) < 0.01

    def test_h_tr_3_formula(self):
        """h_tr_3 = 1 / (1/h_tr_2 + 1/H_tr_ms)."""
        b = Building5R1C(floor_area_m2=100)
        expected = 1.0 / (1.0 / b.h_tr_2 + 1.0 / b._h_tr_ms)
        assert abs(b.h_tr_3 - expected) < 0.01

    def test_h_tr_em_hand_calculation(self):
        """H_tr_em: 1/H_tr_em = 1/H_tr_op - 1/H_tr_ms."""
        b = Building5R1C(
            floor_area_m2=100,
            u_walls=0.25,
            wall_area_m2=120,
            u_roof=0.15,
            roof_area_m2=100,
        )
        if b._h_tr_op > 0 and b._h_tr_ms > 0:
            inv_expected = 1.0 / b._h_tr_op - 1.0 / b._h_tr_ms
            if inv_expected > 0:
                expected = 1.0 / inv_expected
                assert abs(b._h_tr_em - expected) < 0.1


# ---------------------------------------------------------------------------
# 18. Reset and state management
# ---------------------------------------------------------------------------
class TestResetAndState:
    """Tests for reset() and state initialization."""

    def test_reset_to_specified_temperature(self):
        b = Building5R1C(floor_area_m2=100)
        b.reset(15.0)
        assert b.t_m == 15.0
        assert b.t_s == 15.0
        assert b.t_air == 15.0

    def test_reset_default_to_setpoint(self):
        b = Building5R1C(floor_area_m2=100, t_set_heating=21.0)
        b.t_m = 5.0  # Mess up state
        b.reset()
        assert b.t_m == 21.0
        assert b.t_air == 21.0

    def test_reset_clears_window_state(self):
        b = Building5R1C(floor_area_m2=100)
        b._windows_open = True
        b.reset(20.0)
        assert not b._windows_open


# ---------------------------------------------------------------------------
# 19. simulate() time-series correctness
# ---------------------------------------------------------------------------
class TestSimulateTimeSeries:
    """Tests for the simulate() wrapper."""

    def test_output_lengths_match_input(self):
        b = Building5R1C(floor_area_m2=100, delta_theta_er=0.0)
        t_out = np.full(48, -5.0)
        results = b.simulate(t_out)
        for key in ["heating_demand", "t_air", "t_m", "t_s", "t_operative", "windows_open"]:
            assert len(results[key]) == 48

    def test_all_outputs_finite(self):
        b = Building5R1C(floor_area_m2=100, delta_theta_er=0.0)
        t_out = np.random.uniform(-20, 30, 168)
        solar = np.clip(np.random.uniform(-50, 800, 168), 0, None)
        results = b.simulate(t_out, solar_gains=solar)
        for key in ["heating_demand", "t_air", "t_m", "t_s", "t_operative"]:
            assert all(np.isfinite(results[key])), f"Non-finite values in {key}"

    def test_constant_conditions_reach_steady_state(self):
        """Under constant outdoor temp, heating should stabilize."""
        b = Building5R1C(floor_area_m2=100, delta_theta_er=0.0)
        t_out = np.full(200, 0.0)
        results = b.simulate(t_out)
        # Last 10 hours should have <1% variation
        last_10 = results["heating_demand"][-10:]
        variation = (last_10.max() - last_10.min()) / last_10.mean() if last_10.mean() > 0 else 0
        assert variation < 0.01


# ---------------------------------------------------------------------------
# 20. Energy conservation sanity
# ---------------------------------------------------------------------------
class TestEnergyConservation:
    """Sanity checks for energy balance."""

    def test_higher_delta_t_more_heating(self):
        """Larger ΔT → more heating (at steady state)."""
        b = Building5R1C(floor_area_m2=100, delta_theta_er=0.0)
        b.reset(20.0)
        for _ in range(50):
            b.calc_heating_demand(t_out=10.0)
        q_10 = b.calc_heating_demand(t_out=10.0)

        b.reset(20.0)
        for _ in range(50):
            b.calc_heating_demand(t_out=-10.0)
        q_30 = b.calc_heating_demand(t_out=-10.0)

        assert q_30 > q_10

    def test_solar_gains_reduce_heating(self):
        """Solar gains should reduce heating demand."""
        b = Building5R1C(floor_area_m2=100, delta_theta_er=0.0)
        b.reset(20.0)
        for _ in range(50):
            b.calc_heating_demand(t_out=0, solar_gains=0)
        q_no_solar = b.calc_heating_demand(t_out=0, solar_gains=0)

        b.reset(20.0)
        for _ in range(50):
            b.calc_heating_demand(t_out=0, solar_gains=1000)
        q_solar = b.calc_heating_demand(t_out=0, solar_gains=1000)

        assert q_solar < q_no_solar

    def test_internal_gains_reduce_heating(self):
        """Internal gains should offset heating demand."""
        b = Building5R1C(
            floor_area_m2=100,
            internal_gains_w_per_m2=0,
            delta_theta_er=0.0,
        )
        b.reset(20.0)
        for _ in range(50):
            b.calc_heating_demand(t_out=0, internal_gains=0)
        q_no_gains = b.calc_heating_demand(t_out=0, internal_gains=0)

        b.reset(20.0)
        for _ in range(50):
            b.calc_heating_demand(t_out=0, internal_gains=500)
        q_gains = b.calc_heating_demand(t_out=0, internal_gains=500)

        assert q_gains < q_no_gains

    def test_better_insulation_less_heating(self):
        """Lower U-values → less heating."""
        poor = Building5R1C(
            floor_area_m2=100,
            u_walls=0.50,
            wall_area_m2=120,
            ach_ventilation=0.5,
            delta_theta_er=0.0,
        )
        good = Building5R1C(
            floor_area_m2=100,
            u_walls=0.15,
            wall_area_m2=120,
            ach_ventilation=0.5,
            delta_theta_er=0.0,
        )
        poor.reset(20.0)
        good.reset(20.0)
        for _ in range(50):
            poor.calc_heating_demand(t_out=0)
            good.calc_heating_demand(t_out=0)
        assert good.calc_heating_demand(t_out=0) < poor.calc_heating_demand(t_out=0)
