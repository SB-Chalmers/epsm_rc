"""
Unit tests for Heat Pump models.

Tests both:
- HeatPumpHPLib: Empirically validated model using hplib (EHPA Keymark data)
- HeatPumpCarnot: Simplified fallback model using Carnot approximation

Reference: hplib validation shows MAPE of 9.8% for COP predictions.
"""

import pytest
import numpy as np

from epsm_rc.heating import (
    HeatPump, 
    HeatPumpCarnot, 
    HPLIB_AVAILABLE,
    list_available_hp_models,
    get_hp_info,
)


# =============================================================================
# Tests for Carnot Fallback Model (always available)
# =============================================================================

class TestCarnotCOP:
    """Tests for simplified Carnot-based COP model.
    
    NOTE: Real heat pumps follow empirical curves:
    COP = p1*T_in + p2*T_out + p3 + p4*T_amb (hplib)
    with MAPE ~9.8% against Keymark test data.
    """
    
    def test_carnot_cop_formula(self):
        """COP should follow Carnot formula: η × T_sink / (T_sink - T_source)."""
        hp = HeatPumpCarnot(
            carnot_efficiency=0.40,
            supply_temp_c=45.0,  # 318.15 K
        )
        
        # At 7°C (280.15 K): COP = 0.40 × 318.15 / (318.15 - 280.15) = 3.35
        cop_7 = hp.cop(np.array([7.0]))[0]
        expected = 0.40 * 318.15 / (318.15 - 280.15)
        assert np.isclose(cop_7, expected, rtol=0.01)
    
    def test_cop_higher_carnot_efficiency(self):
        """Higher Carnot efficiency should give higher COP."""
        hp_old = HeatPumpCarnot(carnot_efficiency=0.32)
        hp_new = HeatPumpCarnot(carnot_efficiency=0.42)
        
        temps = np.array([0.0, -7.0, -15.0])
        
        cops_old = hp_old.cop(temps)
        cops_new = hp_new.cop(temps)
        
        # New HP should have higher COP at all temperatures
        assert all(cops_new > cops_old)
    
    def test_cop_varies_with_supply_temp(self):
        """Higher supply temp (radiators) gives lower COP than floor heating."""
        hp_floor = HeatPumpCarnot(supply_temp_c=35.0)  # Floor heating
        hp_rad = HeatPumpCarnot(supply_temp_c=55.0)    # Radiators
        
        temps = np.array([-7.0])
        
        cop_floor = hp_floor.cop(temps)[0]
        cop_rad = hp_rad.cop(temps)[0]
        
        # Floor heating should be more efficient (lower lift)
        assert cop_floor > cop_rad
    
    def test_cop_typical_values_by_era(self):
        """Test realistic COP values for different building eras."""
        # Pre-1960: η=0.32, expect COP ~1.9-2.0 @ -7°C
        hp_old = HeatPumpCarnot(carnot_efficiency=0.32, supply_temp_c=45.0)
        cop_old = hp_old.cop(np.array([-7.0]))[0]
        assert 1.8 < cop_old < 2.2
        
        # Modern: η=0.42, expect COP ~2.5-2.7 @ -7°C  
        hp_modern = HeatPumpCarnot(carnot_efficiency=0.42, supply_temp_c=45.0)
        cop_modern = hp_modern.cop(np.array([-7.0]))[0]
        assert 2.4 < cop_modern < 2.8
    
    def test_cop_minimum_clamp(self):
        """COP should not fall below min_cop at extreme cold."""
        hp = HeatPumpCarnot(min_cop=1.0, carnot_efficiency=0.30)
        
        extreme_cold = np.array([-40.0, -50.0])
        cops = hp.cop(extreme_cold)
        
        assert all(cops >= hp.min_cop)


class TestCarnotAutoSizing:
    """Tests for Carnot model auto-sizing."""
    
    def test_sizing_based_on_u_value(self):
        """Compressor size should scale with building heat loss coefficient."""
        hp_small = HeatPumpCarnot(u_total_w_per_k=50.0)
        hp_large = HeatPumpCarnot(u_total_w_per_k=200.0)
        
        assert hp_large.compressor_power_max_w > hp_small.compressor_power_max_w
    
    def test_aux_heater_sized_for_design_temp(self):
        """Aux heater should cover remaining demand at design temperature."""
        hp = HeatPumpCarnot(
            u_total_w_per_k=165.0,
            design_temp_c=-20.0,
            bivalent_temp_c=-7.0,
        )
        
        assert hp.aux_power_max_w >= 2000.0
        
        design_demand = hp.heat_demand_w(np.array([-20.0]))[0]
        total_w, comp_w, aux_w = hp.electrical_load(np.array([-20.0]))
        
        cop_design = hp.cop(np.array([-20.0]))[0]
        heat_from_comp = comp_w[0] * cop_design
        heat_from_aux = aux_w[0]
        total_heat = heat_from_comp + heat_from_aux
        
        assert total_heat >= design_demand * 0.9


class TestCarnotCalculateLoad:
    """Tests for Carnot model load calculation."""
    
    def test_zero_load_above_indoor_temp(self):
        """No heating needed when outdoor >= indoor temp."""
        hp = HeatPumpCarnot(indoor_temp_c=21.0)
        
        warm_temps = np.array([21.0, 25.0, 30.0])
        loads = hp.calculate_load(warm_temps)
        
        assert all(loads == 0)
    
    def test_load_increases_with_cold(self):
        """Load should increase as temperature drops."""
        hp = HeatPumpCarnot()
        
        temps = np.array([10.0, 0.0, -10.0, -20.0])
        loads = hp.calculate_load(temps)
        
        assert all(np.diff(loads) > 0)


# =============================================================================
# Tests for hplib Model (requires hplib)
# =============================================================================

@pytest.mark.skipif(not HPLIB_AVAILABLE, reason="hplib not installed")
class TestHPLibModel:
    """Tests for hplib-based heat pump model."""
    
    def test_hplib_model_initializes(self):
        """HeatPumpHPLib should initialize with Generic model."""
        from epsm_rc.heating import HeatPumpHPLib
        
        hp = HeatPumpHPLib(
            u_total_w_per_k=150.0,
            hp_model="Generic",
            group_id=1,
        )
        
        assert hp._hp_params is not None
        assert hp._p_th_ref > 0
        assert hp.aux_heater_max_w > 0
    
    def test_hplib_simulation_returns_sensible_values(self):
        """hplib simulation should return realistic COP and power values."""
        from epsm_rc.heating import HeatPumpHPLib
        
        hp = HeatPumpHPLib(
            u_total_w_per_k=150.0,
            hp_model="Generic",
            group_id=1,
        )
        
        temps = np.array([-7.0, 0.0, 7.0])
        cop, p_el, p_th = hp.simulate_hplib(temps)
        
        # COP should be in sensible range (1-6 for these conditions)
        assert all(cop >= 1.0)
        assert all(cop <= 8.0)
        
        # Power should be positive
        assert all(p_el > 0)
        assert all(p_th > 0)
        
        # P_th = P_el * COP (approximately)
        assert np.allclose(p_th, p_el * cop, rtol=0.1)
    
    def test_hplib_cop_higher_at_mild_temps(self):
        """COP should increase with warmer outdoor temperatures."""
        from epsm_rc.heating import HeatPumpHPLib
        
        hp = HeatPumpHPLib(u_total_w_per_k=150.0)
        
        temps = np.array([-15.0, -7.0, 0.0, 7.0])
        cop, _, _ = hp.simulate_hplib(temps)
        
        # COP should generally increase with temperature
        assert cop[-1] > cop[0]
    
    def test_hplib_calculate_load_realistic(self):
        """Calculate load should return realistic values for Swedish winter."""
        from epsm_rc.heating import HeatPumpHPLib
        
        hp = HeatPumpHPLib(
            u_total_w_per_k=150.0,  # ~100m² modern villa
            indoor_temp_c=21.0,
        )
        
        # Cold day in Stockholm
        cold_temp = np.array([-15.0])
        load = hp.calculate_load(cold_temp)
        
        # Should be in kW range for villa
        assert 1000 < load[0] < 10000


@pytest.mark.skipif(not HPLIB_AVAILABLE, reason="hplib not installed")
class TestHPLibDatabase:
    """Tests for hplib database access functions."""
    
    def test_list_available_models(self):
        """Should return list of available air-source HP models."""
        models = list_available_hp_models("air")
        
        assert isinstance(models, list)
        assert len(models) > 100  # hplib has 5000+ air/water models
        # Models should be formatted as "Manufacturer Model"
        assert all(isinstance(m, str) for m in models)
    
    def test_get_hp_info_generic(self):
        """Should return info for a model from the database."""
        models = list_available_hp_models("air")
        if models and "hplib not installed" not in models[0]:
            info = get_hp_info(models[0])
            assert "manufacturer" in info
            assert "model" in info
            assert "cop_ref" in info
            assert "mape_cop" in info  # Added accuracy metric
            # Verify numeric values are reasonable
            assert info.get("cop_ref", 0) > 0 or "error" in info


# =============================================================================
# Tests for Factory Function
# =============================================================================

class TestHeatPumpFactory:
    """Tests for HeatPump factory function."""
    
    def test_factory_returns_heating_system(self):
        """Factory should return a HeatingSystem instance."""
        from epsm_rc.heating import HeatingSystem
        
        hp = HeatPump(u_total_w_per_k=150.0)
        
        assert isinstance(hp, HeatingSystem)
    
    def test_factory_respects_parameters(self):
        """Factory should pass parameters to underlying model."""
        hp = HeatPump(
            indoor_temp_c=22.0,
            u_total_w_per_k=100.0,
            supply_temp_c=40.0,
        )
        
        assert hp.indoor_temp_c == 22.0
        assert hp.u_total_w_per_k == 100.0
        assert hp.supply_temp_c == 40.0
    
    def test_factory_calculate_load_works(self):
        """Factory-created HP should have working calculate_load."""
        hp = HeatPump(u_total_w_per_k=150.0)
        
        temps = np.array([-10.0, 0.0, 10.0])
        loads = hp.calculate_load(temps)
        
        assert len(loads) == 3
        assert all(loads >= 0)
        assert loads[0] > loads[2]  # Colder = more load


# =============================================================================
# Integration Tests
# =============================================================================

class TestHeatPumpIntegration:
    """Integration tests for heat pump with building models."""
    
    def test_pre1960_villa_peak_load(self):
        """Pre-1960 villa should have significant peak load."""
        # Pre-1960 U-values (typical Swedish villa)
        u_values = {"u_walls": 0.60, "u_roof": 0.40, "u_floor": 0.50, "u_windows": 2.80}
        # Approximate U-total for 150m² villa
        # Area-weighted: walls ~200m², roof ~75m², floor ~75m², windows ~22m²
        u_total = (
            200 * u_values["u_walls"] +
            75 * u_values["u_roof"] +
            75 * u_values["u_floor"] +
            22 * u_values["u_windows"]
        )
        hp = HeatPump(u_total_w_per_k=u_total, carnot_efficiency=0.32)
        
        peak = hp.calculate_load(np.array([-20.0]))[0]
        
        # Should be substantial: 4-12 kW electrical
        assert 3000 < peak < 15000
    
    def test_modern_villa_lower_peak(self):
        """Modern BBR 2012 villa should have much lower peak."""
        # Post-2010 U-values (modern Swedish villa)
        u_values = {"u_walls": 0.18, "u_roof": 0.13, "u_floor": 0.15, "u_windows": 1.20}
        # Approximate U-total for 150m² villa
        u_total = (
            200 * u_values["u_walls"] +
            75 * u_values["u_roof"] +
            75 * u_values["u_floor"] +
            22 * u_values["u_windows"]
        )
        hp = HeatPump(u_total_w_per_k=u_total, carnot_efficiency=0.42)
        
        peak = hp.calculate_load(np.array([-20.0]))[0]
        
        # Should be low: 500-4000W electrical
        assert 300 < peak < 5000
    
    def test_weekly_simulation_energy(self):
        """Test energy consumption over a week with varying temperatures."""
        hp = HeatPump(u_total_w_per_k=100.0, carnot_efficiency=0.40)
        
        temps = np.linspace(-15, -5, 7*24*6)  # 10-min resolution, 7 days
        loads = hp.calculate_load(temps)
        
        weekly_kwh = loads.sum() * (10/60) / 1000
        
        # Should be in realistic range for Swedish winter week
        assert 50 < weekly_kwh < 600
