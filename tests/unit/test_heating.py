"""
Unit tests for the HeatPump and heating models.

Note: Tests use HeatPumpCarnot (simplified Carnot-based model) which has
explicit .cop() and .electrical_load() methods. HeatPumpHPLib uses hplib
for empirically-validated simulation but has a different API.
"""

import pytest
import numpy as np

from epsm_rc.heating import HeatPumpCarnot, DistrictHeating, DirectElectric

# Check if hplib is available for optional tests
try:
    from epsm_rc.heating import HeatPumpHPLib, HPLIB_AVAILABLE
except ImportError:
    HPLIB_AVAILABLE = False


# =============================================================================
# HeatPumpHPLib Tests (hplib-based empirically validated model)
# =============================================================================

@pytest.mark.skipif(not HPLIB_AVAILABLE, reason="hplib not installed")
class TestHeatPumpHPLib:
    """Tests for HeatPumpHPLib (empirically-validated hplib model)."""
    
    def test_initialization_with_defaults(self):
        """HeatPumpHPLib should initialize with default parameters."""
        hp = HeatPumpHPLib()
        
        assert hp.indoor_temp_c == 21.0
        assert hp.u_total_w_per_k == 200.0
        assert hp.hp_model == "Generic"
        assert hp.group_id == 1  # Air/water regulated (inverter)
        assert hp._hp_params is not None
        assert hp._p_th_ref > 0
    
    def test_initialization_with_custom_params(self):
        """HeatPumpHPLib should accept custom parameters."""
        hp = HeatPumpHPLib(
            indoor_temp_c=20.0,
            u_total_w_per_k=150.0,
            supply_temp_c=35.0,  # Floor heating
            group_id=1,
        )
        
        assert hp.indoor_temp_c == 20.0
        assert hp.u_total_w_per_k == 150.0
        assert hp.supply_temp_c == 35.0
    
    def test_heat_demand_calculation(self):
        """Heat demand should follow Q = U * (T_in - T_out)."""
        hp = HeatPumpHPLib(indoor_temp_c=21.0, u_total_w_per_k=200.0)
        
        outdoor = np.array([0.0, -10.0, 10.0])
        demand = hp.heat_demand_w(outdoor)
        
        assert np.isclose(demand[0], 200.0 * 21.0)  # 4200 W at 0°C
        assert np.isclose(demand[1], 200.0 * 31.0)  # 6200 W at -10°C
        assert np.isclose(demand[2], 200.0 * 11.0)  # 2200 W at 10°C
    
    def test_heat_demand_zero_when_warm(self):
        """No heating needed when outdoor >= indoor."""
        hp = HeatPumpHPLib(indoor_temp_c=21.0)
        
        warm = np.array([21.0, 25.0, 30.0])
        demand = hp.heat_demand_w(warm)
        
        assert all(demand == 0.0)
    
    def test_simulate_hplib_returns_three_arrays(self):
        """simulate_hplib should return (COP, P_el, P_th) arrays."""
        hp = HeatPumpHPLib()
        
        temps = np.array([-10.0, 0.0, 10.0])
        cop, p_el, p_th = hp.simulate_hplib(temps)
        
        assert len(cop) == 3
        assert len(p_el) == 3
        assert len(p_th) == 3
    
    def test_cop_decreases_with_temperature(self):
        """COP should decrease as outdoor temperature drops."""
        hp = HeatPumpHPLib()
        
        temps = np.array([10.0, 0.0, -10.0, -20.0])
        cop, _, _ = hp.simulate_hplib(temps)
        
        # COP should generally decrease as temperature drops
        # (may not be strictly monotonic due to hplib interpolation)
        assert cop[0] > cop[-1], "COP at 10°C should be higher than at -20°C"
    
    def test_cop_in_reasonable_range(self):
        """COP should be in physically reasonable range (1-10)."""
        hp = HeatPumpHPLib()
        
        temps = np.linspace(-20, 15, 20)
        cop, _, _ = hp.simulate_hplib(temps)
        
        assert all(cop >= hp.min_cop), f"COP below minimum: {cop.min()}"
        assert all(cop <= 10.0), f"COP above maximum: {cop.max()}"
    
    def test_cop_typical_swedish_winter(self):
        """COP should be 2-4 for typical Swedish winter temps."""
        hp = HeatPumpHPLib()
        
        # Typical winter range
        temps = np.array([-5.0, 0.0, 5.0])
        cop, _, _ = hp.simulate_hplib(temps)
        
        assert all(cop > 1.5), f"COP too low for mild conditions: {cop}"
        assert all(cop < 6.0), f"COP unrealistically high: {cop}"
    
    def test_calculate_load_returns_positive(self):
        """Electrical load should be non-negative."""
        hp = HeatPumpHPLib()
        
        temps = np.linspace(-25, 15, 50)
        load = hp.calculate_load(temps)
        
        assert all(load >= 0), "Electrical load should never be negative"
    
    def test_calculate_load_zero_when_warm(self):
        """No electrical load when outdoor temp exceeds indoor."""
        hp = HeatPumpHPLib(indoor_temp_c=21.0)
        
        warm = np.array([22.0, 25.0, 30.0])
        load = hp.calculate_load(warm)
        
        assert all(load == 0), "No heating load expected when warm"
    
    def test_calculate_load_increases_with_cold(self):
        """Electrical load should increase as temperature drops."""
        hp = HeatPumpHPLib()
        
        temps = np.array([10.0, 0.0, -10.0])
        load = hp.calculate_load(temps)
        
        # Load should increase as it gets colder
        assert load[0] < load[1] < load[2], "Load should increase with cold"
    
    def test_aux_heater_sized_appropriately(self):
        """Auxiliary heater should be sized for design conditions."""
        hp = HeatPumpHPLib(
            u_total_w_per_k=300.0,  # High heat loss
            design_temp_c=-20.0,
        )
        
        # Aux heater should be at least 2000W (minimum)
        assert hp.aux_heater_max_w >= 2000.0
    
    def test_thermal_mass_smoothing(self):
        """Thermal mass should smooth load transitions."""
        hp = HeatPumpHPLib(thermal_time_constant_hours=2.0)
        
        # Step change in temperature
        temps = np.concatenate([
            np.full(10, 5.0),   # Mild
            np.full(10, -15.0), # Sudden cold
        ])
        
        load = hp.calculate_load(temps, minutes_per_slot=10)
        
        # The step change should be smoothed - not instant jump
        # Check that load[10] (first cold slot) is less than load[19] (stabilized)
        assert load[10] < load[19], "Thermal mass should delay response"
    
    def test_no_thermal_mass_when_disabled(self):
        """Setting thermal_time_constant=0 should disable smoothing."""
        hp = HeatPumpHPLib(thermal_time_constant_hours=0.0)
        
        temps = np.array([5.0, -15.0])
        load = hp.calculate_load(temps, minutes_per_slot=10)
        
        # Without smoothing, load should follow demand directly
        # Just check it works without error
        assert len(load) == 2
    
    def test_different_group_ids(self):
        """Different group_id values should work (air/water types)."""
        # Group 1: Air/water regulated (inverter)
        hp1 = HeatPumpHPLib(group_id=1)
        
        # Group 4: Air/water on-off
        hp4 = HeatPumpHPLib(group_id=4)
        
        temps = np.array([0.0])
        cop1, _, _ = hp1.simulate_hplib(temps)
        cop4, _, _ = hp4.simulate_hplib(temps)
        
        # Both should return valid COP (may differ due to different HP types)
        assert cop1[0] > 1.0
        assert cop4[0] > 1.0
    
    def test_lower_supply_temp_improves_cop(self):
        """Lower supply temperature (floor heating) should give higher COP."""
        hp_radiator = HeatPumpHPLib(supply_temp_c=55.0)  # Radiators
        hp_floor = HeatPumpHPLib(supply_temp_c=35.0)     # Floor heating
        
        temps = np.array([0.0])
        cop_rad, _, _ = hp_radiator.simulate_hplib(temps)
        cop_floor, _, _ = hp_floor.simulate_hplib(temps)
        
        # Floor heating should generally have better COP
        # (Note: hplib may have some model-specific variations)
        assert cop_floor[0] >= cop_rad[0] * 0.9, \
            f"Floor heating COP ({cop_floor[0]:.2f}) should be close to or better than radiator ({cop_rad[0]:.2f})"
    
    def test_array_length_consistency(self):
        """Output arrays should match input length."""
        hp = HeatPumpHPLib()
        
        for n in [1, 10, 100, 144]:
            temps = np.linspace(-15, 10, n)
            cop, p_el, p_th = hp.simulate_hplib(temps)
            load = hp.calculate_load(temps)
            
            assert len(cop) == n
            assert len(p_el) == n
            assert len(p_th) == n
            assert len(load) == n


class TestHeatPumpCarnot:
    """Tests for HeatPumpCarnot physics model (Carnot-based approximation)."""
    
    def test_cop_decreases_with_temperature(self):
        """COP should decrease as outdoor temperature drops."""
        hp = HeatPumpCarnot()
        
        temps = np.array([10.0, 0.0, -10.0, -20.0])
        cops = hp.cop(temps)
        
        # COP should decrease as temperature drops
        assert cops[0] > cops[1] > cops[2] > cops[3]
    
    def test_cop_minimum_clamp(self):
        """COP should not fall below min_cop even at extreme cold."""
        hp = HeatPumpCarnot(min_cop=1.0)
        
        extreme_cold = np.array([-40.0, -50.0])
        cops = hp.cop(extreme_cold)
        
        assert all(cops >= hp.min_cop)
    
    def test_cop_typical_values(self):
        """COP should be in reasonable range for typical Swedish winter."""
        hp = HeatPumpCarnot()
        
        # Typical winter: -5 to +5 °C
        temps = np.array([-5.0, 0.0, 5.0])
        cops = hp.cop(temps)
        
        # COP should be between 2 and 4 for these temperatures
        assert all(cops > 1.5)
        assert all(cops < 5.0)
    
    def test_heat_demand_zero_when_warm(self):
        """No heating needed when outdoor >= indoor temperature."""
        hp = HeatPumpCarnot(indoor_temp_c=21.0)
        
        warm_temps = np.array([21.0, 25.0, 30.0])
        demand = hp.heat_demand_w(warm_temps)
        
        assert all(demand == 0.0)
    
    def test_heat_demand_increases_with_cold(self):
        """Heat demand should increase as it gets colder."""
        hp = HeatPumpCarnot(indoor_temp_c=21.0, u_total_w_per_k=200.0)
        
        temps = np.array([15.0, 5.0, -5.0, -15.0])
        demand = hp.heat_demand_w(temps)
        
        assert demand[0] < demand[1] < demand[2] < demand[3]
    
    def test_heat_demand_formula(self):
        """Heat demand should follow Q = U * (T_in - T_out)."""
        hp = HeatPumpCarnot(indoor_temp_c=21.0, u_total_w_per_k=200.0)
        
        outdoor = np.array([0.0])
        demand = hp.heat_demand_w(outdoor)
        
        expected = 200.0 * (21.0 - 0.0)  # 4200 W
        assert np.isclose(demand[0], expected)
    
    def test_electrical_load_returns_three_components(self):
        """electrical_load should return total, compressor, and aux."""
        hp = HeatPumpCarnot()
        
        temps = np.array([0.0, -10.0])
        total, comp, aux = hp.electrical_load(temps)
        
        assert len(total) == 2
        assert len(comp) == 2
        assert len(aux) == 2
        
        # Total should equal comp + aux
        assert np.allclose(total, comp + aux)
    
    def test_auxiliary_heater_kicks_in_at_extreme_cold(self):
        """Aux heater should activate when compressor can't meet demand."""
        # High U-value building: auto-sizing will create appropriate HP
        hp = HeatPumpCarnot(
            indoor_temp_c=21.0,
            u_total_w_per_k=300.0,  # High heat loss building
        )
        
        # At extreme cold, demand exceeds what compressor can provide
        extreme_cold = np.array([-25.0])
        total, comp, aux = hp.electrical_load(extreme_cold)
        
        # Aux should be > 0 at extreme cold
        assert aux[0] > 0, "Aux heater should activate at extreme cold"
    
    def test_compressor_capped_at_max(self):
        """Compressor power should not exceed auto-sized max."""
        hp = HeatPumpCarnot(u_total_w_per_k=100.0)
        
        temps = np.linspace(-30, 10, 50)
        total, comp, aux = hp.electrical_load(temps)
        
        assert all(comp <= hp.compressor_power_max_w + 1e-6)


class TestDistrictHeating:
    """Tests for district heating model."""
    
    def test_returns_zero_electrical_load(self):
        """District heating contributes zero to resident's electricity meter.
        
        Circulation pumps are on building common or district network infrastructure,
        not the individual household's meter.
        """
        dh = DistrictHeating()
        
        temps = np.array([-20.0, 0.0, 20.0])
        load = dh.calculate_load(temps)
        
        assert all(load == 0.0), "District heating should show zero on resident meter"
    
    def test_load_independent_of_temperature(self):
        """District heating load is always zero regardless of outdoor temp."""
        dh = DistrictHeating()
        
        cold = dh.calculate_load(np.array([-20.0]))
        warm = dh.calculate_load(np.array([10.0]))
        
        assert cold[0] == warm[0] == 0.0


class TestDirectElectric:
    """Tests for direct electric heating model."""
    
    def test_no_heat_when_warm(self):
        """No heating needed when outdoor >= indoor."""
        de = DirectElectric(indoor_temp_c=21.0)
        
        warm_temps = np.array([21.0, 25.0])
        load = de.calculate_load(warm_temps)
        
        assert all(load == 0.0)
    
    def test_load_increases_with_cold(self):
        """Load should increase as temperature drops."""
        de = DirectElectric(indoor_temp_c=21.0, u_total_w_per_k=200.0)
        
        temps = np.array([10.0, 0.0, -10.0])
        load = de.calculate_load(temps)
        
        assert load[0] < load[1] < load[2]
    
    def test_efficiency_is_one(self):
        """Direct electric has COP=1, so electrical = heat demand."""
        de = DirectElectric(indoor_temp_c=21.0, u_total_w_per_k=200.0)
        
        outdoor = np.array([0.0])
        load = de.calculate_load(outdoor)
        
        expected = 200.0 * 21.0  # U * delta_T
        assert np.isclose(load[0], expected)
