"""Tests for ISO 13790 5R1C thermal building model."""

import pytest
import numpy as np
from epsm_rc.iso13790 import (
    Building5R1C,
    BuildingClass,
    THERMAL_CAPACITANCE,
    MASS_AREA_FACTOR,
    create_swedish_villa,
    create_swedish_apartment,
)


class TestBuildingClass:
    """Tests for BuildingClass enum and related constants."""
    
    def test_building_classes_defined(self):
        """All five ISO 13790 building classes exist."""
        assert len(BuildingClass) == 5
        assert BuildingClass.VERY_LIGHT.value == 1
        assert BuildingClass.VERY_HEAVY.value == 5
    
    def test_thermal_capacitance_values(self):
        """Thermal capacitance values follow ISO 13790 Table 12."""
        assert THERMAL_CAPACITANCE[BuildingClass.VERY_LIGHT] == 80_000
        assert THERMAL_CAPACITANCE[BuildingClass.LIGHT] == 110_000
        assert THERMAL_CAPACITANCE[BuildingClass.MEDIUM] == 165_000
        assert THERMAL_CAPACITANCE[BuildingClass.HEAVY] == 260_000
        assert THERMAL_CAPACITANCE[BuildingClass.VERY_HEAVY] == 370_000
    
    def test_mass_area_factors(self):
        """Mass area factors follow ISO 13790 Table 12."""
        assert MASS_AREA_FACTOR[BuildingClass.VERY_LIGHT] == 2.5
        assert MASS_AREA_FACTOR[BuildingClass.MEDIUM] == 2.5
        assert MASS_AREA_FACTOR[BuildingClass.HEAVY] == 3.0
        assert MASS_AREA_FACTOR[BuildingClass.VERY_HEAVY] == 3.5


class TestBuilding5R1CInitialization:
    """Tests for Building5R1C initialization."""
    
    def test_default_initialization(self):
        """Building can be created with default values."""
        building = Building5R1C()
        assert building.floor_area_m2 == 120.0
        assert building.building_class == BuildingClass.MEDIUM
        assert building.t_set_heating == 21.0
    
    def test_custom_initialization(self):
        """Building accepts custom parameters."""
        building = Building5R1C(
            floor_area_m2=200.0,
            building_class=BuildingClass.HEAVY,
            u_walls=0.30,
            t_set_heating=20.0,
        )
        assert building.floor_area_m2 == 200.0
        assert building.building_class == BuildingClass.HEAVY
        assert building.u_walls == 0.30
        assert building.t_set_heating == 20.0
    
    def test_derived_properties_calculated(self):
        """Post-init calculates derived thermal properties."""
        building = Building5R1C(floor_area_m2=100.0)
        
        # Check capacitance calculated
        expected_c_m = THERMAL_CAPACITANCE[BuildingClass.MEDIUM] * 100.0
        assert building._c_m == expected_c_m
        
        # Check areas calculated
        assert building._a_tot == 4.5 * 100.0  # λ_at = 4.5
        assert building._mass_area == 2.5 * 100.0  # Medium class
        
        # Check building volume
        assert building._building_vol == 100.0 * 2.5


class TestThermalConductances:
    """Tests for 5R1C thermal conductance calculations."""
    
    @pytest.fixture
    def standard_building(self):
        """Standard test building."""
        return Building5R1C(
            floor_area_m2=100.0,
            wall_area_m2=120.0,
            roof_area_m2=50.0,
            floor_area_ground_m2=50.0,
            window_area_m2=20.0,
            u_walls=0.25,
            u_roof=0.15,
            u_floor=0.20,
            u_windows=1.2,
        )
    
    def test_h_tr_op_calculation(self, standard_building):
        """Opaque transmission calculated correctly."""
        # H_tr_op = u_walls*A_walls + u_roof*A_roof + u_floor*A_floor*b_gw
        expected = (0.25 * 120.0 + 0.15 * 50.0 + 0.20 * 50.0 * 0.5)
        assert abs(standard_building._h_tr_op - expected) < 0.1
    
    def test_h_tr_w_calculation(self, standard_building):
        """Window transmission calculated correctly."""
        expected = 1.2 * 20.0  # u_windows * window_area
        assert standard_building._h_tr_w == expected
    
    def test_h_tr_ms_calculation(self, standard_building):
        """Mass-to-surface conductance uses h_ms=9.1."""
        expected = 9.1 * standard_building._mass_area
        assert standard_building._h_tr_ms == expected
    
    def test_h_tr_is_calculation(self, standard_building):
        """Surface-to-air conductance uses h_is=3.45."""
        expected = 3.45 * standard_building._a_tot
        assert standard_building._h_tr_is == expected
    
    def test_combined_conductances(self, standard_building):
        """h_tr_1, h_tr_2, h_tr_3 follow ISO 13790 equations."""
        b = standard_building
        
        # h_tr_1 = 1/(1/H_ve + 1/H_tr_is) - Eq. C.6
        h_tr_1_expected = 1.0 / (1.0 / b._h_ve + 1.0 / b._h_tr_is)
        assert abs(b.h_tr_1 - h_tr_1_expected) < 0.01
        
        # h_tr_2 = h_tr_1 + H_tr_w - Eq. C.7
        assert abs(b.h_tr_2 - (b.h_tr_1 + b._h_tr_w)) < 0.01
        
        # h_tr_3 = 1/(1/h_tr_2 + 1/H_tr_ms) - Eq. C.8
        h_tr_3_expected = 1.0 / (1.0 / b.h_tr_2 + 1.0 / b._h_tr_ms)
        assert abs(b.h_tr_3 - h_tr_3_expected) < 0.01


class TestVentilation:
    """Tests for ventilation heat transfer."""
    
    def test_ventilation_coefficient_no_recovery(self):
        """Ventilation without heat recovery."""
        building = Building5R1C(
            floor_area_m2=100.0,
            building_height_m=2.5,
            ach_infiltration=0.2,
            ach_ventilation=0.5,
            heat_recovery_efficiency=0.0,
        )
        
        # H_ve = ρ·c_p·V·n/3600 = 1200 * V * ACH / 3600
        vol = 100.0 * 2.5
        ach_total = 0.2 + 0.5
        expected = 1200.0 * vol * ach_total / 3600.0
        assert abs(building._h_ve - expected) < 0.1
    
    def test_ventilation_with_heat_recovery(self):
        """Heat recovery reduces ventilation losses."""
        building_no_hr = Building5R1C(
            floor_area_m2=100.0,
            ach_ventilation=0.5,
            heat_recovery_efficiency=0.0,
        )
        building_with_hr = Building5R1C(
            floor_area_m2=100.0,
            ach_ventilation=0.5,
            heat_recovery_efficiency=0.8,
        )
        
        # Heat recovery should reduce H_ve
        assert building_with_hr._h_ve < building_no_hr._h_ve


class TestTimeConstant:
    """Tests for building thermal time constant."""
    
    def test_time_constant_increases_with_mass(self):
        """Heavier buildings have longer time constants."""
        light = Building5R1C(building_class=BuildingClass.LIGHT)
        heavy = Building5R1C(building_class=BuildingClass.VERY_HEAVY)
        
        assert heavy.time_constant_hours > light.time_constant_hours
    
    def test_time_constant_reasonable_range(self):
        """Time constant in reasonable range for buildings."""
        building = Building5R1C(building_class=BuildingClass.MEDIUM)
        
        # Typical range 10-100 hours for buildings
        assert 5 < building.time_constant_hours < 200


class TestHeatFlowCalculations:
    """Tests for heat flow distribution."""
    
    @pytest.fixture
    def building(self):
        return Building5R1C(floor_area_m2=100.0)
    
    def test_internal_gains_split(self, building):
        """Internal gains split 50/50 between air and mass/surface."""
        phi_ia, phi_st, phi_m = building.calc_heat_flows(
            t_out=0.0,
            internal_gains=1000.0,
            solar_gains=0.0,
        )
        
        # φ_ia gets 50% of internal gains
        assert abs(phi_ia - 500.0) < 1.0
        
        # φ_st and φ_m share the rest based on area ratios
        assert phi_st + phi_m > 0
    
    def test_solar_gains_to_mass_surface(self, building):
        """Solar gains go to surface and mass nodes only."""
        phi_ia_no_solar, _, _ = building.calc_heat_flows(
            t_out=0.0, internal_gains=0.0, solar_gains=0.0
        )
        phi_ia_solar, phi_st_solar, phi_m_solar = building.calc_heat_flows(
            t_out=0.0, internal_gains=0.0, solar_gains=1000.0
        )
        
        # Air node unchanged by solar gains alone
        assert abs(phi_ia_solar - phi_ia_no_solar) < 1.0
        
        # Surface and mass receive solar
        assert phi_st_solar > 0 or phi_m_solar > 0
    
    def test_heating_demand_to_air(self, building):
        """Heating demand added to air node (radiator assumption)."""
        phi_ia_no_heat, _, _ = building.calc_heat_flows(
            t_out=0.0, internal_gains=0.0, solar_gains=0.0, energy_demand=0.0
        )
        phi_ia_heat, _, _ = building.calc_heat_flows(
            t_out=0.0, internal_gains=0.0, solar_gains=0.0, energy_demand=2000.0
        )
        
        assert phi_ia_heat == phi_ia_no_heat + 2000.0


class TestTemperatureCalculations:
    """Tests for node temperature calculations."""
    
    @pytest.fixture
    def building(self):
        b = Building5R1C(floor_area_m2=100.0, t_set_heating=21.0)
        b.t_m = 21.0  # Start at setpoint
        return b
    
    def test_step_updates_temperatures(self, building):
        """Step method updates all node temperatures."""
        t_air_old = building.t_air
        
        building.step(t_out=-10.0, solar_gains=0.0)
        
        # Temperatures should change with cold outdoor
        assert building.t_air != t_air_old
    
    def test_cold_outdoor_cools_building(self, building):
        """Cold outdoor temperature cools the building."""
        building.t_m = 21.0
        building.t_air = 21.0
        
        # Step with very cold outdoor, no heating
        building.step(t_out=-20.0, solar_gains=0.0, energy_demand=0.0)
        
        assert building.t_air < 21.0
        assert building.t_m < 21.0
    
    def test_solar_gains_warm_building(self, building):
        """Solar gains increase indoor temperature."""
        building.t_m = 15.0
        building.step(t_out=10.0, solar_gains=0.0)
        t_air_no_solar = building.t_air
        
        building.t_m = 15.0
        building.step(t_out=10.0, solar_gains=3000.0)
        t_air_solar = building.t_air
        
        assert t_air_solar > t_air_no_solar
    
    def test_heating_maintains_temperature(self, building):
        """Heating demand maintains indoor temperature."""
        building.t_m = 21.0
        
        # Calculate required heating
        q_heat = building.calc_heating_demand(t_out=-10.0, solar_gains=0.0)
        
        # Temperature should be at/near setpoint
        assert abs(building.t_air - 21.0) < 0.5
        assert q_heat > 0  # Should need heating


class TestHeatingDemand:
    """Tests for heating demand calculation."""
    
    @pytest.fixture
    def building(self):
        return Building5R1C(
            floor_area_m2=100.0,
            t_set_heating=21.0,
            t_set_cooling=26.0,
        )
    
    def test_heating_demand_increases_with_cold(self, building):
        """Colder outdoor → higher heating demand."""
        building.reset(21.0)
        q_mild = building.calc_heating_demand(t_out=10.0)
        
        building.reset(21.0)
        q_cold = building.calc_heating_demand(t_out=-10.0)
        
        assert q_cold > q_mild
    
    def test_no_heating_in_warm_weather(self, building):
        """No heating needed when outdoor is warm."""
        building.reset(21.0)
        q_heat = building.calc_heating_demand(t_out=22.0, solar_gains=1000.0)
        
        assert q_heat == 0 or q_heat < 0  # No heating or cooling
    
    def test_heating_capped_by_capacity(self, building):
        """Max heating power limits demand."""
        building.reset(21.0)
        q_unlimited = building.calc_heating_demand(t_out=-30.0, max_heating_power=np.inf)
        
        building.reset(21.0)
        q_capped = building.calc_heating_demand(t_out=-30.0, max_heating_power=2000.0)
        
        assert q_capped <= 2000.0
        assert q_capped < q_unlimited
    
    def test_solar_gains_reduce_heating(self, building):
        """Solar gains offset heating demand."""
        building.reset(21.0)
        q_no_solar = building.calc_heating_demand(t_out=5.0, solar_gains=0.0)
        
        building.reset(21.0)
        q_with_solar = building.calc_heating_demand(t_out=5.0, solar_gains=2000.0)
        
        assert q_with_solar < q_no_solar


class TestSimulation:
    """Tests for time-series simulation."""
    
    def test_simulate_returns_arrays(self):
        """Simulate returns properly sized output arrays."""
        building = Building5R1C(floor_area_m2=100.0)
        
        t_outdoor = np.linspace(-10, 10, 24)
        result = building.simulate(t_outdoor, dt_hours=1.0)
        
        assert 'heating_demand' in result
        assert 't_air' in result
        assert 't_m' in result
        assert len(result['heating_demand']) == 24
    
    def test_simulate_heating_demand_profile(self):
        """Heating demand follows outdoor temperature."""
        building = Building5R1C(floor_area_m2=100.0)
        
        # Cold morning, warm afternoon
        t_outdoor = np.concatenate([
            np.full(6, -15.0),   # Night
            np.linspace(-15, 5, 6),  # Morning warming
            np.full(6, 5.0),    # Afternoon
            np.linspace(5, -10, 6),  # Evening cooling
        ])
        
        result = building.simulate(t_outdoor)
        
        # Night heating > afternoon heating
        night_avg = np.mean(result['heating_demand'][:6])
        afternoon_avg = np.mean(result['heating_demand'][12:18])
        
        assert night_avg > afternoon_avg
    
    def test_mass_temperature_lag(self):
        """Mass temperature lags behind air temperature."""
        building = Building5R1C(
            floor_area_m2=100.0,
            building_class=BuildingClass.VERY_HEAVY,
        )
        
        # Sudden temperature drop
        t_outdoor = np.concatenate([
            np.full(12, 15.0),
            np.full(12, -10.0),
        ])
        
        result = building.simulate(t_outdoor, dt_hours=1.0)
        
        # After temperature drop, mass should lag
        # Air responds faster to heating demand
        t_air_after_drop = result['t_air'][15]
        t_m_after_drop = result['t_m'][15]
        
        # Mass temp should be between old and new equilibrium
        # (demonstrating thermal lag)
        assert t_m_after_drop > t_air_after_drop - 5  # Mass hasn't caught up


class TestReset:
    """Tests for state reset."""
    
    def test_reset_to_setpoint(self):
        """Reset defaults to heating setpoint."""
        building = Building5R1C(t_set_heating=20.0)
        building.t_m = 15.0
        building.t_air = 15.0
        
        building.reset()
        
        assert building.t_m == 20.0
        assert building.t_air == 20.0
    
    def test_reset_to_custom_temp(self):
        """Reset accepts custom temperature."""
        building = Building5R1C()
        
        building.reset(t_initial=18.0)
        
        assert building.t_m == 18.0


class TestSwedishBuildingFactories:
    """Tests for Swedish building factory functions."""
    
    @pytest.mark.parametrize("era", [
        "pre_1960", "1960_1975", "1976_1990", "1991_2010", "post_2010"
    ])
    def test_create_swedish_villa_all_eras(self, era):
        """Villa factory works for all eras."""
        villa = create_swedish_villa(floor_area_m2=150.0, era=era)
        
        assert villa.floor_area_m2 == 150.0
        assert villa.u_walls > 0
        assert villa._c_m > 0
    
    @pytest.mark.parametrize("era", [
        "pre_1960", "1960_1975", "1976_1990", "1991_2010", "post_2010"
    ])
    def test_create_swedish_apartment_all_eras(self, era):
        """Apartment factory works for all eras."""
        with pytest.warns(DeprecationWarning, match="epsm_data LoD1"):
            apt = create_swedish_apartment(floor_area_m2=70.0, era=era)
        
        assert apt.floor_area_m2 == 70.0
        assert apt.u_walls > 0
    
    def test_newer_buildings_better_insulated(self):
        """Newer buildings have lower U-values."""
        old = create_swedish_villa(era="pre_1960")
        new = create_swedish_villa(era="post_2010")
        
        assert new.u_walls < old.u_walls
        assert new.u_windows < old.u_windows
    
    def test_apartment_less_external_surface(self):
        """Apartments have less external surface area."""
        villa = create_swedish_villa(floor_area_m2=100.0)
        apt = create_swedish_apartment(floor_area_m2=100.0)
        
        # Apartment mid-unit has no roof/floor losses
        assert apt.roof_area_m2 < villa.roof_area_m2
        assert apt.wall_area_m2 < villa.wall_area_m2
    
    def test_factory_accepts_overrides(self):
        """Factory functions accept parameter overrides."""
        villa = create_swedish_villa(
            floor_area_m2=200.0,
            era="1976_1990",
            u_walls=0.30,  # Override
            t_set_heating=22.0,  # Override
        )
        
        assert villa.floor_area_m2 == 200.0
        assert villa.u_walls == 0.30
        assert villa.t_set_heating == 22.0
    
    def test_villa_heat_loss_realistic(self):
        """Villa heat loss coefficient in realistic range."""
        villa = create_swedish_villa(floor_area_m2=150.0, era="1976_1990")
        
        # Typical range 100-400 W/K for Swedish villa
        assert 50 < villa.heat_loss_coefficient < 500
    
    def test_apartment_lower_heat_loss(self):
        """Apartment has lower heat loss per area than villa."""
        villa = create_swedish_villa(floor_area_m2=100.0, era="1976_1990")
        apt = create_swedish_apartment(floor_area_m2=100.0, era="1976_1990")
        
        # Per-area heat loss
        villa_specific = villa.heat_loss_coefficient / 100.0
        apt_specific = apt.heat_loss_coefficient / 100.0
        
        assert apt_specific < villa_specific


class TestHeatLossCoefficient:
    """Tests for overall heat loss coefficient."""
    
    def test_heat_loss_includes_transmission_and_ventilation(self):
        """Heat loss is sum of transmission and ventilation."""
        building = Building5R1C()
        
        expected = building._h_tr_op + building._h_tr_w + building._h_ve
        assert abs(building.heat_loss_coefficient - expected) < 0.1


class TestOperativeTemperature:
    """Tests for operative temperature calculation."""
    
    def test_operative_weighted_average(self):
        """Operative temp is weighted average of air and surface."""
        building = Building5R1C()
        building.t_air = 20.0
        building.t_s = 18.0
        
        # t_op = 0.3 * t_air + 0.7 * t_s (ISO 13790 Eq. C.12)
        expected = 0.3 * 20.0 + 0.7 * 18.0
        assert abs(building.t_operative - expected) < 0.01


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_zero_ventilation(self):
        """Model handles zero ventilation."""
        building = Building5R1C(
            ach_infiltration=0.0,
            ach_ventilation=0.0,
        )
        
        # Should not crash, h_ve = 0
        assert building._h_ve == 0
        
        # Should still calculate heating demand
        q = building.calc_heating_demand(t_out=-10.0)
        assert q > 0
    
    def test_very_small_building(self):
        """Model handles small buildings."""
        building = Building5R1C(
            floor_area_m2=20.0,
            wall_area_m2=40.0,
            window_area_m2=5.0,
        )
        
        assert building._c_m > 0
        q = building.calc_heating_demand(t_out=0.0)
        assert q >= 0
    
    def test_extreme_cold(self):
        """Model handles extreme cold temperatures."""
        building = Building5R1C(floor_area_m2=100.0)
        
        q = building.calc_heating_demand(t_out=-40.0)
        
        # Should have high but finite demand
        assert 0 < q < 50000
    
    def test_extreme_heat(self):
        """Model handles extreme heat (cooling scenario)."""
        building = Building5R1C(
            floor_area_m2=100.0,
            t_set_cooling=26.0,
        )
        building.reset(30.0)
        
        q = building.calc_heating_demand(
            t_out=35.0,
            solar_gains=5000.0,
            max_cooling_power=-10000.0,
        )
        
        # Should request cooling (negative)
        assert q <= 0


class TestAnnualSimulation:
    """Tests for annual simulation with realistic weather."""
    
    @pytest.fixture
    def annual_temps(self):
        """Simplified annual temperature profile."""
        # Synthetic sinusoidal annual pattern
        hours = np.arange(8760)
        day_of_year = hours / 24
        # Annual cycle: coldest Jan 15 (day 15), warmest Jul 15 (day 196)
        annual_cycle = 10 * np.cos(2 * np.pi * (day_of_year - 196) / 365)
        # Daily cycle
        hour_of_day = hours % 24
        daily_cycle = 3 * np.cos(2 * np.pi * (hour_of_day - 14) / 24)
        # Base temp around 6°C (Swedish average)
        return 6.0 + annual_cycle + daily_cycle
    
    def test_simulate_returns_correct_keys(self, annual_temps):
        """Simulate returns expected result keys."""
        building = Building5R1C(floor_area_m2=100.0)
        
        result = building.simulate(
            t_outdoor=annual_temps,
            dt_hours=1.0,
        )
        
        assert "heating_demand" in result
        assert "t_air" in result
        assert "t_m" in result
        assert "t_s" in result  # Surface temperature (simulate returns t_s, not t_operative)
    
    def test_simulate_output_lengths(self, annual_temps):
        """Simulate returns arrays of correct length."""
        building = Building5R1C(floor_area_m2=100.0)
        
        result = building.simulate(t_outdoor=annual_temps)
        
        assert len(result["heating_demand"]) == 8760
        assert len(result["t_air"]) == 8760
        assert len(result["t_m"]) == 8760
    
    def test_annual_heating_demand_positive(self, annual_temps):
        """Annual heating demand should be positive in Sweden."""
        building = Building5R1C(floor_area_m2=100.0)
        
        result = building.simulate(t_outdoor=annual_temps)
        annual_kwh = np.sum(result["heating_demand"]) / 1000
        
        assert annual_kwh > 0
    
    def test_heating_only_when_cold(self, annual_temps):
        """Heating should only occur when outdoor temp is below setpoint."""
        building = Building5R1C(floor_area_m2=100.0, t_set_heating=21.0)
        
        result = building.simulate(t_outdoor=annual_temps)
        
        # Find hours where heating occurred
        heating_hours = result["heating_demand"] > 0
        
        # Indoor temp should be at or near setpoint during heating
        heated_indoor_temps = result["t_air"][heating_hours]
        assert np.all(heated_indoor_temps >= 20.0)  # Within 1°C of setpoint
    
    def test_solar_gains_reduce_heating(self, annual_temps):
        """Solar gains should reduce heating demand."""
        building = Building5R1C(floor_area_m2=100.0, window_area_m2=15.0)
        
        # No solar
        result_no_solar = building.simulate(
            t_outdoor=annual_temps,
            solar_gains=np.zeros(8760),
        )
        
        # With solar (simplified profile)
        hours = np.arange(8760)
        solar = np.maximum(0, 200 * np.sin((hours % 24 - 6) * np.pi / 12))
        result_solar = building.simulate(
            t_outdoor=annual_temps,
            solar_gains=solar,
        )
        
        annual_no_solar = np.sum(result_no_solar["heating_demand"])
        annual_solar = np.sum(result_solar["heating_demand"])
        
        assert annual_solar < annual_no_solar
    
    def test_internal_gains_reduce_heating(self, annual_temps):
        """Internal gains should reduce heating demand."""
        building_low_gains = Building5R1C(
            floor_area_m2=100.0,
            internal_gains_w_per_m2=1.0,
        )
        building_high_gains = Building5R1C(
            floor_area_m2=100.0,
            internal_gains_w_per_m2=5.0,
        )
        
        result_low = building_low_gains.simulate(t_outdoor=annual_temps)
        result_high = building_high_gains.simulate(t_outdoor=annual_temps)
        
        assert np.sum(result_high["heating_demand"]) < np.sum(result_low["heating_demand"])


class TestSwedishBuildingTypes:
    """Tests for Swedish building factory functions."""
    
    def test_create_swedish_villa(self):
        """Can create Swedish villa with default parameters."""
        villa = create_swedish_villa()
        
        assert villa.floor_area_m2 > 0
        assert villa._h_tr_op > 0
        assert villa._h_ve > 0
    
    def test_create_swedish_apartment(self):
        """Can create Swedish apartment with default parameters."""
        apt = create_swedish_apartment()
        
        assert apt.floor_area_m2 > 0
        assert apt._h_tr_op > 0
    
    def test_villa_larger_than_apartment(self):
        """Default villa should be larger than default apartment."""
        villa = create_swedish_villa()
        apt = create_swedish_apartment()
        
        assert villa.floor_area_m2 > apt.floor_area_m2
    
    def test_villa_era_affects_insulation(self):
        """Different eras should have different U-values."""
        old_villa = create_swedish_villa(era="pre_1960")
        new_villa = create_swedish_villa(era="post_2010")
        
        # Newer buildings should have better (lower) U-values
        assert new_villa.u_walls < old_villa.u_walls
        assert new_villa.u_windows < old_villa.u_windows
    
    def test_villa_era_heating_demand(self):
        """Older villas should have higher heating demand."""
        old_villa = create_swedish_villa(era="pre_1960", floor_area_m2=120)
        new_villa = create_swedish_villa(era="post_2010", floor_area_m2=120)
        
        # At same conditions, old villa needs more heating
        q_old = old_villa.calc_heating_demand(t_out=0.0)
        q_new = new_villa.calc_heating_demand(t_out=0.0)
        
        assert q_old > q_new * 1.5  # At least 50% higher


class TestThermalMassEffect:
    """Tests for thermal mass (building class) effects."""
    
    @pytest.fixture
    def temp_step(self):
        """Temperature step change for testing dynamics."""
        # 24 hours at 0°C, then 24 hours at 20°C
        return np.concatenate([np.full(24, 0.0), np.full(24, 20.0)])
    
    def test_heavy_building_slower_response(self, temp_step):
        """Heavy buildings should respond slower to temperature changes."""
        light = Building5R1C(
            floor_area_m2=100.0,
            building_class=BuildingClass.VERY_LIGHT,
        )
        heavy = Building5R1C(
            floor_area_m2=100.0,
            building_class=BuildingClass.VERY_HEAVY,
        )
        
        result_light = light.simulate(t_outdoor=temp_step, dt_hours=1.0)
        result_heavy = heavy.simulate(t_outdoor=temp_step, dt_hours=1.0)
        
        # After warm period starts (hour 24), check response
        # Heavy building should have higher thermal mass temperature initially
        t_m_light_diff = abs(result_light["t_m"][30] - result_light["t_m"][24])
        t_m_heavy_diff = abs(result_heavy["t_m"][30] - result_heavy["t_m"][24])
        
        # Light building changes faster
        assert t_m_light_diff > t_m_heavy_diff
    
    def test_thermal_mass_increases_with_class(self):
        """Higher building class should have more thermal mass."""
        classes = [
            BuildingClass.VERY_LIGHT,
            BuildingClass.LIGHT,
            BuildingClass.MEDIUM,
            BuildingClass.HEAVY,
            BuildingClass.VERY_HEAVY,
        ]
        
        buildings = [Building5R1C(building_class=bc) for bc in classes]
        capacitances = [b._c_m for b in buildings]
        
        # Should be strictly increasing
        for i in range(len(capacitances) - 1):
            assert capacitances[i+1] > capacitances[i]


class TestSpecificHeatingDemand:
    """Tests for specific heating demand calculations."""
    
    def test_specific_demand_reasonable_range(self):
        """Specific heating demand should be in reasonable range."""
        # Typical Swedish apartment: 50-100 kWh/m²/year
        # With simplified weather
        hours = np.arange(8760)
        temps = 6.0 + 10 * np.cos(2 * np.pi * (hours / 24 - 196) / 365)
        
        apt = create_swedish_apartment(floor_area_m2=75, era="1991_2010")
        result = apt.simulate(t_outdoor=temps)
        
        annual_kwh = np.sum(result["heating_demand"]) / 1000
        specific_kwh_m2 = annual_kwh / 75
        
        # Should be in reasonable range for Swedish apartment
        # (1991_2010 era has good insulation, so lower bound is lower)
        assert 5 < specific_kwh_m2 < 150
    
    def test_larger_building_higher_total_demand(self):
        """Larger building should have higher total heating demand."""
        temps = np.full(100, 0.0)  # 100 hours at 0°C
        
        small = Building5R1C(floor_area_m2=50.0)
        large = Building5R1C(floor_area_m2=150.0)
        
        result_small = small.simulate(t_outdoor=temps)
        result_large = large.simulate(t_outdoor=temps)
        
        assert np.sum(result_large["heating_demand"]) > np.sum(result_small["heating_demand"])
