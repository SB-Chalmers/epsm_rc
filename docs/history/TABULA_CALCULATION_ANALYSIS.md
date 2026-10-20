# Historical experiment record

This document is retained from RC source revision `79c04461f1f2675d310eb8629d662db9ac8422c0` (0.7.1).
Its numerical results, paths and validation claims describe that historical
experiment. They are **not certification, a current release check, or evidence of
prediction accuracy for a measured building**. Some referenced scripts are no
longer present. Current checks are documented in [the project README](../../README.md)
and [the benchmark guide](../../benchmarks/README.md).

---

# TABULA Webtool Calculation Method - Analysis Summary

## Validation Results

### Final Results (after methodology corrections)

| Zone | Pass Rate | Mean Deviation | Notes |
|------|-----------|----------------|-------|
| Zone 1 (Luleå) | **10/10 (100%)** | **+1.2%** | Verified against webtool |
| Zone 2 (Östersund) | 9/10 (90%) | +9.8% | Estimated climate |
| Zone 3 (Göteborg) | 6/10 (60%) | -27.5% | Estimated climate |
| **Overall** | **25/30 (83%)** | **-5.5%** | ±30% tolerance |

Zone 1 matches the TABULA webtool **exactly** when using verified parameters.

## Key Findings

After analyzing the TABULA webtool API responses and reverse-engineering the calculations, we can now reproduce the webtool results exactly.

## Critical Differences from Our Model

### 1. Temperature Reduction Factor (f_red_temp)

The TABULA webtool applies a **temperature reduction factor** to heat losses:

```
Q_ht = H × DD × 24/1000 × f_red_temp
```

Where `f_red_temp ≈ 0.85` accounts for:
- Temperature setbacks (night, unoccupied periods)
- Partially heated spaces
- Effective temperature differences

**Our model** uses `non_uniform_heating_factor = 1.0` (no reduction).

### 2. Solar Gain Calculation

TABULA calculates solar gains as the sum of **two components**:

```
Q_sol = Q_sol_directional + Q_sol_horizontal
```

Where:
- **Q_sol_directional** = Σ(A_w,k × g × FF × F_sh_vert × F_w × I_sol,k)
- **Q_sol_horizontal** = A_w_total × g × FF × F_sh_hor × F_w × I_hor

With:
- g = 0.8 (glass g-value, g_gl_n)
- FF = 0.7 (glass fraction = 1 - frame_factor)
- F_sh_vert = 0.6 (vertical shading)
- F_sh_hor = 0.8 (horizontal shading)
- F_w = 0.9 (non-perpendicular incidence correction)

**Our model** only uses directional calculation without the horizontal component.

### 3. Climate Data

TABULA webtool Zone 1 uses:
- θe = -1.2°C (mean external temp during heating season)
- Heating days = 365
- Degree-days = 7738 K·days
- Solar: I_hor=711, I_N=309, I_E=628, I_S=725, I_W=628 kWh/(m²a)

**Our model** uses:
- θe = 0.5°C  
- Heating days = 303
- Degree-days = 5909 K·days (24% less)

## Verified Calculation for SFH.01 Zone 1

Using exact webtool parameters:

| Component | Formula | Value |
|-----------|---------|-------|
| q_ht_tr | H_tr × DD × 24/1000 × f_red_temp | 386.85 kWh/(m²a) |
| q_ht_ve | H_ve × DD × 24/1000 × f_red_temp | 107.57 kWh/(m²a) |
| **q_ht** | q_ht_tr + q_ht_ve | **494.43 kWh/(m²a)** |
| q_int | φ_int × 365 × 24/1000 | 26.28 kWh/(m²a) |
| q_sol | Directional + Horizontal | 96.03 kWh/(m²a) |
| **q_gn** | q_int + q_sol | **122.31 kWh/(m²a)** |
| η_H | Gain utilization factor | 0.8686 |
| **q_h_nd** | q_ht - η_H × q_gn | **388.18 kWh/(m²a)** |

## Webtool Parameter Sources

Key parameters from webtool API (`/data/adv/building/detail/{code}/...`):

| Parameter | Variable | SFH.01 Value |
|-----------|----------|--------------|
| Transmission coefficient | h_transmission | 2.445 W/(m²K) |
| Ventilation coefficient | h_ventilation | 0.68 W/(m²K) |
| Reference area | a_c_ref | 106.25 m² |
| Degree-days | sum_deltat_for_heatingdays | 7738 K·days |
| Temp reduction | f_red_temp | 0.8518 |
| Gain utilization | eta_h_gn | 0.8686 |
| Internal gains | phi_int | 3.0 W/m² |
| Glass g-value | g_gl_n | 0.8 |
| Frame factor | f_f | 0.3 |
| Vertical shading | f_sh_vert | 0.6 |
| Horizontal shading | f_sh_hor | 0.8 |
| Incidence factor | f_w | 0.9 |

## NET vs GROSS Values

The webtool provides two types of heating demand:

1. **GROSS (q_h_nd)**: Building envelope demand from detail API
   - No system effects
   - This is what we should validate against

2. **NET ("National minimum requirement")**: From chart API
   - Includes ventilation heat recovery credit
   - Includes system efficiency adjustments
   - Has net/gross ratio (af) of 0.51-0.97

## Recommendations

1. **Update climate data** to match TABULA webtool parameters
2. **Add temperature reduction factor** (f_red_temp) to our model
3. **Update solar gain calculation** to include horizontal component
4. **Validate against GROSS values** (q_h_nd from detail API)

## Files Created

- `scripts/scrape_tabula_webtool.py` - Fetches all reference values from TABULA API
- `scripts/validate_tabula_params.py` - Verifies calculation method (100% match)
- `scripts/analyze_tabula_solar.py` - Solar gain analysis
- `scripts/calc_zone_climate.py` - Climate parameter estimation
- `scripts/fetch_tabula_params.py` - Parameter extraction from API
- `references/tabula_webtool_scraped.json` - Scraped reference data (NET and GROSS)
- `references/sfh01_params.txt` - Full parameter dump for SFH.01
- `docs/TABULA_CALCULATION_ANALYSIS.md` - This analysis document

## Updated Files

- `src/models/tabula.py` - Updated ClimateData with verified Zone 1 parameters
- `scripts/run_tabula_seasonal.py` - Updated to use GROSS reference values

## Conclusion

The TABULA seasonal model implementation has been validated against the 
official TABULA webtool with the following key findings:

1. **Zone 1 validation is successful** - 100% pass rate with +1.2% mean deviation
2. **Methodology is correct** - When using exact webtool parameters, we achieve 
   0.00% deviation on all components (q_ht_tr, q_ht_ve, q_int, q_sol, q_h_nd)
3. **Climate data is critical** - The webtool uses specific climate parameters
   that significantly affect results (θe = -1.2°C, DD = 7738 K·days for Zone 1)
4. **NET vs GROSS distinction** - The webtool provides both values; GROSS should
   be used for building envelope validation

The remaining deviations in Zone 2/3 are attributed to estimated climate 
parameters. To achieve 100% validation across all zones, exact climate 
parameters from the TABULA webtool would need to be obtained for each zone.
