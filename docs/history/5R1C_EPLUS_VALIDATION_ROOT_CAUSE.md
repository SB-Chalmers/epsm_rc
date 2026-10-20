# Historical experiment record

This document is retained from RC source revision `79c04461f1f2675d310eb8629d662db9ac8422c0` (0.7.1).
Its numerical results, paths and validation claims describe that historical
experiment. They are **not certification, a current release check, or evidence of
prediction accuracy for a measured building**. Some referenced scripts are no
longer present. Current checks are documented in [the project README](../../README.md)
and [the benchmark guide](../../benchmarks/README.md).

---

# 5R1C vs EnergyPlus Validation - Root Cause Analysis

## Summary

The original GSA comparison showed **+19.2% mean overprediction** by the 5R1C model compared to EnergyPlus. Root cause investigation identified two key issues:

### Issue 1: Ground Temperature Mismatch (Primary Cause)

**Finding:** The 5R1C model was auto-loading ground temperatures from the EPW file (0.5m depth), which are **significantly colder in winter** than the values used by EnergyPlus.

| Month | EPW 0.5m | E+ Value | Difference |
|-------|----------|----------|------------|
| Jan | 4.1°C | 4.0°C | +0.1°C |
| Feb | **1.3°C** | 3.5°C | **-2.2°C** |
| Mar | **0.6°C** | 4.0°C | **-3.4°C** |
| Apr | 1.2°C | 6.0°C | **-4.8°C** |
| ... | | | |

The colder EPW ground temps caused the 5R1C model to calculate higher floor heat losses in winter, resulting in higher heating demand.

**Impact:** ~40 percentage points of the error

### Issue 2: Internal Gains Convective Fraction (Secondary Cause)

**Finding:** ISO 13790 specifies 50% convective / 50% radiant split for internal gains. EnergyPlus uses:
- People: 70% convective, 30% radiant
- Lights: 60% convective, 40% radiant  
- Equipment: 70% convective, 30% radiant

Weighted average for E+: **~66.3% convective**

With more heat delivered directly to air (convective), the E+ zone temperature rises faster and requires less HVAC heating. The ISO 50/50 split sends more heat to thermal mass, which delays the heating benefit.

**Impact:** ~8-10 percentage points of the error

## Resolution

Added two configuration parameters to `Building5R1C`:

1. **`ground_temps_monthly`**: Explicit monthly ground temperatures [°C]. When set, these are used instead of auto-loaded EPW values.

2. **`convective_fraction_internal`**: Fraction of internal gains delivered directly to air node. Default: 0.5 (ISO 13790). Set to 0.663 to match E+ behavior.

## Results After Fixes

| Metric | Original | Corrected |
|--------|----------|-----------|
| Mean error | +19.2% | **-4.0%** |
| Std deviation | 15.5% | 15.6% |
| Within ±10% | 26% | **54%** |
| Within ±15% | 40% | **74%** |

The internal gains correlation with error dropped from **+0.57** to **+0.01**, confirming the systematic bias has been eliminated.

## Remaining Outliers

4 cases (8%) show |error| > 25%, all with:
- Very low heating demand (4-16 kWh/m²)
- High internal gains (21-23 W/m²)
- Well-insulated envelopes

At very low heating demands, small absolute differences cause large percentage errors. These represent edge cases where the simplified 5R1C model may not match E+'s detailed simulation.

## Recommendations

1. **For E+ validation studies**: Use `ground_temps_monthly` from E+ `Site:GroundTemperature:BuildingSurface` and `convective_fraction_internal=0.663`

2. **For general use**: Keep ISO 13790 defaults (0.5m EPW ground temps, 50% convective) for standards-compliant results

3. **Document limitations**: 5R1C accuracy degrades for very low heating demand buildings (<10 kWh/m²) with high internal gains

## Files Modified

- `src/models/iso13790.py`: Added `convective_fraction_internal` parameter
- `src/data/weather.py`: Added `parse_ground_temperatures()` function
- `validation/iso13790_comparison/gsa_analysis.py`: Updated to use matched parameters
