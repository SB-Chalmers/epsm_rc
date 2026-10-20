# Historical experiment record

This document is retained from RC source revision `79c04461f1f2675d310eb8629d662db9ac8422c0` (0.7.1).
Its numerical results, paths and validation claims describe that historical
experiment. They are **not certification, a current release check, or evidence of
prediction accuracy for a measured building**. Some referenced scripts are no
longer present. Current checks are documented in [the project README](../../README.md)
and [the benchmark guide](../../benchmarks/README.md).

---

# TABULA 5R1C Model Validation Results

## Summary

The ISO 13790 5R1C thermal model implementation has been validated against Swedish TABULA reference building data.

| Metric | Value |
|--------|-------|
| **Pass Rate** | 45/45 (100%) |
| **Correlation (r)** | 0.977 |
| **R-squared (R²)** | 0.954 |
| **MAPE** | 6.8% |
| **CV-RMSE** | 8.3% |
| **Mean Bias Error** | -0.4 kWh/m²/year |

## Validation Criteria

- **Pass threshold**: ±25% deviation from TABULA GROSS reference values
- **Reference standard**: ASHRAE Guideline 14 recommends CV-RMSE < 15% for calibrated models

## Results by Climate Zone

| Zone | Location | Pass Rate | Mean Deviation |
|------|----------|-----------|----------------|
| Zone 1 | Northern Sweden (Kiruna) | 15/15 | -1.7% |
| Zone 2 | Central Sweden (Östersund) | 15/15 | +5.2% |
| Zone 3 | Southern Sweden (Uppsala) | 15/15 | -2.2% |

## Results by Building Type

| Building Type | Pass Rate | Mean Deviation |
|---------------|-----------|----------------|
| Single Family 125m² | 15/15 | +6.3% |
| Single Family 160m² | 15/15 | -0.5% |
| Multi Family | 15/15 | -2.8% |

## Results by Construction Period

| Period | Pass Rate | Mean Deviation |
|--------|-----------|----------------|
| ...1960 | 9/9 | -3.8% |
| 1961-1975 | 9/9 | -3.6% |
| 1976-1985 | 9/9 | +3.4% |
| 1986-1995 | 9/9 | +1.6% |
| 1996-2005 | 9/9 | +5.3% |

## EPW Weather File Mapping

The validation uses TMYx EPW files from [climate.onebuilding.org](https://climate.onebuilding.org) matched to TABULA reference climate conditions:

| Zone | EPW File | θe_hs (EPW) | θe_hs (TABULA) | Heating Days |
|------|----------|-------------|----------------|--------------|
| 1 | kiruna.epw | -0.7°C | -1.2°C | 354 |
| 2 | ostersund.epw | 2.4°C | 0.3°C | 334 |
| 3 | uppsala.epw | 4.5°C | 2.2°C | 282 |

Note: θe_hs = heating season mean external temperature

## Methodology

1. **Model**: ISO 13790 5R1C simplified hourly method
2. **Reference**: Swedish TABULA GROSS heating demand values (includes system losses)
3. **Weather**: Hourly EPW data with heating season calculations
4. **Gains**: Internal gains from TABULA (occupancy, appliances, lighting)
5. **Solar**: Irradiance from EPW with orientation-specific window areas

## Key Parameters

From TABULA/ISO 13790 methodology:
- Gain utilization factor: a₀ = 0.8, τ₀ = 30 hours
- Internal temperature setpoint: 20°C
- Building thermal mass: Based on construction type

## References

- [TABULA WebTool](https://webtool.building-typology.eu/)
- ISO 13790:2008 - Energy performance of buildings
- Swedish TABULA National Scientific Report
