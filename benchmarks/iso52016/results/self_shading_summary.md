# Self-shading against EnergyPlus

128 conditioned cases: 4 climates, 2 archetypes, 2 forms (3 storeys WWR 0.2, 6 storeys WWR 0.45), orientations 0, 30 degrees. Values are medians of RC / EnergyPlus annual totals; EnergyPlus uses FullExterior polygon clipping.

## Window incident beam

| Shape | ignore | iso13790_hourly_horizon | hourly_sunlit_fraction |
|---|---|---|---|
| rectangle | 1.002 | 1.002 | 1.002 |
| l_shape | 1.049 | 1.010 | 1.002 |
| u_shape | 1.142 | 1.035 | 1.003 |
| courtyard | 1.278 | 0.987 | 1.003 |

## Window incident sky diffuse

| Shape | ignore | iso13790_hourly_horizon | hourly_sunlit_fraction |
|---|---|---|---|
| rectangle | 0.997 | 0.997 | 0.997 |
| l_shape | 1.051 | 1.051 | 1.002 |
| u_shape | 1.134 | 1.134 | 1.020 |
| courtyard | 1.291 | 1.291 | 1.019 |

## Window transmitted solar

| Shape | ignore | iso13790_hourly_horizon | hourly_sunlit_fraction |
|---|---|---|---|
| rectangle | 1.001 | 1.001 | 1.001 |
| l_shape | 1.041 | 1.024 | 1.003 |
| u_shape | 1.114 | 1.064 | 1.010 |
| courtyard | 1.233 | 1.101 | 1.009 |

## Heating

| Shape | ignore | iso13790_hourly_horizon | hourly_sunlit_fraction |
|---|---|---|---|
| rectangle | 1.025 | 1.025 | 1.025 |
| l_shape | 1.022 | 1.026 | 1.036 |
| u_shape | 1.004 | 1.019 | 1.037 |
| courtyard | 0.982 | 1.015 | 1.040 |

## Sensible cooling

| Shape | ignore | iso13790_hourly_horizon | hourly_sunlit_fraction |
|---|---|---|---|
| rectangle | 1.034 | 1.034 | 1.034 |
| l_shape | 1.094 | 1.068 | 1.031 |
| u_shape | 1.236 | 1.159 | 1.032 |
| courtyard | 1.896 | 1.457 | 1.019 |

Ratios exclude cases where the EnergyPlus total is below 0.05 kWh/m².
These are model-to-model comparisons, not validation.
