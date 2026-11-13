# Conditioned basements and cold attics against EnergyPlus

64 cases: 4 climates, 2 archetypes, 4 topologies, conditioned and free-floating. Failures: 0.

## Energy (conditioned), RC / EnergyPlus summed over cases

| Topology | Model | Heating | Sensible cooling | Heating hourly CV(RMSE) |
|---|---|---|---|---|
| Flat roof, no basement | 5R1C | 1.017 | 0.94 | 66.5 % |
| Flat roof, no basement | ISO 52016-1 | 1.017 | 0.84 | 48.8 % |
| Conditioned basement | 5R1C | 1.042 | 0.98 | 73.1 % |
| Conditioned basement | ISO 52016-1 | 1.040 | 0.87 | 46.0 % |
| Cold attic | 5R1C | 1.019 | 0.95 | 71.5 % |
| Cold attic | ISO 52016-1 | 1.018 | 0.83 | 49.5 % |
| Basement and attic | 5R1C | 1.043 | 0.98 | 78.2 % |
| Basement and attic | ISO 52016-1 | 1.041 | 0.86 | 46.5 % |

## Zone air temperature (conditioned), bias / RMSE in K, mean over cases

| Topology | Model | Main | Basement | Attic |
|---|---|---|---|---|
| Flat roof, no basement | 5R1C | +0.00 / 0.28 | – | – |
| Flat roof, no basement | ISO 52016-1 | -0.04 / 0.21 | – | – |
| Conditioned basement | 5R1C | +0.04 / 0.33 | -0.45 / 0.78 | – |
| Conditioned basement | ISO 52016-1 | -0.00 / 0.18 | -0.47 / 0.75 | – |
| Cold attic | 5R1C | +0.00 / 0.29 | – | – |
| Cold attic | ISO 52016-1 | -0.05 / 0.20 | – | +0.17 / 0.72 |
| Basement and attic | 5R1C | +0.04 / 0.34 | -0.45 / 0.78 | – |
| Basement and attic | ISO 52016-1 | -0.01 / 0.16 | -0.47 / 0.75 | +0.19 / 0.72 |

## Zone air temperature (free), bias / RMSE in K, mean over cases

| Topology | Model | Main | Basement | Attic |
|---|---|---|---|---|
| Flat roof, no basement | 5R1C | -0.05 / 0.90 | – | – |
| Flat roof, no basement | ISO 52016-1 | -0.39 / 0.92 | – | – |
| Conditioned basement | 5R1C | +0.13 / 1.09 | -1.21 / 1.75 | – |
| Conditioned basement | ISO 52016-1 | -0.20 / 0.54 | -1.50 / 1.98 | – |
| Cold attic | 5R1C | -0.07 / 0.96 | – | – |
| Cold attic | ISO 52016-1 | -0.42 / 0.91 | – | -0.12 / 0.75 |
| Basement and attic | 5R1C | +0.11 / 1.15 | -1.20 / 1.75 | – |
| Basement and attic | ISO 52016-1 | -0.23 / 0.53 | -1.49 / 1.98 | -0.07 / 0.70 |

## Per climate (conditioned, ISO 52016-1), RC / EnergyPlus

| Climate | Heating | Sensible cooling | Attic RMSE (K) | Basement RMSE (K) |
|---|---|---|---|---|
| goteborg | 0.999 | 1.03 | 0.73 | 0.76 |
| kiruna | 1.035 | 0.27 | 0.93 | 0.03 |
| chicago | 1.044 | 0.91 | 0.59 | 0.95 |
| tampa | 1.237 | 0.84 | 0.63 | 1.25 |

## Reading

- The zones add no error of their own. With ISO 52016-1, heating, cooling and
  main-zone temperatures with an attic or basement match the flat-roof
  reference of the same building within about 2 %, and the main-zone RMSE in
  free-floating mode drops from 0.92 K to 0.53 K once the basement is a zone.
- The attic air temperature follows EnergyPlus's attic zone with a bias of
  +0.2 K and an hourly RMSE of 0.7 K. The 5R1C quasi-steady buffer has no attic
  temperature to compare.
- Both RC models run the basement 0.5 K (conditioned) to 1.2–1.5 K
  (free-floating) colder than EnergyPlus. The two models share the ISO 13370
  ground reduction, while EnergyPlus uses the two-dimensional Kiva ground model,
  so this is a ground-model difference rather than the zone coupling.
- On this smaller footprint both models underpredict Tampa cooling (ISO 52016-1
  0.84, 5R1C 0.94 of EnergyPlus) already in the flat-roof reference; the
  38-case stress set with the preset footprint gives 1.00 and 1.17.
- Heating in Tampa is small (a few kWh/m²), so its 1.24 ratio is a small
  absolute error. January free-floating temperatures differ because
  EnergyPlus warms up on repeated first days while RC repeats the whole year.
