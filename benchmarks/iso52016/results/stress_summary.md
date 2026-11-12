# Stress comparison with EnergyPlus (144 cases)

## Free-floating air temperature

### All (72 cases)

| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5R1C | 1.21 | 1.96 | +0.25 | +0.41 | +0.03 | 0.87 | 0.992 | +1.30 | 1.06 |
| 52016 (operative) | 0.84 | 2.01 | -0.18 | -0.10 | -0.61 | 0.95 | 0.997 | -0.64 | 0.96 |
| 52016 (air) | 0.84 | 2.01 | -0.18 | -0.10 | -0.61 | 0.95 | 0.997 | -0.64 | 0.96 |

### climate = chicago (12 cases)

| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5R1C | 1.35 | 1.70 | +0.25 | +0.57 | +0.11 | 0.87 | 0.993 | +1.29 | 1.05 |
| 52016 (operative) | 0.84 | 1.10 | -0.31 | -0.10 | -0.64 | 0.91 | 0.997 | -0.55 | 0.95 |
| 52016 (air) | 0.84 | 1.10 | -0.31 | -0.10 | -0.64 | 0.91 | 0.997 | -0.55 | 0.95 |

### climate = golden (12 cases)

| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5R1C | 1.51 | 1.76 | -0.56 | -0.24 | -0.41 | 0.92 | 0.988 | +0.32 | 0.91 |
| 52016 (operative) | 1.17 | 1.52 | -1.05 | -0.98 | -0.94 | 0.98 | 0.998 | -1.24 | 0.81 |
| 52016 (air) | 1.17 | 1.52 | -1.05 | -0.98 | -0.94 | 0.98 | 0.998 | -1.24 | 0.81 |

### climate = goteborg (12 cases)

| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5R1C | 1.19 | 1.53 | +0.70 | +1.01 | +0.37 | 0.86 | 0.994 | +1.44 | 1.59 |
| 52016 (operative) | 0.67 | 0.95 | +0.20 | +0.54 | -0.12 | 0.90 | 0.997 | -0.21 | 1.23 |
| 52016 (air) | 0.67 | 0.95 | +0.20 | +0.54 | -0.12 | 0.90 | 0.997 | -0.21 | 1.23 |

### climate = kiruna (12 cases)

| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5R1C | 1.43 | 1.96 | +0.28 | +0.20 | +0.25 | 0.92 | 0.992 | +2.34 | 3.41 |
| 52016 (operative) | 1.27 | 2.01 | -0.29 | -0.18 | -0.97 | 0.89 | 0.993 | -0.60 | 0.78 |
| 52016 (air) | 1.27 | 2.01 | -0.29 | -0.18 | -0.97 | 0.89 | 0.993 | -0.60 | 0.78 |

### climate = malmo (12 cases)

| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5R1C | 1.02 | 1.38 | +0.44 | +0.80 | -0.05 | 0.87 | 0.994 | +2.18 | 1.80 |
| 52016 (operative) | 0.92 | 1.26 | -0.09 | +0.24 | -0.77 | 0.93 | 0.992 | -0.57 | 1.15 |
| 52016 (air) | 0.92 | 1.26 | -0.09 | +0.24 | -0.77 | 0.93 | 0.992 | -0.57 | 1.15 |

### climate = tampa (12 cases)

| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5R1C | 0.86 | 1.20 | +0.27 | +0.33 | +0.02 | 0.85 | 0.990 | +0.64 | 1.02 |
| 52016 (operative) | 0.46 | 0.78 | -0.12 | -0.16 | -0.19 | 0.91 | 0.997 | -0.24 | 0.98 |
| 52016 (air) | 0.46 | 0.78 | -0.12 | -0.16 | -0.19 | 0.91 | 0.997 | -0.24 | 0.98 |

### archetype = mf_se02 (24 cases)

| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5R1C | 1.28 | 1.76 | +0.06 | +0.31 | -0.12 | 0.89 | 0.992 | +1.19 | 1.02 |
| 52016 (operative) | 0.90 | 1.53 | -0.38 | -0.14 | -0.80 | 0.95 | 0.998 | -0.60 | 0.91 |
| 52016 (air) | 0.90 | 1.53 | -0.38 | -0.14 | -0.80 | 0.95 | 0.998 | -0.60 | 0.91 |

### archetype = office_se06 (24 cases)

| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5R1C | 1.45 | 1.96 | +0.45 | +0.61 | +0.10 | 0.75 | 0.988 | +1.48 | 1.08 |
| 52016 (operative) | 1.02 | 2.01 | -0.10 | +0.04 | -0.72 | 0.68 | 0.995 | -0.70 | 1.01 |
| 52016 (air) | 1.02 | 2.01 | -0.10 | +0.04 | -0.72 | 0.68 | 0.995 | -0.70 | 1.01 |

### archetype = sf_se05 (24 cases)

| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5R1C | 0.96 | 1.28 | +0.28 | +0.39 | +0.23 | 1.05 | 0.997 | +1.33 | 1.05 |
| 52016 (operative) | 0.72 | 1.42 | -0.15 | -0.06 | -0.34 | 1.04 | 0.997 | -0.49 | 0.95 |
| 52016 (air) | 0.72 | 1.42 | -0.15 | -0.06 | -0.34 | 1.04 | 0.997 | -0.49 | 0.95 |

### form = reference (36 cases)

| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5R1C | 0.96 | 1.76 | +0.06 | +0.06 | -0.13 | 0.80 | 0.992 | +0.77 | 1.01 |
| 52016 (operative) | 1.07 | 2.01 | -0.34 | -0.24 | -0.78 | 0.85 | 0.996 | -0.94 | 0.91 |
| 52016 (air) | 1.07 | 2.01 | -0.34 | -0.24 | -0.78 | 0.85 | 0.996 | -0.94 | 0.91 |

### form = tall_glazed (36 cases)

| Model | Median RMSE K | Worst RMSE K | Bias K | Summer bias | Winter bias | Daily swing ratio | r | Peak T error K | Hours >27 °C ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5R1C | 1.23 | 1.96 | +0.54 | +0.67 | +0.31 | 1.00 | 0.992 | +1.98 | 1.10 |
| 52016 (operative) | 0.74 | 1.34 | -0.08 | +0.13 | -0.40 | 0.98 | 0.998 | -0.12 | 1.00 |
| 52016 (air) | 0.74 | 1.34 | -0.08 | +0.13 | -0.40 | 0.98 | 0.998 | -0.12 | 1.00 |

## Conditioned loads

\* over cases where E+ needs more than 1 kWh/m² of that load.

Cooling is the EnergyPlus sensible coil energy. The latent share below is
excluded because both RC models are sensible only.

| Climate | E+ sensible cooling kWh/m² | Latent share of E+ total |
|---|---|---|
| chicago | 13.4 | 30% |
| golden | 11.0 | 1% |
| goteborg | 2.6 | 9% |
| kiruna | 0.2 | 0% |
| malmo | 1.6 | 10% |
| tampa | 50.0 | 43% |

### All: cooling (72 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.17 | 1.15 | 27/48 | 2.07 | 1.29 | 0.95 |
| 52016 (operative) | 1.10 | 1.11 | 32/48 | 1.50 | 1.33 | 0.94 |
| 52016 (air) | 1.00 | 1.00 | 31/48 | 0.98 | 1.12 | 0.97 |

### climate = chicago: cooling (12 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.23 | 1.15 | 6/12 | 2.42 | 1.21 | 0.96 |
| 52016 (operative) | 1.13 | 1.07 | 10/12 | 1.52 | 1.27 | 0.95 |
| 52016 (air) | 1.02 | 0.98 | 11/12 | 0.82 | 1.12 | 0.98 |

### climate = golden: cooling (12 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.14 | 1.04 | 9/12 | 1.59 | 1.30 | 0.95 |
| 52016 (operative) | 1.03 | 0.96 | 10/12 | 1.19 | 1.35 | 0.94 |
| 52016 (air) | 0.92 | 0.88 | 8/12 | 1.06 | 1.14 | 0.97 |

### climate = goteborg: cooling (12 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.88 | 2.00 | 0/8 | 1.91 | 1.41 | 0.91 |
| 52016 (operative) | 1.59 | 1.61 | 0/8 | 1.25 | 1.39 | 0.93 |
| 52016 (air) | 1.34 | 1.36 | 0/8 | 0.73 | 1.16 | 0.96 |

### climate = kiruna: cooling (12 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 3.22 | – | 0/0 | 0.37 | – | – |
| 52016 (operative) | 1.35 | – | 0/0 | 0.06 | – | – |
| 52016 (air) | 1.00 | – | 0/0 | 0.01 | – | – |

### climate = malmo: cooling (12 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.96 | 1.92 | 0/4 | 1.25 | 1.37 | 0.93 |
| 52016 (operative) | 1.52 | 1.55 | 0/4 | 0.66 | 1.36 | 0.94 |
| 52016 (air) | 1.26 | 1.29 | 0/4 | 0.35 | 1.13 | 0.97 |

### climate = tampa: cooling (12 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.09 | 1.06 | 12/12 | 4.90 | 1.21 | 0.96 |
| 52016 (operative) | 1.07 | 1.04 | 12/12 | 4.35 | 1.27 | 0.93 |
| 52016 (air) | 0.99 | 0.97 | 12/12 | 2.93 | 1.08 | 0.97 |

### archetype = mf_se02: cooling (24 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.15 | 1.14 | 9/14 | 1.83 | 1.33 | 0.95 |
| 52016 (operative) | 1.08 | 1.09 | 10/14 | 1.39 | 1.45 | 0.93 |
| 52016 (air) | 0.96 | 0.94 | 9/14 | 0.96 | 1.16 | 0.96 |

### archetype = office_se06: cooling (24 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.16 | 1.13 | 10/18 | 2.34 | 1.26 | 0.95 |
| 52016 (operative) | 1.10 | 1.11 | 12/18 | 1.68 | 1.26 | 0.96 |
| 52016 (air) | 1.02 | 1.03 | 12/18 | 1.16 | 1.08 | 0.99 |

### archetype = sf_se05: cooling (24 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.21 | 1.24 | 8/16 | 2.06 | 1.26 | 0.97 |
| 52016 (operative) | 1.13 | 1.14 | 10/16 | 1.44 | 1.28 | 0.93 |
| 52016 (air) | 1.01 | 1.02 | 10/16 | 0.83 | 1.11 | 0.96 |

### All: heating (72 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.05 | 1.03 | 50/68 | 4.30 | 1.09 | 0.98 |
| 52016 (operative) | 1.11 | 1.08 | 52/68 | 7.33 | 1.13 | 0.98 |
| 52016 (air) | 1.06 | 1.03 | 53/68 | 4.37 | 1.03 | 0.99 |

### climate = chicago: heating (12 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.04 | 1.03 | 12/12 | 2.75 | 1.07 | 0.99 |
| 52016 (operative) | 1.09 | 1.08 | 12/12 | 6.09 | 1.10 | 0.98 |
| 52016 (air) | 1.04 | 1.04 | 12/12 | 2.92 | 1.02 | 0.99 |

### climate = golden: heating (12 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.26 | 1.26 | 0/12 | 13.31 | 1.24 | 0.97 |
| 52016 (operative) | 1.32 | 1.32 | 0/12 | 15.84 | 1.31 | 0.96 |
| 52016 (air) | 1.27 | 1.26 | 0/12 | 13.33 | 1.17 | 0.98 |

### climate = goteborg: heating (12 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 0.99 | 0.99 | 12/12 | 1.79 | 1.07 | 0.98 |
| 52016 (operative) | 1.05 | 1.04 | 12/12 | 4.23 | 1.13 | 0.98 |
| 52016 (air) | 1.00 | 1.00 | 12/12 | 1.84 | 1.03 | 0.99 |

### climate = kiruna: heating (12 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.04 | 1.03 | 12/12 | 5.93 | 1.06 | 0.99 |
| 52016 (operative) | 1.09 | 1.08 | 12/12 | 13.15 | 1.12 | 0.99 |
| 52016 (air) | 1.04 | 1.03 | 12/12 | 6.14 | 1.04 | 0.99 |

### climate = malmo: heating (12 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.00 | 0.99 | 12/12 | 1.26 | 1.06 | 0.98 |
| 52016 (operative) | 1.06 | 1.05 | 12/12 | 4.24 | 1.11 | 0.98 |
| 52016 (air) | 1.01 | 1.01 | 12/12 | 1.56 | 1.01 | 0.99 |

### climate = tampa: heating (12 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.31 | 1.26 | 2/8 | 0.74 | 1.26 | 0.95 |
| 52016 (operative) | 1.17 | 1.19 | 4/8 | 0.40 | 1.31 | 0.95 |
| 52016 (air) | 1.18 | 1.15 | 5/8 | 0.45 | 1.15 | 0.97 |

### archetype = mf_se02: heating (24 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.07 | 1.06 | 16/24 | 6.42 | 1.14 | 0.98 |
| 52016 (operative) | 1.14 | 1.12 | 16/24 | 12.55 | 1.21 | 0.98 |
| 52016 (air) | 1.07 | 1.06 | 17/24 | 6.61 | 1.06 | 0.99 |

### archetype = office_se06: heating (24 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.03 | 0.99 | 16/20 | 2.24 | 1.07 | 0.93 |
| 52016 (operative) | 1.05 | 1.03 | 16/20 | 2.50 | 1.10 | 0.97 |
| 52016 (air) | 1.04 | 1.01 | 16/20 | 2.22 | 1.01 | 0.96 |

### archetype = sf_se05: heating (24 cases)

| Model | Total / E+ | Median ratio* | Within ±20 %* | MAE kWh/m² | Peak ratio* | r* |
|---|---|---|---|---|---|---|
| 5R1C | 1.04 | 1.02 | 18/24 | 4.23 | 1.07 | 0.98 |
| 52016 (operative) | 1.09 | 1.06 | 20/24 | 6.94 | 1.12 | 0.98 |
| 52016 (air) | 1.05 | 1.03 | 20/24 | 4.27 | 1.03 | 0.99 |

