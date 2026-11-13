# Overhang response against EnergyPlus

54 conditioned cases: 3 Swedish climates, 3 archetypes, 2 forms, overhang depth 0, 0.5, 1 m. Changes are relative to the same building without an overhang, in kWh/m².

| Depth | Quantity | E+ change, median | RC change, median | Same sign | RC / E+ change |
|---|---|---|---|---|---|
| 0.5 m | heating | +2.59 | +4.44 | 18/18 | 1.68 |
| 0.5 m | cooling | -0.26 | -0.29 | 12/12 | 1.38 |
| 1 m | heating | +5.06 | +7.65 | 18/18 | 1.54 |
| 1 m | cooling | -0.37 | -0.38 | 12/12 | 1.37 |

Same sign and the ratio count cases where EnergyPlus changes by more than 0.1 kWh/m².
These are model-to-model comparisons, not validation.
