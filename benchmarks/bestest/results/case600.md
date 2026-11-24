# BESTEST Case 600

Weather: `USA_CO_Denver.Intl.AP.725650_TMY3.epw`. All engines run the same EPSM Core Building.
Range: ASHRAE 140-2017 Table B8-2 example results (older DRYCOLD weather).
RC cells give the difference from the EnergyPlus run in brackets.

| Metric | Standard 140 range | EnergyPlus 24.2 | 5R1C | ISO 52016-1 |
|---|---|---|---|---|
| Annual heating (MWh) | 4.30 – 5.71 | 3.86 ¹ | 4.17 (+8 %) ¹ | 3.96 (+2 %) ¹ |
| Annual cooling (MWh) | 6.14 – 8.45 | 6.81 | 6.59 (-3 %) | 6.17 (-9 %) |
| Peak heating (kW) | 3.44 – 4.35 | 2.99 ¹ | 3.10 (+4 %) ¹ | 3.04 (+2 %) ¹ |
| Peak cooling (kW) | 5.96 – 7.19 | 6.75 | 6.71 (-1 %) | 6.24 (-8 %) |

¹ outside the Standard 140 range.

Hourly air temperature RMSE against EnergyPlus: 5R1C 0.32 K, ISO 52016-1 0.37 K.
