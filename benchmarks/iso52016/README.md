# 5R1C, ISO 52016-1 and EnergyPlus over Studio options

`compare_energyplus.py` builds 38 buildings from the parameters EPSM Studio
exposes: seven archetypes (function and era), three fabric states and two
orientations. Each uses the illustrative-reference geometry, the bundled
Gothenburg EPW and an explicit 26 °C cooling setpoint. Every building runs once in
EnergyPlus 24.2 and in nine RC variants: 5R1C or ISO 52016-1, v1 or v2 solar
optics, and continuous or load-dependent heat recovery. `summarise.py` writes
`results/summary.md`. These are model-to-model comparisons, not validation.
Office and school calendars are synthetic. EnergyPlus cooling is ideal-loads
supply-coil cooling.

```sh
python benchmarks/iso52016/compare_energyplus.py --directory outputs/iso52016 \
    --energyplus /path/to/energyplus --workers 4
python benchmarks/iso52016/summarise.py outputs/iso52016/results.json --output outputs/iso52016/summary
```
