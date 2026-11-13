# 5R1C, ISO 52016-1 and EnergyPlus over Studio options

`compare_energyplus.py` builds 38 buildings from the parameters EPSM Studio
exposes: seven archetypes (function and era), three fabric states and two
orientations. Each uses the illustrative-reference geometry, the bundled
Gothenburg EPW and an explicit 26 °C cooling setpoint. Every building runs once in
EnergyPlus 24.2 and in nine RC variants: 5R1C or ISO 52016-1, v1 or v2 solar
optics, and continuous or load-dependent heat recovery. `summarise.py` writes
`results/summary.md`. These are model-to-model comparisons, not validation.
Office and school calendars are synthetic. EnergyPlus cooling is the sensible
part of the ideal-loads supply-coil energy, because both RC models are sensible
only; the total coil energy includes latent load (7 % here, 43 % in Tampa).

`stress_energyplus.py` runs 144 cases: six climates (Gothenburg, Kiruna, Malmö,
Chicago, Golden, Tampa), three archetypes, two forms (3 storeys and 20 % glazing,
6 storeys and 45 % glazing), two orientations, conditioned and free-floating.
`summarise_stress.py` writes `results/stress_summary.md`. `benchmark_speed.py`
times the Python and Rust recurrences (`results/speed.md`).

```sh
python benchmarks/iso52016/compare_energyplus.py --directory outputs/iso52016 \
    --energyplus /path/to/energyplus --workers 4
python benchmarks/iso52016/summarise.py outputs/iso52016/results.json --output outputs/iso52016/summary
python benchmarks/iso52016/stress_energyplus.py --directory outputs/stress \
    --energyplus /path/to/energyplus --workers 4
python benchmarks/iso52016/summarise_stress.py outputs/stress/results.json --output outputs/stress
python benchmarks/iso52016/benchmark_speed.py --repeats 5
```
