# BESTEST Case 600

`case600.py` builds ASHRAE Standard 140 Case 600 as one EPSM Core `Building`
and runs it through both RC thermal models (`iso13790_5r1c` and `iso52016_1`)
with every other setting shared. With `--energyplus`, the same Building also
runs in EnergyPlus 24.2 through Core, so all three engines see identical inputs
and weather.

```sh
python benchmarks/bestest/case600.py \
    --epw USA_CO_Denver.Intl.AP.725650_TMY3.epw \
    --energyplus /path/to/energyplus --output outputs/bestest
```

The EPW is the Denver International Airport TMY3 file (WMO 725650), the
Standard 140-2020 site. It is not bundled; the EnergyPlus weather site provides
it. `results/` holds the latest run.

## Inputs and assumptions

Geometry, layer materials, south glazing (12 m²), 0.5 ACH infiltration,
200 W of continuous gains (60 % radiative) and the 20/27 °C thermostat follow
Standard 140 Section 5.2.1. These are choices of this script:

- Window: Core simple glazing with U 2.74 W/(m² K) and SHGC 0.760. Those are
  the values EnergyPlus 24.2 computes for the Standard 140 double clear pane
  (two 3.048 mm panes, 12 mm air gap).
- Floor: constant 10 °C ground boundary in all engines, as in the Modelica
  Buildings Library Case 600.
- RC infiltration uses the air heat capacity at 1650 m (992 J/(m³ K)).
  EnergyPlus applies the site elevation itself.
- Overhangs (610, 910) use `overhang_shading_method="geometric_profile_angle"`,
  the exact shadow of an infinitely wide overhang. ISO 13790 G.4-G.6 shade too
  much in winter (8 % less transmitted solar than EnergyPlus over the year).
- Films, convection and sky exchange come from each engine.

## Reference results and acceptance criteria

`ashrae140.py` compares every case with the ASHRAE 140-2020 example-result
ranges on the same Denver TMY3 weather and applies the Standard 140-2020
Addendum b test acceptance criteria to annual heating and cooling. Both tables
are taken from the Modelica Buildings Library BESTEST validation package
(`Buildings.ThermalZones.Detailed.Validation.BESTEST`). The ranges are
informative; the acceptance criteria are the pass/fail test. `--check` exits
non-zero when an RC model misses a criterion.

Cooling is compared on the ideal-loads supply-air sensible energy, the RC
boundary.

Core's warning review rejects some EnergyPlus runs for psychrometric warnings
(a wet-bulb iteration in very dry air at 82.6 kPa) that it cannot waive.
EnergyPlus completes with no severe errors, so the script reads the
ideal-loads energies from `eplusout.sql` and records the rejection.

## Section 5.2 case family

`ashrae140.py` runs 14 variations of Case 600: 600, 610 (south overhang),
620 (east/west glazing), 640 (heating setback), 650 (night ventilation,
cooling by day), the heavyweight 900, 910, 920, 940 and 950, and the
free-float cases 600FF, 650FF, 900FF and 950FF. Cases 630 and 930 (side fins)
and 960 (sunspace) are not run: RC does not model fins or a sunspace. Results
are in `results/ashrae140.md`.

```sh
python benchmarks/bestest/ashrae140.py --epw USA_CO_Denver.Intl.AP.725650_TMY3.epw \
    --energyplus /path/to/energyplus --output outputs/ashrae140 --check
```
