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

- Window: Core simple glazing with U 2.10 W/(m² K) and SHGC 0.769 (the
  Standard 140-2020 WINDOW 7 file, glazing ID 23). Core places one 12 m² pane,
  not two 3 m × 2 m panes.
- Floor: constant 10 °C ground boundary in all engines, as in the Modelica
  Buildings Library Case 600. The 1 m floor insulation makes it about 2 % of
  the heat loss.
- RC infiltration uses the air heat capacity at 1650 m (992 J/(m³ K)).
  EnergyPlus applies the site elevation itself.
- Films, convection and sky exchange come from each engine.

## Reading the results

Cooling is compared on the zone sensible ideal-loads energy, the RC boundary.

The Standard 140 range in the table is from the example results in ASHRAE
140-2017 Table B8-2. Those programs used the older DRYCOLD.TMY weather, whose
coldest hour is −24.4 °C; the TMY3 file reaches −19.4 °C. That is why the
EnergyPlus run on the same Building also falls below the heating range.
The EnergyPlus column is the like-for-like reference here. A
result inside the range means agreement with other programs, not
certification.

Core's warning review rejects the EnergyPlus run for four psychrometric
warnings (a wet-bulb iteration in very dry air at 82.6 kPa) that it cannot
waive. EnergyPlus completes with no severe errors, so the script reads the
ideal-loads energies from `eplusout.sql` and records the rejection in
`results/case600.json`.

## Section 5.2 case family

`ashrae140.py` runs 14 variations of Case 600 on the same three engines:
600, 610 (south overhang), 620 (east/west glazing), 640 (heating setback),
650 (night ventilation, cooling by day), the heavyweight 900, 910, 920, 940
and 950, and the free-float cases 600FF, 650FF, 900FF and 950FF. Cases 630
and 930 (side fins) and 960 (sunspace) are not run: RC does not model fins
or a sunspace. Results are in `results/ashrae140.md`.

```sh
python benchmarks/bestest/ashrae140.py --epw USA_CO_Denver.Intl.AP.725650_TMY3.epw \
    --energyplus /path/to/energyplus --output outputs/ashrae140
```

EnergyPlus heating and cooling are the ideal-loads supply-air (coil) sensible
energies, so the free cooling of night ventilation air is not counted as load.
These runs compare engines; they are not yet checked against the Standard 140
acceptance criteria.
