# EPSM RC

Reduced-order thermal calculation using the existing ISO 13790 5R1C implementation.
The package and import are now `epsm_rc` (previously `SweThermalModel` /
`swethermalmodel`). Numerical solver behavior was preserved during the rename.

```python
from epsm_rc import create_swedish_villa
villa = create_swedish_villa(floor_area_m2=150, era="1976_1990")
result = villa.simulate_with_epw("weather.epw")
print(result["heating_demand_kwh"])
```

Install with `python -m pip install .`; tests with `python -m pytest tests`.
The platform adapter is documented separately as it is introduced. The standalone
factories retain historical defaults and should not be used to assign evidence
in the harmonised Core Building model.

Code includes adaptations credited in the source to DIBS/IWU and
RC_BuildingSimulator/ETH. Retain their original notices. EPSM-owned additions use
MIT; see LICENSE. Local ISO reference extracts are excluded from publication.
Bundled weather inputs retain their source headers.
