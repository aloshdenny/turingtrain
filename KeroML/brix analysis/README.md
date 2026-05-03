# BRIX Analysis

This folder contains BI/BRIX-integrated scenario analysis for KeroML.

## Files

- `keroml_brix_scenario_analysis.ipynb`
  - Integrates theoretical BRIX limits with KeroML composition data.
  - Produces least/mean/most branching CN scenarios.
  - Runs Monte Carlo BI perturbations and exports uncertainty bands.

## Outputs Produced by Notebook

- `keroml_brix_scenarios.csv`
- `keroml_brix_monte_carlo.csv`

## Notes

The BI perturbation currently uses a uniform distribution between theoretical BRIX min and max values for branching-sensitive classes.
