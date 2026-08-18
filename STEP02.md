# Step 02 — NACE Harmonisation and Panel Diagnostics

## Objective

Build a defensible **country × NACE Rev. 2 × year** panel linking enterprise AI adoption to greenhouse-gas emission intensity.

## Core methodological rule

The project **does not average GHG-intensity ratios across different NACE groups**.

The first empirical panel uses only observations with the same:

- country (`geo`)
- NACE Rev. 2 code (`nace_r2`)
- year (`time`)

This prevents invalid aggregation of intensity ratios. If broader NACE harmonisation becomes necessary later, it should be reconstructed from emissions totals and gross value added, not from a simple mean of intensity ratios.

## Core years

Primary panel window:

**2021–2024**

AI observations for 2025 remain available in the raw data but are not paired with a 2024 environmental outcome.

## Run

```powershell
python src\02_build_core_panel.py
python src\03_panel_diagnostics.py
```

## Main outputs

### Harmonisation

- `outputs/tables/nace_inventory.csv`
- `outputs/tables/nace_match_report.csv`
- `outputs/tables/duplicate_key_report.csv`

### Panel

- `data/processed/core_panel.csv`

### Coverage

- `outputs/tables/panel_coverage_by_year.csv`
- `outputs/tables/panel_coverage_by_country.csv`
- `outputs/tables/panel_coverage_by_nace.csv`

### Diagnostics

- `outputs/tables/descriptive_statistics.csv`
- `outputs/tables/missingness_report.csv`
- `outputs/tables/correlation_matrix.csv`
- `outputs/tables/panel_balance_report.csv`
- `outputs/tables/descriptives_by_year.csv`
- `outputs/tables/descriptives_by_nace.csv`

## Decision gate before econometrics

Do not start fixed-effects estimation until these checks pass:

1. No duplicated `geo × nace_r2 × time` key.
2. Sufficient number of exact NACE matches.
3. Adequate country and sector coverage.
4. Acceptable panel balance.
5. No unexplained missing values in the core variables.

Once these checks pass, the next stage will add:
- two-way fixed effects,
- AI-squared rebound specification,
- clustered standard errors,
- and later the ML/SHAP layer.
