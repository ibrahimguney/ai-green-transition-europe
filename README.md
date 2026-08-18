# AI Green Transition Europe

Research project:

**Artificial Intelligence and the Green Transition: Does AI Adoption Reduce Carbon Intensity or Create a Digital Rebound Effect in Europe?**

## Research design

Unit of analysis:

**country × NACE Rev. 2 economic activity × year**

Primary estimation window:

**2021–2024**

Main outcome:
- greenhouse-gas emission intensity of gross value added

Main explanatory variable:
- enterprise AI adoption (% of enterprises)

The core panel uses **exact NACE Rev. 2 matches only**. GHG-intensity ratios are not averaged across incompatible NACE groups.

## 1. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Step 01 — Download Eurostat data

```powershell
python src\01_download_eurostat.py
```

Expected raw files:

- `data/raw/eurostat_ai_nace.csv`
- `data/raw/eurostat_ghg_intensity_nace.csv`
- `data/raw/eurostat_digital_intensity_nace.csv`
- `data/raw/eurostat_cloud_nace.csv`

## 3. Step 02 — NACE harmonisation and core panel

```powershell
python src\02_build_core_panel.py
```

Main output:

- `data/processed/core_panel.csv`

Additional reports:

- `outputs/tables/nace_inventory.csv`
- `outputs/tables/nace_match_report.csv`
- `outputs/tables/duplicate_key_report.csv`
- `outputs/tables/panel_coverage_by_year.csv`
- `outputs/tables/panel_coverage_by_country.csv`
- `outputs/tables/panel_coverage_by_nace.csv`

## 4. Panel diagnostics and descriptive statistics

```powershell
python src\03_panel_diagnostics.py
```

Outputs include:

- descriptive statistics
- missingness report
- correlation matrix
- balanced-panel diagnostics
- year-level descriptives
- NACE-level descriptives

See [`STEP02.md`](STEP02.md) for methodological details.

## Eurostat datasets

- `isoc_eb_ain2` — Artificial intelligence by NACE Rev. 2 activity
- `env_ac_aeint_r2` — Air emissions intensities by NACE Rev. 2 activity
- `isoc_e_diin2` — Digital Intensity by NACE Rev. 2 activity
- `isoc_cicce_usen2` — Cloud computing services by NACE Rev. 2 activity

## Planned next stage

After Step 02 diagnostics pass:

1. two-way fixed-effects model
2. nonlinear `AI + AI²` digital-rebound model
3. clustered standard errors
4. robustness controls
5. XGBoost + SHAP explanatory layer
