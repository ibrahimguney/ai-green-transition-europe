# AI Green Transition Europe

Research project:

**Artificial Intelligence and the Green Transition: Does AI Adoption Reduce Carbon Intensity or Create a Digital Rebound Effect in Europe?**

## Research design

Unit of analysis:

**country × NACE Rev. 2 economic activity × year**

Current comparable AI estimation years:

**2021, 2023, 2024**

Main outcome:
- greenhouse-gas emission intensity of gross value added

Main explanatory variable:
- enterprise AI adoption (% of enterprises)

The core panel uses **exact NACE Rev. 2 matches only**. GHG-intensity ratios are not averaged across incompatible NACE groups. Main regressions use broad, non-overlapping one-letter NACE sections.

## 1. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Step 01 — Download Eurostat AI/GHG data

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
python src\03_panel_diagnostics.py
```

Main output:

- `data/processed/core_panel.csv`

See [`STEP02.md`](STEP02.md) for methodological details.

## 4. Step 03 — Baseline fixed-effects models

```powershell
python src\04_fixed_effects.py
```

Models include:

- two-way fixed effects
- linear AI effect
- quadratic `AI + AI²` specification
- log-GHG primary outcome
- winsorised level robustness
- entity- and country-clustered inference checks

Main outputs:

- `outputs/tables/fixed_effects_coefficients.csv`
- `outputs/tables/quadratic_turning_point.csv`
- `outputs/tables/step03_sample_diagnostics.csv`
- `outputs/tables/step03_model_summary.txt`

## 5. Step 04 — Add country-year controls

Download controls:

```powershell
python src\05_download_controls.py
```

Merge controls:

```powershell
python src\06_merge_controls.py
```

Estimate controlled fixed-effects models:

```powershell
python src\07_controlled_fixed_effects.py
```

Controls:

- renewable-energy share (`nrg_ind_ren`)
- real GDP per capita (`nama_10_pc`)
- R&D intensity (`rd_e_gerdtot`)
- non-household electricity price, band IC (`nrg_pc_205`)

Step 04 also estimates France/Sweden-excluded robustness models because Eurostat documents a 2023 break in the enterprise ICT time series for these countries.

See [`STEP04.md`](STEP04.md) for the complete model sequence and interpretation rules.

## Eurostat datasets

Core:

- `isoc_eb_ain2` — Artificial intelligence by NACE Rev. 2 activity
- `env_ac_aeint_r2` — Air emissions intensities by NACE Rev. 2 activity
- `isoc_e_diin2` — Digital Intensity by NACE Rev. 2 activity
- `isoc_cicce_usen2` — Cloud computing services by NACE Rev. 2 activity

Controls:

- `nrg_ind_ren` — Share of energy from renewable sources
- `nama_10_pc` — GDP and main components per capita
- `rd_e_gerdtot` — GERD by sector of performance
- `nrg_pc_205` — Non-household electricity prices

## Current research sequence

1. Core data download — complete
2. Exact NACE panel construction — complete
3. Panel diagnostics — complete
4. Baseline two-way FE — complete
5. Country-year controls and controlled FE — **current stage**
6. XGBoost + SHAP explanatory layer — next
7. Robustness synthesis and manuscript tables — later

## Interpretation principle

The current panel has only three comparable AI survey years. Results are therefore treated as **associational fixed-effects evidence**, not definitive causal estimates. Dynamic GMM is postponed until a longer comparable AI series can be constructed.
