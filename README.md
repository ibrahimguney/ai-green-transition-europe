# AI Green Transition Europe

Research project:

**Artificial Intelligence and the Green Transition: Does Enterprise AI Adoption Reduce Carbon Intensity or Create a Digital Rebound Effect in Europe?**

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

## 3. Step 02 — NACE harmonisation and panel diagnostics

```powershell
python src\02_build_core_panel.py
python src\03_panel_diagnostics.py
```

See [`STEP02.md`](STEP02.md).

## 4. Step 03 — Baseline fixed-effects models

```powershell
python src\04_fixed_effects.py
```

Models include two-way fixed effects, linear AI, quadratic `AI + AI²`, log-GHG primary outcome, winsorised level robustness, and clustered-inference checks.

## 5. Step 04 — Country-year controls and controlled FE

```powershell
python src\05_download_controls.py
python src\06_merge_controls.py
python src\07_controlled_fixed_effects.py
```

Controls:
- renewable-energy share (`nrg_ind_ren`)
- real GDP per capita (`nama_10_pc`)
- R&D intensity (`rd_e_gerdtot`)
- non-household electricity price (`nrg_pc_205`, optional when coverage is available)

France/Sweden-excluded robustness models address the documented 2023 ICT-series break. Current controlled-FE evidence does not support a statistically significant linear AI effect or the hypothesised U-shaped digital-rebound effect.

See [`STEP04.md`](STEP04.md).

## 6. Step 05 — XGBoost + SHAP

```powershell
python src\08_xgboost_shap.py
python src\09_grouped_shap.py
```

The ML layer uses XGBoost with country-grouped cross-validation (`GroupKFold`). Country is not used as a predictor. SHAP values are aggregated back to original conceptual feature groups.

Current grouped-SHAP ranking:
1. NACE sector
2. Real GDP per capita
3. Renewable-energy share
4. R&D intensity
5. AI adoption
6. Year

## 7. Step 06 — Econometric + ML synthesis

```powershell
python src\10_synthesize_results.py
```

Main outputs:
- `outputs/tables/manuscript_table_fe.csv`
- `outputs/tables/manuscript_table_ml.csv`
- `outputs/tables/manuscript_table_grouped_shap.csv`
- `outputs/tables/evidence_synthesis.csv`
- `manuscript/results_draft.md`

See [`STEP06.md`](STEP06.md).

## 8. Step 07 — Full manuscript assembly

Narrative manuscript modules:
- `manuscript/front_matter.md`
- `manuscript/introduction_draft.md`
- `manuscript/literature_review_draft.md`
- `manuscript/methods_draft.md`
- `manuscript/results_draft.md` (generated from Step 06)
- `manuscript/discussion_draft.md`
- `manuscript/conclusion_draft.md`

Assemble the full manuscript with:

```powershell
python src\11_build_full_manuscript.py
```

Output:
- `manuscript/manuscript_full.md`

Journal targeting notes:
- `manuscript/journal_targets.md`

References:
- `references/references_step07.bib`

## Current research sequence

1. Core data download — complete
2. Exact NACE panel construction — complete
3. Panel diagnostics — complete
4. Baseline two-way FE — complete
5. Country-year controls and controlled FE — complete
6. XGBoost + SHAP explanatory layer — complete
7. Econometric/ML synthesis and manuscript-ready tables — complete
8. Full manuscript assembly and journal targeting — **current stage**
9. Journal-specific formatting, figures, highlights, cover letter, and submission package — next

## Interpretation principle

The current panel has only three comparable AI survey years. Results are therefore treated as **associational fixed-effects evidence** plus **predictive machine-learning evidence**, not definitive causal estimates. The current evidence is best described as an **absence of robust evidence for an independent AI green effect in the available data**, not as proof that AI has no environmental consequences.
