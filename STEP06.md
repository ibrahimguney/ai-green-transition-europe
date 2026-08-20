# Step 06 — Econometric + ML Results Synthesis

## Objective

Combine the fixed-effects, robustness, XGBoost, and grouped-SHAP evidence into reproducible manuscript-ready tables and an automatically generated Results draft.

## Run

```powershell
python src\10_synthesize_results.py
```

## Required prior outputs

The script expects the outputs produced by Steps 04 and 05:

- `outputs/tables/controlled_fe_coefficients.csv`
- `outputs/tables/controlled_fe_model_diagnostics.csv`
- `outputs/tables/controlled_quadratic_turning_points.csv`
- `outputs/tables/xgb_cv_metrics.csv`
- `outputs/tables/xgb_cv_predictions.csv`
- `outputs/tables/shap_grouped_importance.csv`

## Generated manuscript tables

### Table A — Controlled fixed-effects results

`outputs/tables/manuscript_table_fe.csv`

Includes:
- model specification,
- sample size,
- country/entity counts,
- AI coefficient and p-value,
- AI-squared coefficient and p-value,
- renewable-energy, GDP, R&D, and electricity-price terms where available,
- approximate 10-percentage-point AI semi-elasticity for linear specifications.

### Table B — XGBoost validation

`outputs/tables/manuscript_table_ml.csv`

Reports pooled out-of-country:
- RMSE,
- MAE,
- R-squared,
- average RMSE improvement relative to the fold-specific mean baseline.

### Table C — Grouped SHAP importance

`outputs/tables/manuscript_table_grouped_shap.csv`

Ranks conceptual feature groups after aggregating one-hot transformed NACE and year columns back to their original variables.

### Evidence matrix

`outputs/tables/evidence_synthesis.csv`

Provides a transparent H1/H2 decision summary based on the pre-specified sign and statistical-credibility rules.

## Results draft

`manuscript/results_draft.md`

The Results text is generated directly from the numerical output files. This reduces transcription error and keeps the manuscript synchronized with rerun analyses.

## Current substantive interpretation

The current evidence does **not** support either:

1. a robust negative independent association between AI adoption and GHG intensity, or
2. the pre-specified U-shaped digital rebound effect.

The ML model nevertheless predicts GHG intensity strongly out of country. Grouped SHAP indicates that NACE sector, real GDP per capita, and renewable-energy share are more important predictive feature groups than AI adoption.

This should be framed as an **absence of robust evidence for an independent AI green effect in the available three-wave panel**, not as proof that AI has no environmental effects.

## Next stage

After reviewing the generated tables and Results draft:

1. construct publication-quality figures and consolidated tables,
2. draft the Methods section with data provenance and estimators,
3. position the null/weak AI finding against the recent AI-and-sustainability literature,
4. draft Discussion, limitations, and policy implications,
5. assemble the full manuscript.
