# Step 05 — XGBoost + SHAP

## Objective

Use a nonlinear machine-learning model to complement the fixed-effects results and test whether AI adoption shows a meaningful predictive relationship with greenhouse-gas intensity after accounting for green-transition and macro controls.

This stage is **explanatory/predictive**, not causal.

## Input

`data/processed/controlled_broad_panel.csv`

Primary outcome:

`log_ghg = log(1 + ghg_intensity)`

## Features

Core numeric features:

- `ai_adoption`
- `renewable_share`
- `log_real_gdp_pc`
- `rd_intensity_pct_gdp`

Optional numeric feature:

- `electricity_price_eur_kwh`

The electricity-price feature is included only if at least 80% of the ML sample has non-missing values.

Categorical features:

- `nace_r2`
- `time`

Country is deliberately **not** used as a predictor.

## Validation strategy

The model uses **GroupKFold by country**.

This is more conservative than a random split because observations from a country cannot simultaneously appear in both the training and test portions of the same fold. The resulting metrics therefore measure how well the fitted relationship generalises to held-out countries rather than merely to held-out country-sector-year rows.

## Model

`XGBRegressor`

The initial specification uses moderate tree depth and regularisation rather than an aggressive hyperparameter search. The purpose at this stage is transparent nonlinear diagnostics and SHAP interpretation, not leaderboard optimisation.

## Outputs

### Cross-validation

- `outputs/tables/xgb_cv_metrics.csv`
- `outputs/tables/xgb_cv_predictions.csv`

Metrics include:

- RMSE on the log outcome
- MAE on the log outcome
- median absolute error
- R²
- improvement relative to a training-mean baseline

### Feature and SHAP outputs

- `outputs/tables/xgb_feature_set.csv`
- `outputs/tables/shap_feature_importance.csv`
- `outputs/tables/shap_ai_binned.csv`
- `outputs/tables/xgb_model_params.json`
- `outputs/tables/step05_ml_summary.txt`

Figures:

- `outputs/figures/shap_summary.png`
- `outputs/figures/shap_ai_dependence.png`

## Interpretation

The SHAP summary plot identifies which variables the fitted XGBoost model uses most strongly for prediction.

The AI dependence plot shows whether the contribution of AI adoption to predicted log-GHG intensity appears:

- approximately monotonic,
- U-shaped,
- inverted-U,
- threshold-like,
- or weak/flat.

A nonlinear SHAP pattern does **not** demonstrate a causal rebound effect. A substantive rebound claim would require consistency with the econometric specification, credible temporal ordering, robustness, and preferably a stronger identification design.

## Run

```powershell
pip install -r requirements.txt
python src\08_xgboost_shap.py
```

## Decision gate for Step 06

Proceed to manuscript-ready synthesis after checking:

1. Out-of-country cross-validation performance is better than the mean baseline.
2. AI's SHAP importance rank is not trivially low.
3. The AI dependence curve is stable enough to interpret.
4. The SHAP pattern is compared explicitly with the fixed-effects results.
5. Any apparent nonlinear shape is described as predictive rather than causal.
