# Step 04 — Country-Year Controls and Controlled Fixed Effects

## Objective

Extend the AI–GHG panel with macro controls that capture the green transition, economic development, innovation capacity, and energy-cost environment.

## Eurostat controls

### 1. Renewable-energy share
- Dataset: `nrg_ind_ren`
- Filter: `nrg_bal=REN`, `unit=PC`
- Variable: `renewable_share`
- Interpretation: renewable energy as a percentage of gross final energy consumption.

### 2. Real GDP per capita
- Dataset: `nama_10_pc`
- Filter: `na_item=B1GQ`, `unit=CLV10_EUR_HAB`
- Variable: `real_gdp_pc_eur2010`
- Regression transform: `log_real_gdp_pc`

### 3. R&D intensity
- Dataset: `rd_e_gerdtot`
- Filter: `sectperf=TOTAL`, `unit=PC_GDP`
- Variable: `rd_intensity_pct_gdp`

### 4. Non-household electricity price
- Dataset: `nrg_pc_205`
- Filters:
  - `product=6000`
  - `nrg_cons=MWH500-1999` (band IC)
  - `unit=KWH`
  - `tax=I_TAX`
- Semester observations are averaged to calendar-year means.
- Variable: `electricity_price_eur_kwh`

**API note:** do not send `currency=EUR` to the Eurostat Statistics API for this query. Eurostat validates filter names against dataset dimensions; the extra `currency` selector caused an HTTP 400 response in the first Step 04 run. The energy-prices visualisation can expose a currency UI selector without that selector being a valid Statistics API filter dimension.

## Run order

```powershell
python src\05_download_controls.py
python src\06_merge_controls.py
python src\07_controlled_fixed_effects.py
```

## Model sequence

### M0 — Baseline
`log(GHG) ~ AI + entity FE + year FE`

### M1 — Green transition + development
`log(GHG) ~ AI + renewables + log(real GDP pc) + entity FE + year FE`

### M2 — Innovation control
M1 + R&D intensity.

### M3 — Energy-price robustness
M1 + non-household electricity price.

### M4 — Nonlinear AI specification
M2 + AI².

### M5 — France/Sweden robustness
M4 excluding France and Sweden because Eurostat reports a 2023 break in the enterprise ICT time series for these countries.

### M6 — Linear France/Sweden robustness
Linear version of M2 excluding France and Sweden.

## Inference

Primary standard errors are clustered by **country** because the macro controls vary at country-year level and are repeated across NACE sectors within each country-year. Entity-clustered AI coefficients are also exported as a sensitivity check.

## Interpretation limits

The core AI panel currently contains 2021, 2023 and 2024. With only three AI survey years, the controlled FE models should be interpreted as **associational**, not as a definitive causal design. Dynamic GMM is postponed until a longer comparable AI time series can be constructed.

## Main outputs

- `data/processed/controlled_panel.csv`
- `data/processed/controlled_broad_panel.csv`
- `outputs/tables/control_merge_coverage.csv`
- `outputs/tables/control_missingness_by_year.csv`
- `outputs/tables/control_country_coverage.csv`
- `outputs/tables/controlled_fe_coefficients.csv`
- `outputs/tables/controlled_fe_model_diagnostics.csv`
- `outputs/tables/controlled_quadratic_turning_points.csv`
- `outputs/tables/step04_model_summary.txt`

## Decision gate for Step 05

Before adding XGBoost/SHAP, inspect:

1. Control-variable coverage by country and year.
2. Whether the AI coefficient changes materially after controls.
3. Whether AI² remains statistically credible and has the hypothesised sign.
4. Whether the France/Sweden exclusion changes the conclusion.
5. Whether electricity-price inclusion materially reduces the sample.

Only after these checks should the machine-learning layer be interpreted alongside the econometric results.
