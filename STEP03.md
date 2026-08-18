# Step 03 — Two-Way Fixed Effects and Digital Rebound Test

## Why the main sample changes

The Step 02 merged panel contains both broad NACE sections and narrower child groups.
Examples include `C` together with `C19`, `C20`, etc., and `G` together with `G45`, `G46`, and `G47`.

Using parents and children in the same regression can partially represent the same economic activity twice.
Therefore the **primary econometric sample uses only one-letter NACE Rev. 2 sections**:

`C, D, E, F, G, H, I, J, L, M, N`

Detailed NACE groups are reserved for separate robustness analyses.

## Available time structure

The current merged panel contains:

- 2021
- 2023
- 2024

There is no common 2022 AI observation in the current core panel.

Because the effective time dimension is only three waves, Step 03 focuses on static two-way fixed effects.
Dynamic GMM is deferred until a longer comparable series is available.

## Primary outcome

GHG intensity is strongly right-skewed, so the primary outcome is:

`log_ghg = log(1 + ghg_intensity)`

A winsorised level outcome is retained as robustness.

## Models

### M1 — Linear two-way FE

`log_ghg_it = beta1 * AI_it + entity_FE + year_FE + error_it`

### M2 — Quadratic two-way FE

`log_ghg_it = beta1 * AI_it + beta2 * AI_it^2 + entity_FE + year_FE + error_it`

AI is mean-centered before squaring.

### Fixed effects

Entity is:

`country × NACE section`

Year fixed effects absorb common macro shocks.

### Inference

Primary standard errors are clustered at the country-sector entity level.

A country-clustered sensitivity specification is also produced.

## Rebound-effect decision rule

Do **not** claim a digital rebound effect merely because the squared coefficient has the expected sign.

A rebound interpretation requires all of the following:

1. the quadratic term is statistically credible;
2. the implied turning point lies within the observed AI-adoption range;
3. the result is stable in the balanced panel;
4. the result is not driven by extreme GHG-intensity observations;
5. the sign pattern survives later control-variable specifications.

## Run

```powershell
git pull origin main
pip install -r requirements.txt
python src\04_fixed_effects.py
```

## Outputs

- `outputs/tables/fixed_effects_coefficients.csv`
- `outputs/tables/quadratic_turning_point.csv`
- `outputs/tables/step03_sample_diagnostics.csv`
- `outputs/tables/step03_model_summary.txt`

## Next stage

Step 04 will add economically meaningful controls and robustness checks, prioritising variables that can be harmonised with the same country-sector-year structure.
