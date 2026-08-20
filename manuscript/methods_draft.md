# Methods

## Research design

This study examines whether enterprise adoption of artificial intelligence (AI) is associated with greenhouse-gas (GHG) emission intensity across European economies. The empirical unit is a **country × NACE Rev. 2 economic activity × year** observation. The analysis links Eurostat enterprise ICT statistics on AI use with Eurostat air-emission intensity accounts using exact country, year, and NACE identifiers.

The comparable AI waves available in the core panel are **2021, 2023, and 2024**. Because the short time dimension does not support a strong dynamic causal design, the econometric results are interpreted as **associational fixed-effects evidence** rather than causal estimates.

## Data sources

### AI adoption

Enterprise AI adoption is obtained from Eurostat enterprise ICT statistics (`isoc_eb_ain2`). The principal indicator is the percentage of enterprises using at least one AI technology (`E_AI_TANY`, unit `PC_ENT`). The enterprise ICT survey primarily covers enterprises with at least 10 persons employed in selected non-financial NACE Rev. 2 activities.

The AI measure captures **observed enterprise adoption** rather than AI patents, scientific publications, investment flows, compute capacity, or a latent AI-development index. This distinction is central to interpretation because much of the existing cross-country literature relies on broader proxies for AI technological development.

### GHG intensity

The dependent variable is GHG emission intensity of gross value added from Eurostat air-emission intensity accounts (`env_ac_aeint_r2`). The preferred specification uses `airpol=GHG`, `na_item=B1G`, and the chain-linked-volume intensity unit `G_EUR_CLV20` where available.

Because the raw intensity distribution is highly right-skewed, the primary outcome is transformed as

\[
\log(1+GHGIntensity_{ijt}).
\]

Level models using winsorised GHG intensity are retained as robustness checks.

### Country-year controls

The controlled specifications merge four Eurostat macroeconomic and energy variables at country-year level:

1. **Renewable-energy share** (`nrg_ind_ren`, `nrg_bal=REN`, `unit=PC`), measured as renewables in gross final energy consumption.
2. **Real GDP per capita** (`nama_10_pc`, `na_item=B1GQ`, `unit=CLV10_EUR_HAB`), entered in logarithmic form.
3. **R&D intensity** (`rd_e_gerdtot`, `sectperf=TOTAL`, `unit=PC_GDP`).
4. **Non-household electricity price** (`nrg_pc_205`, consumption band `MWH500-1999`), treated as an optional robustness control when coverage is sufficient.

## NACE harmonisation

A key design rule is that GHG-intensity ratios are **not averaged mechanically across incompatible NACE aggregates**. The core merge therefore begins with exact NACE matches. The main regressions use broad, non-overlapping one-letter NACE sections to avoid simultaneously including an aggregate sector and its detailed subsectors.

The principal broad-NACE sample contains 11 economic sections. This restriction preserves a coherent sectoral comparison and reduces double representation of the same economic activity.

## Fixed-effects models

The baseline model is

\[
Y_{ijt}=\beta_1 AI_{ijt}+\alpha_{ij}+\gamma_t+\varepsilon_{ijt},
\]

where \(Y_{ijt}=\log(1+GHGIntensity_{ijt})\), \(\alpha_{ij}\) denotes country-sector fixed effects, and \(\gamma_t\) denotes year fixed effects.

The nonlinear specification tests the pre-specified digital-rebound hypothesis:

\[
Y_{ijt}=\beta_1 AI_{ijt}+\beta_2 AI_{ijt}^2+\mathbf{X}_{it}'\delta+\alpha_{ij}+\gamma_t+\varepsilon_{ijt}.
\]

A U-shaped rebound pattern would require a substantively credible combination of \(\beta_1<0\) and \(\beta_2>0\), with a turning point inside the observed AI-adoption range and robustness to alternative specifications.

Controlled models add renewable-energy share, log real GDP per capita, and R&D intensity. Because these controls vary at country-year level and are repeated across sectors within countries, the primary controlled-model inference uses **country-clustered standard errors**. Entity-clustered estimates are retained as sensitivity checks.

Eurostat documents a 2023 break in the enterprise ICT time series for France and Sweden. Accordingly, the analysis includes robustness models excluding these two countries.

## Machine-learning model

The econometric analysis is complemented by an XGBoost regression model predicting the log-transformed GHG-intensity outcome. The core predictors are:

- enterprise AI adoption,
- renewable-energy share,
- log real GDP per capita,
- R&D intensity,
- NACE section,
- year.

Country is deliberately **not** included as a predictor. Predictive performance is evaluated using **GroupKFold cross-validation grouped by country**, so observations from the same country do not appear simultaneously in training and validation folds. This provides a more demanding out-of-country generalisation test than a random train/test split.

The model is evaluated using RMSE, MAE, and \(R^2\) on the log outcome, together with improvement over a fold-specific mean baseline.

## SHAP interpretation

SHAP values are used to describe how the fitted XGBoost model allocates predictive contributions across features. Because NACE and year are one-hot encoded, transformed dummy-level SHAP values are also aggregated back to their **original conceptual feature groups** at the observation level before computing global importance.

Grouped SHAP importance therefore compares six conceptual predictors: NACE sector, real GDP per capita, renewable-energy share, R&D intensity, AI adoption, and year.

SHAP results are interpreted as **predictive/explanatory**, not causal. They are used to assess whether nonlinear predictive evidence is consistent with or contrary to the fixed-effects findings.

## Reproducibility

All data-download, harmonisation, fixed-effects, XGBoost, SHAP, and synthesis scripts are version-controlled in the project repository. Intermediate and manuscript-ready tables are generated programmatically to reduce transcription error and keep the written results synchronised with rerun analyses.
