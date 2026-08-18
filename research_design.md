# Research Design — AI and the Green Transition in Europe

## Proposed title
**Artificial Intelligence and the Green Transition: Does AI Adoption Reduce Carbon Intensity or Create a Digital Rebound Effect in Europe?**

## Unit of analysis
Country × NACE Rev. 2 economic activity × year.

## Core period
Primary estimation window: **2021–2024**, restricted to years and NACE groups jointly available in the AI-adoption and GHG-intensity datasets.

2025 AI data are downloaded when available, but should not be merged into the main annual carbon-intensity panel unless the corresponding annual GHG-intensity observation is available.

## Dependent variable
**GHG intensity of gross value added**

Eurostat dataset: `env_ac_aeint_r2`

Preferred filters:
- `airpol = GHG`
- `na_item = B1G` (gross value added)
- `unit = G_EUR_CLV20` (grams per euro, chain-linked volumes, 2020)

Alternative robustness:
- `unit = G_EUR_CP` (grams per euro, current prices)

## Main explanatory variable
**AI adoption (% of enterprises)**

Eurostat dataset: `isoc_eb_ain2`

Preferred indicator:
- `indic_is = E_AI_TANY`
- `unit = PC_ENT`

Definition: enterprises using at least one listed AI technology.

## Controls / robustness variables
1. Digital intensity by NACE: `isoc_e_diin2`
2. Cloud computing by NACE: `isoc_cicce_usen2`
3. Gross value added by detailed industry: `nama_10_a64` (if needed)
4. Additional country-level controls may later be added from Eurostat/OECD/World Bank.

## Hypotheses
**H1. AI adoption and carbon intensity**
Higher AI adoption is associated with lower greenhouse-gas intensity of economic activity.

Expected sign:
β1 < 0

**H2. Digital rebound effect**
The relationship between AI adoption and GHG intensity is nonlinear.

Model:
GHGIntensity = β0 + β1 AI + β2 AI² + controls + fixed effects + ε

Digital rebound pattern:
β1 < 0 and β2 > 0

**H3. Digital maturity moderation**
The carbon-reducing association of AI is stronger in sectors/countries with higher digital maturity.

Interaction:
AI × DigitalIntensity

**H4. Cloud dependence moderation**
The environmental effect of AI varies with cloud-computing intensity because cloud services can both improve resource efficiency and raise computing demand.

## Baseline econometric models

### Model 1 — Two-way fixed effects
GHGIntensity_ijt =
β1 AI_ijt + β2 X_ijt + α_ij + γ_t + ε_ijt

where:
- i = country
- j = NACE sector
- t = year
- α_ij = country-sector fixed effect
- γ_t = year fixed effect

### Model 2 — Nonlinear rebound model
GHGIntensity_ijt =
β1 AI_ijt + β2 AI²_ijt + β3 X_ijt + α_ij + γ_t + ε_ijt

### Model 3 — Dynamic robustness
GHGIntensity_ijt =
ρ GHGIntensity_ij,t-1 + β1 AI_ijt + β2 X_ijt + α_ij + γ_t + ε_ijt

Potential estimator: Arellano–Bond / System GMM, subject to final T and N structure.

## Machine-learning layer
- XGBoost regression
- SHAP global feature importance
- SHAP dependence plot for AI adoption
- Partial dependence / ALE as robustness

The ML layer is explanatory/predictive support, not a substitute for causal econometric identification.

## Main methodological cautions
1. ICT enterprise statistics cover enterprises with at least 10 employees/self-employed persons and selected non-financial NACE activities.
2. The Digital Intensity Index changes composition across survey years; use cautiously in longitudinal models.
3. NACE aggregation must be harmonised before merging.
4. 2025 AI observations should not be paired with 2024 emissions.
5. Causal language should be avoided unless a stronger identification strategy is added later.
