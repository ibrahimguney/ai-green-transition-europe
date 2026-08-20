# Discussion

## Main finding

The empirical evidence does not support a robust independent association between enterprise AI adoption and lower GHG intensity in the available European country-sector panel. Across baseline and controlled fixed-effects models, the AI coefficient is small and statistically imprecise. Adding renewable-energy share, real GDP per capita, and R&D intensity does not materially change this conclusion. The pre-specified U-shaped digital-rebound hypothesis is likewise unsupported: the estimated quadratic term is not statistically credible and does not exhibit the hypothesised sign pattern.

The machine-learning evidence points in the same substantive direction. XGBoost predicts out-of-country variation in log GHG intensity strongly, but grouped SHAP analysis places AI adoption below NACE sector, real GDP per capita, renewable-energy share, and R&D intensity in global predictive importance. Thus, the absence of a robust AI coefficient in the fixed-effects models is not simply a consequence of imposing linearity; a flexible nonlinear model also assigns greater explanatory weight to structural and macroeconomic features.

## Why the result differs from parts of the recent literature

Recent international studies have reported stronger and often nonlinear environmental effects of AI. Wang, Li and Li (2024), using a long multi-country panel and system-GMM/threshold models, report that AI development can reduce carbon emissions and ecological footprints and that the magnitude of these effects varies with industrial structure, trade openness, AI development, and the energy transition. Zhang, Wang and Li (2025) identify an inverted-U relationship in which AI initially raises emissions but later reduces them, with renewable and nuclear energy influencing the transition point. Alnafrah (2025), by contrast, reports a digital rebound pattern in which high AI intensity worsens environmental performance in a global panel.

The divergence from those findings need not be contradictory. The present study measures **enterprise AI adoption directly**, whereas several long-run studies operationalise AI through patent-based, technological-intensity, or composite development measures. Adoption and technological capacity are not equivalent. A firm can report using AI without the intensity, purpose, compute demand, or operational integration being large enough to generate measurable sector-level changes in emissions. Conversely, a patent-based AI measure may capture underlying technological capability long before widespread enterprise diffusion.

The time horizon also differs sharply. The current comparable European adoption panel contains only three survey waves (2021, 2023, and 2024), whereas recent global studies often span decades. If environmental effects emerge only after organisational learning, capital replacement, process redesign, or energy-system adaptation, a short adoption panel may reasonably produce weak contemporaneous associations even when long-run effects exist.

## Sector structure appears more important than AI adoption

The grouped SHAP results indicate that NACE sector is by far the strongest predictive feature group. This is substantively plausible because GHG intensity differs fundamentally between energy-intensive industries and service-oriented activities. Real GDP per capita and renewable-energy share also rank above AI adoption.

This suggests that the environmental consequences of AI should not be studied independently of the production structure in which AI is embedded. The same level of AI adoption may have very different environmental implications in manufacturing, transport, information services, or professional services. AI may improve process efficiency in one setting while raising electricity or compute demand in another.

Accordingly, a useful interpretation of the current findings is not that AI is environmentally irrelevant, but that **sectoral production structure and energy context dominate the cross-sector signal over the short adoption period currently observable**.

## The energy-AI duality

The International Energy Agency characterises AI as having a dual relationship with the energy system. AI applications can improve forecasting, optimisation, grid operation, maintenance, and process efficiency, but the compute infrastructure supporting AI also requires substantial electricity. The IEA estimates that data centres accounted for roughly 1.5% of global electricity consumption in 2024 and projects strong growth through 2030, with AI as a major driver.

This duality provides a useful framework for the non-result in the present panel. Efficiency gains and additional electricity demand can operate simultaneously and may partially offset one another. At aggregate sector level, the net contemporaneous effect may therefore be small even when important mechanisms exist in opposite directions.

The rapid growth of enterprise AI adoption further complicates interpretation. Eurostat reports that the share of EU enterprises using AI rose markedly between 2023 and 2025. Such rapid diffusion implies that the current period may be a transitional phase in which adoption is moving faster than measurable capital, production, and energy adjustments.

## Implications for the digital rebound hypothesis

The results do not support the pre-specified U-shaped rebound relationship. This is important because digital rebound should not be assumed merely from the existence of energy-intensive computing. A rebound claim requires evidence that efficiency gains are systematically offset or reversed as AI adoption rises.

In the current data, neither the controlled fixed-effects models nor the grouped SHAP evidence provides a robust basis for such a claim. This differs from studies using longer global panels and AI-intensity measures that identify threshold or rebound regimes. One interpretation is that rebound may depend on **AI intensity, compute demand, or maturity**, none of which is directly measured by the binary/percentage enterprise-adoption indicator used here.

Future work should therefore distinguish at least three concepts: AI adoption, AI operational intensity, and AI infrastructure intensity. Treating them as interchangeable may obscure the conditions under which rebound effects arise.

## Policy implications

The results argue against a simple policy assumption that increasing enterprise AI adoption will automatically lower carbon intensity. Digitalisation policy and decarbonisation policy should be coordinated rather than treated as inherently aligned.

First, the strong role of sector structure indicates that sector-specific AI strategies are likely to be more informative than economy-wide claims. Second, the predictive importance of renewable-energy share suggests that the carbon consequences of digital expansion depend partly on the energy system supporting it. Third, AI-related environmental policy should consider both operational efficiency gains and upstream electricity demand from digital infrastructure.

The implication is not to slow AI adoption per se, but to accompany AI diffusion with clean electricity, energy-efficiency standards, transparent measurement of compute-related energy use, and sector-specific monitoring of environmental outcomes.

## Limitations

Several limitations constrain inference. First, the panel contains only three comparable AI waves, limiting the ability to estimate long-run dynamics or causal lag structures. Second, the enterprise AI variable measures the share of firms using at least one AI technology but does not capture intensity of use, compute consumption, model size, or whether AI is deployed for energy-saving applications. Third, country-year controls are more aggregated than the sector-level dependent and AI variables. Fourth, electricity-price coverage was insufficient for the planned main robustness specification. Fifth, Eurostat documents a 2023 time-series break for France and Sweden, although excluding those countries does not materially change the conclusion.

Finally, SHAP values describe predictive contributions in a fitted model and should not be interpreted as causal effects.

## Contribution

The study contributes to the emerging AI-environment literature by using a different empirical lens from many existing cross-country studies: directly observed enterprise AI adoption linked to official country-sector GHG-intensity statistics. The combination of exact NACE harmonisation, country-sector fixed effects, country-grouped machine-learning validation, and grouped SHAP analysis provides a transparent test of whether the rapid diffusion of enterprise AI is already associated with measurable carbon-intensity changes in Europe.

The central result is deliberately modest but policy-relevant: **in the currently observable three-wave European panel, there is no robust evidence that enterprise AI adoption independently lowers GHG intensity, and no robust evidence of the hypothesised U-shaped digital rebound effect. Structural sector and energy-economic conditions appear more important in explaining observed variation.**
