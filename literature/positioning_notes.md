# Literature Positioning Notes — AI and Environmental Performance

## Why the literature is contested

Recent empirical work does not point to a single universal AI–environment relationship. Studies report emission-reducing effects, inverted-U relationships, threshold effects, and digital rebound patterns. These differences appear to depend on how AI is measured, the time horizon, country coverage, energy mix, industrial structure, and the environmental outcome used.

## Key recent studies

### Wang, Li & Li (2024)
**Ecological footprints, carbon emissions, and energy transitions: the impact of artificial intelligence (AI).** Humanities and Social Sciences Communications, 11, 1043. DOI: 10.1057/s41599-024-03520-5.

- Panel: 67 countries, 1993–2019.
- Methods: System GMM and dynamic panel threshold models.
- Main result: AI development is associated with lower ecological footprint and carbon emissions and stronger energy transition, with heterogeneous/threshold effects.
- Relevance to our paper: long-run technological-development measure and dynamic design; contrasts with our short enterprise-adoption panel.

### Zhang, Wang & Li (2025)
**How does clean energy reshape the relationship between artificial intelligence and carbon emissions? Evidence from renewable and nuclear energy.** Energy Economics, 149, 108785. DOI: 10.1016/j.eneco.2025.108785.

- Panel: 62 countries, 1995–2023.
- Main result: inverted-U relationship; AI initially raises emissions and later reduces them.
- Mechanism: energy rebound in early stages; renewable energy advances the turning point toward emission reduction.
- Relevance: demonstrates that nonlinear AI effects may require long time horizons and measures of AI technological development rather than simple adoption shares.

### Alnafrah (2025)
**The Two Tales of AI: A Global assessment of the environmental impacts of artificial intelligence from a multidimensional policy perspective.** Journal of Environmental Management, 392, 126813. DOI: 10.1016/j.jenvman.2025.126813.

- Panel: 56 countries, 2013–2023.
- Methods: MMQR, system GMM, dynamic threshold modelling.
- Main result: high AI intensity is associated with worse environmental performance; the study interprets this as a digital rebound effect.
- Relevance: offers a direct contrast to our non-significant U-shaped rebound test and highlights the importance of AI-intensity measurement.

## Energy-system context

### International Energy Agency (2025), Energy and AI

The IEA frames AI as a two-sided energy phenomenon. AI can improve forecasting, optimisation, maintenance, grid operation, and energy efficiency, but AI workloads also increase electricity demand through data-centre expansion. The IEA estimates that data centres represented about 1.5% of global electricity consumption in 2024 and projects a strong rise through 2030, with AI as a major driver.

This dual mechanism provides a theoretical explanation for weak aggregate contemporaneous coefficients: efficiency gains and additional electricity demand can coexist and partly offset one another.

### International Energy Agency (2026), Key Questions on Energy and AI

The IEA reports that data-centre electricity demand continued to grow rapidly in 2025 and that AI-focused data centres expanded even faster. At the same time, energy efficiency per AI task is improving quickly. This is a classic setting in which technical efficiency and scale expansion may move in opposite directions.

## European adoption context

Eurostat reports rapid growth in enterprise AI use. AI adoption among EU enterprises with at least 10 employees rose from 8.0% in 2023 to 13.5% in 2024, and to 20.0% in 2025. Eurostat also documents a break in the enterprise ICT time series for France and Sweden in 2023.

This fast diffusion means that the currently observed period may capture an early transition stage rather than a mature long-run equilibrium relationship between AI and sectoral environmental performance.

## Positioning statement for our manuscript

Our study differs from much of the recent literature in four ways:

1. **AI measure:** directly observed enterprise adoption rather than patents, investment, publications, or composite AI-development indices.
2. **Unit of analysis:** country × NACE sector × year rather than country-year only.
3. **Outcome:** official GHG intensity of gross value added rather than national total CO2 alone.
4. **Validation strategy:** fixed effects plus out-of-country XGBoost validation and grouped SHAP.

This allows the paper to ask a narrower but policy-relevant question:

> Has the recent diffusion of enterprise AI already translated into measurable reductions—or rebound effects—in sectoral GHG intensity across Europe?

The current answer is: **not robustly, within the available three-wave panel.**

## Recommended Discussion framing

Avoid framing the null result as evidence that AI has no environmental impact. Prefer:

> The findings suggest that enterprise AI adoption, as currently measured and over the short observable European adoption window, does not yet exhibit a robust independent association with sectoral GHG intensity. This differs from several long-run global studies based on AI-development or intensity proxies, implying that measurement, maturity, sector composition, and energy-system context are central to the AI–environment nexus.
