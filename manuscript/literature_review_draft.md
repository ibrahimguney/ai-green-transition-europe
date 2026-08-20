# Literature Review and Hypotheses

## AI as an efficiency-enhancing technology

The environmental case for AI rests primarily on efficiency. AI systems can improve prediction, scheduling, maintenance, process control, logistics, and the allocation of energy and materials. In energy and industrial systems, better information and optimisation can reduce waste, lower downtime, improve load management, and support the integration of renewable resources. From this perspective, AI adoption may reduce the GHG intensity of value creation even if total output continues to expand.

Recent international evidence provides support for this mechanism. Wang, Li and Li [@WangLiLi2024] find that AI development is associated with reductions in carbon emissions and ecological footprints and with progress in energy transition across a 67-country panel. Their threshold results also imply that the environmental return to AI is conditional on structural characteristics such as industrial composition, trade openness, and the energy transition. More broadly, the emerging literature increasingly treats AI as a general-purpose technology whose environmental effect depends on complementary institutions, capital, skills, and energy systems rather than on digitalisation alone.

If efficiency and optimisation dominate the additional resource requirements associated with AI, higher enterprise adoption should be associated with lower emission intensity. This motivates the first hypothesis:

**H1. Higher enterprise AI adoption is associated with lower GHG emission intensity.**

## AI, electricity demand, and rebound mechanisms

The opposite mechanism arises from the physical infrastructure required to train and deploy AI. Data centres, specialised processors, networking, cooling, and storage require electricity and embodied resources. The IEA [@IEA2025EnergyAI] describes this relationship as a two-sided energy–AI nexus: AI may improve energy-system efficiency while simultaneously increasing electricity demand through the expansion of computing infrastructure. The IEA's 2026 update [@IEA2026KeyQuestions] reinforces this tension by showing that energy use per AI task is falling rapidly while the scale and energy intensity of AI applications are rising.

This tension is closely related to the rebound effect. Efficiency improvements can reduce the effective cost of an activity, stimulating additional use and partially or fully offsetting the initial savings. Applied to AI, a digital rebound effect can arise when more efficient computing, cheaper AI services, or productivity improvements increase the scale of digital activity sufficiently to raise total electricity demand or associated emissions.

Recent empirical studies report different nonlinear forms. Zhang, Wang and Li [@ZhangWangLi2025] find an inverted-U relationship between AI development and carbon emissions: AI raises emissions at earlier stages but reduces them after a development threshold, with renewable energy bringing forward the turning point. Alnafrah [@Alnafrah2025] instead identifies a digital rebound regime in which high AI intensity is associated with worsening environmental outcomes. These differences suggest that nonlinear effects may depend on the AI measure, energy mix, regulatory conditions, and technological maturity.

The present study pre-specifies a U-shaped rebound test that is particularly relevant to enterprise adoption. At low or moderate levels of adoption, optimisation gains may dominate. At higher adoption levels, additional computing and induced activity may offset those gains. In a quadratic specification, this pattern requires a negative linear term, a positive squared term, and a turning point within the observed adoption range.

**H2. The relationship between enterprise AI adoption and GHG intensity is U-shaped, consistent with a digital rebound effect at higher adoption levels.**

## The moderating role of the energy and production system

AI does not operate independently of the energy system. A given increment in electricity demand has different carbon consequences depending on the generation mix, while a given AI application has different environmental effects depending on the production process into which it is introduced. Renewable-energy availability can therefore reduce the emissions associated with digital expansion and may also strengthen the environmental return to AI-enabled electrification and optimisation.

Similarly, sector composition is likely to be central. Heavy industry, utilities, transport, information services, real estate, and professional services differ substantially in baseline emission intensity, capital intensity, and opportunities for AI-enabled efficiency. An economy-wide AI coefficient may therefore conceal large differences across productive contexts. The current design addresses this heterogeneity through country-sector fixed effects and by retaining NACE sector in the machine-learning model.

Economic development and innovation capacity also matter. Higher-income economies may have cleaner capital stocks, stronger environmental regulation, more efficient infrastructure, and greater capacity to deploy AI productively. At the same time, they may operate more digital infrastructure and consume more services. R&D intensity can proxy absorptive capacity and the ability to combine AI with complementary organisational and technical innovation.

For these reasons, renewable-energy share, real GDP per capita, and R&D intensity are treated as important controls rather than as auxiliary variables. The empirical strategy asks whether AI retains an independent association with GHG intensity after these structural conditions are taken into account.

## Measurement: AI adoption versus AI technological intensity

A central issue in interpreting the literature is that 'AI' is not a single empirical construct. Patent counts and patent quality measure technological development; scientific publications measure knowledge production; investment measures financial commitment; compute capacity measures infrastructure; and enterprise surveys measure adoption. These indicators can move at different speeds and may capture different environmental mechanisms.

The Eurostat indicator used here measures the percentage of enterprises using at least one AI technology. It therefore captures diffusion rather than intensity. A firm using AI for document processing and a firm operating energy-intensive generative models both count as adopters, even though their electricity requirements and environmental consequences may differ substantially. This measurement distinction may explain why short-run enterprise-adoption results differ from long-run studies based on AI patents or composite technological indices.

The study therefore treats its hypotheses as tests of **enterprise AI diffusion**, not of the full environmental footprint of the AI technology system. This narrower interpretation is also the basis for the machine-learning analysis: if AI adoption is already an important determinant of GHG intensity, a flexible nonlinear model should assign it meaningful predictive importance after sector and macroeconomic conditions are included.
