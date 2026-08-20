# Journal of Cleaner Production — Submission Checklist

## Manuscript

- [ ] Run `python src\10_synthesize_results.py` after any analysis change.
- [ ] Run `python src\11_build_full_manuscript.py` to rebuild `manuscript_full.md`.
- [ ] Confirm title and abstract against `submission/jcp/title_abstract.md`.
- [ ] Confirm all numerical claims match generated CSV tables.
- [ ] Confirm H1 and H2 are described as *not supported*, not as proof of no effect.
- [ ] Check all in-text citations against the bibliography.
- [ ] Add table and figure callouts in the manuscript at appropriate locations.
- [ ] Replace provisional JEL codes if the target journal does not use them.

## Author information and declarations

- [ ] Final author names and order.
- [ ] Institutional affiliations and postal addresses.
- [ ] Corresponding-author email and ORCID.
- [ ] CRediT author-contribution statement.
- [ ] Funding statement.
- [ ] Conflict-of-interest declaration.
- [ ] Data/code availability statement.
- [ ] Generative-AI / AI-assisted-writing disclosure, if required by current journal policy.

## Separate submission files

- [ ] Main manuscript.
- [ ] Title page if the submission system requests it separately.
- [ ] Highlights (`submission/jcp/highlights.md`).
- [ ] Cover letter (`submission/jcp/cover_letter.md`).
- [ ] Graphical abstract prepared from `submission/jcp/graphical_abstract_plan.md` if requested/recommended by the current portal.
- [ ] Main figures at publication quality.
- [ ] Supplementary tables or code/data statement if needed.

## Statistical and methodological checks

- [ ] Report that comparable AI years are 2021, 2023, and 2024.
- [ ] Report the exact NACE matching rule and broad-sector restriction.
- [ ] Explain why log(1 + GHG intensity) is the primary outcome.
- [ ] State that controlled-model standard errors are clustered by country.
- [ ] State that France/Sweden exclusion is a robustness check for the 2023 series break.
- [ ] State that XGBoost uses GroupKFold by country and that country is not a predictor.
- [ ] State that SHAP values are predictive/explanatory, not causal.
- [ ] Explain why dynamic GMM is not used with only three comparable AI waves.

## Figures recommended for the paper

1. Research design / data-linkage flowchart.
2. AI adoption versus GHG intensity descriptive plot by year.
3. XGBoost observed-versus-predicted plot under grouped cross-validation.
4. Grouped SHAP importance figure.
5. AI SHAP dependence figure as main-text or supplementary evidence.

## Final journal-fit check

Journal of Cleaner Production describes itself as an international, transdisciplinary journal covering cleaner production, environmental and sustainability research and practice, including energy/resource efficiency and sustainability assessment. Before submission, recheck its current Guide for Authors, article type, formatting rules, file requirements, and disclosure policies in the live submission system.