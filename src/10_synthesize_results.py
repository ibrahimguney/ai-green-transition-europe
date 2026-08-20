"""
Step 06 - Synthesize econometric and machine-learning results into
manuscript-ready tables and a reproducible Results draft.

Inputs
------
outputs/tables/controlled_fe_coefficients.csv
outputs/tables/controlled_fe_model_diagnostics.csv
outputs/tables/controlled_quadratic_turning_points.csv
outputs/tables/xgb_cv_metrics.csv
outputs/tables/xgb_cv_predictions.csv
outputs/tables/shap_grouped_importance.csv
outputs/tables/shap_ai_binned.csv (optional)

Outputs
-------
outputs/tables/manuscript_table_fe.csv
outputs/tables/manuscript_table_ml.csv
outputs/tables/manuscript_table_grouped_shap.csv
outputs/tables/evidence_synthesis.csv
manuscript/results_draft.md

Run
---
    python src/10_synthesize_results.py
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
MANUSCRIPT = ROOT / "manuscript"
MANUSCRIPT.mkdir(parents=True, exist_ok=True)

FE_COEF = TABLES / "controlled_fe_coefficients.csv"
FE_DIAG = TABLES / "controlled_fe_model_diagnostics.csv"
FE_TP = TABLES / "controlled_quadratic_turning_points.csv"
XGB_METRICS = TABLES / "xgb_cv_metrics.csv"
XGB_PRED = TABLES / "xgb_cv_predictions.csv"
GROUPED_SHAP = TABLES / "shap_grouped_importance.csv"
AI_BINNED = TABLES / "shap_ai_binned.csv"

MODEL_ORDER = [
    "M0_AI_only",
    "M1_AI_REN_GDP",
    "M2_AI_REN_GDP_RD",
    "M3_AI_REN_GDP_EPRICE",
    "M4_quadratic_REN_GDP_RD",
    "M5_quadratic_exFRSE",
    "M6_linear_exFRSE",
]

MODEL_LABELS = {
    "M0_AI_only": "AI only",
    "M1_AI_REN_GDP": "AI + renewables + GDP",
    "M2_AI_REN_GDP_RD": "AI + renewables + GDP + R&D",
    "M3_AI_REN_GDP_EPRICE": "AI + renewables + GDP + electricity price",
    "M4_quadratic_REN_GDP_RD": "Quadratic AI + renewables + GDP + R&D",
    "M5_quadratic_exFRSE": "Quadratic model, France/Sweden excluded",
    "M6_linear_exFRSE": "Linear model, France/Sweden excluded",
}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required Step 06 input not found: {path}")
    return path


def fmt_num(x, digits=4):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def fmt_p(x):
    if pd.isna(x):
        return "NA"
    x = float(x)
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def build_fe_table() -> pd.DataFrame:
    coef = pd.read_csv(require(FE_COEF))
    diag = pd.read_csv(require(FE_DIAG))

    rows = []
    for model in MODEL_ORDER:
        sub = coef[coef["model"] == model]
        if sub.empty:
            continue

        d = diag[diag["model"] == model]
        drow = d.iloc[0] if not d.empty else pd.Series(dtype=object)

        ai = sub[sub["term"] == "ai_c"]
        ai2 = sub[sub["term"] == "ai_c_sq"]
        ren = sub[sub["term"] == "renewable_share"]
        gdp = sub[sub["term"] == "log_real_gdp_pc"]
        rd = sub[sub["term"] == "rd_intensity_pct_gdp"]
        eprice = sub[sub["term"] == "electricity_price_eur_kwh"]

        def get(frame, col):
            return float(frame.iloc[0][col]) if not frame.empty else np.nan

        ai_beta = get(ai, "coef")
        ai_p = get(ai, "p_value")
        approx_10pp = (math.exp(10 * ai_beta) - 1) * 100 if np.isfinite(ai_beta) and ai2.empty else np.nan

        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "n_obs": int(drow.get("rows", sub["n_obs"].max())) if len(sub) else np.nan,
                "countries": int(drow.get("countries", sub["countries"].max())) if len(sub) else np.nan,
                "entities": int(drow.get("entities", sub["entities"].max())) if len(sub) else np.nan,
                "ai_beta": ai_beta,
                "ai_p": ai_p,
                "ai2_beta": get(ai2, "coef"),
                "ai2_p": get(ai2, "p_value"),
                "renewable_beta": get(ren, "coef"),
                "renewable_p": get(ren, "p_value"),
                "log_gdp_beta": get(gdp, "coef"),
                "log_gdp_p": get(gdp, "p_value"),
                "rd_beta": get(rd, "coef"),
                "rd_p": get(rd, "p_value"),
                "electricity_price_beta": get(eprice, "coef"),
                "electricity_price_p": get(eprice, "p_value"),
                "approx_pct_change_for_10pp_ai_linear": approx_10pp,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "manuscript_table_fe.csv", index=False)
    return out


def build_ml_table() -> pd.DataFrame:
    metrics = pd.read_csv(require(XGB_METRICS))
    pred = pd.read_csv(require(XGB_PRED))

    y = pred["log_ghg"].to_numpy(dtype=float)
    yhat = pred["pred_log_ghg"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mae = float(np.mean(np.abs(y - yhat)))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    out = pd.DataFrame(
        [
            {
                "validation": "GroupKFold by country",
                "folds": int(metrics["fold"].nunique()),
                "rows": int(len(pred)),
                "countries": int(pred["geo"].nunique()),
                "pooled_rmse_log": rmse,
                "pooled_mae_log": mae,
                "pooled_r2_log": r2,
                "mean_fold_rmse_improvement_pct_vs_mean_baseline": float(
                    metrics["rmse_improvement_pct_vs_mean"].mean()
                ),
            }
        ]
    )
    out.to_csv(TABLES / "manuscript_table_ml.csv", index=False)
    return out


def build_shap_table() -> pd.DataFrame:
    shap = pd.read_csv(require(GROUPED_SHAP)).copy()
    shap = shap.sort_values("rank").reset_index(drop=True)
    shap.to_csv(TABLES / "manuscript_table_grouped_shap.csv", index=False)
    return shap


def evidence_synthesis(fe: pd.DataFrame, shap: pd.DataFrame) -> pd.DataFrame:
    m2 = fe[fe["model"] == "M2_AI_REN_GDP_RD"]
    m4 = fe[fe["model"] == "M4_quadratic_REN_GDP_RD"]
    m5 = fe[fe["model"] == "M5_quadratic_exFRSE"]
    m6 = fe[fe["model"] == "M6_linear_exFRSE"]

    def val(frame, col):
        return float(frame.iloc[0][col]) if not frame.empty and pd.notna(frame.iloc[0][col]) else np.nan

    m2_b, m2_p = val(m2, "ai_beta"), val(m2, "ai_p")
    m6_b, m6_p = val(m6, "ai_beta"), val(m6, "ai_p")
    m4_b2, m4_p2 = val(m4, "ai2_beta"), val(m4, "ai2_p")
    m5_b2, m5_p2 = val(m5, "ai2_beta"), val(m5, "ai2_p")

    ai_shap = shap[shap["feature_group"] == "AI adoption"]
    ai_rank = int(ai_shap.iloc[0]["rank"]) if not ai_shap.empty else np.nan
    ai_importance = float(ai_shap.iloc[0]["mean_abs_group_shap"]) if not ai_shap.empty else np.nan

    linear_supported = bool(
        np.isfinite(m2_b) and np.isfinite(m2_p) and m2_b < 0 and m2_p < 0.05
        and np.isfinite(m6_b) and np.isfinite(m6_p) and m6_b < 0 and m6_p < 0.05
    )
    rebound_supported = bool(
        np.isfinite(m4_b2) and np.isfinite(m4_p2) and m4_b2 > 0 and m4_p2 < 0.05
        and np.isfinite(m5_b2) and np.isfinite(m5_p2) and m5_b2 > 0 and m5_p2 < 0.05
    )

    rows = [
        {
            "question": "H1: AI adoption is associated with lower GHG intensity",
            "econometric_evidence": (
                f"M2 beta={m2_b:.6f}, p={m2_p:.4f}; "
                f"FR/SE-excluded M6 beta={m6_b:.6f}, p={m6_p:.4f}"
            ),
            "ml_evidence": f"AI grouped SHAP rank={ai_rank}; mean|SHAP|={ai_importance:.6f}",
            "conclusion": "Supported" if linear_supported else "Not supported",
        },
        {
            "question": "H2: AI exhibits a U-shaped digital rebound effect",
            "econometric_evidence": (
                f"M4 AI^2 beta={m4_b2:.8f}, p={m4_p2:.4f}; "
                f"FR/SE-excluded M5 AI^2 beta={m5_b2:.8f}, p={m5_p2:.4f}"
            ),
            "ml_evidence": "SHAP is used only as nonlinear predictive evidence, not as a causal rebound test",
            "conclusion": "Supported" if rebound_supported else "Not supported",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "evidence_synthesis.csv", index=False)
    return out


def build_results_markdown(fe: pd.DataFrame, ml: pd.DataFrame, shap: pd.DataFrame, evidence: pd.DataFrame):
    m0 = fe[fe["model"] == "M0_AI_only"].iloc[0]
    m1 = fe[fe["model"] == "M1_AI_REN_GDP"].iloc[0]
    m2 = fe[fe["model"] == "M2_AI_REN_GDP_RD"].iloc[0]
    m4 = fe[fe["model"] == "M4_quadratic_REN_GDP_RD"].iloc[0]
    m5 = fe[fe["model"] == "M5_quadratic_exFRSE"].iloc[0]
    m6 = fe[fe["model"] == "M6_linear_exFRSE"].iloc[0]
    mlr = ml.iloc[0]

    def srow(name):
        r = shap[shap["feature_group"] == name]
        return r.iloc[0] if not r.empty else None

    nace = srow("NACE sector")
    gdp = srow("Real GDP per capita")
    ren = srow("Renewable energy share")
    rd = srow("R&D intensity")
    ai = srow("AI adoption")
    year = srow("Year")

    text = f"""# Results

## Fixed-effects estimates

The baseline two-way fixed-effects model included {int(m0['n_obs'])} observations from {int(m0['countries'])} countries and {int(m0['entities'])} country-sector entities. The coefficient on enterprise AI adoption was negative but statistically indistinguishable from zero (beta = {m0['ai_beta']:.6f}, p = {fmt_p(m0['ai_p'])}). Adding renewable-energy share and real GDP per capita modestly increased the magnitude of the negative AI coefficient (beta = {m1['ai_beta']:.6f}, p = {fmt_p(m1['ai_p'])}), but the estimate remained statistically non-significant. The preferred controlled linear specification, which additionally included R&D intensity, produced a similar result (beta = {m2['ai_beta']:.6f}, p = {fmt_p(m2['ai_p'])}). Interpreted semi-elastically, a 10-percentage-point increase in AI adoption corresponded to an estimated {m2['approx_pct_change_for_10pp_ai_linear']:.3f}% change in GHG intensity; however, the associated uncertainty precludes interpreting this estimate as evidence of a systematic relationship.

The nonlinear specification also failed to support the hypothesised digital rebound mechanism. In the controlled quadratic model, the squared AI term was negative rather than positive (beta = {m4['ai2_beta']:.8f}, p = {fmt_p(m4['ai2_p'])}), which is inconsistent with the pre-specified U-shaped rebound hypothesis. Excluding France and Sweden to address the documented 2023 ICT-series break did not change this conclusion: the squared term remained negative and non-significant (beta = {m5['ai2_beta']:.8f}, p = {fmt_p(m5['ai2_p'])}). The corresponding France/Sweden-excluded linear model also yielded a negative but non-significant AI coefficient (beta = {m6['ai_beta']:.6f}, p = {fmt_p(m6['ai_p'])}). Overall, neither H1 nor H2 is supported by the fixed-effects evidence.

## Out-of-country predictive performance

The XGBoost model was evaluated with five-fold GroupKFold cross-validation in which countries, rather than individual country-sector-year rows, defined the validation groups. Across {int(mlr['rows'])} out-of-country predictions from {int(mlr['countries'])} countries, the pooled RMSE on the log-GHG outcome was {mlr['pooled_rmse_log']:.4f}, the pooled MAE was {mlr['pooled_mae_log']:.4f}, and the pooled R-squared was {mlr['pooled_r2_log']:.4f}. Relative to a fold-specific mean-outcome baseline, the model reduced RMSE by an average of {mlr['mean_fold_rmse_improvement_pct_vs_mean_baseline']:.2f}%. These results indicate that the observed covariates contain substantial information about cross-country and cross-sector differences in GHG intensity, even though the fixed-effects models do not identify a statistically credible marginal association for AI adoption.

## Grouped SHAP importance

Grouped SHAP analysis shows that the predictive model relies primarily on structural sectoral and macroeconomic information. NACE sector ranked first (mean absolute grouped SHAP = {nace['mean_abs_group_shap']:.6f}), followed by real GDP per capita ({gdp['mean_abs_group_shap']:.6f}) and renewable-energy share ({ren['mean_abs_group_shap']:.6f}). R&D intensity ranked fourth ({rd['mean_abs_group_shap']:.6f}), while AI adoption ranked fifth of six original feature groups ({ai['mean_abs_group_shap']:.6f}). Year effects ranked last ({year['mean_abs_group_shap']:.6f}). The closeness of the grouped SHAP values for R&D intensity and AI adoption suggests that AI is not irrelevant to prediction, but its contribution is considerably smaller than that of sectoral composition, income level, and renewable-energy penetration.

## Synthesis

The econometric and machine-learning results point in the same substantive direction. The fixed-effects models provide no statistically significant evidence that greater AI adoption lowers GHG intensity, and the hypothesised U-shaped digital rebound effect is not observed. At the same time, XGBoost achieves strong out-of-country predictive performance, with grouped SHAP attributing most predictive importance to NACE sector, real GDP per capita, and renewable-energy share rather than to AI adoption. The combined evidence therefore suggests that, over the currently comparable 2021, 2023, and 2024 AI-survey years, structural sectoral and green-transition conditions dominate AI adoption as correlates and predictors of GHG intensity.

These findings should be interpreted as associational rather than causal. The short three-wave AI panel limits temporal identification, and SHAP values describe predictive model behaviour rather than causal effects. The null AI findings are therefore best read as an absence of robust evidence for an independent green effect of AI in the available data, not as proof that AI has no environmental consequences.
"""

    (MANUSCRIPT / "results_draft.md").write_text(text, encoding="utf-8")


def main():
    fe = build_fe_table()
    ml = build_ml_table()
    shap = build_shap_table()
    evidence = evidence_synthesis(fe, shap)
    build_results_markdown(fe, ml, shap, evidence)

    print("STEP 06 - RESULTS SYNTHESIS")
    print("=" * 42)
    print("Generated manuscript-ready outputs:")
    for p in [
        TABLES / "manuscript_table_fe.csv",
        TABLES / "manuscript_table_ml.csv",
        TABLES / "manuscript_table_grouped_shap.csv",
        TABLES / "evidence_synthesis.csv",
        MANUSCRIPT / "results_draft.md",
    ]:
        print(f"- {p}")

    print("\nEvidence conclusions:")
    print(evidence[["question", "conclusion"]].to_string(index=False))
    print("\nNext: review results_draft.md before manuscript integration.")


if __name__ == "__main__":
    main()
