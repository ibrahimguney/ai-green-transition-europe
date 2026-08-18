"""
Step 03 - Two-way fixed-effects models for AI adoption and GHG intensity.

Main sample:
- broad, non-overlapping one-letter NACE Rev. 2 sections only
- years available in the core panel
- country-sector entities with at least two observed years

Primary dependent variable:
    log_ghg = log(1 + ghg_intensity)

Models:
1) Linear AI effect
2) Quadratic AI effect (digital rebound test)
3) Balanced-panel linear robustness
4) Balanced-panel quadratic robustness
5) Level-outcome robustness with 1%/99% winsorisation

Fixed effects:
- country-sector entity fixed effects
- year fixed effects

Inference:
- cluster-robust standard errors by country-sector entity
- additional country-clustered results for sensitivity

Run:
    python src/04_fixed_effects.py
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

INPUT = PROCESSED / "core_panel.csv"

BROAD_NACE_REGEX = r"^[A-Z]$"
MIN_PERIODS = 2


def prepare_sample(df: pd.DataFrame) -> pd.DataFrame:
    required = {"panel_id", "geo", "nace_r2", "time", "ghg_intensity", "ai_adoption"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out = out[out["nace_r2"].astype(str).str.fullmatch(BROAD_NACE_REGEX)].copy()

    n_periods = out.groupby("panel_id")["time"].transform("nunique")
    out = out[n_periods >= MIN_PERIODS].copy()

    out["log_ghg"] = np.log1p(out["ghg_intensity"])

    ai_mean = out["ai_adoption"].mean()
    out["ai_c"] = out["ai_adoption"] - ai_mean
    out["ai_c_sq"] = out["ai_c"] ** 2

    lo = out["ghg_intensity"].quantile(0.01)
    hi = out["ghg_intensity"].quantile(0.99)
    out["ghg_intensity_w"] = out["ghg_intensity"].clip(lo, hi)

    out.attrs["ai_mean"] = float(ai_mean)
    out.attrs["winsor_lo"] = float(lo)
    out.attrs["winsor_hi"] = float(hi)

    return out.sort_values(["panel_id", "time"]).reset_index(drop=True)


def fit_model(formula: str, data: pd.DataFrame, cluster: str):
    model = smf.ols(formula, data=data)
    return model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cluster], "use_correction": True},
    )


def extract_terms(result, model_name: str, terms: list[str], cluster: str, data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for term in terms:
        if term not in result.params.index:
            continue
        beta = float(result.params[term])
        se = float(result.bse[term])
        p = float(result.pvalues[term])
        ci_low, ci_high = map(float, result.conf_int().loc[term])
        rows.append({
            "model": model_name,
            "cluster": cluster,
            "term": term,
            "coef": beta,
            "std_error": se,
            "p_value": p,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "n_obs": int(result.nobs),
            "n_entities": int(data["panel_id"].nunique()),
            "n_countries": int(data["geo"].nunique()),
            "n_nace": int(data["nace_r2"].nunique()),
            "r_squared": float(result.rsquared),
            "adj_r_squared": float(result.rsquared_adj),
        })
    return pd.DataFrame(rows)


def turning_point(result, ai_mean: float, data: pd.DataFrame) -> dict:
    b1 = float(result.params.get("ai_c", np.nan))
    b2 = float(result.params.get("ai_c_sq", np.nan))
    if not np.isfinite(b1) or not np.isfinite(b2) or abs(b2) < 1e-12:
        raw_tp = np.nan
    else:
        raw_tp = ai_mean - b1 / (2.0 * b2)

    ai_min = float(data["ai_adoption"].min())
    ai_max = float(data["ai_adoption"].max())
    return {
        "beta_ai_centered": b1,
        "beta_ai_squared": b2,
        "p_ai_centered": float(result.pvalues.get("ai_c", np.nan)),
        "p_ai_squared": float(result.pvalues.get("ai_c_sq", np.nan)),
        "turning_point_ai_pct": raw_tp,
        "ai_min_pct": ai_min,
        "ai_max_pct": ai_max,
        "turning_point_in_observed_range": bool(np.isfinite(raw_tp) and ai_min <= raw_tp <= ai_max),
        "shape": "U-shaped" if b2 > 0 else "inverted-U" if b2 < 0 else "linear/flat",
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run Step 02 first: python src\\02_build_core_panel.py"
        )

    raw = pd.read_csv(INPUT)
    sample = prepare_sample(raw)

    all_years = sorted(sample["time"].dropna().unique().tolist())
    required_periods = len(all_years)
    entity_periods = sample.groupby("panel_id")["time"].nunique()
    balanced_ids = entity_periods[entity_periods == required_periods].index
    balanced = sample[sample["panel_id"].isin(balanced_ids)].copy()

    formulas = {
        "M1_log_linear": "log_ghg ~ ai_c + C(panel_id) + C(time)",
        "M2_log_quadratic": "log_ghg ~ ai_c + ai_c_sq + C(panel_id) + C(time)",
        "M3_level_winsor_linear": "ghg_intensity_w ~ ai_c + C(panel_id) + C(time)",
        "M4_level_winsor_quadratic": "ghg_intensity_w ~ ai_c + ai_c_sq + C(panel_id) + C(time)",
    }

    records = []
    primary_results = {}

    for name, formula in formulas.items():
        result = fit_model(formula, sample, cluster="panel_id")
        primary_results[name] = result
        records.append(extract_terms(result, name, ["ai_c", "ai_c_sq"], "panel_id", sample))

    for name, formula in formulas.items():
        result = fit_model(formula, sample, cluster="geo")
        records.append(extract_terms(result, f"{name}_country_cluster", ["ai_c", "ai_c_sq"], "geo", sample))

    if not balanced.empty and balanced["panel_id"].nunique() > 1:
        for name, formula in {
            "M5_balanced_log_linear": formulas["M1_log_linear"],
            "M6_balanced_log_quadratic": formulas["M2_log_quadratic"],
        }.items():
            result = fit_model(formula, balanced, cluster="panel_id")
            records.append(extract_terms(result, name, ["ai_c", "ai_c_sq"], "panel_id", balanced))

    coef_table = pd.concat(records, ignore_index=True)
    coef_table.to_csv(TABLES / "fixed_effects_coefficients.csv", index=False)

    tp = turning_point(
        primary_results["M2_log_quadratic"],
        ai_mean=float(sample.attrs["ai_mean"]),
        data=sample,
    )
    pd.DataFrame([tp]).to_csv(TABLES / "quadratic_turning_point.csv", index=False)

    diagnostics = pd.DataFrame([
        {
            "sample": "main_broad_nace_min2periods",
            "rows": len(sample),
            "countries": sample["geo"].nunique(),
            "nace_groups": sample["nace_r2"].nunique(),
            "entities": sample["panel_id"].nunique(),
            "years": ",".join(map(str, all_years)),
            "ai_mean_pct": sample.attrs["ai_mean"],
            "ghg_winsor_p01": sample.attrs["winsor_lo"],
            "ghg_winsor_p99": sample.attrs["winsor_hi"],
        },
        {
            "sample": "balanced_all_available_years",
            "rows": len(balanced),
            "countries": balanced["geo"].nunique() if not balanced.empty else 0,
            "nace_groups": balanced["nace_r2"].nunique() if not balanced.empty else 0,
            "entities": balanced["panel_id"].nunique() if not balanced.empty else 0,
            "years": ",".join(map(str, all_years)),
            "ai_mean_pct": balanced["ai_adoption"].mean() if not balanced.empty else np.nan,
            "ghg_winsor_p01": np.nan,
            "ghg_winsor_p99": np.nan,
        },
    ])
    diagnostics.to_csv(TABLES / "step03_sample_diagnostics.csv", index=False)

    m1 = primary_results["M1_log_linear"]
    m2 = primary_results["M2_log_quadratic"]
    beta10 = float(m1.params["ai_c"]) * 10
    approx_pct_10pp = (math.exp(beta10) - 1.0) * 100.0

    lines = [
        "STEP 03 - TWO-WAY FIXED EFFECTS",
        "=" * 40,
        f"Rows: {len(sample):,}",
        f"Entities: {sample['panel_id'].nunique():,}",
        f"Countries: {sample['geo'].nunique()}",
        f"Broad NACE sections: {sample['nace_r2'].nunique()}",
        f"Years: {all_years}",
        "",
        "PRIMARY LINEAR LOG MODEL",
        f"AI coefficient: {m1.params['ai_c']:.6f}",
        f"Clustered SE: {m1.bse['ai_c']:.6f}",
        f"p-value: {m1.pvalues['ai_c']:.6f}",
        f"Approx. change in GHG intensity for +10 pp AI: {approx_pct_10pp:.3f}%",
        "",
        "PRIMARY QUADRATIC LOG MODEL",
        f"AI coefficient: {m2.params['ai_c']:.6f}",
        f"AI^2 coefficient: {m2.params['ai_c_sq']:.8f}",
        f"AI p-value: {m2.pvalues['ai_c']:.6f}",
        f"AI^2 p-value: {m2.pvalues['ai_c_sq']:.6f}",
        f"Estimated shape: {tp['shape']}",
        f"Turning point (% AI): {tp['turning_point_ai_pct']}",
        f"Turning point in observed range: {tp['turning_point_in_observed_range']}",
        "",
        "INTERPRETATION RULE",
        "Do not claim a digital rebound effect unless the quadratic term is",
        "statistically credible, the turning point lies within the observed AI",
        "range, and the result survives the planned robustness checks.",
    ]

    (TABLES / "step03_model_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nSaved:")
    print(f"- {TABLES / 'fixed_effects_coefficients.csv'}")
    print(f"- {TABLES / 'quadratic_turning_point.csv'}")
    print(f"- {TABLES / 'step03_sample_diagnostics.csv'}")
    print(f"- {TABLES / 'step03_model_summary.txt'}")


if __name__ == "__main__":
    main()
