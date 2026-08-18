"""
Step 04C - Controlled two-way fixed-effects models.

Main sample:
- broad, non-overlapping one-letter NACE sections
- at least two observed years per country-sector entity
- complete cases for the controls required by each model

Dependent variable:
    log_ghg = log(1 + GHG intensity)

Fixed effects:
- country-sector entity fixed effects
- year fixed effects

Primary inference:
- cluster-robust standard errors by country
  because country-year controls are repeated across sectors.

Models:
M0: AI only
M1: AI + renewable share + real GDP per capita
M2: M1 + R&D intensity
M3: M1 + electricity price
M4: quadratic AI + renewable + GDP + R&D
M5: M4 excluding France and Sweden (2023 ICT break robustness)
M6: linear AI + renewable + GDP + R&D excluding France and Sweden

Outputs:
    outputs/tables/controlled_fe_coefficients.csv
    outputs/tables/controlled_fe_model_diagnostics.csv
    outputs/tables/controlled_quadratic_turning_points.csv
    outputs/tables/step04_model_summary.txt

Run:
    python src/07_controlled_fixed_effects.py
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

INPUT = PROCESSED / "controlled_broad_panel.csv"
MIN_PERIODS = 2


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "panel_id",
        "geo",
        "nace_r2",
        "time",
        "ghg_intensity",
        "ai_adoption",
        "renewable_share",
        "real_gdp_pc_eur2010",
        "rd_intensity_pct_gdp",
        "electricity_price_eur_kwh",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["log_ghg"] = np.log1p(out["ghg_intensity"])
    out["log_real_gdp_pc"] = np.log(
        out["real_gdp_pc_eur2010"].where(out["real_gdp_pc_eur2010"] > 0)
    )

    periods = out.groupby("panel_id")["time"].transform("nunique")
    out = out[periods >= MIN_PERIODS].copy()

    ai_mean = out["ai_adoption"].mean()
    out["ai_c"] = out["ai_adoption"] - ai_mean
    out["ai_c_sq"] = out["ai_c"] ** 2
    out.attrs["ai_mean"] = float(ai_mean)

    return out.sort_values(["panel_id", "time"]).reset_index(drop=True)


def complete_case(data: pd.DataFrame, needed: list[str]) -> pd.DataFrame:
    cols = ["log_ghg", "panel_id", "geo", "time", *needed]
    return data.dropna(subset=cols).copy()


def fit(formula: str, data: pd.DataFrame, cluster: str = "geo"):
    if data[cluster].nunique() < 2:
        raise ValueError(f"Too few {cluster} clusters")
    model = smf.ols(formula, data=data)
    return model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": data[cluster],
            "use_correction": True,
            "df_correction": True,
        },
    )


def extract(result, model_name: str, data: pd.DataFrame, terms: list[str], cluster: str):
    rows = []
    ci = result.conf_int()
    for term in terms:
        if term not in result.params.index:
            continue
        rows.append(
            {
                "model": model_name,
                "cluster": cluster,
                "term": term,
                "coef": float(result.params[term]),
                "std_error": float(result.bse[term]),
                "p_value": float(result.pvalues[term]),
                "ci95_low": float(ci.loc[term, 0]),
                "ci95_high": float(ci.loc[term, 1]),
                "n_obs": int(result.nobs),
                "countries": int(data["geo"].nunique()),
                "entities": int(data["panel_id"].nunique()),
                "nace_sections": int(data["nace_r2"].nunique()),
                "r_squared": float(result.rsquared),
                "adj_r_squared": float(result.rsquared_adj),
            }
        )
    return pd.DataFrame(rows)


def turning_point(result, ai_mean: float, data: pd.DataFrame, model_name: str) -> dict:
    b1 = float(result.params.get("ai_c", np.nan))
    b2 = float(result.params.get("ai_c_sq", np.nan))
    tp = np.nan if (not np.isfinite(b2) or abs(b2) < 1e-12) else ai_mean - b1 / (2 * b2)
    ai_min = float(data["ai_adoption"].min())
    ai_max = float(data["ai_adoption"].max())
    return {
        "model": model_name,
        "beta_ai": b1,
        "p_ai": float(result.pvalues.get("ai_c", np.nan)),
        "beta_ai_sq": b2,
        "p_ai_sq": float(result.pvalues.get("ai_c_sq", np.nan)),
        "shape": "U-shaped" if b2 > 0 else "inverted-U" if b2 < 0 else "linear/flat",
        "turning_point_ai_pct": tp,
        "ai_min_pct": ai_min,
        "ai_max_pct": ai_max,
        "turning_point_in_observed_range": bool(np.isfinite(tp) and ai_min <= tp <= ai_max),
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run:\n"
            "  python src\\05_download_controls.py\n"
            "  python src\\06_merge_controls.py"
        )

    raw = pd.read_csv(INPUT)
    sample = prepare(raw)

    specs = [
        {
            "name": "M0_AI_only",
            "needed": ["ai_c"],
            "formula": "log_ghg ~ ai_c + C(panel_id) + C(time)",
            "terms": ["ai_c"],
            "exclude_fr_se": False,
        },
        {
            "name": "M1_AI_REN_GDP",
            "needed": ["ai_c", "renewable_share", "log_real_gdp_pc"],
            "formula": (
                "log_ghg ~ ai_c + renewable_share + log_real_gdp_pc "
                "+ C(panel_id) + C(time)"
            ),
            "terms": ["ai_c", "renewable_share", "log_real_gdp_pc"],
            "exclude_fr_se": False,
        },
        {
            "name": "M2_AI_REN_GDP_RD",
            "needed": [
                "ai_c",
                "renewable_share",
                "log_real_gdp_pc",
                "rd_intensity_pct_gdp",
            ],
            "formula": (
                "log_ghg ~ ai_c + renewable_share + log_real_gdp_pc "
                "+ rd_intensity_pct_gdp + C(panel_id) + C(time)"
            ),
            "terms": [
                "ai_c",
                "renewable_share",
                "log_real_gdp_pc",
                "rd_intensity_pct_gdp",
            ],
            "exclude_fr_se": False,
        },
        {
            "name": "M3_AI_REN_GDP_EPRICE",
            "needed": [
                "ai_c",
                "renewable_share",
                "log_real_gdp_pc",
                "electricity_price_eur_kwh",
            ],
            "formula": (
                "log_ghg ~ ai_c + renewable_share + log_real_gdp_pc "
                "+ electricity_price_eur_kwh + C(panel_id) + C(time)"
            ),
            "terms": [
                "ai_c",
                "renewable_share",
                "log_real_gdp_pc",
                "electricity_price_eur_kwh",
            ],
            "exclude_fr_se": False,
        },
        {
            "name": "M4_quadratic_REN_GDP_RD",
            "needed": [
                "ai_c",
                "ai_c_sq",
                "renewable_share",
                "log_real_gdp_pc",
                "rd_intensity_pct_gdp",
            ],
            "formula": (
                "log_ghg ~ ai_c + ai_c_sq + renewable_share + log_real_gdp_pc "
                "+ rd_intensity_pct_gdp + C(panel_id) + C(time)"
            ),
            "terms": [
                "ai_c",
                "ai_c_sq",
                "renewable_share",
                "log_real_gdp_pc",
                "rd_intensity_pct_gdp",
            ],
            "exclude_fr_se": False,
        },
        {
            "name": "M5_quadratic_exFRSE",
            "needed": [
                "ai_c",
                "ai_c_sq",
                "renewable_share",
                "log_real_gdp_pc",
                "rd_intensity_pct_gdp",
            ],
            "formula": (
                "log_ghg ~ ai_c + ai_c_sq + renewable_share + log_real_gdp_pc "
                "+ rd_intensity_pct_gdp + C(panel_id) + C(time)"
            ),
            "terms": [
                "ai_c",
                "ai_c_sq",
                "renewable_share",
                "log_real_gdp_pc",
                "rd_intensity_pct_gdp",
            ],
            "exclude_fr_se": True,
        },
        {
            "name": "M6_linear_exFRSE",
            "needed": [
                "ai_c",
                "renewable_share",
                "log_real_gdp_pc",
                "rd_intensity_pct_gdp",
            ],
            "formula": (
                "log_ghg ~ ai_c + renewable_share + log_real_gdp_pc "
                "+ rd_intensity_pct_gdp + C(panel_id) + C(time)"
            ),
            "terms": [
                "ai_c",
                "renewable_share",
                "log_real_gdp_pc",
                "rd_intensity_pct_gdp",
            ],
            "exclude_fr_se": True,
        },
    ]

    coefficient_tables = []
    diagnostics = []
    turning_points = []
    fitted = {}

    for spec in specs:
        data = sample.copy()
        if spec["exclude_fr_se"]:
            data = data[~data["geo"].isin(["FR", "SE"])].copy()
        data = complete_case(data, spec["needed"])

        if data.empty:
            print(f"SKIP {spec['name']}: no complete cases")
            continue

        result = fit(spec["formula"], data, cluster="geo")
        fitted[spec["name"]] = (result, data)

        coefficient_tables.append(
            extract(result, spec["name"], data, spec["terms"], cluster="geo")
        )

        diagnostics.append(
            {
                "model": spec["name"],
                "rows": len(data),
                "countries": data["geo"].nunique(),
                "entities": data["panel_id"].nunique(),
                "nace_sections": data["nace_r2"].nunique(),
                "years": ",".join(map(str, sorted(data["time"].unique().tolist()))),
                "excluded_fr_se": spec["exclude_fr_se"],
                "r_squared": float(result.rsquared),
                "adj_r_squared": float(result.rsquared_adj),
            }
        )

        if "ai_c_sq" in spec["terms"]:
            turning_points.append(
                turning_point(
                    result,
                    ai_mean=float(sample.attrs["ai_mean"]),
                    data=data,
                    model_name=spec["name"],
                )
            )

        entity_result = fit(spec["formula"], data, cluster="panel_id")
        coefficient_tables.append(
            extract(
                entity_result,
                f"{spec['name']}_entity_cluster",
                data,
                [term for term in ["ai_c", "ai_c_sq"] if term in spec["terms"]],
                cluster="panel_id",
            )
        )

    if not coefficient_tables:
        raise RuntimeError("No models could be estimated. Check control coverage.")

    coef = pd.concat(coefficient_tables, ignore_index=True)
    diag = pd.DataFrame(diagnostics)
    tp = pd.DataFrame(turning_points)

    coef.to_csv(TABLES / "controlled_fe_coefficients.csv", index=False)
    diag.to_csv(TABLES / "controlled_fe_model_diagnostics.csv", index=False)
    tp.to_csv(TABLES / "controlled_quadratic_turning_points.csv", index=False)

    lines = [
        "STEP 04 - CONTROLLED TWO-WAY FIXED EFFECTS",
        "=" * 48,
        "",
        "Primary inference clusters standard errors by country because",
        "the macro controls vary at country-year level and are repeated",
        "across NACE sectors within each country-year.",
        "",
    ]

    for model_name in [
        "M0_AI_only",
        "M1_AI_REN_GDP",
        "M2_AI_REN_GDP_RD",
        "M3_AI_REN_GDP_EPRICE",
        "M4_quadratic_REN_GDP_RD",
        "M5_quadratic_exFRSE",
        "M6_linear_exFRSE",
    ]:
        if model_name not in fitted:
            continue
        result, data = fitted[model_name]
        lines.extend(
            [
                model_name,
                f"  N = {int(result.nobs):,}; countries = {data['geo'].nunique()}; "
                f"entities = {data['panel_id'].nunique()}",
                f"  AI beta = {result.params.get('ai_c', np.nan):.6f}; "
                f"p = {result.pvalues.get('ai_c', np.nan):.4f}",
            ]
        )
        if "ai_c_sq" in result.params.index:
            lines.append(
                f"  AI^2 beta = {result.params['ai_c_sq']:.8f}; "
                f"p = {result.pvalues['ai_c_sq']:.4f}"
            )
        if "ai_c" in result.params.index:
            ten_pp = (math.exp(float(result.params["ai_c"]) * 10) - 1) * 100
            lines.append(f"  Approx. +10 pp AI effect = {ten_pp:.3f}%")
        lines.append("")

    lines.extend(
        [
            "INTERPRETATION RULE",
            "Do not interpret control coefficients causally. With only three AI",
            "survey years in the core panel, these models are associational fixed-",
            "effects models. The France/Sweden exclusion is a robustness check for",
            "the documented 2023 ICT survey series break.",
        ]
    )

    summary_path = TABLES / "step04_model_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nSaved:")
    print(f"- {TABLES / 'controlled_fe_coefficients.csv'}")
    print(f"- {TABLES / 'controlled_fe_model_diagnostics.csv'}")
    print(f"- {TABLES / 'controlled_quadratic_turning_points.csv'}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
