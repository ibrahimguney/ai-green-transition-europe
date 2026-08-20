"""
Step 05B - Grouped SHAP importance by original conceptual feature.

Why this step exists
--------------------
The Step 05 model one-hot encodes NACE and year. Ranking transformed columns
one-by-one can make a categorical variable appear as many separate features.
This script recomputes SHAP values and aggregates them at the ROW level back to
original feature families before taking mean absolute SHAP.

Groups
------
- AI adoption
- Renewable energy share
- Real GDP per capita
- R&D intensity
- Electricity price (only if sufficiently observed)
- NACE sector
- Year

Input
-----
data/processed/controlled_broad_panel.csv

Outputs
-------
outputs/tables/shap_grouped_importance.csv
outputs/tables/shap_grouped_row_values.csv
outputs/figures/shap_grouped_importance.png
outputs/tables/step05b_grouped_shap_summary.txt

Run
---
    python src/09_grouped_shap.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

INPUT = PROCESSED / "controlled_broad_panel.csv"
OPTIONAL_COVERAGE_THRESHOLD = 0.80

BASE_NUMERIC = [
    "ai_adoption",
    "renewable_share",
    "log_real_gdp_pc",
    "rd_intensity_pct_gdp",
]
OPTIONAL_NUMERIC = ["electricity_price_eur_kwh"]
CATEGORICAL = ["nace_r2", "time"]
TARGET = "log_ghg"

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 2,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "log_ghg" not in out.columns:
        out["log_ghg"] = np.log1p(out["ghg_intensity"])
    if "log_real_gdp_pc" not in out.columns:
        out["log_real_gdp_pc"] = np.log(
            out["real_gdp_pc_eur2010"].where(out["real_gdp_pc_eur2010"] > 0)
        )
    return out


def select_numeric(df: pd.DataFrame) -> list[str]:
    numeric = list(BASE_NUMERIC)
    for col in BASE_NUMERIC:
        if col not in df.columns:
            raise ValueError(f"Required feature missing: {col}")
    for col in OPTIONAL_NUMERIC:
        if col in df.columns and df[col].notna().mean() >= OPTIONAL_COVERAGE_THRESHOLD:
            numeric.append(col)
    return numeric


def make_preprocessor(numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def conceptual_group(transformed_name: str) -> str:
    if transformed_name.endswith("ai_adoption"):
        return "AI adoption"
    if transformed_name.endswith("renewable_share"):
        return "Renewable energy share"
    if transformed_name.endswith("log_real_gdp_pc"):
        return "Real GDP per capita"
    if transformed_name.endswith("rd_intensity_pct_gdp"):
        return "R&D intensity"
    if transformed_name.endswith("electricity_price_eur_kwh"):
        return "Electricity price"
    if transformed_name.startswith("cat__nace_r2_"):
        return "NACE sector"
    if transformed_name.startswith("cat__time_"):
        return "Year"
    return transformed_name


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run Step 04 first.")

    raw = ensure_columns(pd.read_csv(INPUT))
    numeric = select_numeric(raw)
    needed = [TARGET, *numeric, *CATEGORICAL]
    sample = raw.dropna(subset=needed).copy()
    sample["nace_r2"] = sample["nace_r2"].astype(str)
    sample["time"] = sample["time"].astype(str)

    X = sample[numeric + CATEGORICAL].copy()
    y = sample[TARGET].to_numpy(dtype=float)

    preprocessor = make_preprocessor(numeric)
    X_t = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_t, y)

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X_t))

    groups = [conceptual_group(name) for name in feature_names]
    unique_groups = list(dict.fromkeys(groups))

    grouped_rows = pd.DataFrame(index=np.arange(len(sample)))
    for group in unique_groups:
        idx = [i for i, g in enumerate(groups) if g == group]
        grouped_rows[group] = shap_values[:, idx].sum(axis=1)

    meta_cols = [c for c in ["geo", "nace_r2", "time", "ai_adoption", "ghg_intensity"] if c in sample.columns]
    row_output = pd.concat(
        [sample[meta_cols].reset_index(drop=True), grouped_rows.reset_index(drop=True)],
        axis=1,
    )
    row_output.to_csv(TABLES / "shap_grouped_row_values.csv", index=False)

    importance_rows = []
    for group in unique_groups:
        vals = grouped_rows[group].to_numpy(dtype=float)
        importance_rows.append(
            {
                "feature_group": group,
                "mean_abs_group_shap": float(np.mean(np.abs(vals))),
                "mean_group_shap": float(np.mean(vals)),
                "median_abs_group_shap": float(np.median(np.abs(vals))),
                "n_transformed_columns": int(sum(g == group for g in groups)),
            }
        )

    importance = (
        pd.DataFrame(importance_rows)
        .sort_values("mean_abs_group_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance.to_csv(TABLES / "shap_grouped_importance.csv", index=False)

    plot_df = importance.sort_values("mean_abs_group_shap", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["feature_group"], plot_df["mean_abs_group_shap"])
    plt.xlabel("Mean absolute grouped SHAP value")
    plt.ylabel("")
    plt.title("Grouped SHAP importance by original feature")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_grouped_importance.png", dpi=220, bbox_inches="tight")
    plt.close()

    ai_row = importance.loc[importance["feature_group"] == "AI adoption"]
    ai_rank = int(ai_row["rank"].iloc[0]) if not ai_row.empty else -1
    ai_value = float(ai_row["mean_abs_group_shap"].iloc[0]) if not ai_row.empty else np.nan

    lines = [
        "STEP 05B - GROUPED SHAP",
        "=" * 42,
        f"Rows used: {len(sample):,}",
        f"Original feature groups: {len(importance)}",
        f"AI grouped mean(|SHAP|): {ai_value:.6f}",
        f"AI grouped importance rank: {ai_rank} of {len(importance)} original feature groups",
        "",
        "GROUPED IMPORTANCE",
        *[
            f"{int(r.rank)}. {r.feature_group}: {r.mean_abs_group_shap:.6f} "
            f"({int(r.n_transformed_columns)} transformed column(s))"
            for r in importance.itertuples(index=False)
        ],
        "",
        "INTERPRETATION",
        "Grouping is performed at the observation level by summing SHAP values",
        "across all transformed columns belonging to the same original variable,",
        "then taking mean absolute grouped SHAP. This avoids treating each NACE",
        "or year dummy as an independent conceptual predictor.",
    ]

    (TABLES / "step05b_grouped_shap_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\n".join(lines))
    print("\nSaved:")
    print(f"- {TABLES / 'shap_grouped_importance.csv'}")
    print(f"- {TABLES / 'shap_grouped_row_values.csv'}")
    print(f"- {FIGURES / 'shap_grouped_importance.png'}")
    print(f"- {TABLES / 'step05b_grouped_shap_summary.txt'}")


if __name__ == "__main__":
    main()
