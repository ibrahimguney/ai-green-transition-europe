"""
Step 05 - XGBoost + SHAP explanatory layer.

Purpose
-------
Complement the fixed-effects analysis with a nonlinear predictive model.
This is NOT a causal model.

Design
------
- Input: data/processed/controlled_broad_panel.csv
- Outcome: log_ghg = log(1 + ghg_intensity)
- Core numeric features:
    ai_adoption
    renewable_share
    log_real_gdp_pc
    rd_intensity_pct_gdp
- Optional numeric feature:
    electricity_price_eur_kwh, only if >= 80% non-missing
- Categorical features:
    nace_r2
    time
- Country is NOT used as a predictor.
- Validation: GroupKFold by country to reduce leakage across sectors/years.
- Final explanatory model: fit on all complete cases, then compute SHAP values.

Outputs
-------
outputs/tables/xgb_cv_metrics.csv
outputs/tables/xgb_cv_predictions.csv
outputs/tables/xgb_feature_set.csv
outputs/tables/shap_feature_importance.csv
outputs/tables/shap_ai_binned.csv
outputs/tables/step05_ml_summary.txt
outputs/figures/shap_summary.png
outputs/figures/shap_ai_dependence.png

Run
---
    python src/08_xgboost_shap.py
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

INPUT = PROCESSED / "controlled_broad_panel.csv"
RANDOM_STATE = 42
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
GROUP = "geo"

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
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "log_ghg" not in out.columns:
        out["log_ghg"] = np.log1p(out["ghg_intensity"])
    if "log_real_gdp_pc" not in out.columns:
        out["log_real_gdp_pc"] = np.log(
            out["real_gdp_pc_eur2010"].where(out["real_gdp_pc_eur2010"] > 0)
        )
    return out


def select_features(df: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    numeric = list(BASE_NUMERIC)
    feature_rows = []

    for col in BASE_NUMERIC:
        if col not in df.columns:
            raise ValueError(f"Required ML feature missing: {col}")
        coverage = float(df[col].notna().mean())
        feature_rows.append(
            {
                "feature": col,
                "type": "numeric",
                "required": True,
                "coverage_pct": coverage * 100,
                "included": True,
            }
        )

    for col in OPTIONAL_NUMERIC:
        coverage = float(df[col].notna().mean()) if col in df.columns else 0.0
        include = coverage >= OPTIONAL_COVERAGE_THRESHOLD
        if include:
            numeric.append(col)
        feature_rows.append(
            {
                "feature": col,
                "type": "numeric",
                "required": False,
                "coverage_pct": coverage * 100,
                "included": include,
            }
        )

    for col in CATEGORICAL:
        if col not in df.columns:
            raise ValueError(f"Required categorical ML feature missing: {col}")
        feature_rows.append(
            {
                "feature": col,
                "type": "categorical",
                "required": True,
                "coverage_pct": float(df[col].notna().mean() * 100),
                "included": True,
            }
        )

    return numeric, list(CATEGORICAL), pd.DataFrame(feature_rows)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_model() -> XGBRegressor:
    return XGBRegressor(**XGB_PARAMS)


def metric_row(
    fold: int,
    y_true: np.ndarray,
    pred: np.ndarray,
    baseline_pred: np.ndarray,
    n_train: int,
    n_test: int,
    test_countries: list[str],
) -> dict:
    rmse = math.sqrt(mean_squared_error(y_true, pred))
    baseline_rmse = math.sqrt(mean_squared_error(y_true, baseline_pred))
    mae = mean_absolute_error(y_true, pred)
    medae = median_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)

    return {
        "fold": fold,
        "n_train": n_train,
        "n_test": n_test,
        "test_countries": ",".join(sorted(test_countries)),
        "rmse_log": rmse,
        "mae_log": mae,
        "median_ae_log": medae,
        "r2_log": r2,
        "baseline_rmse_log": baseline_rmse,
        "rmse_improvement_pct_vs_mean": (
            (baseline_rmse - rmse) / baseline_rmse * 100
            if baseline_rmse > 0
            else np.nan
        ),
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run Step 04 first:\n"
            "  python src\\05_download_controls.py\n"
            "  python src\\06_merge_controls.py"
        )

    raw = pd.read_csv(INPUT)
    raw = ensure_derived_columns(raw)

    numeric, categorical, feature_table = select_features(raw)
    feature_table.to_csv(TABLES / "xgb_feature_set.csv", index=False)

    needed = [TARGET, GROUP, *numeric, *categorical]
    sample = raw.dropna(subset=needed).copy()
    sample["time"] = sample["time"].astype(str)
    sample["nace_r2"] = sample["nace_r2"].astype(str)

    n_countries = sample[GROUP].nunique()
    if n_countries < 3:
        raise ValueError("Need at least 3 countries for grouped cross-validation.")

    n_splits = min(5, n_countries)
    cv = GroupKFold(n_splits=n_splits)

    X = sample[numeric + categorical].copy()
    y = sample[TARGET].to_numpy(dtype=float)
    groups = sample[GROUP].astype(str).to_numpy()

    pred_records = []
    metric_records = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        preprocessor = make_preprocessor(numeric, categorical)
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        model = make_model()
        model.fit(X_train_t, y_train)
        pred = model.predict(X_test_t)
        baseline_pred = np.repeat(y_train.mean(), len(test_idx))

        test_countries = sorted(sample.iloc[test_idx][GROUP].astype(str).unique().tolist())
        metric_records.append(
            metric_row(
                fold,
                y_test,
                pred,
                baseline_pred,
                len(train_idx),
                len(test_idx),
                test_countries,
            )
        )

        fold_frame = sample.iloc[test_idx][
            ["geo", "nace_r2", "time", "ai_adoption", "ghg_intensity", TARGET]
        ].copy()
        fold_frame["fold"] = fold
        fold_frame["pred_log_ghg"] = pred
        fold_frame["pred_ghg_intensity"] = np.expm1(pred)
        fold_frame["residual_log"] = fold_frame[TARGET] - fold_frame["pred_log_ghg"]
        pred_records.append(fold_frame)

    metrics = pd.DataFrame(metric_records)
    predictions = pd.concat(pred_records, ignore_index=True)
    metrics.to_csv(TABLES / "xgb_cv_metrics.csv", index=False)
    predictions.to_csv(TABLES / "xgb_cv_predictions.csv", index=False)

    # Final full-sample model for SHAP explanation.
    final_preprocessor = make_preprocessor(numeric, categorical)
    X_all_t = final_preprocessor.fit_transform(X)
    feature_names = final_preprocessor.get_feature_names_out().tolist()

    final_model = make_model()
    final_model.fit(X_all_t, y)

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_all_t)
    shap_values = np.asarray(shap_values)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(TABLES / "shap_feature_importance.csv", index=False)

    # SHAP summary plot.
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_all_t,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png", dpi=220, bbox_inches="tight")
    plt.close()

    # AI dependence plot and binned AI-SHAP table.
    ai_candidates = [i for i, name in enumerate(feature_names) if name.endswith("ai_adoption")]
    if not ai_candidates:
        raise RuntimeError("AI feature not found after preprocessing.")
    ai_idx = ai_candidates[0]

    plt.figure()
    shap.dependence_plot(
        ai_idx,
        shap_values,
        X_all_t,
        feature_names=feature_names,
        interaction_index=None,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_ai_dependence.png", dpi=220, bbox_inches="tight")
    plt.close()

    ai_shap = pd.DataFrame(
        {
            "ai_adoption": sample["ai_adoption"].to_numpy(dtype=float),
            "ai_shap": shap_values[:, ai_idx],
        }
    )
    q = min(10, ai_shap["ai_adoption"].nunique())
    if q >= 2:
        ai_shap["ai_bin"] = pd.qcut(
            ai_shap["ai_adoption"], q=q, duplicates="drop"
        ).astype(str)
        ai_binned = (
            ai_shap.groupby("ai_bin", as_index=False, observed=True)
            .agg(
                n=("ai_adoption", "size"),
                mean_ai_adoption=("ai_adoption", "mean"),
                median_ai_adoption=("ai_adoption", "median"),
                mean_ai_shap=("ai_shap", "mean"),
                median_ai_shap=("ai_shap", "median"),
            )
        )
    else:
        ai_binned = pd.DataFrame()
    ai_binned.to_csv(TABLES / "shap_ai_binned.csv", index=False)

    params_path = TABLES / "xgb_model_params.json"
    params_path.write_text(json.dumps(XGB_PARAMS, indent=2), encoding="utf-8")

    avg = metrics.mean(numeric_only=True)
    weighted_rmse = math.sqrt(
        mean_squared_error(predictions[TARGET], predictions["pred_log_ghg"])
    )
    weighted_mae = mean_absolute_error(predictions[TARGET], predictions["pred_log_ghg"])
    weighted_r2 = r2_score(predictions[TARGET], predictions["pred_log_ghg"])

    top_features = importance.head(10)
    ai_rank = int(importance.reset_index(drop=True).index[importance["feature"].str.endswith("ai_adoption")][0] + 1)
    ai_importance = float(
        importance.loc[importance["feature"].str.endswith("ai_adoption"), "mean_abs_shap"].iloc[0]
    )

    lines = [
        "STEP 05 - XGBOOST + SHAP",
        "=" * 42,
        f"Rows used: {len(sample):,}",
        f"Countries: {n_countries}",
        f"NACE sections: {sample['nace_r2'].nunique()}",
        f"Years: {sorted(sample['time'].unique().tolist())}",
        f"GroupKFold splits: {n_splits} (group = country)",
        "",
        "FEATURES",
        f"Numeric: {numeric}",
        f"Categorical: {categorical}",
        "Country is not used as a predictor.",
        "",
        "OUT-OF-COUNTRY CROSS-VALIDATION",
        f"Pooled RMSE (log outcome): {weighted_rmse:.4f}",
        f"Pooled MAE  (log outcome): {weighted_mae:.4f}",
        f"Pooled R2   (log outcome): {weighted_r2:.4f}",
        f"Mean fold RMSE improvement vs mean baseline: {avg['rmse_improvement_pct_vs_mean']:.2f}%",
        "",
        "SHAP",
        f"AI mean(|SHAP|): {ai_importance:.6f}",
        f"AI importance rank: {ai_rank} of {len(importance)} transformed features",
        "Top 10 transformed features:",
        *[
            f"  {row.feature}: mean|SHAP|={row.mean_abs_shap:.6f}"
            for row in top_features.itertuples(index=False)
        ],
        "",
        "INTERPRETATION RULE",
        "SHAP values describe how the fitted predictive model uses variables;",
        "they do not establish causal effects. Compare the AI SHAP pattern with",
        "the fixed-effects results before drawing substantive conclusions.",
    ]

    (TABLES / "step05_ml_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nSaved:")
    for path in [
        TABLES / "xgb_cv_metrics.csv",
        TABLES / "xgb_cv_predictions.csv",
        TABLES / "xgb_feature_set.csv",
        TABLES / "shap_feature_importance.csv",
        TABLES / "shap_ai_binned.csv",
        TABLES / "xgb_model_params.json",
        TABLES / "step05_ml_summary.txt",
        FIGURES / "shap_summary.png",
        FIGURES / "shap_ai_dependence.png",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
