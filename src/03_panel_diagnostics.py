"""
Step 03 - Panel diagnostics and descriptive statistics.

Input:
    data/processed/core_panel.csv

Outputs:
    outputs/tables/descriptive_statistics.csv
    outputs/tables/missingness_report.csv
    outputs/tables/correlation_matrix.csv
    outputs/tables/panel_balance_report.csv
    outputs/tables/descriptives_by_year.csv
    outputs/tables/descriptives_by_nace.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

CORE_YEARS = [2021, 2022, 2023, 2024]
VARS = ["ghg_intensity", "ai_adoption", "ai_adoption_sq"]


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    desc = df[VARS].describe(
        percentiles=[0.25, 0.50, 0.75]
    ).T.reset_index()
    desc = desc.rename(
        columns={
            "index": "variable",
            "25%": "q25",
            "50%": "median",
            "75%": "q75",
            "std": "sd",
        }
    )
    return desc


def panel_balance(df: pd.DataFrame) -> pd.DataFrame:
    x = (
        df.groupby(["panel_id", "geo", "nace_r2"], as_index=False)
        .agg(
            first_year=("time", "min"),
            last_year=("time", "max"),
            n_years=("time", "nunique"),
            observations=("time", "size"),
        )
    )
    x["balanced_2021_2024"] = x["n_years"].eq(len(CORE_YEARS))
    return x.sort_values(["balanced_2021_2024", "n_years", "panel_id"],
                         ascending=[False, False, True])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    panel_path = PROCESSED / "core_panel.csv"

    if not panel_path.exists():
        raise FileNotFoundError(
            "core_panel.csv not found. Run: python src/02_build_core_panel.py"
        )

    df = pd.read_csv(panel_path)
    required = ["panel_id", "geo", "nace_r2", "time", *VARS]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"core_panel.csv missing columns: {missing_cols}")

    key_dup = df.duplicated(["geo", "nace_r2", "time"], keep=False)
    if key_dup.any():
        raise ValueError(
            f"Panel has {int(key_dup.sum())} duplicated country-sector-year rows."
        )

    descriptive_statistics(df).to_csv(
        TABLES / "descriptive_statistics.csv", index=False
    )

    missingness = pd.DataFrame(
        {
            "variable": df.columns,
            "missing_n": [int(df[c].isna().sum()) for c in df.columns],
            "missing_pct": [float(df[c].isna().mean() * 100) for c in df.columns],
        }
    )
    missingness.to_csv(TABLES / "missingness_report.csv", index=False)

    df[VARS].corr(method="pearson").to_csv(
        TABLES / "correlation_matrix.csv"
    )

    balance = panel_balance(df)
    balance.to_csv(TABLES / "panel_balance_report.csv", index=False)

    by_year = (
        df.groupby("time")[["ghg_intensity", "ai_adoption"]]
        .agg(["count", "mean", "std", "median", "min", "max"])
    )
    by_year.to_csv(TABLES / "descriptives_by_year.csv")

    by_nace = (
        df.groupby("nace_r2")[["ghg_intensity", "ai_adoption"]]
        .agg(["count", "mean", "std", "median", "min", "max"])
    )
    by_nace.to_csv(TABLES / "descriptives_by_nace.csv")

    balanced_n = int(balance["balanced_2021_2024"].sum())
    total_units = len(balance)
    balanced_pct = (balanced_n / total_units * 100) if total_units else 0.0

    print("\nSTEP 03 DIAGNOSTICS COMPLETE")
    print("----------------------------")
    print(f"Observations: {len(df):,}")
    print(f"Countries: {df['geo'].nunique()}")
    print(f"NACE groups: {df['nace_r2'].nunique()}")
    print(f"Panel units: {total_units}")
    print(f"Balanced 2021-2024 units: {balanced_n} ({balanced_pct:.1f}%)")
    print(f"Missing core-variable values: {int(df[VARS].isna().sum().sum())}")
    print("Saved diagnostic tables under outputs/tables/")


if __name__ == "__main__":
    main()
