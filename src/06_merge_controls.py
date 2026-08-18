"""
Step 04B - Merge Eurostat country-year controls into the core AI-GHG panel.

The controls vary at country-year level and are intentionally repeated
across NACE sectors within a country-year after the merge.

Outputs:
    data/processed/controlled_panel.csv
    data/processed/controlled_broad_panel.csv
    outputs/tables/control_merge_coverage.csv
    outputs/tables/control_missingness_by_year.csv
    outputs/tables/control_country_coverage.csv

Run:
    python src/06_merge_controls.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
PROCESSED.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

CORE = PROCESSED / "core_panel.csv"

CONTROL_FILES = {
    "renewable_share": RAW / "control_renewable_share.csv",
    "real_gdp_pc_eur2010": RAW / "control_real_gdp_pc.csv",
    "rd_intensity_pct_gdp": RAW / "control_rd_intensity.csv",
    "electricity_price_eur_kwh": RAW / "control_electricity_price_annual.csv",
}

BROAD_NACE_REGEX = r"^[A-Z]$"


def read_control(path: Path, value_col: str) -> pd.DataFrame:
    if not path.exists():
        print(f"WARNING: missing optional control file: {path}")
        return pd.DataFrame(columns=["geo", "time", value_col])

    df = pd.read_csv(path)
    required = {"geo", "time", value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    out = df[["geo", "time", value_col]].copy()
    out["time"] = pd.to_numeric(out["time"], errors="raise").astype(int)

    if out.duplicated(["geo", "time"]).any():
        dup = out[out.duplicated(["geo", "time"], keep=False)]
        raise ValueError(
            f"{path.name}: duplicated geo-time keys found:\n{dup.head(20).to_string(index=False)}"
        )
    return out


def main():
    if not CORE.exists():
        raise FileNotFoundError(
            f"{CORE} not found. Run Step 02 first: python src\\02_build_core_panel.py"
        )

    panel = pd.read_csv(CORE)
    panel["time"] = pd.to_numeric(panel["time"], errors="raise").astype(int)

    for value_col, path in CONTROL_FILES.items():
        control = read_control(path, value_col)
        panel = panel.merge(control, on=["geo", "time"], how="left", validate="many_to_one")

    panel["log_ghg"] = np.log1p(panel["ghg_intensity"])
    panel["log_real_gdp_pc"] = np.log(panel["real_gdp_pc_eur2010"].where(panel["real_gdp_pc_eur2010"] > 0))
    panel["exclude_fr_se"] = panel["geo"].isin(["FR", "SE"]).astype(int)

    panel = panel.sort_values(["panel_id", "time"]).reset_index(drop=True)
    broad = panel[panel["nace_r2"].astype(str).str.fullmatch(BROAD_NACE_REGEX)].copy()

    panel.to_csv(PROCESSED / "controlled_panel.csv", index=False)
    broad.to_csv(PROCESSED / "controlled_broad_panel.csv", index=False)

    controls = list(CONTROL_FILES)
    coverage_rows = []
    for col in controls:
        coverage_rows.append(
            {
                "control": col,
                "nonmissing_rows_full_panel": int(panel[col].notna().sum()),
                "coverage_pct_full_panel": float(panel[col].notna().mean() * 100),
                "countries_with_any_data": int(panel.loc[panel[col].notna(), "geo"].nunique()),
                "years_with_any_data": ",".join(
                    map(str, sorted(panel.loc[panel[col].notna(), "time"].unique().tolist()))
                ),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(TABLES / "control_merge_coverage.csv", index=False)

    missing_by_year = (
        panel.groupby("time")[controls]
        .agg(lambda s: s.isna().mean() * 100)
        .reset_index()
    )
    missing_by_year.to_csv(TABLES / "control_missingness_by_year.csv", index=False)

    country_rows = []
    for geo, group in panel.groupby("geo"):
        row = {"geo": geo, "panel_rows": len(group)}
        for col in controls:
            row[f"{col}_coverage_pct"] = float(group[col].notna().mean() * 100)
        country_rows.append(row)
    pd.DataFrame(country_rows).to_csv(TABLES / "control_country_coverage.csv", index=False)

    print("STEP 04B - CONTROL MERGE")
    print("=" * 40)
    print(f"Full panel rows: {len(panel):,}")
    print(f"Broad-NACE rows: {len(broad):,}")
    print(f"Countries: {panel['geo'].nunique()}")
    print(f"Years: {sorted(panel['time'].unique().tolist())}")
    print("\nControl coverage:")
    print(pd.DataFrame(coverage_rows).to_string(index=False))
    print("\nSaved:")
    print(f"- {PROCESSED / 'controlled_panel.csv'}")
    print(f"- {PROCESSED / 'controlled_broad_panel.csv'}")
    print(f"- {TABLES / 'control_merge_coverage.csv'}")
    print(f"- {TABLES / 'control_missingness_by_year.csv'}")
    print(f"- {TABLES / 'control_country_coverage.csv'}")


if __name__ == "__main__":
    main()
