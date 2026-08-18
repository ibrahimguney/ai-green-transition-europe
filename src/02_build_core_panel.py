"""
Step 02 - Build the harmonised AI × GHG country-sector-year panel.

Principle:
    Never average emission-intensity ratios across NACE groups.
    The core panel uses exact NACE Rev. 2 code matches only.

Inputs (created by Step 01):
    data/raw/eurostat_ai_nace.csv
    data/raw/eurostat_ghg_intensity_nace.csv

Outputs:
    data/processed/core_panel.csv
    outputs/tables/nace_inventory.csv
    outputs/tables/nace_match_report.csv
    outputs/tables/panel_coverage_by_year.csv
    outputs/tables/panel_coverage_by_country.csv
    outputs/tables/panel_coverage_by_nace.csv
    outputs/tables/duplicate_key_report.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

CORE_YEARS = [2021, 2022, 2023, 2024]
KEYS = ["geo", "nace_r2", "time"]

# Eurostat aggregate geographies that should not be mixed with countries.
AGGREGATE_GEOS = {
    "EU27_2020", "EU28", "EU15", "EA", "EA19", "EA20",
    "EEA30_2007", "EEA31", "EFTA", "EURO_AREA",
}


def require_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def normalise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    require_columns(out, ["geo", "nace_r2", "time", "value"], source)

    out["geo"] = out["geo"].astype(str).str.strip()
    out["nace_r2"] = out["nace_r2"].astype(str).str.strip().str.upper()
    out["time"] = pd.to_numeric(out["time"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out[out["time"].isin(CORE_YEARS)].copy()
    out = out[~out["geo"].isin(AGGREGATE_GEOS)].copy()
    out = out.dropna(subset=["geo", "nace_r2", "time", "value"])

    return out.reset_index(drop=True)


def label_lookup(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["nace_r2"]
    if "nace_r2_label" in df.columns:
        cols.append("nace_r2_label")
    lookup = df[cols].drop_duplicates()
    if "nace_r2_label" not in lookup.columns:
        lookup["nace_r2_label"] = lookup["nace_r2"]
    return lookup


def duplicate_report(df: pd.DataFrame, source: str) -> pd.DataFrame:
    counts = (
        df.groupby(KEYS, dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    dup = counts[counts["n_rows"] > 1].copy()
    dup.insert(0, "source", source)
    return dup


def build_nace_inventory(ai: pd.DataFrame, ghg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, df in [("AI", ai), ("GHG", ghg)]:
        labels = label_lookup(df)
        counts = (
            df.groupby("nace_r2", as_index=False)
            .agg(
                observations=("value", "size"),
                countries=("geo", "nunique"),
                years=("time", "nunique"),
            )
        )
        x = labels.merge(counts, on="nace_r2", how="right")
        x.insert(0, "source", source)
        rows.append(x)

    return pd.concat(rows, ignore_index=True).sort_values(["nace_r2", "source"])


def build_match_report(ai: pd.DataFrame, ghg: pd.DataFrame) -> pd.DataFrame:
    ai_codes = label_lookup(ai).rename(columns={"nace_r2_label": "ai_label"})
    ghg_codes = label_lookup(ghg).rename(columns={"nace_r2_label": "ghg_label"})

    report = ai_codes.merge(
        ghg_codes,
        on="nace_r2",
        how="outer",
        indicator=True,
    )
    report["exact_code_match"] = report["_merge"].eq("both")
    report["available_in_ai"] = report["_merge"].isin(["both", "left_only"])
    report["available_in_ghg"] = report["_merge"].isin(["both", "right_only"])
    report = report.drop(columns="_merge")

    return report.sort_values(["exact_code_match", "nace_r2"], ascending=[False, True])


def compact(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    keep = KEYS + ["value"]
    out = df[keep].copy()
    return out.rename(columns={"value": value_name})


def save_coverage_tables(panel: pd.DataFrame) -> None:
    by_year = (
        panel.groupby("time", as_index=False)
        .agg(
            observations=("panel_id", "size"),
            countries=("geo", "nunique"),
            nace_groups=("nace_r2", "nunique"),
            panel_units=("panel_id", "nunique"),
            mean_ai_adoption=("ai_adoption", "mean"),
            mean_ghg_intensity=("ghg_intensity", "mean"),
        )
    )
    by_year.to_csv(TABLES / "panel_coverage_by_year.csv", index=False)

    by_country = (
        panel.groupby("geo", as_index=False)
        .agg(
            observations=("panel_id", "size"),
            years=("time", "nunique"),
            nace_groups=("nace_r2", "nunique"),
            mean_ai_adoption=("ai_adoption", "mean"),
            mean_ghg_intensity=("ghg_intensity", "mean"),
        )
        .sort_values(["observations", "geo"], ascending=[False, True])
    )
    by_country.to_csv(TABLES / "panel_coverage_by_country.csv", index=False)

    by_nace = (
        panel.groupby("nace_r2", as_index=False)
        .agg(
            observations=("panel_id", "size"),
            countries=("geo", "nunique"),
            years=("time", "nunique"),
            mean_ai_adoption=("ai_adoption", "mean"),
            mean_ghg_intensity=("ghg_intensity", "mean"),
        )
        .sort_values(["observations", "nace_r2"], ascending=[False, True])
    )
    by_nace.to_csv(TABLES / "panel_coverage_by_nace.csv", index=False)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    ai_path = RAW / "eurostat_ai_nace.csv"
    ghg_path = RAW / "eurostat_ghg_intensity_nace.csv"

    if not ai_path.exists() or not ghg_path.exists():
        raise FileNotFoundError(
            "Step 01 outputs are missing. Run: python src/01_download_eurostat.py"
        )

    ai = normalise(pd.read_csv(ai_path), "AI")
    ghg = normalise(pd.read_csv(ghg_path), "GHG intensity")

    inventory = build_nace_inventory(ai, ghg)
    inventory.to_csv(TABLES / "nace_inventory.csv", index=False)

    match_report = build_match_report(ai, ghg)
    match_report.to_csv(TABLES / "nace_match_report.csv", index=False)

    dup = pd.concat(
        [duplicate_report(ai, "AI"), duplicate_report(ghg, "GHG")],
        ignore_index=True,
    )
    dup.to_csv(TABLES / "duplicate_key_report.csv", index=False)

    if not dup.empty:
        raise ValueError(
            "Duplicate geo × NACE × year keys detected. "
            "Inspect outputs/tables/duplicate_key_report.csv before merging."
        )

    panel = compact(ghg, "ghg_intensity").merge(
        compact(ai, "ai_adoption"),
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    if panel.empty:
        raise ValueError(
            "No exact geo × NACE × year matches were found. "
            "Inspect outputs/tables/nace_match_report.csv."
        )

    panel["panel_id"] = panel["geo"] + "__" + panel["nace_r2"]
    panel["ai_adoption_sq"] = panel["ai_adoption"] ** 2
    panel = panel[
        ["panel_id", "geo", "nace_r2", "time",
         "ghg_intensity", "ai_adoption", "ai_adoption_sq"]
    ].sort_values(["geo", "nace_r2", "time"])

    panel.to_csv(PROCESSED / "core_panel.csv", index=False)
    save_coverage_tables(panel)

    n_exact_codes = int(match_report["exact_code_match"].sum())
    years = sorted(panel["time"].dropna().unique().tolist())

    print("\nSTEP 02 COMPLETE")
    print("----------------")
    print(f"Exact NACE codes shared by AI and GHG: {n_exact_codes}")
    print(f"Panel rows: {len(panel):,}")
    print(f"Countries: {panel['geo'].nunique()}")
    print(f"NACE groups: {panel['nace_r2'].nunique()}")
    print(f"Panel units (country × NACE): {panel['panel_id'].nunique()}")
    print(f"Years: {years}")
    print("Saved: data/processed/core_panel.csv")
    print("Diagnostics: outputs/tables/")


if __name__ == "__main__":
    main()
