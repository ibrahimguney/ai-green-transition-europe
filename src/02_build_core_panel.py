"""
Step 02 preview - build a harmonised country × NACE × year panel.

This file is included now so Step 01 outputs can immediately be checked.
It only uses exact common keys; a more explicit NACE crosswalk can be
added after inspecting downloaded category codes.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def compact(df, value_name):
    keys = [c for c in ["geo", "nace_r2", "time"] if c in df.columns]
    keep = keys + ["value"]
    out = df[keep].copy()
    out = out.rename(columns={"value": value_name})
    return out


def main():
    ai = pd.read_csv(RAW / "eurostat_ai_nace.csv")
    ghg = pd.read_csv(RAW / "eurostat_ghg_intensity_nace.csv")

    a = compact(ai, "ai_adoption")
    g = compact(ghg, "ghg_intensity")

    panel = g.merge(a, on=["geo", "nace_r2", "time"], how="inner")
    panel = panel.sort_values(["geo", "nace_r2", "time"])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "core_panel.csv"
    panel.to_csv(out, index=False)

    print(f"Panel rows: {len(panel):,}")
    print(f"Countries: {panel['geo'].nunique() if 'geo' in panel else 0}")
    print(f"NACE groups: {panel['nace_r2'].nunique() if 'nace_r2' in panel else 0}")
    print(f"Years: {sorted(panel['time'].dropna().unique().tolist())}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
