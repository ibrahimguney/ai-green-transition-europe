"""
Step 01 - Download Eurostat data for the AI–Green Transition project.

Downloads:
1) AI adoption by NACE Rev.2: isoc_eb_ain2
2) GHG intensity by NACE Rev.2: env_ac_aeint_r2
3) Digital intensity by NACE Rev.2: isoc_e_diin2
4) Cloud computing by NACE Rev.2: isoc_cicce_usen2

Outputs are saved as CSV and Parquet under data/raw/.

Run:
    python src/01_download_eurostat.py
"""

from __future__ import annotations

from pathlib import Path
from itertools import product
import time
import requests
import pandas as pd
import numpy as np

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2021, 2022, 2023, 2024, 2025]

DATASETS = {
    "ai": "isoc_eb_ain2",
    "ghg_intensity": "env_ac_aeint_r2",
    "digital_intensity": "isoc_e_diin2",
    "cloud": "isoc_cicce_usen2",
}


def _ordered_codes(index_obj):
    """
    JSON-stat category.index may be a dict {code: position}
    or a list of codes. Return codes in positional order.
    """
    if isinstance(index_obj, dict):
        return [k for k, _ in sorted(index_obj.items(), key=lambda kv: kv[1])]
    if isinstance(index_obj, list):
        return index_obj
    raise TypeError(f"Unsupported category.index type: {type(index_obj)}")


def jsonstat_to_dataframe(js: dict) -> pd.DataFrame:
    """
    Convert a Eurostat JSON-stat 2.0 dataset to a tidy DataFrame.
    Handles sparse value dictionaries and dense value arrays.
    """
    dims = js["id"]
    sizes = js["size"]
    dim_codes = {}
    dim_labels = {}

    for d in dims:
        cat = js["dimension"][d]["category"]
        codes = _ordered_codes(cat["index"])
        labels = cat.get("label", {})
        dim_codes[d] = codes
        dim_labels[d] = {c: labels.get(c, c) for c in codes}

    total_cells = int(np.prod(sizes))
    values = np.full(total_cells, np.nan, dtype=float)

    raw_values = js.get("value", {})
    if isinstance(raw_values, dict):
        for k, v in raw_values.items():
            values[int(k)] = v
    elif isinstance(raw_values, list):
        arr = np.array([np.nan if v is None else v for v in raw_values], dtype=float)
        values[: len(arr)] = arr
    else:
        raise TypeError("Unsupported JSON-stat value container")

    records = []
    for flat_idx, combo in enumerate(product(*[dim_codes[d] for d in dims])):
        val = values[flat_idx]
        if np.isnan(val):
            continue
        row = {d: code for d, code in zip(dims, combo)}
        for d, code in zip(dims, combo):
            row[f"{d}_label"] = dim_labels[d].get(code, code)
        row["value"] = val
        records.append(row)

    return pd.DataFrame.from_records(records)


def get_json(dataset: str, params: dict | None = None, retries: int = 4) -> dict:
    url = f"{BASE}/{dataset}"
    params = {"lang": "EN", **(params or {})}
    headers = {
        "User-Agent": "Mozilla/5.0 Eurostat-research-downloader/1.0",
        "Accept": "application/json",
    }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_error = exc
            print(f"[retry {attempt}/{retries}] {dataset}: {exc}")
            time.sleep(2 * attempt)

    raise RuntimeError(f"Download failed for {dataset}: {last_error}")


def download_years(dataset: str, years: list[int], fixed_filters: dict | None = None) -> pd.DataFrame:
    frames = []
    for year in years:
        params = {"time": year}
        if fixed_filters:
            params.update(fixed_filters)
        print(f"Downloading {dataset}, {year} ...")
        try:
            js = get_json(dataset, params=params)
            df = jsonstat_to_dataframe(js)
            if not df.empty:
                frames.append(df)
                print(f"  -> {len(df):,} observations")
            else:
                print("  -> no observations")
        except requests.HTTPError as exc:
            print(f"  -> skipped: {exc}")
        except Exception as exc:
            print(f"  -> skipped: {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save(df: pd.DataFrame, stem: str):
    csv_path = RAW_DIR / f"{stem}.csv"
    pq_path = RAW_DIR / f"{stem}.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as exc:
        print(f"Parquet skipped for {stem}: {exc}")
    print(f"Saved: {csv_path}")


def filter_ai(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "indic_is" in out.columns:
        mask = out["indic_is"].eq("E_AI_TANY")
        if mask.any():
            out = out[mask]
    if "unit" in out.columns:
        mask = out["unit"].eq("PC_ENT")
        if mask.any():
            out = out[mask]
    return out.reset_index(drop=True)


def filter_ghg_intensity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    filters = {
        "airpol": "GHG",
        "na_item": "B1G",
        "unit": "G_EUR_CLV20",
    }
    for col, code in filters.items():
        if col in out.columns:
            mask = out[col].eq(code)
            if mask.any():
                out = out[mask]

    return out.reset_index(drop=True)


def choose_by_label(df: pd.DataFrame, dimension: str, phrases: list[str]) -> pd.DataFrame:
    """
    Robust helper for datasets whose indicator code may vary by survey version.
    It searches the label column and keeps the first matching indicator family.
    """
    label_col = f"{dimension}_label"
    if label_col not in df.columns:
        return df

    labels = df[[dimension, label_col]].drop_duplicates()
    score = pd.Series(False, index=labels.index)

    for phrase in phrases:
        score = score | labels[label_col].str.contains(phrase, case=False, na=False)

    candidates = labels.loc[score]
    if candidates.empty:
        print(f"WARNING: no label match for {dimension}: {phrases}")
        return df

    print(f"Matched {dimension} values:")
    print(candidates.to_string(index=False))

    codes = candidates[dimension].tolist()
    return df[df[dimension].isin(codes)].reset_index(drop=True)


def main():
    # 1) AI adoption
    ai = download_years(DATASETS["ai"], YEARS)
    ai = filter_ai(ai)
    save(ai, "eurostat_ai_nace")

    # 2) GHG intensity
    # Use server-side filters to keep the payload small.
    ghg = download_years(
        DATASETS["ghg_intensity"],
        YEARS,
        fixed_filters={
            "airpol": "GHG",
            "na_item": "B1G",
            "unit": "G_EUR_CLV20",
        },
    )
    ghg = filter_ghg_intensity(ghg)
    save(ghg, "eurostat_ghg_intensity_nace")

    # 3) Digital intensity
    dii = download_years(DATASETS["digital_intensity"], YEARS)
    if "indic_is" in dii.columns:
        dii = choose_by_label(
            dii,
            "indic_is",
            [
                "high digital intensity",
                "very high digital intensity",
                "basic digital intensity",
            ],
        )
    save(dii, "eurostat_digital_intensity_nace")

    # 4) Cloud computing
    cloud = download_years(DATASETS["cloud"], YEARS)
    if "indic_is" in cloud.columns:
        cloud = choose_by_label(
            cloud,
            "indic_is",
            [
                "buying cloud computing services",
                "using paid cloud computing services",
                "purchased cloud computing services",
            ],
        )
    save(cloud, "eurostat_cloud_nace")

    print("\nDONE.")
    print("Next: inspect common geo × nace_r2 × time cells and build the harmonised panel.")


if __name__ == "__main__":
    main()
