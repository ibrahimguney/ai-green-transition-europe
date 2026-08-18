"""
Step 04A - Download country-year control variables from Eurostat.

Controls:
1) Renewable energy share: nrg_ind_ren
   - nrg_bal = REN
   - unit = PC

2) Real GDP per capita: nama_10_pc
   - na_item = B1GQ
   - unit = CLV10_EUR_HAB

3) R&D intensity: rd_e_gerdtot
   - sectperf = TOTAL
   - unit = PC_GDP

4) Non-household electricity price: nrg_pc_205
   - product = 6000 (electrical energy)
   - nrg_cons = MWH500-1999 (band IC)
   - unit = KWH
   - tax = I_TAX
   - semester values are averaged to calendar-year means

Important:
The Eurostat Statistics API validates query parameters against actual dataset
Dimensions. `currency=EUR` is intentionally NOT sent for nrg_pc_205 because
`currency` is a visualisation/UI selector rather than a valid filter dimension
for this Statistics API query; including it can produce HTTP 400.

Outputs:
    data/raw/control_renewable_share.csv
    data/raw/control_real_gdp_pc.csv
    data/raw/control_rd_intensity.csv
    data/raw/control_electricity_price_semester.csv
    data/raw/control_electricity_price_annual.csv

Run:
    python src/05_download_controls.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

YEARS = [2021, 2022, 2023, 2024]
SEMESTERS = [f"{year}-S{half}" for year in YEARS for half in (1, 2)]


def _ordered_codes(index_obj):
    if isinstance(index_obj, dict):
        return [k for k, _ in sorted(index_obj.items(), key=lambda kv: kv[1])]
    if isinstance(index_obj, list):
        return index_obj
    raise TypeError(f"Unsupported category.index type: {type(index_obj)}")


def jsonstat_to_dataframe(js: dict) -> pd.DataFrame:
    dims = js["id"]
    sizes = js["size"]
    dim_codes = {}
    dim_labels = {}

    for dim in dims:
        cat = js["dimension"][dim]["category"]
        codes = _ordered_codes(cat["index"])
        labels = cat.get("label", {})
        dim_codes[dim] = codes
        dim_labels[dim] = {code: labels.get(code, code) for code in codes}

    total_cells = int(np.prod(sizes))
    values = np.full(total_cells, np.nan, dtype=float)
    raw_values = js.get("value", {})

    if isinstance(raw_values, dict):
        for idx, value in raw_values.items():
            values[int(idx)] = value
    elif isinstance(raw_values, list):
        arr = np.array(
            [np.nan if value is None else value for value in raw_values],
            dtype=float,
        )
        values[: len(arr)] = arr
    else:
        raise TypeError("Unsupported JSON-stat value container")

    records = []
    code_lists = [dim_codes[d] for d in dims]
    for flat_idx, combo in enumerate(product(*code_lists)):
        value = values[flat_idx]
        if np.isnan(value):
            continue
        row = {dim: code for dim, code in zip(dims, combo)}
        for dim, code in zip(dims, combo):
            row[f"{dim}_label"] = dim_labels[dim].get(code, code)
        row["value"] = value
        records.append(row)

    return pd.DataFrame.from_records(records)


def get_json(dataset: str, params: dict, retries: int = 4) -> dict:
    url = f"{BASE}/{dataset}"
    headers = {
        "User-Agent": "Mozilla/5.0 AI-Green-Transition-Research/1.0",
        "Accept": "application/json",
    }
    params = {"lang": "EN", **params}

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=120)

            # 4xx errors normally mean an invalid dataset filter/code. Retrying the
            # same request four times will not help, so fail immediately and print
            # Eurostat's response text for diagnosis.
            if 400 <= response.status_code < 500:
                detail = response.text.strip().replace("\n", " ")[:500]
                raise ValueError(
                    f"Eurostat HTTP {response.status_code} for {response.url}. "
                    f"Response: {detail or '<empty>'}"
                )

            response.raise_for_status()
            return response.json()

        except ValueError:
            raise
        except Exception as exc:
            last_error = exc
            print(f"[retry {attempt}/{retries}] {dataset}: {exc}")
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(f"Download failed for {dataset}: {last_error}")


def download_periods(dataset: str, periods: list[str | int], filters: dict) -> pd.DataFrame:
    frames = []
    for period in periods:
        params = {"time": period, **filters}
        print(f"Downloading {dataset}, {period} ...")
        try:
            js = get_json(dataset, params)
            frame = jsonstat_to_dataframe(js)
            if frame.empty:
                print("  -> no observations")
                continue
            frames.append(frame)
            print(f"  -> {len(frame):,} observations")
        except Exception as exc:
            print(f"  -> skipped: {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def standardise_annual(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["geo", "time", value_name])

    required = {"geo", "time", "value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{value_name}: missing columns {sorted(missing)}")

    out = df[["geo", "time", "value"]].copy()
    out["time"] = pd.to_numeric(out["time"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["time"])
    out["time"] = out["time"].astype(int)

    duplicate_count = int(out.duplicated(["geo", "time"], keep=False).sum())
    if duplicate_count:
        raise ValueError(
            f"{value_name}: {duplicate_count} duplicated geo-time rows after filtering. "
            "Inspect Eurostat dimensions before continuing."
        )

    return out.rename(columns={"value": value_name}).sort_values(["geo", "time"])


def standardise_electricity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        empty_sem = pd.DataFrame(
            columns=["geo", "semester", "time", "electricity_price_eur_kwh"]
        )
        empty_ann = pd.DataFrame(
            columns=["geo", "time", "electricity_price_eur_kwh"]
        )
        return empty_sem, empty_ann

    out = df[["geo", "time", "value"]].copy()
    out = out.rename(
        columns={
            "time": "semester",
            "value": "electricity_price_eur_kwh",
        }
    )
    out["semester"] = out["semester"].astype(str)
    out["time"] = pd.to_numeric(
        out["semester"].str.extract(r"^(\d{4})", expand=False),
        errors="coerce",
    )
    out = out.dropna(subset=["time"])
    out["time"] = out["time"].astype(int)

    duplicate_count = int(out.duplicated(["geo", "semester"], keep=False).sum())
    if duplicate_count:
        raise ValueError(
            f"Electricity price: {duplicate_count} duplicated geo-semester rows after filtering."
        )

    annual = (
        out.groupby(["geo", "time"], as_index=False)["electricity_price_eur_kwh"]
        .mean()
        .sort_values(["geo", "time"])
    )

    return out.sort_values(["geo", "semester"]), annual


def save(df: pd.DataFrame, filename: str):
    path = RAW / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(df):,} rows)")


def main():
    renewable_raw = download_periods(
        "nrg_ind_ren",
        YEARS,
        {"nrg_bal": "REN", "unit": "PC"},
    )
    renewable = standardise_annual(renewable_raw, "renewable_share")
    save(renewable, "control_renewable_share.csv")

    gdp_raw = download_periods(
        "nama_10_pc",
        YEARS,
        {"na_item": "B1GQ", "unit": "CLV10_EUR_HAB"},
    )
    gdp = standardise_annual(gdp_raw, "real_gdp_pc_eur2010")
    save(gdp, "control_real_gdp_pc.csv")

    rd_raw = download_periods(
        "rd_e_gerdtot",
        YEARS,
        {"sectperf": "TOTAL", "unit": "PC_GDP"},
    )
    rd = standardise_annual(rd_raw, "rd_intensity_pct_gdp")
    save(rd, "control_rd_intensity.csv")

    electricity_raw = download_periods(
        "nrg_pc_205",
        SEMESTERS,
        {
            "product": "6000",
            "nrg_cons": "MWH500-1999",
            "unit": "KWH",
            "tax": "I_TAX",
        },
    )
    electricity_semester, electricity_annual = standardise_electricity(electricity_raw)
    save(electricity_semester, "control_electricity_price_semester.csv")
    save(electricity_annual, "control_electricity_price_annual.csv")

    summary = pd.DataFrame(
        [
            [
                "renewable_share",
                len(renewable),
                renewable["geo"].nunique() if not renewable.empty else 0,
            ],
            [
                "real_gdp_pc_eur2010",
                len(gdp),
                gdp["geo"].nunique() if not gdp.empty else 0,
            ],
            [
                "rd_intensity_pct_gdp",
                len(rd),
                rd["geo"].nunique() if not rd.empty else 0,
            ],
            [
                "electricity_price_eur_kwh",
                len(electricity_annual),
                electricity_annual["geo"].nunique()
                if not electricity_annual.empty
                else 0,
            ],
        ],
        columns=["control", "country_year_rows", "countries"],
    )
    save(summary, "control_download_summary.csv")

    print("\nControl download complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
