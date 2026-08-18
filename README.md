# AI Green Transition Europe — Step 01

This starter package creates the data foundation for:

**Artificial Intelligence and the Green Transition: Does AI Adoption Reduce Carbon Intensity or Create a Digital Rebound Effect in Europe?**

## 1. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Download Eurostat data

```powershell
python src\01_download_eurostat.py
```

Expected files:

- `data/raw/eurostat_ai_nace.csv`
- `data/raw/eurostat_ghg_intensity_nace.csv`
- `data/raw/eurostat_digital_intensity_nace.csv`
- `data/raw/eurostat_cloud_nace.csv`

## 3. Build first common panel

```powershell
python src\02_build_core_panel.py
```

Expected output:

- `data/processed/core_panel.csv`

## 4. Inspect the panel

Check:
- number of countries
- NACE groups
- common years
- missing values
- duplicated country-sector-year cells

The next research step is to construct an explicit NACE harmonisation table and run descriptive diagnostics before econometric estimation.

## Eurostat datasets

- `isoc_eb_ain2` — Artificial intelligence by NACE Rev. 2 activity
- `env_ac_aeint_r2` — Air emissions intensities by NACE Rev. 2 activity
- `isoc_e_diin2` — Digital Intensity by NACE Rev. 2 activity
- `isoc_cicce_usen2` — Cloud computing services by NACE Rev. 2 activity
