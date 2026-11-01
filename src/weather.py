"""Daily ERA5 weather per district from the Open-Meteo historical API,
plus the NOAA Oceanic Nino Index (ENSO).

    python -m src.weather            # all districts -> data/raw/weather/<district>.csv
    python -m src.weather --update   # existing files: fetch only the last UPDATE_OVERLAP_DAYS and append
    python -m src.weather --force    # download the full history again
    python -m src.weather oni        # NOAA ONI -> data/raw/oni/oni.csv
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

from . import config as C

# Open-Meteo weights a request by how much data it returns, so a full 1979-today history
# uses up the hourly fair-use limit after about 10 districts. --update re-fetches only
# the recent tail (ERA5T values for the last weeks are revised) and keeps the rest.
UPDATE_OVERLAP_DAYS = 60


def _request(name: str, params: dict) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(C.OPEN_METEO_ARCHIVE, params=params, timeout=120)
        except requests.Timeout:
            wait = 20 * (attempt + 1)
            print(f"  timed out on {name}, waiting {wait}s ...")
            time.sleep(wait)
            continue
        if r.status_code == 429:                       # fair-use limit: back off
            wait = 20 * (attempt + 1)
            print(f"  rate-limited on {name}, waiting {wait}s ...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"still failing after 3 attempts on {name}")


def fetch_district(name: str, lat: float, lon: float, force: bool = False, update: bool = False) -> pd.DataFrame:
    out = C.RAW_WEATHER / f"{name.replace(' ', '_')}.csv"
    old = pd.read_csv(out, parse_dates=["date"]) if out.exists() and not force else None
    if old is not None and not update:
        print(f"  cached  {out.name}")
        return old
    start = C.WEATHER_START
    if old is not None:
        start = (old.date.max() - timedelta(days=UPDATE_OVERLAP_DAYS)).date().isoformat()
    end = (date.today() - timedelta(days=7)).isoformat()   # ERA5 lags ~5 days
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ",".join(C.WEATHER_VARS),
        "timezone": "Asia/Colombo",
    }
    js = _request(name, params)
    df = pd.DataFrame(js["daily"]).rename(columns={
        "time": "date", "precipitation_sum": "rain_mm", "temperature_2m_max": "tmax_c",
        "temperature_2m_min": "tmin_c", "et0_fao_evapotranspiration": "et0_mm"})
    df["date"] = pd.to_datetime(df["date"])
    df.insert(0, "district", name)
    if old is not None:
        df = pd.concat([old[old.date < df.date.min()], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"  saved   {out.name}  {df.date.min().date()} -> {df.date.max().date()}  ({len(df)} days)")
    return df


def fetch_all(force: bool = False, update: bool = False) -> list[str]:
    print("Downloading ERA5 daily weather from Open-Meteo ...")
    failed = []
    for name, (lat, lon) in C.DISTRICTS.items():
        try:
            fetch_district(name, lat, lon, force, update)
        except Exception as e:  # noqa: BLE001
            print(f"  skipped {name}: {e}")
            failed.append(name)
        time.sleep(5)
    if failed:
        print(f"Skipped {len(failed)} district(s), rerun `python -m src.weather` later to retry: {', '.join(failed)}")
    return failed


def fetch_oni() -> pd.DataFrame:
    """ONI = 3-month running mean SST anomaly in Nino 3.4. 'DJF 1980' is centred on Jan."""
    txt = requests.get(C.ONI_URL, timeout=60).text
    rows = [ln.split() for ln in txt.splitlines()[1:] if ln.strip()]
    centre = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
              "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    df = pd.DataFrame([(int(y), centre[s], float(a)) for s, y, _, a in rows],
                      columns=["year", "month", "oni"])
    out = C.RAW_ONI / "oni.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out.relative_to(C.ROOT)} ({len(df)} months)")
    return df


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "oni":
        fetch_oni()
    else:
        fetch_all(force="--force" in sys.argv, update="--update" in sys.argv)
        try:
            fetch_oni()
        except Exception as e:  # noqa: BLE001
            print(f"ONI download failed ({e}); the ENSO feature will be skipped.")
