"""
AWC (Aviation Weather Center) Near-Real-Time Extractor
-------------------------------------------------------
Polls the FAA AWC API for the latest METAR and TAF reports
for all 10 target airports and loads them into BigQuery.

METAR: current observed conditions (updated ~hourly)
TAF:   terminal aerodrome forecast (updated ~4x/day)

BigQuery targets:
  raw_weather.metar_raw
  raw_weather.taf_raw

Usage (manual run):
    python extract/awc_extractor.py

Called by Airflow DAG:
    airflow/dags/awc_polling_dag.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_to_bigquery import load_dataframe_to_bigquery

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# ICAO codes for our 10 target airports
AIRPORT_ICAO = ["KATL", "KDFW", "KDEN", "KORD", "KLAX", "KJFK", "KSFO", "KSEA", "KLAS", "KMCO"]

# IATA ↔ ICAO mapping (for joining back to flight/weather tables)
ICAO_TO_IATA = {
    "KATL": "ATL", "KDFW": "DFW", "KDEN": "DEN", "KORD": "ORD", "KLAX": "LAX",
    "KJFK": "JFK", "KSFO": "SFO", "KSEA": "SEA", "KLAS": "LAS", "KMCO": "MCO",
}

AWC_BASE = "https://aviationweather.gov/api/data"
REQUEST_TIMEOUT = 30

METAR_DATASET = "raw_weather"
METAR_TABLE = "metar_raw"
TAF_DATASET = "raw_weather"
TAF_TABLE = "taf_raw"


# ---------------------------------------------------------------------------
# METAR
# ---------------------------------------------------------------------------

def fetch_metar(hours_back: int = 2) -> pd.DataFrame:
    """Fetch the latest METAR for all target airports."""
    ids = ",".join(AIRPORT_ICAO)
    url = f"{AWC_BASE}/metar"
    params = {"ids": ids, "format": "json", "hours": hours_back}

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    records = response.json()

    if not records:
        return pd.DataFrame()

    rows = []
    ingested_at = datetime.now(timezone.utc).isoformat()

    for r in records:
        icao = r.get("icaoId", "")
        rows.append({
            "icao_id":          icao,
            "iata_code":        ICAO_TO_IATA.get(icao, icao),
            "obs_time":         r.get("reportTime"),
            "receipt_time":     r.get("receiptTime"),
            "temp_c":           r.get("temp"),
            "dewpoint_c":       r.get("dewp"),
            "wind_dir_deg":     r.get("wdir"),
            "wind_speed_knots": r.get("wspd"),
            "wind_gust_knots":  r.get("wgst"),
            "visibility_sm":    r.get("visib"),
            "altimeter_hpa":    r.get("altim"),
            "sea_level_pres_hpa": r.get("slp"),
            "flight_category":  r.get("fltCat"),
            "cloud_cover":      r.get("cover"),
            "metar_type":       r.get("metarType"),
            "raw_metar":        r.get("rawOb"),
            "latitude":         r.get("lat"),
            "longitude":        r.get("lon"),
            "elevation_m":      r.get("elev"),
            "source_name":      "awc_metar",
            "ingested_at":      ingested_at,
        })

    df = pd.DataFrame(rows)
    # wind_dir_deg can be "VRB" (variable) — store as STRING to avoid type conflicts
    df["wind_dir_deg"] = df["wind_dir_deg"].astype(str)
    return df


# ---------------------------------------------------------------------------
# TAF
# ---------------------------------------------------------------------------

def fetch_taf() -> pd.DataFrame:
    """Fetch the most recent TAF for all target airports."""
    ids = ",".join(AIRPORT_ICAO)
    url = f"{AWC_BASE}/taf"
    params = {"ids": ids, "format": "json"}

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    records = response.json()

    if not records:
        return pd.DataFrame()

    rows = []
    ingested_at = datetime.now(timezone.utc).isoformat()

    for r in records:
        icao = r.get("icaoId", "")
        # Flatten: one row per forecast period
        fcsts = r.get("fcsts", [{}])
        for fcst in fcsts:
            rows.append({
                "icao_id":              icao,
                "iata_code":            ICAO_TO_IATA.get(icao, icao),
                "issue_time":           r.get("issueTime"),
                "bulletin_time":        r.get("bulletinTime"),
                "valid_from":           fcst.get("timeFrom"),
                "valid_to":             fcst.get("timeTo"),
                "forecast_change_type": fcst.get("fcstChange"),
                "wind_dir_deg":         fcst.get("wdir"),
                "wind_speed_knots":     fcst.get("wspd"),
                "wind_gust_knots":      fcst.get("wgst"),
                "visibility_sm":        fcst.get("visib"),
                "wx_string":            fcst.get("wxString"),
                "flight_category":      fcst.get("fltCat"),
                "raw_taf":              r.get("rawTAF"),
                "latitude":             r.get("lat"),
                "longitude":            r.get("lon"),
                "source_name":          "awc_taf",
                "ingested_at":          ingested_at,
            })

    df = pd.DataFrame(rows)
    # wind_dir_deg can be "VRB" (variable) — store as STRING
    df["wind_dir_deg"] = df["wind_dir_deg"].astype(str)
    # valid_from/valid_to are unix epoch ints — cast to Int64 (nullable) for consistency
    df["valid_from"] = pd.to_numeric(df["valid_from"], errors="coerce").astype("Int64")
    df["valid_to"] = pd.to_numeric(df["valid_to"], errors="coerce").astype("Int64")
    # visibility may be mixed string/numeric — normalize to string
    df["visibility_sm"] = df["visibility_sm"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(write_mode: str = "APPEND") -> None:
    print(f"[AWC Extractor] Fetching METAR and TAF — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # METAR
    print("  Fetching METAR...")
    metar_df = fetch_metar(hours_back=2)
    if metar_df.empty:
        print("  No METAR data returned.")
    else:
        print(f"  {len(metar_df)} METAR records")
        load_dataframe_to_bigquery(metar_df, METAR_DATASET, METAR_TABLE, write_mode)

    # TAF
    print("  Fetching TAF...")
    taf_df = fetch_taf()
    if taf_df.empty:
        print("  No TAF data returned.")
    else:
        print(f"  {len(taf_df)} TAF forecast periods")
        load_dataframe_to_bigquery(taf_df, TAF_DATASET, TAF_TABLE, write_mode)

    print("  Done.")


if __name__ == "__main__":
    run(write_mode="APPEND")
