"""
BTS On-Time Performance Pipeline
---------------------------------
Downloads monthly ZIP files from BTS TranStats (Feb 2022 – Jan 2026),
extracts the needed columns, and loads them to BigQuery raw_aviation.bts_raw.

Each month is processed and loaded individually (APPEND mode after first batch)
so we never hold more than one month of CSV on disk at a time.

Usage:
    python extract/bts_pipeline.py
"""

import io
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests

from load_to_bigquery import get_bigquery_client, PROJECT_ID
from google.cloud import bigquery

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ID = "raw_aviation"
TABLE_ID = "bts_raw"

# Feb 2022 → Jan 2026  (inclusive)
START_YEAR, START_MONTH = 2022, 2
END_YEAR, END_MONTH = 2026, 1

# BTS PREZIP download URL pattern (no leading zero on month)
BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Actual column names as they appear in BTS PREZIP CSV files
PREFERRED_COLUMNS = [
    "FlightDate",
    "Year",
    "Month",
    "DayofMonth",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Tail_Number",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "AirTime",
    "Distance",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _months_in_range(start_year, start_month, end_year, end_month):
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1


def _download_and_parse(year: int, month: int) -> pd.DataFrame | None:
    url = BTS_URL.format(year=year, month=month)
    print(f"  Downloading {year}-{month:02d} from BTS...", end=" ", flush=True)

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
        return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

    print(f"{len(response.content) / 1_048_576:.1f} MB downloaded.", end=" ", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # Extract the first CSV found in the archive
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                print("No CSV found in ZIP.")
                return None
            csv_name = csv_names[0]
            zf.extract(csv_name, tmp_path)

        extracted_csv = tmp_path / csv_name
        df = pd.read_csv(extracted_csv, low_memory=False)

    # Keep only columns that exist in this month's file
    available = [c for c in PREFERRED_COLUMNS if c in df.columns]
    df = df[available].copy()

    # Standardise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    df["source_name"] = "bts"
    df["ingestion_stage"] = "raw"

    print(f"{len(df):,} rows, {len(df.columns)} columns.")
    return df


def _load_to_bigquery(df: pd.DataFrame, write_mode: str) -> None:
    client = get_bigquery_client()
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    disposition_map = {
        "TRUNCATE": bigquery.WriteDisposition.WRITE_TRUNCATE,
        "APPEND": bigquery.WriteDisposition.WRITE_APPEND,
    }

    job_config = bigquery.LoadJobConfig(
        write_disposition=disposition_map[write_mode],
        autodetect=True,
    )
    load_job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    load_job.result()
    print(f"    -> Loaded to {table_ref} ({write_mode})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    months = list(_months_in_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH))
    total = len(months)
    print(f"BTS Pipeline: {total} months to process ({START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d})\n")

    total_rows = 0
    failed = []

    for i, (year, month) in enumerate(months):
        print(f"[{i+1}/{total}] {year}-{month:02d}")
        df = _download_and_parse(year, month)

        if df is None or df.empty:
            failed.append((year, month))
            continue

        write_mode = "TRUNCATE" if i == 0 else "APPEND"
        _load_to_bigquery(df, write_mode)
        total_rows += len(df)

    print(f"\nDone. {total_rows:,} total rows loaded to {DATASET_ID}.{TABLE_ID}")
    if failed:
        print(f"Failed months: {failed}")


if __name__ == "__main__":
    main()
