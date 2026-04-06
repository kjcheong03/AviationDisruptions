"""
DAG: awc_polling_dag
---------------------
Polls the Aviation Weather Center (AWC) API for near-real-time METAR and TAF
reports for all 10 target airports and appends them to BigQuery.

METAR: observed weather updated every ~20 minutes at major airports.
TAF:   terminal forecasts issued every ~6 hours.

We poll every 30 minutes to capture updates promptly without hammering the API.

Schedule:  */30 * * * *  (every 30 minutes)

BigQuery targets:
  raw_weather.metar_raw   — current observations
  raw_weather.taf_raw     — terminal forecasts
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

EXTRACT_DIR = str(Path(__file__).resolve().parents[2] / "extract")
if EXTRACT_DIR not in sys.path:
    sys.path.insert(0, EXTRACT_DIR)

# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------

default_args = {
    "owner": "role_b",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def poll_metar(**context) -> None:
    """Fetch latest METAR observations and append to BigQuery."""
    from awc_extractor import fetch_metar
    from load_to_bigquery import load_dataframe_to_bigquery

    df = fetch_metar(hours_back=1)
    if df.empty:
        print("No METAR data returned — skipping load.")
        return

    print(f"Fetched {len(df)} METAR records")
    load_dataframe_to_bigquery(df, "raw_weather", "metar_raw", write_mode="APPEND")


def poll_taf(**context) -> None:
    """Fetch latest TAF forecasts and append to BigQuery."""
    from awc_extractor import fetch_taf
    from load_to_bigquery import load_dataframe_to_bigquery

    df = fetch_taf()
    if df.empty:
        print("No TAF data returned — skipping load.")
        return

    print(f"Fetched {len(df)} TAF forecast periods")
    load_dataframe_to_bigquery(df, "raw_weather", "taf_raw", write_mode="APPEND")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="awc_polling_dag",
    description="Near-real-time AWC METAR/TAF polling every 30 minutes",
    default_args=default_args,
    start_date=datetime(2026, 4, 6),
    schedule_interval="*/30 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["role_b", "realtime", "awc", "metar", "taf"],
) as dag:

    fetch_metar_task = PythonOperator(
        task_id="poll_metar",
        python_callable=poll_metar,
    )

    fetch_taf_task = PythonOperator(
        task_id="poll_taf",
        python_callable=poll_taf,
    )

    # METAR and TAF are independent — run in parallel
    [fetch_metar_task, fetch_taf_task]
