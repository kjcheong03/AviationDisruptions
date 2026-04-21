"""
DAG: openmeteo_ingestion_dag
-----------------------------
Fetches historical weather data from the Open-Meteo Archive API for all 10
target airports and loads it into BigQuery.

Runs daily to append the previous day's hourly weather records so the
weather table stays current alongside BTS flight data.

Schedule:  0 5 * * *  (05:00 UTC daily — after midnight UTC ensures the
                        previous day is fully available in the archive API)

BigQuery target: raw_weather.openmeteo_raw
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
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def fetch_and_load_weather(**context) -> None:
    """Fetch yesterday's hourly weather for all 10 airports and append to BQ."""
    import time
    from openmeteo_extractor import fetch_openmeteo_history, AIRPORTS
    from load_to_bigquery import load_dataframe_to_bigquery
    import pandas as pd

    logical_date = context["logical_date"]
    # Fetch the day before the logical date (ensures data is available)
    target_date = (logical_date - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Fetching weather for {target_date}")

    all_dfs = []
    for i, airport_code in enumerate(AIRPORTS):
        if i > 0:
            time.sleep(3)  # avoid rate limiting
        try:
            df = fetch_openmeteo_history(airport_code, target_date, target_date)
            all_dfs.append(df)
            print(f"  {airport_code}: {len(df)} rows")
        except Exception as e:
            print(f"  {airport_code}: FAILED — {e}")

    if not all_dfs:
        raise ValueError(f"No weather data fetched for {target_date}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["time"] = pd.to_datetime(combined["time"], utc=True)
    combined["extract_start_date"] = pd.to_datetime(combined["extract_start_date"])
    combined["extract_end_date"] = pd.to_datetime(combined["extract_end_date"])
    load_dataframe_to_bigquery(combined, "raw_weather", "openmeteo_raw", write_mode="APPEND")
    print(f"Loaded {len(combined):,} rows for {target_date}")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="openmeteo_ingestion_dag",
    description="Daily Open-Meteo weather ingestion for 10 airports to BigQuery",
    default_args=default_args,
    start_date=datetime(2022, 2, 1),
    schedule_interval="0 5 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["role_b", "ingestion", "weather", "openmeteo"],
) as dag:

    fetch_weather = PythonOperator(
        task_id="fetch_and_load_weather",
        python_callable=fetch_and_load_weather,
    )
