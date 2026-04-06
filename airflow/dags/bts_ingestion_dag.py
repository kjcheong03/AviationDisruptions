"""
DAG: bts_ingestion_dag
-----------------------
Downloads and loads BTS On-Time Performance historical data into BigQuery.
Runs once per month on the 5th (data for the prior month is usually available by then).

Schedule:  0 6 5 * *  (06:00 UTC on the 5th of every month)
Backfill:  Set start_date to 2022-02-01 and run with catchup=True to re-load
           the full historical range.

BigQuery target: raw_aviation.bts_raw
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

# Allow imports from extract/
EXTRACT_DIR = str(Path(__file__).resolve().parents[2] / "extract")
if EXTRACT_DIR not in sys.path:
    sys.path.insert(0, EXTRACT_DIR)

# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------

default_args = {
    "owner": "role_b",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def download_and_load_bts(**context) -> None:
    """Download the BTS ZIP for the logical month and load to BigQuery."""
    from bts_pipeline import _download_and_parse, _load_to_bigquery

    # logical_date is the first day of the month being processed
    logical_date = context["logical_date"]
    year = logical_date.year
    month = logical_date.month

    print(f"Processing BTS data for {year}-{month:02d}")
    df = _download_and_parse(year, month)

    if df is None or df.empty:
        raise ValueError(f"No data returned for {year}-{month:02d}")

    # Always APPEND for scheduled runs; the pipeline deduplicates via BQ partitioning
    _load_to_bigquery(df, write_mode="APPEND")
    print(f"Loaded {len(df):,} rows for {year}-{month:02d}")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="bts_ingestion_dag",
    description="Monthly BTS On-Time Performance ingestion to BigQuery",
    default_args=default_args,
    start_date=datetime(2022, 2, 1),
    schedule_interval="0 6 5 * *",
    catchup=False,        # set True to backfill historical months
    max_active_runs=1,
    tags=["role_b", "ingestion", "bts"],
) as dag:

    ingest_bts = PythonOperator(
        task_id="download_and_load_bts",
        python_callable=download_and_load_bts,
    )
