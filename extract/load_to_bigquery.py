import os
from pathlib import Path

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT_ID = "is3107-aviation-pipeline"
# Resolve relative to the project root (one level up from this file's directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FALLBACK_SA_PATH = _PROJECT_ROOT / "dbt" / "gcp-service-account.json"


def get_bigquery_client() -> bigquery.Client:
    # Prefer the standard GCP env var (set in Docker / Airflow containers).
    # Fall back to the hardcoded local path for direct script execution.
    sa_path_str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path_str:
        sa_path = Path(sa_path_str)
    else:
        sa_path = _FALLBACK_SA_PATH

    if not sa_path.exists():
        raise FileNotFoundError(
            f"Service account file not found at: {sa_path}\n"
            "Set GOOGLE_APPLICATION_CREDENTIALS or place the key at "
            f"{_FALLBACK_SA_PATH}"
        )
    credentials = service_account.Credentials.from_service_account_file(
        str(sa_path)
    )
    return bigquery.Client(project=PROJECT_ID, credentials=credentials)


def load_csv_to_bigquery(
    csv_path: str | Path,
    dataset_id: str,
    table_id: str,
    write_mode: str = "TRUNCATE",
) -> None:
    """Load a CSV file into a BigQuery table.

    Args:
        csv_path: Path to the CSV file.
        dataset_id: BigQuery dataset (e.g. 'raw_weather', 'raw_aviation').
        table_id: BigQuery table name (e.g. 'openmeteo_raw', 'bts_raw').
        write_mode: 'TRUNCATE' replaces the table, 'APPEND' adds rows.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    client = get_bigquery_client()
    table_ref = f"{PROJECT_ID}.{dataset_id}.{table_id}"

    disposition_map = {
        "TRUNCATE": bigquery.WriteDisposition.WRITE_TRUNCATE,
        "APPEND": bigquery.WriteDisposition.WRITE_APPEND,
    }
    write_disposition = disposition_map.get(write_mode.upper())
    if write_disposition is None:
        raise ValueError(f"write_mode must be 'TRUNCATE' or 'APPEND', got: {write_mode!r}")

    print(f"Loading {csv_path.name} -> {table_ref} (mode={write_mode})")

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
    )

    with open(csv_path, "rb") as source_file:
        load_job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

    load_job.result()

    table = client.get_table(table_ref)
    print(f"Done. Table {table_ref} now has {table.num_rows:,} rows.")
    print("Schema:")
    for field in table.schema:
        print(f"  {field.name}: {field.field_type}")


def load_dataframe_to_bigquery(
    df: pd.DataFrame,
    dataset_id: str,
    table_id: str,
    write_mode: str = "APPEND",
) -> None:
    """Load a pandas DataFrame into a BigQuery table."""
    client = get_bigquery_client()
    table_ref = f"{PROJECT_ID}.{dataset_id}.{table_id}"

    disposition_map = {
        "TRUNCATE": bigquery.WriteDisposition.WRITE_TRUNCATE,
        "APPEND": bigquery.WriteDisposition.WRITE_APPEND,
    }
    write_disposition = disposition_map.get(write_mode.upper())
    if write_disposition is None:
        raise ValueError(f"write_mode must be 'TRUNCATE' or 'APPEND', got: {write_mode!r}")

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
    )

    load_job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    load_job.result()
    print(f"Loaded {len(df):,} rows -> {table_ref} (mode={write_mode})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python load_to_bigquery.py <csv_path> <dataset_id> <table_id> [TRUNCATE|APPEND]")
        print("Example: python load_to_bigquery.py extract/output/openmeteo_all.csv raw_weather openmeteo_raw TRUNCATE")
        sys.exit(1)

    csv_path = sys.argv[1]
    dataset_id = sys.argv[2]
    table_id = sys.argv[3]
    write_mode = sys.argv[4] if len(sys.argv) > 4 else "TRUNCATE"

    load_csv_to_bigquery(csv_path, dataset_id, table_id, write_mode)
