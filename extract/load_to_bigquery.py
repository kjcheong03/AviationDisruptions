from pathlib import Path
import os

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT_ID = "is3107-aviation-pipeline"
DATASET_ID = "raw_weather"
TABLE_ID = "openmeteo_raw"

CSV_PATH = Path("extract/output/openmeteo_jfk_2025-01-01_2025-01-07.csv")
SERVICE_ACCOUNT_PATH = Path("dbt/gcp-service-account.json")


def get_bigquery_client() -> bigquery.Client:
    if not SERVICE_ACCOUNT_PATH.exists():
        raise FileNotFoundError(
            f"Service account file not found at: {SERVICE_ACCOUNT_PATH}"
        )

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH
    )

    client = bigquery.Client(
        project=PROJECT_ID,
        credentials=credentials,
    )
    return client


def load_csv_to_bigquery() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

    client = get_bigquery_client()

    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded local CSV with {len(df)} rows and {len(df.columns)} columns.")
    print(f"Target table: {table_ref}")

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
    )

    with open(CSV_PATH, "rb") as source_file:
        load_job = client.load_table_from_file(
            source_file,
            table_ref,
            job_config=job_config,
        )

    load_job.result()

    table = client.get_table(table_ref)
    print("BigQuery load completed successfully.")
    print(f"Table now has {table.num_rows} rows.")
    print(f"Schema:")
    for field in table.schema:
        print(f"  - {field.name}: {field.field_type}")


if __name__ == "__main__":
    load_csv_to_bigquery()