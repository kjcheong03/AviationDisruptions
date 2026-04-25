# AviationDisruptions

End-to-end cloud-native ELT pipeline that fuses BTS On-Time Performance, Open-Meteo, and AWC METAR/TAF data in BigQuery, then trains Gradient Boosting models to predict per-flight delay risk and surfaces the results through a Streamlit dashboard.

## How to run

Requires Docker Desktop, Python 3.10+, and a GCP service account key at `dbt/gcp-service-account.json`.

```bash
# 1. Start Airflow + Postgres + dbt containers
docker compose up -d

# 2. Train models and generate risk scores (~30 min)
python scripts/generate_risk_scores.py

# 3. Launch the dashboard
streamlit run dashboard/app.py
```
