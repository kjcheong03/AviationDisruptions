# Ingestion Log

## 2026-04-01 — Open-Meteo historical weather ingestion

### Source
Open-Meteo Archive API

### Purpose
Initial Week 1 historical weather ingestion for Role B.
Tested end-to-end flow:
API -> local CSV -> BigQuery raw table

### Date Range
2025-01-01 to 2025-01-07

### Airports
- JFK
- LAX
- SFO

### Local Output Files
- `extract/output/openmeteo_jfk_2025-01-01_2025-01-07.csv`
- `extract/output/openmeteo_lax_2025-01-01_2025-01-07.csv`
- `extract/output/openmeteo_sfo_2025-01-01_2025-01-07.csv`

### Sample Schema
Columns extracted:
- `time`
- `temperature_2m`
- `relative_humidity_2m`
- `precipitation`
- `visibility`
- `wind_speed_10m`
- `weather_code`
- `airport_code`
- `source_name`
- `extract_start_date`
- `extract_end_date`

### Row Counts
- JFK: 168 rows
- LAX: 168 rows
- SFO: 168 rows

### BigQuery Target
- Project: `is3107-aviation-pipeline`
- Dataset: `raw_weather`
- Table: `openmeteo_raw`

### Load Result
BigQuery load completed successfully.

### Notes
- Initial test used only JFK and successfully loaded 168 rows.
- Re-running with append mode caused duplicate rows.
- Loader was updated to use `WRITE_TRUNCATE` during testing to avoid duplicates.
- Weather ingestion path is now validated for:
  - extraction
  - local raw staging
  - BigQuery loading

### Issues Encountered
1. Docker setup issues on Apple Silicon
   - Fixed by adding:
     `platform: linux/amd64`
     to `docker-compose.yml`

2. Missing dbt config files
   - Created/placed:
     - `dbt/profiles.yml`
     - `dbt/gcp-service-account.json`

3. Local Python package install issue
   - Resolved by using a virtual environment:
     - `.venv`

4. Duplicate BigQuery rows during repeated test loads
   - Resolved by changing load mode from `WRITE_APPEND` to `WRITE_TRUNCATE` for testing

### Current Status
Completed:
- Open-Meteo extraction
- Local CSV creation
- BigQuery raw table load

Next:
- Generalize weather ingestion for combined multi-airport load
- Start BTS historical extraction
- Reuse loader for additional raw sources