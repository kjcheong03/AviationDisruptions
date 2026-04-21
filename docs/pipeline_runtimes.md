# Pipeline Runtime Documentation
**Week 3 — Airflow DAG Evidence**
Role B | Captured: [DATE]

---

## How to Capture Screenshots

After running `docker-compose up -d` and visiting `http://localhost:8080` (login: admin / admin):

1. **Graph View** — Click a DAG → click the "Graph" tab → screenshot the full task dependency graph
2. **Tree View / Grid View** — Click a DAG → click the "Grid" tab → screenshot showing run history and task states
3. **Task Instance Log** — Click a green/coloured task box → "Log" → screenshot showing duration and success message

---

## DAG 1: `bts_ingestion_dag`

**Schedule:** `0 6 5 * *` (monthly, 5th of each month at 06:00 UTC)
**Purpose:** Downloads BTS On-Time Performance ZIP for the prior month and appends to `raw_aviation.bts_raw` in BigQuery.

| Step | Task ID | Status | Duration |
|------|---------|--------|----------|
| 1 | `download_and_load_bts` | ✅ success | [fill in] |

**Total DAG runtime:** [fill in]

**Screenshot files:**
- `docs/screenshots/bts_dag_graph.png`
- `docs/screenshots/bts_dag_grid.png`

---

## DAG 2: `openmeteo_ingestion_dag`

**Schedule:** `0 5 * * *` (daily at 05:00 UTC)
**Purpose:** Fetches yesterday's hourly weather for 10 airports and appends to `raw_weather.openmeteo_raw`.

| Step | Task ID | Status | Duration |
|------|---------|--------|----------|
| 1 | `fetch_and_load_weather` | ✅ success | [fill in] |

**Total DAG runtime:** [fill in]

**Screenshot files:**
- `docs/screenshots/openmeteo_dag_graph.png`
- `docs/screenshots/openmeteo_dag_grid.png`

---

## DAG 3: `awc_polling_dag`

**Schedule:** `*/30 * * * *` (every 30 minutes)
**Purpose:** Polls AWC API for METAR observations and TAF forecasts, appends to `raw_weather.metar_raw` and `raw_weather.taf_raw`.

| Step | Task ID | Status | Duration |
|------|---------|--------|----------|
| 1 | `poll_metar` | ✅ success | [fill in] |
| 2 | `poll_taf` | ✅ success | [fill in] |

**Note:** `poll_metar` and `poll_taf` run in parallel (no dependency between them).

**Total DAG runtime:** [fill in]

**Screenshot files:**
- `docs/screenshots/awc_dag_graph.png`
- `docs/screenshots/awc_dag_grid.png`

---

## Notes

- All three DAGs are tagged `role_b` and visible under the "role_b" tag filter in the Airflow UI.
- `max_active_runs=1` on each DAG prevents overlapping executions.
- BTS DAG uses `catchup=False`; to backfill historical months set `catchup=True` and trigger manually.
- AWC DAG `start_date` is `2026-04-06` (Week 3 deployment date).
