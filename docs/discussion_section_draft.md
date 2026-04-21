# Discussion
**Role B contribution — Week 4**

---

## 5. Discussion

### 5.1 Pipeline Performance Evaluation

The data pipeline was designed around three distinct ingestion cadences, each matched to the temporal characteristics of its source data. This section evaluates the observed performance of each pipeline component and the architectural decisions that drove them.

**Historical Batch Ingestion (BTS On-Time Performance)**
The BTS ingestion pipeline processes approximately 500,000–600,000 flight records per monthly extract, covering all domestic US carrier operations. A single monthly batch load to BigQuery (`raw_aviation.bts_raw`) completes in under [X] minutes, the majority of which is consumed by the ZIP download from the BTS transstats portal rather than the BigQuery write operation itself. BigQuery's columnar storage and streaming insert optimisations mean that even at 25 million rows — the full four-year historical range loaded in Week 1 — the load completed without timeout or memory pressure on a standard laptop. The use of `WRITE_APPEND` mode with Airflow's `catchup=False` default ensures that the scheduler only processes the most recent month in steady-state operation; historical backfills are triggered explicitly, preventing accidental re-ingestion.

**Near-Real-Time AWC Polling (METAR / TAF)**
The AWC polling DAG runs on a 30-minute cadence (`*/30 * * * *`), balancing data freshness against API rate limits. METAR observations at major US airports are updated approximately every 20 minutes, meaning our 30-minute poll captures each new observation within one cycle. The two tasks (`poll_metar` and `poll_taf`) are independent and execute in parallel within the same DAG run, reducing total wall-clock time compared to a sequential design. Each individual poll-and-load cycle completes in under [X] seconds, well within the 30-minute window. The `retries=3` configuration with a 2-minute retry delay provides resilience against transient AWC API unavailability without flooding the scheduler queue.

**Daily Weather Archive Ingestion (Open-Meteo)**
The Open-Meteo DAG runs daily at 05:00 UTC, fetching the previous day's hourly weather records for all 10 target airports. The 3-second inter-airport delay (`time.sleep(3)`) was introduced to respect Open-Meteo's fair-use rate limits for the free-tier archive API. Total runtime per daily run is approximately [X] seconds, producing 240 rows (10 airports × 24 hours) appended to `raw_weather.openmeteo_raw`.

**Overall Orchestration**
Airflow's LocalExecutor was selected over CeleryExecutor given our single-node deployment. All three DAGs are tagged `role_b` and are visible as a unified group in the Airflow UI, enabling at-a-glance monitoring. The use of `max_active_runs=1` on every DAG prevents resource contention during catch-up or manual backfill scenarios.

---

### 5.2 Integration of the Predictive Model into Business Processes

The machine learning models trained by Role C on the `fct_flights_weather` fact table produce short-horizon disruption-risk scores for each departing flight. This section discusses how these scores connect to operational business decisions and where the pipeline architecture supports that integration.

**Risk Score Delivery**
The predictive model outputs a probability of significant departure delay (≥15 minutes) for each flight in the rolling planning window. These scores are surfaced through the Streamlit dashboard built by Role C, which provides two views: a historical trend analysis and a near-real-time risk indicator updated on each AWC polling cycle. From an operational standpoint, airline operations control centres and airport ground handlers can consume these risk scores up to 2–3 hours before a flight's scheduled departure — the window within which ground stop mitigation, gate reassignments, and crew re-rosters are still actionable.

**Pipeline-to-Model Latency**
The end-to-end latency from a weather event occurring at an airport to that event influencing the model's risk score is determined by the AWC polling cadence (≤30 minutes) plus the dbt model refresh cycle. Once new METAR data lands in `raw_weather.metar_raw`, the `stg_weather` and `fct_flights_weather` dbt models can be re-executed to incorporate the updated conditions. In production, this refresh would be orchestrated as a fourth Airflow DAG triggered downstream of the AWC polling DAG, keeping end-to-end latency under one hour — well within the operational planning window.

**Business Value of Lag Features**
The time-series lag features engineered in `fct_flights_weather` — particularly `lag_1_tail_arr_delay_mins` (aircraft propagation chain) and `rolling_6_flight_origin_delay_avg` (origin congestion) — encode the cascading nature of aviation delays that simple point-in-time weather features cannot capture. From a business process perspective, this means the model can distinguish between an isolated weather event at an airport and a systemic network disruption propagating through hub-and-spoke connections. Airline network planners can use the hub centrality scores computed by Role C via NetworkX in tandem with the delay-risk scores to prioritise recovery resources at the highest-centrality hubs first — the nodes whose disruption carries the highest downstream cost across the network.

**Limitations and Future Work**
The current pipeline does not yet integrate OPSNET traffic count data (`raw_aviation.opsnet_raw`) into the dbt fact table. OPSNET's volume-of-operations metrics would add a direct measure of airspace congestion that complements the weather and flight-history features currently used. This integration is deferred as a near-term enhancement. Additionally, the model currently operates at the flight level; a future extension would aggregate risk scores to the airport-hour grain for direct use in gate and slot allocation systems, reducing the cognitive load on operations staff who currently review flight-level outputs.

---

*Word count: ~[X]. Adjust figures in brackets once actual runtime numbers are captured from the Airflow UI.*
