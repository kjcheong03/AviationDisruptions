"""
BigQuery data loader for the AviRisk dashboard.
All queries run aggregations server-side — only summary rows pulled into pandas.
Falls back gracefully to None when credentials are missing.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

ROOT = Path(__file__).parent.parent
KEY_PATH = ROOT / "dbt" / "gcp-service-account.json"
PROJECT = "is3107-aviation-pipeline"
BTS = f"`{PROJECT}.raw_aviation.bts_raw`"


def _client():
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
        if KEY_PATH.exists():
            creds = service_account.Credentials.from_service_account_file(
                str(KEY_PATH),
                scopes=["https://www.googleapis.com/auth/bigquery"],
            )
            return bigquery.Client(project=PROJECT, credentials=creds)
        return bigquery.Client(project=PROJECT)   # falls back to ADC
    except Exception as e:
        print(f"[BQ] _client() failed: {e}")
        return None


def is_available() -> bool:
    c = _client()
    return c is not None


def _run(sql: str) -> Optional[pd.DataFrame]:
    client = _client()
    if client is None:
        return None
    try:
        return client.query(sql).to_dataframe()
    except Exception as e:
        print(f"[BQ] Query failed: {e}")
        return None


def _where(months=None, airlines=None, airports=None, date_range=None,
           extra="Cancelled = 0 AND DepDelay IS NOT NULL") -> str:
    """Build a WHERE clause from sidebar filter selections.
    date_range: tuple of (start_date, end_date) — python date / datetime / 'YYYY-MM-DD' string.
    """
    clauses = [extra] if extra else []
    if date_range:
        start, end = date_range
        clauses.append(
            f"CAST(FlightDate AS DATE) BETWEEN DATE('{start}') AND DATE('{end}')"
        )
    if months:
        month_list = ", ".join(str(m) for m in months)
        clauses.append(f"EXTRACT(MONTH FROM CAST(FlightDate AS DATE)) IN ({month_list})")
    if airlines:
        al = ", ".join(f"'{a}'" for a in airlines)
        clauses.append(f"Reporting_Airline IN ({al})")
    if airports:
        ap = ", ".join(f"'{a}'" for a in airports)
        clauses.append(f"Origin IN ({ap})")
    return "WHERE " + " AND ".join(clauses) if clauses else ""


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_kpis(months=None, airlines=None, airports=None, date_range=None) -> Optional[dict]:
    w = _where(months, airlines, airports, date_range, extra=None)
    sql = f"""
    SELECT
      COUNT(*)                                                        AS total_flights,
      COUNTIF(Cancelled = 1.0)                                        AS cancelled,
      COUNTIF(DepDelay > 15 AND Cancelled != 1.0)                     AS delayed,
      SAFE_DIVIDE(
        COUNTIF(DepDelay > 15 AND Cancelled != 1.0),
        COUNTIF(Cancelled != 1.0)
      )                                                               AS delay_rate
    FROM {BTS}
    {w}
    """
    df = _run(sql)
    if df is None or df.empty:
        return None
    r = df.iloc[0]
    return {k: (int(r[k]) if k != "delay_rate" else float(r[k])) for k in r.index}


def load_delay_by_hour(months=None, airlines=None, airports=None, date_range=None) -> Optional[pd.DataFrame]:
    w = _where(months, airlines, airports, date_range)
    sql = f"""
    SELECT
      CAST(FLOOR(CRSDepTime / 100) AS INT64)        AS hour,
      SAFE_DIVIDE(COUNTIF(DepDelay > 15), COUNT(*)) AS delay_rate,
      COUNT(*)                                      AS n
    FROM {BTS}
    {w}
      AND CRSDepTime IS NOT NULL
    GROUP BY hour
    HAVING hour BETWEEN 0 AND 23
    ORDER BY hour
    """
    return _run(sql)


def load_delay_by_airline(months=None, airports=None, date_range=None) -> Optional[pd.DataFrame]:
    # note: no airline filter here — we want to compare all airlines
    w = _where(months, None, airports, date_range)
    sql = f"""
    SELECT
      Reporting_Airline                              AS airline_code,
      SAFE_DIVIDE(COUNTIF(DepDelay > 15), COUNT(*)) AS delay_rate,
      COUNT(*)                                      AS n
    FROM {BTS}
    {w}
    GROUP BY airline_code
    HAVING n >= 500
    ORDER BY delay_rate DESC
    LIMIT 20
    """
    return _run(sql)


def load_delay_by_airport(months=None, airlines=None, date_range=None) -> Optional[pd.DataFrame]:
    w = _where(months, airlines, None, date_range)
    sql = f"""
    SELECT
      Origin                                         AS airport,
      SAFE_DIVIDE(COUNTIF(DepDelay > 15), COUNT(*))  AS delay_rate,
      COUNT(*)                                       AS n
    FROM {BTS}
    {w}
    GROUP BY airport
    HAVING n >= 300
    ORDER BY delay_rate DESC
    LIMIT 30
    """
    return _run(sql)


def load_monthly_trend(airlines=None, airports=None, date_range=None, months=None) -> Optional[pd.DataFrame]:
    w = _where(months, airlines, airports, date_range)
    sql = f"""
    SELECT
      EXTRACT(YEAR  FROM CAST(FlightDate AS DATE)) AS year,
      EXTRACT(MONTH FROM CAST(FlightDate AS DATE)) AS month,
      SAFE_DIVIDE(COUNTIF(DepDelay > 15), COUNT(*)) AS delay_rate,
      COUNT(*)                                      AS n
    FROM {BTS}
    {w}
    GROUP BY year, month
    ORDER BY year, month
    """
    df = _run(sql)
    if df is None:
        return None
    df["year"]  = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["label"] = df.apply(
        lambda r: pd.Timestamp(year=int(r["year"]), month=int(r["month"]), day=1).strftime("%b %Y"), axis=1
    )
    return df


def load_delay_causes(months=None, airlines=None, airports=None, date_range=None) -> Optional[pd.DataFrame]:
    w = _where(months, airlines, airports, date_range, extra="Cancelled != 1.0")
    sql = f"""
    SELECT
      SUM(CarrierDelay)      AS Carrier,
      SUM(WeatherDelay)      AS Weather,
      SUM(NASDelay)          AS NAS,
      SUM(SecurityDelay)     AS Security,
      SUM(LateAircraftDelay) AS `Late Aircraft`
    FROM {BTS}
    {w}
    """
    df = _run(sql)
    if df is None:
        return None
    return df.melt(var_name="Cause", value_name="Total Min").dropna()


def load_delay_distribution_sample(months=None, airlines=None, airports=None,
                                     date_range=None, n: int = 80_000) -> Optional[pd.DataFrame]:
    w = _where(months, airlines, airports, date_range)
    sql = f"""
    SELECT DepDelay
    FROM {BTS}
    {w}
      AND DepDelay BETWEEN -60 AND 180
    ORDER BY RAND()
    LIMIT {n}
    """
    return _run(sql)


def load_airline_list() -> list[str]:
    sql = f"""
    SELECT DISTINCT Reporting_Airline AS airline
    FROM {BTS}
    WHERE Reporting_Airline IS NOT NULL
    ORDER BY airline
    """
    df = _run(sql)
    return sorted(df["airline"].tolist()) if df is not None else []


METAR = f"`{PROJECT}.raw_weather.metar_raw`"


def load_latest_metar() -> Optional[pd.DataFrame]:
    sql = f"""
    SELECT iata_code, obs_time, temp_c, wind_speed_knots, wind_dir_deg,
           visibility_sm, flight_category, dewpoint_c, ingested_at
    FROM (
      SELECT *,
        ROW_NUMBER() OVER (PARTITION BY iata_code ORDER BY ingested_at DESC) AS rn
      FROM {METAR}
      WHERE iata_code IS NOT NULL
    )
    WHERE rn = 1
    ORDER BY iata_code
    """
    return _run(sql)


def load_cancellations_by_airline(months=None, airports=None, date_range=None) -> Optional[pd.DataFrame]:
    w = _where(months, None, airports, date_range, extra=None)
    sql = f"""
    SELECT
      Reporting_Airline AS airline_code,
      COUNTIF(Cancelled = 1.0) AS cancelled,
      COUNT(*) AS total,
      SAFE_DIVIDE(COUNTIF(Cancelled = 1.0), COUNT(*)) AS cancel_rate
    FROM {BTS}
    {w}
    GROUP BY airline_code
    HAVING total >= 500
    ORDER BY cancel_rate DESC
    LIMIT 20
    """
    return _run(sql)


def load_cancellations_by_airport(months=None, airlines=None, date_range=None) -> Optional[pd.DataFrame]:
    w = _where(months, airlines, None, date_range, extra=None)
    sql = f"""
    SELECT
      Origin AS airport,
      COUNTIF(Cancelled = 1.0) AS cancelled,
      COUNT(*) AS total,
      SAFE_DIVIDE(COUNTIF(Cancelled = 1.0), COUNT(*)) AS cancel_rate
    FROM {BTS}
    {w}
    GROUP BY airport
    HAVING total >= 300
    ORDER BY cancel_rate DESC
    LIMIT 20
    """
    return _run(sql)


def load_cancellation_codes(months=None, airlines=None, airports=None, date_range=None) -> Optional[pd.DataFrame]:
    w = _where(months, airlines, airports, date_range,
               extra="Cancelled = 1.0 AND CancellationCode IS NOT NULL AND CancellationCode != ''")
    sql = f"""
    SELECT CancellationCode AS code, COUNT(*) AS n
    FROM {BTS}
    {w}
    GROUP BY code
    ORDER BY n DESC
    """
    return _run(sql)


def load_cancellation_trend(airlines=None, airports=None, date_range=None, months=None) -> Optional[pd.DataFrame]:
    w = _where(months, airlines, airports, date_range, extra=None)
    sql = f"""
    SELECT
      EXTRACT(YEAR  FROM CAST(FlightDate AS DATE)) AS year,
      EXTRACT(MONTH FROM CAST(FlightDate AS DATE)) AS month,
      SAFE_DIVIDE(COUNTIF(Cancelled = 1.0), COUNT(*)) AS cancel_rate,
      COUNT(*) AS n
    FROM {BTS}
    {w}
    GROUP BY year, month
    ORDER BY year, month
    """
    df = _run(sql)
    if df is None:
        return None
    df["year"]  = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["label"] = df.apply(
        lambda r: pd.Timestamp(year=int(r["year"]), month=int(r["month"]), day=1).strftime("%b %Y"), axis=1
    )
    return df


def load_delay_by_dow_hour(months=None, airlines=None, airports=None, date_range=None) -> Optional[pd.DataFrame]:
    w = _where(months, airlines, airports, date_range)
    sql = f"""
    SELECT
      EXTRACT(DAYOFWEEK FROM CAST(FlightDate AS DATE)) AS dow,
      CAST(FLOOR(CRSDepTime / 100) AS INT64)           AS hour,
      SAFE_DIVIDE(COUNTIF(DepDelay > 15), COUNT(*))    AS delay_rate,
      COUNT(*)                                         AS n
    FROM {BTS}
    {w}
      AND CRSDepTime IS NOT NULL
    GROUP BY dow, hour
    HAVING hour BETWEEN 0 AND 23 AND n >= 30
    ORDER BY dow, hour
    """
    df = _run(sql)
    if df is None:
        return None
    dow_map = {1:"Sun", 2:"Mon", 3:"Tue", 4:"Wed", 5:"Thu", 6:"Fri", 7:"Sat"}
    df["dow_label"] = df["dow"].map(dow_map)
    return df
