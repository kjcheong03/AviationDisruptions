"""
Week 3 Role C: Train finalized ML models and generate short-horizon disruption-risk scores.
Pulls feature-engineered data directly from analytics_staging.fct_flights_weather in BigQuery
(Role A's dbt fact table), adds NetworkX centrality features, trains three classifiers,
and saves scored flight records for the Streamlit dashboard.

Runtime: ~20-30 min (BQ download + GradientBoosting training on ~5M rows)
Run from project root: python scripts/generate_risk_scores.py
"""

import json
import sys
import joblib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

ROOT      = Path(__file__).parent.parent
CACHE     = ROOT / "eda" / "cache"
MODELS    = ROOT / "models"
KEY_PATH  = ROOT / "dbt" / "gcp-service-account.json"
PROJECT   = "is3107-aviation-pipeline"
FCT_TABLE = f"`{PROJECT}.analytics_staging.fct_flights_weather`"
BTS_TABLE = f"`{PROJECT}.raw_aviation.bts_raw`"

MODELS.mkdir(exist_ok=True)

DELAY_THRESHOLD = 15

WEATHER_MEDIANS = {
    "temperature_c":       24.50,
    "precipitation_mm":     0.00,
    "wind_speed_knots":     5.00,
    "cloud_cover_total_pct":22.00,
    "cloud_cover_low_pct":  0.00,
}

# Features that come pre-built from Role A's dbt fct_flights_weather
FEATURE_COLS = [
    "flight_month", "day_of_week", "scheduled_dep_hour", "Distance",
    "temperature_c", "precipitation_mm", "wind_speed_knots",
    "cloud_cover_total_pct", "cloud_cover_low_pct",
    "rolling_6_flight_origin_delay_avg", "rolling_3_flight_weather_delay_flag",
    "wind_speed_delta", "lag_1_tail_arr_delay_mins",
    "origin_pagerank", "origin_betweenness", "dest_pagerank",
]


# ── BigQuery client ───────────────────────────────────────────────────────────
def bq_client() -> bigquery.Client:
    if not KEY_PATH.exists():
        sys.exit(f"ERROR: Service account key not found at {KEY_PATH}")
    creds = service_account.Credentials.from_service_account_file(
        str(KEY_PATH),
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )
    return bigquery.Client(project=PROJECT, credentials=creds)


def run_query(client: bigquery.Client, sql: str) -> pd.DataFrame:
    print(f"  Running BQ query ({len(sql)} chars)…")
    return client.query(sql).to_dataframe()


# ── Step 1: Load feature data from fct_flights_weather ───────────────────────
def load_from_bigquery(client: bigquery.Client, sample_pct: int = 20) -> pd.DataFrame:
    """
    Pull the pre-engineered feature columns from Role A's dbt fact table.
    Uses a deterministic ~sample_pct% sample to keep memory and runtime manageable
    while covering all 4 years of data (vs 3-month local cache).
    """
    print(f"Loading fct_flights_weather from BigQuery (~{sample_pct}% sample)…")
    sql = f"""
    SELECT
        date_id                                        AS flight_date,
        flight_month,
        day_of_week,
        hour_id                                        AS scheduled_dep_hour,
        airline_id                                     AS airline_code,
        tail_number,
        origin_airport_id                              AS Origin,
        dest_airport_id                                AS Dest,
        distance_miles                                 AS Distance,

        -- weather (NULL where Open-Meteo has no coverage → imputed later)
        temperature_c,
        precipitation_mm,
        wind_speed_knots,
        cloud_cover_total_pct,
        cloud_cover_low_pct,

        -- lag / time-series features (pre-built by Role A dbt)
        COALESCE(lag_1_tail_arr_delay_mins,        0)  AS lag_1_tail_arr_delay_mins,
        COALESCE(rolling_6_flight_origin_delay_avg, 0) AS rolling_6_flight_origin_delay_avg,
        COALESCE(wind_speed_delta,                  0) AS wind_speed_delta,
        COALESCE(rolling_3_flight_weather_delay_flag,0) AS rolling_3_flight_weather_delay_flag,

        -- targets
        dep_delay_minutes,
        arr_delay_minutes,
        weather_delay_mins

    FROM {FCT_TABLE}
    WHERE is_cancelled = FALSE
      AND dep_delay_minutes IS NOT NULL
      AND MOD(ABS(FARM_FINGERPRINT(flight_id)), 100) < {sample_pct}
    """
    df = run_query(client, sql)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Impute missing weather with medians
    for col, val in WEATHER_MEDIANS.items():
        df[col] = df[col].fillna(val)

    # Cap delays
    df["dep_delay_minutes"] = df["dep_delay_minutes"].clip(-720, 720)
    return df.reset_index(drop=True)


# ── Step 2: Build network + centrality from BQ ───────────────────────────────
def build_network(client: bigquery.Client) -> pd.DataFrame:
    """Build aviation network from full 4-year route data and compute centrality."""
    print("Building aviation network from BigQuery…")
    sql = f"""
    SELECT origin_airport_id AS Origin, dest_airport_id AS Dest, COUNT(*) AS n
    FROM {FCT_TABLE}
    WHERE is_cancelled = FALSE
    GROUP BY 1, 2
    """
    routes = run_query(client, sql)
    print(f"  {len(routes):,} routes loaded")

    G = nx.DiGraph()
    for _, r in routes.iterrows():
        G.add_edge(r["Origin"], r["Dest"], weight=int(r["n"]))
    print(f"  Network: {G.number_of_nodes()} airports, {G.number_of_edges()} routes")

    print("  Computing PageRank…")
    pagerank = nx.pagerank(G, weight="weight")

    print("  Computing Betweenness Centrality (k=100 approximate)…")
    betweenness = nx.betweenness_centrality(G, k=100, normalized=True, weight="weight", seed=42)

    metrics = pd.DataFrame({
        "airport":    list(pagerank.keys()),
        "pagerank":   [pagerank[a] for a in pagerank],
        "betweenness":[betweenness.get(a, 0) for a in pagerank],
    })
    metrics.to_csv(MODELS / "centrality_metrics.csv", index=False)
    print(f"  Saved centrality for {len(metrics)} airports → models/centrality_metrics.csv")
    return metrics


# ── Step 3: Add centrality features ──────────────────────────────────────────
def add_centrality(df: pd.DataFrame, centrality: pd.DataFrame) -> pd.DataFrame:
    print("Adding centrality features…")
    cent = centrality.set_index("airport")
    df["origin_pagerank"]    = df["Origin"].map(cent["pagerank"]).fillna(0)
    df["origin_betweenness"] = df["Origin"].map(cent["betweenness"]).fillna(0)
    df["dest_pagerank"]      = df["Dest"].map(cent["pagerank"]).fillna(0)
    return df


# ── Step 4: Train models ──────────────────────────────────────────────────────
def train_models(df: pd.DataFrame):
    print(f"\nTraining models on {len(df):,} rows…")
    df["is_delayed"] = (df["dep_delay_minutes"] > DELAY_THRESHOLD).astype(int)
    X = df[FEATURE_COLS].fillna(0)
    y = df["is_delayed"]
    print(f"  Delay rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    print("  [1/3] Logistic Regression…")
    scaler = StandardScaler()
    Xs_tr  = scaler.fit_transform(X_train)
    Xs_te  = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1)
    lr.fit(Xs_tr, y_train)
    p = lr.predict_proba(Xs_te)[:, 1]
    results["Logistic Regression"] = dict(
        accuracy=accuracy_score(y_test, lr.predict(Xs_te)),
        f1=f1_score(y_test, lr.predict(Xs_te)),
        auc=roc_auc_score(y_test, p),
    )
    joblib.dump(scaler, MODELS / "scaler.joblib")
    joblib.dump(lr,     MODELS / "logistic_regression.joblib")
    print(f"    AUC={results['Logistic Regression']['auc']:.4f}")

    print("  [2/3] Random Forest (200 trees)…")
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    p = rf.predict_proba(X_test)[:, 1]
    results["Random Forest"] = dict(
        accuracy=accuracy_score(y_test, rf.predict(X_test)),
        f1=f1_score(y_test, rf.predict(X_test)),
        auc=roc_auc_score(y_test, p),
    )
    joblib.dump(rf, MODELS / "random_forest.joblib")
    print(f"    AUC={results['Random Forest']['auc']:.4f}")

    print("  [3/3] Gradient Boosting (200 trees)…")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     random_state=42)
    gb.fit(X_train, y_train)
    p = gb.predict_proba(X_test)[:, 1]
    results["Gradient Boosting"] = dict(
        accuracy=accuracy_score(y_test, gb.predict(X_test)),
        f1=f1_score(y_test, gb.predict(X_test)),
        auc=roc_auc_score(y_test, p),
    )
    joblib.dump(gb, MODELS / "gradient_boosting.joblib")
    print(f"    AUC={results['Gradient Boosting']['auc']:.4f}")

    with open(MODELS / "feature_cols.json", "w") as fh:
        json.dump(FEATURE_COLS, fh)
    with open(MODELS / "feature_medians.json", "w") as fh:
        json.dump({c: float(X[c].median()) for c in FEATURE_COLS}, fh, indent=2)

    print("\nResults:")
    for name, m in results.items():
        print(f"  {name:22s}  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    return gb


# ── Step 5: Score all flights ─────────────────────────────────────────────────
def score_flights(df: pd.DataFrame, model) -> pd.DataFrame:
    print("\nGenerating risk scores…")
    X = df[FEATURE_COLS].fillna(0)
    df["risk_score"] = model.predict_proba(X)[:, 1]
    df["risk_label"] = pd.cut(df["risk_score"], bins=[0, 0.3, 0.6, 1.0],
                               labels=["Low", "Medium", "High"])
    df["is_delayed"] = (df["dep_delay_minutes"] > DELAY_THRESHOLD).astype(int)

    keep = ["flight_date", "airline_code", "Origin", "Dest", "scheduled_dep_hour",
            "Distance", "dep_delay_minutes", "arr_delay_minutes", "is_delayed",
            "risk_score", "risk_label",
            "rolling_6_flight_origin_delay_avg", "lag_1_tail_arr_delay_mins",
            "wind_speed_knots", "temperature_c", "precipitation_mm",
            "cloud_cover_total_pct", "origin_pagerank", "origin_betweenness",
            "flight_month", "day_of_week", "weather_delay_mins"]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_parquet(MODELS / "risk_scores.parquet", index=False)

    high = (df["risk_score"] >= 0.6).sum()
    print(f"  Scored {len(df):,} flights — High risk: {high:,} ({high/len(df):.1%})")
    print(f"  Saved → models/risk_scores.parquet")
    return df


# ── Step 6: Export report charts ─────────────────────────────────────────────
def export_charts(df: pd.DataFrame):
    print("\nExporting report charts…")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Aviation Disruption Risk — Summary Report Charts", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    data = (df.groupby("airline_code")["is_delayed"]
              .agg(["mean", "count"]).query("count >= 1000")
              .sort_values("mean").tail(15))
    data["mean"].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Delay Rate by Airline (≥1 000 flights, top 15)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax = axes[0, 1]
    hourly = df.groupby("scheduled_dep_hour")["is_delayed"].mean()
    ax.plot(hourly.index, hourly.values, marker="o", color="coral", linewidth=2)
    ax.fill_between(hourly.index, hourly.values, alpha=0.25, color="coral")
    ax.set_title("Delay Rate by Departure Hour")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xticks(range(0, 24, 2))

    ax = axes[1, 0]
    ax.hist(df["risk_score"], bins=60, color="mediumseagreen", edgecolor="none", alpha=0.85)
    ax.axvline(0.3, color="orange", linestyle="--", linewidth=1.5, label="Low/Medium")
    ax.axvline(0.6, color="red",    linestyle="--", linewidth=1.5, label="Medium/High")
    ax.set_title("Disruption Risk Score Distribution")
    ax.set_xlabel("Risk Score"); ax.set_ylabel("Flights"); ax.legend()

    ax = axes[1, 1]
    data = (df.groupby("Origin")["is_delayed"]
              .agg(["mean", "count"]).query("count >= 500")
              .sort_values("mean").tail(20))
    data["mean"].plot(kind="barh", ax=ax, color="mediumpurple")
    ax.set_title("Delay Rate by Origin Airport (top 20)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    plt.tight_layout()
    out = CACHE / "risk_analysis_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {out.name}")

    # Monthly trend line
    fig, ax = plt.subplots(figsize=(12, 4))
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    monthly = (df.groupby(df["flight_date"].dt.to_period("M"))["is_delayed"]
                 .mean().reset_index())
    monthly["flight_date"] = monthly["flight_date"].dt.to_timestamp()
    ax.plot(monthly["flight_date"], monthly["is_delayed"], marker="o", color="#00b4d8", linewidth=2)
    ax.fill_between(monthly["flight_date"], monthly["is_delayed"], alpha=0.15, color="#00b4d8")
    ax.set_title("Monthly Delay Rate — Feb 2022 to Jan 2026")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    out = CACHE / "monthly_delay_trend.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Week 3 Role C — Disruption Risk Score Generation (BigQuery)")
    print("=" * 60)

    client = bq_client()

    df         = load_from_bigquery(client, sample_pct=20)
    centrality = build_network(client)
    df         = add_centrality(df, centrality)
    model      = train_models(df)
    df         = score_flights(df, model)
    export_charts(df)

    print("\n" + "=" * 60)
    print("Artifacts saved to models/:")
    for f in sorted(MODELS.iterdir()):
        print(f"  {f.name:45s} {f.stat().st_size // 1024:>6} KB")
    print("=" * 60)
    print("\nNext: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
