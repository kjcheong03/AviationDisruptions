from pathlib import Path
import time
import requests
import pandas as pd

OUTPUT_DIR = Path("extract/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# All 10 target airports with lat/lon coordinates
AIRPORTS = {
    "ATL": {"lat": 33.6407, "lon": -84.4277},
    "DFW": {"lat": 32.8998, "lon": -97.0403},
    "DEN": {"lat": 39.8561, "lon": -104.6737},
    "ORD": {"lat": 41.9742, "lon": -87.9073},
    "LAX": {"lat": 33.9416, "lon": -118.4085},
    "JFK": {"lat": 40.6413, "lon": -73.7781},
    "SFO": {"lat": 37.6213, "lon": -122.3790},
    "SEA": {"lat": 47.4502, "lon": -122.3088},
    "LAS": {"lat": 36.0840, "lon": -115.1537},
    "MCO": {"lat": 28.4312, "lon": -81.3081},
}

START_DATE = "2022-02-01"
END_DATE = "2026-01-31"


def fetch_openmeteo_history(airport_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    airport_code = airport_code.upper()

    if airport_code not in AIRPORTS:
        raise ValueError(f"Unsupported airport code: {airport_code}")

    airport = AIRPORTS[airport_code]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": airport["lat"],
        "longitude": airport["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "visibility",
            "wind_speed_10m",
            "weather_code",
            "cloud_cover",
            "cloud_cover_low",
        ],
        "wind_speed_unit": "kn",  # return wind speed in knots
        "timezone": "UTC",
    }

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()

    hourly = data.get("hourly")
    if not hourly:
        raise ValueError(f"No hourly data returned for {airport_code}")

    df = pd.DataFrame(hourly)
    df["airport_code"] = airport_code
    df["source_name"] = "openmeteo"
    df["extract_start_date"] = start_date
    df["extract_end_date"] = end_date

    return df


def main():
    all_dfs: list[pd.DataFrame] = []

    for i, airport_code in enumerate(AIRPORTS):
        if i > 0:
            time.sleep(3)  # avoid rate limiting
        print(f"Fetching {airport_code} ({START_DATE} to {END_DATE})...")
        try:
            df = fetch_openmeteo_history(airport_code, START_DATE, END_DATE)
            all_dfs.append(df)
            print(f"  -> {len(df):,} rows")
        except Exception as e:
            print(f"  -> FAILED: {e}")

    if not all_dfs:
        raise RuntimeError("No data fetched for any airport.")

    combined = pd.concat(all_dfs, ignore_index=True)
    output_path = OUTPUT_DIR / f"openmeteo_all_{START_DATE}_{END_DATE}.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved {len(combined):,} rows across {len(all_dfs)} airports to {output_path}")
    print(combined.head())
    return output_path


if __name__ == "__main__":
    main()
