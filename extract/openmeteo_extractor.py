from pathlib import Path
import requests
import pandas as pd

OUTPUT_DIR = Path("extract/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AIRPORTS = {
    "JFK": {"lat": 40.6413, "lon": -73.7781},
    "LAX": {"lat": 33.9416, "lon": -118.4085},
    "SFO": {"lat": 37.6213, "lon": -122.3790},
}

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
        ],
        "timezone": "UTC",
    }

    response = requests.get(url, params=params, timeout=60)
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

def save_to_csv(df: pd.DataFrame, airport_code: str, start_date: str, end_date: str) -> Path:
    output_path = OUTPUT_DIR / f"openmeteo_{airport_code.lower()}_{start_date}_{end_date}.csv"
    df.to_csv(output_path, index=False)
    return output_path

def main():
    airport_codes = ["JFK", "LAX", "SFO"]
    start_date = "2025-01-01"
    end_date = "2025-01-07"

    for airport_code in airport_codes:
        try:
            df = fetch_openmeteo_history(airport_code, start_date, end_date)
            output_path = save_to_csv(df, airport_code, start_date, end_date)

            print(f"\nSaved {len(df)} rows to {output_path}")
            print(df.head())
        except Exception as e:
            print(f"\nFailed for {airport_code}: {e}")

if __name__ == "__main__":
    main()