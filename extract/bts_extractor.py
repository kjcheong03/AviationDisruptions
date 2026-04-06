from pathlib import Path
from typing import Iterable
import pandas as pd

INPUT_DIR = Path("extract/input/bts")
OUTPUT_DIR = Path("extract/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep this list small and practical for Week 1.
# Adjust based on the exact BTS export columns your team downloads.
PREFERRED_COLUMNS = [
    "FL_DATE",
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_FL_NUM",
    "TAIL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "CRS_ARR_TIME",
    "ARR_TIME",
    "ARR_DELAY",
    "CANCELLED",
    "CANCELLATION_CODE",
    "DIVERTED",
    "AIR_TIME",
    "DISTANCE",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]

def find_input_files() -> list[Path]:
    patterns = ["*.csv", "*.txt"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(INPUT_DIR.glob(pattern))
    return sorted(files)

def normalize_column_names(columns: Iterable[str]) -> list[str]:
    return [str(c).strip().lower() for c in columns]

def read_bts_file(path: Path) -> pd.DataFrame:
    # BTS downloads are commonly CSV. Some may be comma-delimited text.
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", low_memory=False)

    original_cols = list(df.columns)
    selected = [c for c in original_cols if c in PREFERRED_COLUMNS]
    if selected:
        df = df[selected].copy()

    df.columns = normalize_column_names(df.columns)

    df["source_name"] = "bts"
    df["source_file"] = path.name
    df["ingestion_stage"] = "raw"

    return df

def main() -> None:
    files = find_input_files()
    if not files:
        raise FileNotFoundError(
            "No BTS input files found. Put exported BTS CSV/TXT files in extract/input/bts/"
        )

    all_dfs: list[pd.DataFrame] = []

    for path in files:
        df = read_bts_file(path)
        all_dfs.append(df)
        print(f"Read {len(df)} rows from {path}")

    combined = pd.concat(all_dfs, ignore_index=True)
    output_path = OUTPUT_DIR / "bts_raw_combined.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved {len(combined)} rows to {output_path}")
    print(combined.head())

if __name__ == "__main__":
    main()