from pathlib import Path
from typing import Iterable
import pandas as pd

INPUT_DIR = Path("extract/input/opsnet")
OUTPUT_DIR = Path("extract/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OPSNET Data Download commonly returns spreadsheets by email link.
# Support csv/xlsx/xls.
PREFERRED_COLUMNS = [
    "Date",
    "Airport",
    "Facility",
    "State",
    "Region",
    "Service Area",
    "Class",
    "Local Hour",
    "Quarter Hour",
    "Arrival Operations",
    "Departure Operations",
    "Total Operations",
    "Delay Count",
    "Delay Minutes",
]

def find_input_files() -> list[Path]:
    patterns = ["*.csv", "*.xlsx", "*.xls"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(INPUT_DIR.glob(pattern))
    return sorted(files)

def normalize_column_names(columns: Iterable[str]) -> list[str]:
    return [str(c).strip().lower().replace(" ", "_").replace("/", "_") for c in columns]

def read_opsnet_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif suffix == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl")
    elif suffix == ".xls":
        # FAA OPSNET exports HTML disguised as .xls — try xlrd first, fall back to HTML
        try:
            df = pd.read_excel(path, engine="xlrd")
        except Exception:
            tables = pd.read_html(path, header=[0, 1, 2])
            df = tables[0]
            # Flatten multi-level columns — keep only the deepest meaningful level
            df.columns = [
                str(c[-1]).strip() if not str(c[-1]).startswith("Unnamed") else str(c[-2]).strip()
                for c in df.columns
            ]
            # Drop fully empty columns
            df = df.dropna(axis=1, how="all")
    else:
        raise ValueError(f"Unsupported OPSNET file type: {path}")

    original_cols = list(df.columns)
    selected = [c for c in original_cols if c in PREFERRED_COLUMNS]
    if selected:
        df = df[selected].copy()

    df.columns = normalize_column_names(df.columns)
    df["source_name"] = "opsnet"
    df["source_file"] = path.name
    df["ingestion_stage"] = "raw"

    return df

def main() -> None:
    files = find_input_files()
    if not files:
        raise FileNotFoundError(
            "No OPSNET input files found. Put exported OPSNET CSV/XLSX/XLS files in extract/input/opsnet/"
        )

    all_dfs: list[pd.DataFrame] = []

    for path in files:
        df = read_opsnet_file(path)
        all_dfs.append(df)
        print(f"Read {len(df)} rows from {path}")

    combined = pd.concat(all_dfs, ignore_index=True)
    output_path = OUTPUT_DIR / "opsnet_raw_combined.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved {len(combined)} rows to {output_path}")
    print(combined.head())

if __name__ == "__main__":
    main()