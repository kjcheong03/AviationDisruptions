from pathlib import Path
from typing import Iterable
import pandas as pd

INPUT_DIR = Path("extract/input/aspm")
OUTPUT_DIR = Path("extract/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ASPM Data Download supports comma-separated, tab-separated, fixed-length, and DBF,
# but CSV is the best first target for Week 1.
# These are common flight-level fields from ASPM documentation; keep only what exists.
PREFERRED_COLUMNS = [
    "Dep_YYYYMM",
    "Dep_DAY",
    "Dep_HOUR",
    "Dep_QTR",
    "Arr_YYYYMM",
    "Arr_DAY",
    "Arr_HOUR",
    "Arr_QTR",
    "Off_YYYYMM",
    "Off_DAY",
    "Off_HOUR",
    "Off_QTR",
    "On_YYYYMM",
    "On_DAY",
    "On_HOUR",
    "On_QTR",
    "Carrier",
    "Flight_Number",
    "Dep_Airport",
    "Arr_Airport",
    "DepSchDT",
    "DepSchTM",
    "ArrSchDT",
    "ArrSchTM",
    "ActTI",
    "ActTO",
    "AIRBORNE",
]

def find_input_files() -> list[Path]:
    patterns = ["*.csv", "*.txt", "*.tsv"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(INPUT_DIR.glob(pattern))
    return sorted(files)

def normalize_column_names(columns: Iterable[str]) -> list[str]:
    return [str(c).strip().lower() for c in columns]

def read_aspm_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t", low_memory=False)
    elif suffix == ".txt":
        try:
            df = pd.read_csv(path, sep=",", low_memory=False)
        except Exception:
            df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        raise ValueError(f"Unsupported ASPM file type: {path}")

    original_cols = list(df.columns)
    selected = [c for c in original_cols if c in PREFERRED_COLUMNS]
    if selected:
        df = df[selected].copy()

    df.columns = normalize_column_names(df.columns)
    df["source_name"] = "aspm"
    df["source_file"] = path.name
    df["ingestion_stage"] = "raw"

    return df

def main() -> None:
    files = find_input_files()
    if not files:
        raise FileNotFoundError(
            "No ASPM input files found. Put exported ASPM CSV/TXT/TSV files in extract/input/aspm/"
        )

    all_dfs: list[pd.DataFrame] = []

    for path in files:
        df = read_aspm_file(path)
        all_dfs.append(df)
        print(f"Read {len(df)} rows from {path}")

    combined = pd.concat(all_dfs, ignore_index=True)
    output_path = OUTPUT_DIR / "aspm_raw_combined.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved {len(combined)} rows to {output_path}")
    print(combined.head())

if __name__ == "__main__":
    main()