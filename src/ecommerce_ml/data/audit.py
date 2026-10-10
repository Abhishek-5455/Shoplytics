from pathlib import Path

from .profiler import profile_data

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def audit_csv():

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}, please download the data first.")

    for file_path in csv_files:
        try:
            profile = profile_data(file_path)
            print(f"Profile for {profile['file']}:")
            print(f"Shape: {profile['shape']}")
            print(f"Columns: {profile['columns']}")
            print(f"Dtypes: {profile['dtypes']}")
            print(f"Missing Values: {profile['missing_values']}")
            print(f"Duplicates: {profile['duplicates']}\n")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            print(f"Error profiling {file_path}: {exc}")


if __name__ == "__main__":
    audit_csv()
