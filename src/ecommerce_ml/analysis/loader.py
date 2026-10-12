from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"

def load_csv(file_name: str) -> pd.DataFrame:
	path = RAW_DIR / file_name

	if not Path(path).is_file():
		raise FileNotFoundError(f"CSV file not found: {file_name}")

	return pd.read_csv(path)

