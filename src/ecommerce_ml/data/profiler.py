from pathlib import Path

import pandas as pd


def profile_data(file_path: str) -> dict:
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    df = pd.read_csv(file_path)

    return {
        'file': Path(file_path).name,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum()
    }