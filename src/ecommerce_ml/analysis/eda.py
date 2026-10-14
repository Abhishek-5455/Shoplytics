import pandas as pd


def summarise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	return pd.DataFrame({
		'dtype': df.dtypes.astype(str),
		'missing_count': df.isna().sum(),
		'missing_percent': df.isna().mean().mul(100).round(2),
		'unique_count': df.nunique()
	})

def numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
	return df.describe().T

