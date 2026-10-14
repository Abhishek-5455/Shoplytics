import pandas as pd
import pytest

from ecommerce_ml.analysis.eda import numerical_summary, summarise_dataframe
from ecommerce_ml.analysis.loader import load_csv


def test_summarise_dataframe():

    df = pd.DataFrame(
        {
            "price": [10, 20, 30],
            "category": ["A", None, "A"],
        }
    )
    result = summarise_dataframe(df)
    print(result)

    assert result.loc["price", "missing_count"] == 0
    assert result.loc["category", "missing_count"] == 1
    assert result.loc["category", "missing_percent"] == 33.33
    assert result.loc["category", "unique_count"] == 1

def test_numerical_summary():
    df = pd.DataFrame({
        'price': [10,20,30]
    })

    result = numerical_summary(df)

    assert result.loc["price", "mean"] == 20
    assert result.loc["price", "min"] == 10
    assert result.loc["price", "max"] == 30

def test_load_csv():
    with pytest.raises(FileNotFoundError):
        load_csv('missing.csv')