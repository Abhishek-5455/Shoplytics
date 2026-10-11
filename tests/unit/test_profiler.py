import pandas as pd
import pytest

from ecommerce_ml.data.profiler import profile_data


def test_profile_csv(tmp_path):
    csv_path = tmp_path / "test.csv"
    
    pd.DataFrame({
        'id': [1, 2, 2],
        'city': ['A', 'B', 'B']
    }).to_csv(csv_path, index=False)

    result = profile_data(csv_path)

    assert result['shape'] == (3,2)
    assert result['duplicates'] == 1

def test_missing_file():
    with pytest.raises(FileNotFoundError):
        profile_data(pd.io.common.stringify_path("missing.csv"))