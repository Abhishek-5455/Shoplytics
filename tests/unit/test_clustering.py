import pandas as pd
import pytest

from ecommerce_ml.analysis.cluster_customers import cluster_customers, scale_rfm


def test_scale_rfm():
    rfm = pd.DataFrame({
        "customer_unique_id": ["u1", "u2", "u3"],
        "recency": [10, 20, 30],
        "frequency": [1, 2, 3],
        "monetary": [100, 200, 300],
    }).set_index("customer_unique_id")

    result = scale_rfm(rfm)

    assert list(result.columns) == [
        "recency", "frequency", "monetary"
    ]
    assert result.mean().abs().max() < 1e-10
    assert ((result.std(ddof=0) - 1).abs() < 1e-10).all()

def test_empty_rfm():
    rfm = pd.DataFrame(
        columns=['recency','frequency', 'monetary']
    )

    with pytest.raises(ValueError, match='empty'):
        scale_rfm(rfm)

def test_customer_cluster_and_labels():
    rfm = pd.DataFrame({
        "customer_unique_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
        "recency": [1, 2, 3, 30, 40, 50],
        "frequency": [10, 9, 8, 2, 1, 1],
        "monetary": [1000, 900, 800, 100, 50, 40],
    }).set_index("customer_unique_id")

    result = cluster_customers(rfm, n_clusters=2)

    assert 'cluster' in result.columns
    assert len(result) == len(rfm)
    assert result['cluster'].nunique() == 2



    