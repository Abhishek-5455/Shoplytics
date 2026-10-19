import numpy as np
import pandas as pd
import pytest

from ecommerce_ml.models import predict_delivery_time


FEATURE_COLUMNS = [
    "purchase_hour",
    "purchase_day_of_week",
    "purchase_month",
    "estimated_delivery_days",
    "item_count",
    "total_price",
    "total_freight",
]


class FakeModel:
    def predict(self, X):
        return np.array([8.5, 12.0])


@pytest.fixture
def sample_features():
    return pd.DataFrame([
        {
            "purchase_hour": 10,
            "purchase_day_of_week": 6,
            "purchase_month": 6,
            "estimated_delivery_days": 15.0,
            "item_count": 2,
            "total_price": 250.0,
            "total_freight": 35.0,
        },
        {
            "purchase_hour": 18,
            "purchase_day_of_week": 5,
            "purchase_month": 11,
            "estimated_delivery_days": 25.0,
            "item_count": 3,
            "total_price": 450.0,
            "total_freight": 65.0,
        },
    ])


def test_predict_delivery_time_returns_predictions(
    monkeypatch, sample_features
):
    monkeypatch.setattr(
        predict_delivery_time,
        "load_delivery_time_model",
        lambda: (FakeModel(), FEATURE_COLUMNS),
    )

    result = predict_delivery_time.predict_delivery_time(sample_features)

    assert list(result.columns) == ["predicted_delivery_days"]
    assert result["predicted_delivery_days"].tolist() == [8.5, 12.0]
    assert result.index.equals(sample_features.index)


def test_predict_delivery_time_rejects_missing_features():
    features = pd.DataFrame([{"purchase_hour": 10}])

    with pytest.raises(ValueError, match="Missing required columns"):
        predict_delivery_time.predict_delivery_time(features)


def test_predict_delivery_time_rejects_missing_values(
    monkeypatch, sample_features
):
    monkeypatch.setattr(
        predict_delivery_time,
        "load_delivery_time_model",
        lambda: (FakeModel(), FEATURE_COLUMNS),
    )

    sample_features.loc[0, "total_price"] = np.nan

    with pytest.raises(ValueError, match="Input features cannot be empty"):
        predict_delivery_time.predict_delivery_time(sample_features)