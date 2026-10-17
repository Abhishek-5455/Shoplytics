import pandas as pd
import pytest

from ecommerce_ml.models import predict_delivery


def test_predict_delivery_returns_expected_columns(monkeypatch):
    class FakeModel:
        def predict_proba(self, X):
            return [
                [0.8, 0.2],
                [0.3, 0.7],
            ]

    monkeypatch.setattr(
        predict_delivery,
        "load_model",
        lambda: (
            FakeModel(),
            [
                "purchase_hour",
                "purchase_day_of_week",
                "purchase_month",
                "days_to_estimated_delivery",
                "item_count",
                "total_price",
                "total_freight",
            ],
            0.5,
        ),
    )

    X = pd.DataFrame([
        {
            "purchase_hour": 9,
            "purchase_day_of_week": 1,
            "purchase_month": 3,
            "days_to_estimated_delivery": 7.0,
            "item_count": 1,
            "total_price": 50.0,
            "total_freight": 12.0,
        },
        {
            "purchase_hour": 14,
            "purchase_day_of_week": 5,
            "purchase_month": 11,
            "days_to_estimated_delivery": 25.0,
            "item_count": 4,
            "total_price": 450.0,
            "total_freight": 65.0,
        },
    ])

    result = predict_delivery.predict_delivery_risk(X)

    assert list(result.columns) == [
        "probabilities",
        "predictions",
    ]
    assert result["predictions"].tolist() == [0, 1]
    assert result["probabilities"].tolist() == [0.2, 0.7]


def test_predict_delivery_rejects_missing_features():
    X = pd.DataFrame([{"purchase_hour": 10}])

    with pytest.raises(ValueError, match="Missing required features"):
        predict_delivery.predict_delivery_risk(X)