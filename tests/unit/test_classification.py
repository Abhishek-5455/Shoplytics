import pandas as pd

from ecommerce_ml.models.random_forest_classifier import (
    FEATURES,
    train_delivery_classifier,
)


def test_train_deliver_classifier():
    rows = []

    for i in range(100):
        rows.append({
            "purchase_hour": i % 24,
            "purchase_day_of_week": i % 7,
            "purchase_month": i % 12,
            "days_to_estimated_delivery": 10 + i % 5,
            "item_count": i % 4 + 1,
            "total_price": 50.0 + i * 2,
            "total_freight": 5.0 + i % 10,
            "is_late": i % 2
        })

    data = pd.DataFrame(rows)

    model, X_test, y_test = train_delivery_classifier(data)

    assert len(X_test) == 20
    assert len(y_test) == 20
    assert list(X_test.columns) == FEATURES 
