import pandas as pd

from ecommerce_ml.features.late_delivery import build_delivery_features


def test_build_delivery_features(monkeypatch):
    orders = pd.DataFrame({
        "order_id": ["o1", "o2"],
        "order_status": ["delivered", "delivered"],
        "order_purchase_timestamp": [
            "2018-01-01 10:00:00",
            "2018-01-02 14:00:00",
        ],
        "order_delivered_customer_date": [
            "2018-01-12 10:00:00",
            "2018-01-09 14:00:00",
        ],
        "order_estimated_delivery_date": [
            "2018-01-10 00:00:00",
            "2018-01-12 00:00:00",
        ],
    })

    items = pd.DataFrame({
        "order_id": ["o1", "o1", "o2"],
        "order_item_id": [1, 2, 1],
        "price": [100.0, 50.0, 75.0],
        "freight_value": [10.0, 5.0, 8.0],
    })

    def fake_load_csv(filename):
        if filename == "olist_orders_dataset.csv":
            return orders.copy()
        if filename == "olist_order_items_dataset.csv":
            return items.copy()
        raise ValueError(f"Unexpected file: {filename}")

    monkeypatch.setattr(
        "ecommerce_ml.features.engineering.load_csv",
        fake_load_csv,
    )

    result = build_delivery_features().set_index("order_id")

    assert len(result) == 2
    assert result.loc["o1", "is_late"] == 1
    assert result.loc["o2", "is_late"] == 0
    assert result.loc["o1", "item_count"] == 2
    assert result.loc["o1", "total_price"] == 150.0
    assert result.loc["o1", "total_freight"] == 15.0
