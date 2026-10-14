import pandas as pd

from ecommerce_ml.analysis.rfm import build_rfm


def test_rfm(monkeypatch):
    orders = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_id": ["c1", "c1", "c2"],
        "order_status": ["delivered", "delivered", "canceled"],
        "order_purchase_timestamp": [
            "2018-01-01 10:00:00",
            "2018-01-10 10:00:00",
            "2018-01-15 10:00:00",
        ],
    })

    customers = pd.DataFrame({
        "customer_id": ["c1", "c2"],
        "customer_unique_id": ["u1", "u2"],
    })

    items = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "price": [100.0, 200.0, 500.0],
    })

    datasets = {
        'olist_customers_dataset.csv': customers,
        'olist_orders_dataset.csv': orders,
        'olist_order_items_dataset.csv': items
    }

    monkeypatch.setattr(
        "ecommerce_ml.analysis.rfm.load_csv",
        lambda filename: datasets[filename].copy()
    )

    result = build_rfm(reference_date="2018-01-11")
    customer = result.set_index("customer_unique_id").loc["u1"]

    assert customer["recency"] == 1
    assert customer["frequency"] == 2
    assert customer["monetary"] == 300.0
    assert "u2" not in result["customer_unique_id"].values