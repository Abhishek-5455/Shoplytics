
import pandas as pd
import pytest

from ecommerce_ml.models import product_recommender


@pytest.fixture
def mock_purchase_data(monkeypatch):
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "order_status": ["delivered"] * 4,
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-02-01",
                    "2018-06-01",
                    "2018-07-01",
                ]
            ),
        }
    )

    items = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "o4"],
            "product_id": ["p1", "p1", "p2", "p2", "p3"],
        }
    )

    def fake_load_csv(filename):
        if filename == "olist_orders_dataset.csv":
            return orders.copy()
        if filename == "olist_order_items_dataset.csv":
            return items.copy()
        raise ValueError(f"Unexpected filename: {filename}")

    monkeypatch.setattr(product_recommender, "load_csv", fake_load_csv)


def test_popularity_respects_cutoff(mock_purchase_data):
    popular = product_recommender.get_popular_products(
        top_n=5, cutoff_date="2018-05-01"
    )

    assert popular == [
        {"product_id": "p1", "purchase_count": 2}
    ]


# def test_popularity_uses_all_dates_when_no_cutoff(mock_purchase_data):
#     popular = product_recommender.get_popular_products(top_n=3)

#     assert popular[0] == {"product_id": "p2", "purchase_count": 2}
#     assert popular[1] == {"product_id": "p1", "purchase_count": 2}


def test_top_n_limits_results(mock_purchase_data):
    popular = product_recommender.get_popular_products(
        top_n=1, cutoff_date="2018-12-31"
    )

    assert len(popular) == 1


@pytest.mark.parametrize("top_n", [0, -1])
def test_invalid_top_n_raises_error(top_n):
    with pytest.raises(ValueError, match="top_n"):
        product_recommender.get_popular_products(top_n=top_n)
