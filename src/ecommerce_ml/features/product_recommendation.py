
import pandas as pd
from scipy.sparse import csr_matrix

from ecommerce_ml.analysis.loader import load_csv


def get_popular_products(top_n=5):
    """Return the most frequently purchased products."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    orders = load_csv("olist_orders_dataset.csv")
    items = load_csv("olist_order_items_dataset.csv")

    delivered_orders = orders.loc[
        orders["order_status"].eq("delivered"),
        ["order_id"],
    ]

    popular = (
        items.merge(
            delivered_orders,
            on="order_id",
            how="inner",
            validate="many_to_one",
        )
        .groupby("product_id")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
    )

    return [
        {
            "product_id": product_id,
            "purchase_count": int(count),
        }
        for product_id, count in popular.items()
    ]


def build_product_purchase_matrix(cutoff_date = None):
    """Build a sparse customer-product purchase-count matrix."""
    orders = load_csv("olist_orders_dataset.csv")
    items = load_csv("olist_order_items_dataset.csv")
    customers = load_csv('olist_customers_dataset.csv')

    orders = orders.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    completed_orders = orders.loc[
        orders["order_status"].eq("delivered"),
        ["order_id", "customer_unique_id", 'order_purchase_timestamp'],
    ].dropna(subset=["order_id", "customer_unique_id", 'order_purchase_timestamp'])

    completed_orders['order_purchase_timestamp'] = pd.to_datetime(
        completed_orders['order_purchase_timestamp'],
        errors='coerce'
    )

    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date)
        completed_orders = completed_orders.loc[
            completed_orders['order_purchase_timestamp'] < cutoff
        ]

    purchase_data = items.merge(
        completed_orders[['order_id', 'customer_unique_id']],
        on="order_id",
        how="inner",
        validate="many_to_one",
    )

    purchase_counts = (
        purchase_data.groupby(["customer_unique_id", "product_id"])["order_id"]
        .nunique()
        .reset_index(name="purchase_count")
    )

    customer_codes, customers = pd.factorize(
        purchase_counts["customer_unique_id"], sort=True
    )
    product_codes, products = pd.factorize(
        purchase_counts["product_id"], sort=True
    )

    matrix = csr_matrix(
        (
            purchase_counts["purchase_count"].to_numpy(dtype="float32"),
            (customer_codes, product_codes),
        ),
        shape=(len(customers), len(products)),
        dtype="float32",
    )

    return matrix, customers, products
