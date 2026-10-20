
import pandas as pd
from scipy.sparse import csr_matrix

from ecommerce_ml.analysis.loader import load_csv


def build_product_purchase_matrix():
    """Build a sparse customer-product purchase-count matrix."""
    orders = load_csv("olist_orders_dataset.csv")
    items = load_csv("olist_order_items_dataset.csv")

    completed_orders = orders.loc[
        orders["order_status"].eq("delivered"),
        ["order_id", "customer_id"],
    ].dropna()

    purchase_data = items.merge(
        completed_orders,
        on="order_id",
        how="inner",
        validate="many_to_one",
    )

    purchase_counts = (
        purchase_data.groupby(["customer_id", "product_id"])["order_id"]
        .nunique()
        .reset_index(name="purchase_count")
    )

    customer_codes, customers = pd.factorize(
        purchase_counts["customer_id"], sort=True
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
