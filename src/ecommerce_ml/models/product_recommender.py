import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from ecommerce_ml.analysis.loader import load_csv
from ecommerce_ml.features.product_recommendation import build_product_purchase_matrix


def get_popular_products(top_n=5, cutoff_date=None):
    """Return the most frequently purchased products before a cutoff."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    orders = load_csv("olist_orders_dataset.csv")
    items = load_csv("olist_order_items_dataset.csv")

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )

    delivered_orders = orders.loc[
        orders["order_status"].eq("delivered"),
        ["order_id", "order_purchase_timestamp"],
    ].dropna(subset=["order_purchase_timestamp"])

    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date)
        delivered_orders = delivered_orders.loc[
            delivered_orders["order_purchase_timestamp"] < cutoff
        ]

    popular = (
        items.merge(
            delivered_orders[["order_id"]],
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
        {"product_id": product_id, "purchase_count": int(count)}
        for product_id, count in popular.items()
    ]


def build_product_similarity():
    purchase_matrix, _, products = build_product_purchase_matrix()

    normalized = normalize(purchase_matrix.T, norm='l2', axis=1)

    similarity = (normalized @ normalized.T).tocsr()
    similarity.setdiag(0)
    similarity.eliminate_zeros()

    return similarity, products

def recommend_similar_products(product_id: str, top_n: int = 5):
    similarity, products = build_product_similarity()

    product_to_index = {product: i for i, product in enumerate(products)}

    if product_id not in product_to_index:
        raise ValueError(f"Unknown product id: {product_id}")

    product_index = product_to_index[product_id]
    scores = similarity.getrow(product_index).toarray().ravel()

    candidate_indices = np.flatnonzero(scores > 0)
    ranked_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]][:top_n]

    return [
        {
            'product_id': products[index],
            'similarity_score': float(scores[index])
        } 
        for index in ranked_indices
    ]
if __name__ == "__main__":
    _, products = build_product_similarity()

    for i in range(20):
        sample_product_id = products[i]
        print(f"{i}. input product id: {sample_product_id}")

        recommendations = recommend_similar_products(sample_product_id)
        if len(recommendations):
            print("Recommendations: ")
            for recommendation in recommendations:
                print(recommendation)
        else:
            print(f"No recommendations for product {sample_product_id}")