
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from ecommerce_ml.analysis.loader import load_csv
from ecommerce_ml.features.product_recommendation import (
    build_product_purchase_matrix,
)


def evaluate_recommender(cutoff_date="2018-05-01", k=5):
    if k < 1:
        raise ValueError("k must be at least 1.")

    orders = load_csv("olist_orders_dataset.csv")
    items = load_csv("olist_order_items_dataset.csv")
    customer_data = load_csv("olist_customers_dataset.csv")

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )

    orders = orders.merge(
        customer_data[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    delivered = orders.loc[
        orders["order_status"].eq("delivered")
    ].dropna(subset=["customer_unique_id", "order_purchase_timestamp"])

    cutoff = pd.Timestamp(cutoff_date)
    train_orders = delivered.loc[
        delivered["order_purchase_timestamp"] < cutoff
    ]
    test_orders = delivered.loc[
        delivered["order_purchase_timestamp"] >= cutoff
    ]

    train_items = items.merge(
        train_orders[["order_id", "customer_unique_id"]],
        on="order_id",
        how="inner",
        validate="many_to_one",
    )
    test_items = items.merge(
        test_orders[["order_id", "customer_unique_id"]],
        on="order_id",
        how="inner",
        validate="many_to_one",
    )

    matrix, train_customers, train_products = (
        build_product_purchase_matrix(cutoff)
    )
    customer_to_index = {
        customer: index for index, customer in enumerate(train_customers)
    }
    product_to_index = {
        product: index for index, product in enumerate(train_products)
    }

    normalized = normalize(matrix.T, norm="l2", axis=1)
    similarity = (normalized @ normalized.T).tocsr()
    similarity.setdiag(0)
    similarity.eliminate_zeros()

    # Use the training period only to establish product popularity.
    popularity = (
        train_items.groupby("product_id")["order_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    popular_products = popularity.index.tolist()

    test_truth = (
        test_items.groupby("customer_unique_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    cf_precision, cf_recall = [], []
    pop_precision, pop_recall = [], []
    customers_with_history = 0
    customers_with_cf_recommendations = 0
    evaluated_customers = 0

    for customer_id, actual_products in test_truth.items():
        customer_index = customer_to_index.get(customer_id)
        if customer_index is None:
            continue

        relevant = actual_products.intersection(product_to_index)
        if not relevant:
            continue

        history = matrix.getrow(customer_index)
        seen_indices = set(history.indices)
        seen_products = {
            train_products[index] for index in seen_indices
        }

        customers_with_history += 1

        # Collaborative-filtering recommendations.
        scores = (history @ similarity).toarray().ravel()
        if seen_indices:
            scores[list(seen_indices)] = 0

        candidate_indices = np.flatnonzero(scores > 0)
        if candidate_indices.size == 0:
            continue

        ranked_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ][:k]
        cf_recommended = {
            train_products[index] for index in ranked_indices
        }

        if not cf_recommended:
            continue

        customers_with_cf_recommendations += 1

        # Popularity recommendations, excluding products already seen.
        pop_recommended = []
        for product in popular_products:
            if product not in seen_products:
                pop_recommended.append(product)
            if len(pop_recommended) == k:
                break

        if not pop_recommended:
            continue

        cf_hits = len(cf_recommended.intersection(relevant))
        pop_hits = len(set(pop_recommended).intersection(relevant))

        cf_precision.append(cf_hits / k)
        cf_recall.append(cf_hits / len(relevant))
        pop_precision.append(pop_hits / k)
        pop_recall.append(pop_hits / len(relevant))
        evaluated_customers += 1

    return {
        "cutoff_date": str(cutoff.date()),
        "k": k,
        "test_customers": len(test_truth),
        "customers_with_training_history": customers_with_history,
        "customers_with_cf_recommendations": customers_with_cf_recommendations,
        "customers_evaluated_for_both": evaluated_customers,
        "collaborative_precision_at_k": (
            float(np.mean(cf_precision)) if cf_precision else 0.0
        ),
        "collaborative_recall_at_k": (
            float(np.mean(cf_recall)) if cf_recall else 0.0
        ),
        "popularity_precision_at_k": (
            float(np.mean(pop_precision)) if pop_precision else 0.0
        ),
        "popularity_recall_at_k": (
            float(np.mean(pop_recall)) if pop_recall else 0.0
        ),
    }


if __name__ == "__main__":
    results = evaluate_recommender(cutoff_date="2018-05-01", k=5)
    print(pd.Series(results).to_string())
