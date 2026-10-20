
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ecommerce_ml.models import product_recommender


@pytest.fixture
def mock_similarity(monkeypatch):
    products = np.array(["product_a", "product_b", "product_c", "product_d"])

    similarity = csr_matrix(
        [
            [0.0, 0.9, 0.3, 0.0],
            [0.9, 0.0, 0.2, 0.0],
            [0.3, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    monkeypatch.setattr(
        product_recommender,
        "build_product_similarity",
        lambda: (similarity, products),
    )


def test_recommend_similar_products_returns_ranked_results(mock_similarity):
    recommendations = product_recommender.recommend_similar_products(
        "product_a", top_n=2
    )

    assert [item["product_id"] for item in recommendations] == [
        "product_b",
        "product_c",
    ]
    assert recommendations[0]["similarity_score"] == pytest.approx(0.9)
    assert recommendations[1]["similarity_score"] == pytest.approx(0.3)


def test_recommendations_exclude_input_product(mock_similarity):
    recommendations = product_recommender.recommend_similar_products(
        "product_a"
    )

    assert all(
        item["product_id"] != "product_a"
        for item in recommendations
    )


def test_unknown_product_raises_error(mock_similarity):
    with pytest.raises(ValueError, match="Unknown product id"):
        product_recommender.recommend_similar_products("unknown_product")


def test_product_without_similar_items_returns_empty_list(mock_similarity):
    recommendations = product_recommender.recommend_similar_products(
        "product_d"
    )

    assert recommendations == []
