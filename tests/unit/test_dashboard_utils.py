from datetime import date

import pandas as pd
import pytest

from ecommerce_ml.dashboard.prediction_logic import (
    DURATION_FEATURES,
    RISK_FEATURES,
    build_duration_features,
    build_risk_features,
    read_duration_result,
    read_late_risk_result,
)

from ecommerce_ml.dashboard.queries import (
    customer_query,
    delivery_query,
    overview_queries,
    payment_queries,
    sales_queries,
)

from ecommerce_ml.dashboard.utils import (
    date_params,
    date_predicate,
    format_money,
    safe_scalar,
)


# -------------------------------------------------
# Date utility tests
# -------------------------------------------------

def test_date_params_single_date_includes_full_day():
    assert date_params(
        date(2017, 1, 1),
        date(2017, 1, 1),
    ) == (
        date(2017, 1, 1),
        date(2017, 1, 2),
    )


def test_date_params_across_month_and_year():
    assert date_params(
        date(2017, 12, 31),
        date(2017, 12, 31),
    )[1] == date(2018, 1, 1)


def test_date_params_leap_day():
    assert date_params(
        date(2020, 2, 29),
        date(2020, 2, 29),
    )[1] == date(2020, 3, 1)


def test_date_params_rejects_reversed_range():
    with pytest.raises(ValueError):
        date_params(
            date(2017, 2, 2),
            date(2017, 2, 1),
        )


@pytest.mark.parametrize(
    "column",
    [
        "order_purchase_timestamp",
        "o.order_purchase_timestamp",
    ],
)
def test_date_predicate_supported_columns(column):
    assert date_predicate(column) == (
        f"{column} >= %s AND {column} < %s"
    )


def test_date_predicate_rejects_untrusted_column():
    with pytest.raises(ValueError):
        date_predicate(
            "order_purchase_timestamp; DROP TABLE public.orders"
        )


# -------------------------------------------------
# Formatting utility tests
# -------------------------------------------------

def test_format_money_handles_null_and_values():
    assert format_money(1250) == "1,250.00"

    assert format_money(float("nan")) == "0.00"

    assert safe_scalar(
        None,
        "fallback",
    ) == "fallback"


# -------------------------------------------------
# SQL query tests
# -------------------------------------------------

def test_analytics_queries_include_parameterized_purchase_window():
    queries = overview_queries()

    assert "%s" in queries["overview"]

    assert "%s" in queries["payment_total"]

    assert "DATE_TRUNC" in queries["monthly"]

    assert "GROUP BY order_status" in queries["statuses"]

    assert "%s" in delivery_query()

    assert all(
        "%s" in query
        for query in sales_queries().values()
    )

    assert "%s" in customer_query()

    assert all(
        "%s" in query
        for query in payment_queries().values()
    )


# -------------------------------------------------
# Late delivery risk feature tests
# -------------------------------------------------

def test_risk_features_order_types_and_values():
    df = build_risk_features(
        purchase_hour=8,
        purchase_day_of_week=1,
        purchase_month=9,
        days_to_estimated_delivery=12,
        item_count=2,
        total_price=150,
        total_freight=25,
    )

    assert list(df.columns) == list(RISK_FEATURES)

    assert len(df) == 1

    assert df.iloc[0]["purchase_hour"] == 8

    assert df.iloc[0]["item_count"] == 2

    assert df.iloc[0]["total_price"] == 150.0


# -------------------------------------------------
# Delivery duration feature tests
# -------------------------------------------------

def test_duration_features_schema():
    df = build_duration_features(
        purchase_hour=8,
        purchase_day_of_week=1,
        purchase_month=9,
        estimated_delivery_days=18,
        item_count=3,
        total_price=190,
        total_freight=31,
    )

    assert list(df.columns) == list(DURATION_FEATURES)

    assert (
        df.iloc[0]["estimated_delivery_days"]
        == 18.0
    )


# -------------------------------------------------
# Risk input validation tests
# -------------------------------------------------

@pytest.mark.parametrize(
    "updates",
    [
        {"purchase_hour": 24},
        {"purchase_day_of_week": 7},
        {"purchase_month": 13},
        {"item_count": 0},
        {"total_price": -1},
        {"total_freight": -1},
        {"days_to_estimated_delivery": -1},
    ],
)
def test_risk_features_reject_invalid_inputs(updates):
    args = {
        "purchase_hour": 10,
        "purchase_day_of_week": 2,
        "purchase_month": 5,
        "days_to_estimated_delivery": 7,
        "item_count": 1,
        "total_price": 10,
        "total_freight": 2,
    }

    args.update(updates)

    with pytest.raises(ValueError):
        build_risk_features(**args)


def test_duration_features_reject_negative_estimate():
    with pytest.raises(ValueError):
        build_duration_features(
            purchase_hour=10,
            purchase_day_of_week=2,
            purchase_month=5,
            estimated_delivery_days=-0.1,
            item_count=1,
            total_price=10,
            total_freight=2,
        )


# -------------------------------------------------
# Prediction output tests
# -------------------------------------------------

def test_read_late_risk_result():
    pred = pd.DataFrame(
        [
            {
                "late_delivery_probability": 0.73,
                "delivery_prediction": 1,
            }
        ]
    )

    assert read_late_risk_result(pred) == (
        0.73,
        1,
    )


def test_read_late_risk_result_rejects_bad_probability():
    pred = pd.DataFrame(
        [
            {
                "late_delivery_probability": 1.4,
                "delivery_prediction": 1,
            }
        ]
    )

    with pytest.raises(ValueError):
        read_late_risk_result(pred)


def test_read_duration_result():
    pred = pd.DataFrame(
        [
            {
                "predicted_delivery_days": 6.25
            }
        ]
    )

    assert read_duration_result(pred) == 6.25


def test_read_duration_result_rejects_negative():
    pred = pd.DataFrame(
        [
            {
                "predicted_delivery_days": -2.0
            }
        ]
    )

    with pytest.raises(ValueError):
        read_duration_result(pred)