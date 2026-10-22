"""Pure prediction-input builders and result readers."""

from __future__ import annotations

import pandas as pd

RISK_FEATURES = (
    "purchase_hour",
    "purchase_day_of_week",
    "purchase_month",
    "days_to_estimated_delivery",
    "item_count",
    "total_price",
    "total_freight",
)

DURATION_FEATURES = (
    "purchase_hour",
    "purchase_day_of_week",
    "purchase_month",
    "estimated_delivery_days",
    "item_count",
    "total_price",
    "total_freight",
)


def _validate_common(
    purchase_hour,
    purchase_day_of_week,
    purchase_month,
    item_count,
    total_price,
    total_freight,
):
    if not 0 <= int(purchase_hour) <= 23:
        raise ValueError(
            "purchase_hour must be between 0 and 23."
        )

    if not 0 <= int(purchase_day_of_week) <= 6:
        raise ValueError(
            "purchase_day_of_week must be between 0 and 6."
        )

    if not 1 <= int(purchase_month) <= 12:
        raise ValueError(
            "purchase_month must be between 1 and 12."
        )

    if int(item_count) < 1:
        raise ValueError(
            "item_count must be at least 1."
        )

    if float(total_price) < 0 or float(total_freight) < 0:
        raise ValueError(
            "Price and freight must be non-negative."
        )


def build_risk_features(
    *,
    purchase_hour,
    purchase_day_of_week,
    purchase_month,
    days_to_estimated_delivery,
    item_count,
    total_price,
    total_freight,
):
    _validate_common(
        purchase_hour,
        purchase_day_of_week,
        purchase_month,
        item_count,
        total_price,
        total_freight,
    )

    if float(days_to_estimated_delivery) < 0:
        raise ValueError(
            "days_to_estimated_delivery must be non-negative."
        )

    return pd.DataFrame(
        [
            {
                "purchase_hour": int(purchase_hour),
                "purchase_day_of_week": int(
                    purchase_day_of_week
                ),
                "purchase_month": int(purchase_month),
                "days_to_estimated_delivery": float(
                    days_to_estimated_delivery
                ),
                "item_count": int(item_count),
                "total_price": float(total_price),
                "total_freight": float(total_freight),
            }
        ],
        columns=RISK_FEATURES,
    )


def build_duration_features(
    *,
    purchase_hour,
    purchase_day_of_week,
    purchase_month,
    estimated_delivery_days,
    item_count,
    total_price,
    total_freight,
):
    _validate_common(
        purchase_hour,
        purchase_day_of_week,
        purchase_month,
        item_count,
        total_price,
        total_freight,
    )

    if float(estimated_delivery_days) < 0:
        raise ValueError(
            "estimated_delivery_days must be non-negative."
        )

    return pd.DataFrame(
        [
            {
                "purchase_hour": int(purchase_hour),
                "purchase_day_of_week": int(
                    purchase_day_of_week
                ),
                "purchase_month": int(purchase_month),
                "estimated_delivery_days": float(
                    estimated_delivery_days
                ),
                "item_count": int(item_count),
                "total_price": float(total_price),
                "total_freight": float(total_freight),
            }
        ],
        columns=DURATION_FEATURES,
    )


def read_late_risk_result(prediction):
    probability = float(
        prediction.iloc[0]["late_delivery_probability"]
    )

    late_flag = int(
        prediction.iloc[0]["delivery_prediction"]
    )

    if not 0 <= probability <= 1:
        raise ValueError(
            "Model returned a probability outside [0, 1]."
        )

    return probability, late_flag


def read_duration_result(prediction) -> float:
    days = float(
        prediction.iloc[0]["predicted_delivery_days"]
    )

    if days < 0:
        raise ValueError(
            "Model returned a negative delivery duration."
        )

    return days