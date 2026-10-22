# utils.py

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

ALLOWED_DATE_COLUMNS = {
    "order_purchase_timestamp",
    "o.order_purchase_timestamp",
}


def date_params(start_date: date, end_date: date) -> tuple[date, date]:
    """Return a half-open date interval that includes the entire end date."""
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    return start_date, end_date + timedelta(days=1)


def date_predicate(column: str) -> str:
    """Build a date predicate for an explicitly allowed SQL column."""
    if column not in ALLOWED_DATE_COLUMNS:
        raise ValueError(f"Unsupported date column: {column}")
    return f"{column} >= %s AND {column} < %s"


def safe_scalar(value: Any, default: Any = 0) -> Any:
    """Replace pandas/null scalar values with a fallback."""
    try:
        return default if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def format_money(value: Any) -> str:
    return f"{float(safe_scalar(value, 0)):,.2f}"