
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ecommerce_ml.dashboard.views import (
    load_cluster_report,
    summarize_cluster_report,
)

ROOT_DIR = Path(__file__).parent.parent.parent

APP_PATH = ROOT_DIR / "src/ecommerce_ml/dashboard/app.py"


def test_load_cluster_report_reads_exported_schema(tmp_path):
    report_path = tmp_path / "customer_cluster_report.csv"
    pd.DataFrame(
        [
            {
                "customer_unique_id": "customer-1",
                "recency": 10,
                "frequency": 2,
                "monetary": 25.0,
                "cluster": 0,
                "segment_names": "Repeat Buyers",
            }
        ]
    ).to_csv(report_path)

    report = load_cluster_report(report_path)

    assert list(report.columns) == [
        "customer_unique_id",
        "recency",
        "frequency",
        "monetary",
        "cluster",
        "segment_names",
    ]
    assert report.iloc[0]["customer_unique_id"] == "customer-1"


def test_load_cluster_report_rejects_missing_columns(tmp_path):
    report_path = tmp_path / "customer_cluster_report.csv"
    pd.DataFrame({"customer_unique_id": ["customer-1"]}).to_csv(
        report_path
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_cluster_report(report_path)


def test_summarize_cluster_report_calculates_cluster_profiles():
    report = pd.DataFrame(
        [
            {
                "customer_unique_id": "customer-1",
                "recency": 10,
                "frequency": 2,
                "monetary": 20.0,
                "cluster": 0,
                "segment_names": "Repeat Buyers",
            },
            {
                "customer_unique_id": "customer-2",
                "recency": 20,
                "frequency": 4,
                "monetary": 40.0,
                "cluster": 0,
                "segment_names": "Repeat Buyers",
            },
            {
                "customer_unique_id": "customer-3",
                "recency": 30,
                "frequency": 1,
                "monetary": 10.0,
                "cluster": 1,
                "segment_names": "Recent One-Time Buyers",
            },
        ]
    )

    summary = summarize_cluster_report(report)

    assert summary["customer_count"].tolist() == [2, 1]
    assert summary["customer_percentage"].tolist() == pytest.approx(
        [200 / 3, 100 / 3]
    )
    assert summary.loc[0, "avg_monetary"] == 30.0


def test_dashboard_loads_business_overview():
    """Dashboard should render without errors."""
    with patch(
        "ecommerce_ml.dashboard.app.get_connection",
        create=True,
    ):
        app = AppTest.from_file(APP_PATH)
        app.run(timeout=30)

    assert not app.exception


def test_date_params_includes_entire_end_date():
    """Date range should include the selected end date."""
    from datetime import date

    from ecommerce_ml.dashboard.app import date_params

    start, end = date_params(
        date(2017, 1, 1),
        date(2017, 1, 31),
    )

    assert start.day == 1
    assert end.day == 1
    assert end.month == 2
