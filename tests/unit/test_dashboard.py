
from unittest.mock import MagicMock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

APP_PATH = ROOT_DIR / "src/ecommerce_ml/dashboard/app.py"


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
