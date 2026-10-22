from __future__ import annotations

import pandas as pd
import streamlit as st

from ecommerce_ml.db import get_connection


@st.cache_data(ttl=60, show_spinner=False)
def read_query(query: str, params: tuple = ()) -> pd.DataFrame:
    """Run an analytics query and return a DataFrame, closing the connection."""
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

            columns = (
                [column.name for column in cursor.description]
                if cursor.description
                else []
            )

        return pd.DataFrame(rows, columns=columns)

    finally:
        connection.close()