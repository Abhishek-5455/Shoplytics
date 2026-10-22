"""Shoplytics Streamlit entry point."""

from __future__ import annotations

from datetime import date

import streamlit as st

from ecommerce_ml.dashboard.utils import date_params

from ecommerce_ml.dashboard.views import (
    render_customers,
    render_delivery,
    render_overview,
    render_payments,
    render_predictions,
    render_sales,
)

from ecommerce_ml.models.predict_delivery_time import (
    predict_delivery_time,
)

from ecommerce_ml.models.predict_late_delivery import (
    predict_delivery_risk,
)


st.set_page_config(
    page_title="Shoplytics | E-commerce Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      .block-container {
          padding-top: 1.4rem;
          padding-bottom: 2rem;
      }

      [data-testid="stSidebar"] {
          border-right: 1px solid rgba(128,128,128,.2);
      }

      [data-testid="stMetric"] {
          background: rgba(128,128,128,.06);
          border: 1px solid rgba(128,128,128,.18);
          padding: 14px 16px;
          border-radius: 12px;
      }

      div[data-testid="stForm"] {
          border: 1px solid rgba(128,128,128,.22);
          padding: 18px;
          border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


st.sidebar.markdown("# SHOPLYTICS")
st.sidebar.caption("E-commerce Intelligence")
st.sidebar.divider()


PAGES = [
    "Business Overview",
    "Delivery Analytics",
    "Sales & Product Analytics",
    "Customer Analytics",
    "Payment Analytics",
    "ML Predictions",
]


page = st.sidebar.radio(
    "Navigate",
    PAGES,
    label_visibility="collapsed",
)


start_date = end_date = None

if page != "ML Predictions":

    st.sidebar.subheader("Purchase Date Range")

    start_date = st.sidebar.date_input(
        "Start date",
        value=date(2017, 1, 1),
        min_value=date(2016, 1, 1),
        max_value=date(2020, 12, 31),
        key="shoplytics_start_date",
    )

    end_date = st.sidebar.date_input(
        "End date",
        value=date(2018, 12, 31),
        min_value=date(2016, 1, 1),
        max_value=date(2020, 12, 31),
        key="shoplytics_end_date",
    )

    if start_date > end_date:
        st.sidebar.error(
            "Start date must be on or before end date."
        )
        st.stop()

    date_params(
        start_date,
        end_date,
    )

    st.sidebar.divider()

    st.sidebar.caption(
        f"Purchase dates: "
        f"{start_date:%d %b %Y} – {end_date:%d %b %Y}"
    )


st.sidebar.divider()

st.sidebar.caption(
    "Analytics cache: 60 seconds"
)


if st.sidebar.button(
    "Refresh dashboard data",
    use_container_width=True,
):
    st.cache_data.clear()
    st.rerun()


if page == "Business Overview":
    render_overview(start_date, end_date)

elif page == "Delivery Analytics":
    render_delivery(start_date, end_date)

elif page == "Sales & Product Analytics":
    render_sales(start_date, end_date)

elif page == "Customer Analytics":
    render_customers(start_date, end_date)

elif page == "Payment Analytics":
    render_payments(start_date, end_date)

elif page == "ML Predictions":
    render_predictions(
        predict_delivery_risk,
        predict_delivery_time,
    )