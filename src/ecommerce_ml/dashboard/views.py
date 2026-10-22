from __future__ import annotations

import pandas as pd
import streamlit as st

from ecommerce_ml.dashboard.data import read_query
from ecommerce_ml.dashboard.prediction_logic import (
    build_duration_features,
    build_risk_features,
    read_duration_result,
    read_late_risk_result
)

from ecommerce_ml.dashboard.queries import (
    customer_query,
    delivery_query,
    overview_queries,
    payment_queries,
    query_params,
    sales_queries
)

from ecommerce_ml.dashboard.utils import format_money

def page_header(title, subtitle):
    st.title(title)
    st.caption(subtitle)

def render_overview(start_date, end_date):
    page_header(
        "Business Overview",
        "A high level view of order activity, payment value, and fulfillment"
    )

    params = query_params(start_date, end_date)
    queries = overview_queries()

    with st.spinner("Loading Overview metrics"):
        overview = read_query(
            queries['overview'], params
        )

        payment_total = read_query(
            queries['payment_total'], params
        )

        monthly = read_query(
            queries['monthly'], params
        )

        statuses = read_query(
            queries['statuses'], params
        )

    if not overview.empty:
        m = overview.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            'Orders',
            f"{int(m['total_orders']):,}"
        )

        c2.metric(
            "Delivered",
            f"{int(m['delivered_orders']):,}"
        )

        c3.metric(
            "Canceled",
            f"{int(m['canceled_orders']):,}"
        )

        amount = (
            payment_total.iloc[0]['total_payment_value']
            if not payment_total.empty else 0
        )

        c4.metric(
            "Recorded Payment Value",
            format_money(amount)
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Monthly Orders")
        if not monthly.empty:
            monthly['month'] = pd.to_datetime(
                monthly['month']
            )

            st.bar_chart(
                monthly.set_index('month')['total_orders']
            )
        else:
            st.info("No orders in this date range")

    with right:
        st.subheader("Order Status")
        if not statuses.empty:
            st.bar_chart(
                statuses.set_index('order_status')['orders']
            )
        else:
            st.info("No order statuses available")

    st.caption(
        "Payment value sums payment rows for orders purchased"
        "in the selected period, it is not net revenue or profit"
    )

def render_delivery(start_date, end_date):
    page_header(
        "Delivery Analytics",
        "Compare delivery duration with estimated delivery dates"
    )

    with st.spinner("Analyzing delivery records"):
        delivery = read_query(
            delivery_query(),
            query_params(start_date, end_date)
        )

    if delivery.empty:
        st.info("No orders found for this period")
        return
    delivered = delivery.dropna(
        subset=['actual_delivery_days']
    ).copy()

    evaluated = delivery[
        delivery['delivery_result'].isin(
            ['Late', 'On time']
        )
    ].copy()

    late_count = int(
        (evaluated['delivery_result'] == 'Late').sum()
    )

    late_rate = (
        late_count / len(evaluated) * 100
        if len(evaluated) else 0
    )

    avg_duration = delivered[
        'actual_delivery_days'
    ].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Orders in Period",
        f"{len(delivery)}"
    )
    c2.metric(
        "Delivered with Dates",
        f"{len(delivered):,}"
    )
    c3.metric(
        "Average Delivery (days)",
        (
            f"{avg_duration:.2f}"
            if pd.notna(avg_duration) else 'N/A'
        )
    )
    c4.metric(
        "Late Rate",
        f"{late_rate:.1f}%"
    )

    st.subheader("Delivery Outcome")

    if not evaluated.empty:
        outcome = (
            evaluated['delivery_result']
            .value_counts()
            .rename_axis('result')
            .to_frame('orders')
        )

        st.bar_chart(outcome)

    st.subheader("Actual vs estimated delivery time")

    duration_df = delivered[
        [
            'actual_delivery_days',
            'estimated_delivery_days'
        ]
    ].dropna()

    if not duration_df.empty:
        st.dataframe(
            duration_df,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Order level Delivery data")
    st.dataframe(
        delivery,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Late rate uses orders with both actual and estimated delivery timestamp."
    )

def render_sales(start_date, end_date):
    page_header(
        "Sales & Product Analysis",
        'Explore monthly payment value, item sales, categories, and products'
    )

    params = query_params(start_date, end_date)
    queries = sales_queries()

    monthly_sales = read_query(
        queries['monthly'], params
    )

    if not monthly_sales.empty:
        a, b = st.columns(2)
        a.metric(
            'Recorded Payment Value',
            format_money(
                monthly_sales['recorded_payment_value'].sum()
            )
        )

        b.metric(
            "Orders",
            f"{int(monthly_sales['total_orders'].sum()):,}"
        )

        monthly_sales['month'] = pd.to_datetime(
            monthly_sales['month']
        )

        left, right = st.columns(2)

        with left:
            st.subheader(
                'Monthly Recorded Payment Value'
            )

            st.line_chart(
                monthly_sales.set_index("month")[
                    'recorded_payment_value'
                ]
            )
        with right:
            st.subheader("Monthly Orders")
            st.bar_chart(
                monthly_sales.set_index('month')['total_orders']
            )
    else:
        st.info(
            "No sales activity in this date range"
        )

    category = read_query(
        queries["category"], params
    )

    st.divider()
    st.subheader("Top Categories by Item Price")

    if not category.empty:
        st.bar_chart(
            category.set_index("category")[
                "item_price_total"
            ]
        )

        st.dataframe(
            category,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("No category records found.")

    products = read_query(
        queries["products"], params
    )

    st.divider()
    st.subheader("Top 10 Products")

    if not products.empty:
        st.dataframe(
            products,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No product records found.")

    st.caption(
        "Item price totals exclude freight, discounts, "
        "refunds, and cost. Payment value is not profit."
    )


def render_customers(start_date, end_date):
    page_header(
        "Customer Analytics",
        "Understand unique customers and purchase "
        "frequency in the selected period.",
    )

    with st.spinner("Analyzing customer behavior..."):
        customer_summary = read_query(
            customer_query(),
            query_params(start_date, end_date),
        )

    if customer_summary.empty:
        st.info(
            "No purchasing customers found in this date range."
        )
        return

    unique_customers = int(
        customer_summary["customer_unique_id"].nunique()
    )

    order_total = int(
        customer_summary["order_count"].sum()
    )

    avg_orders = customer_summary[
        "order_count"
    ].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Purchasing Customers",
        f"{unique_customers:,}",
    )

    c2.metric(
        "Orders",
        f"{order_total:,}",
    )

    c3.metric(
        "Avg Orders / Purchasing Customer",
        f"{avg_orders:.2f}",
    )

    frequency = (
        customer_summary["order_count"]
        .value_counts()
        .sort_index()
        .rename_axis("orders_per_customer")
        .to_frame("customer_count")
    )

    st.subheader("Order Frequency")

    st.bar_chart(frequency)

    st.dataframe(
        frequency.reset_index(),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Customers are grouped by customer_unique_id "
        "and included only when they ordered during "
        "the selected window."
    )


def render_payments(start_date, end_date):
    page_header(
        "Payment Analytics",
        "Compare payment methods and installment patterns "
        "for orders purchased in the selected period.",
    )

    params = query_params(start_date, end_date)
    queries = payment_queries()

    payment = read_query(
        queries["methods"], params
    )

    payment_orders = read_query(
        queries["orders"], params
    )

    if payment.empty:
        st.info(
            "No payment data found for this period."
        )
        return

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Payment Records",
        f"{int(payment['payment_records'].sum()):,}",
    )

    order_count = (
        int(
            payment_orders.iloc[0][
                "orders_with_payment"
            ]
        )
        if not payment_orders.empty
        else 0
    )

    c2.metric(
        "Orders with Payment",
        f"{order_count:,}",
    )

    c3.metric(
        "Recorded Payment Value",
        format_money(
            payment["total_payment_value"].sum()
        ),
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Payment Value by Method")

        st.bar_chart(
            payment.set_index("payment_type")[
                "total_payment_value"
            ]
        )

    with right:
        st.subheader("Payment Record Count")

        st.bar_chart(
            payment.set_index("payment_type")[
                "payment_records"
            ]
        )

    st.subheader("Payment Method Details")

    st.dataframe(
        payment,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Payment records can be multiple per order. "
        "Sum is recorded payment activity, not necessarily net revenue."
    )


def render_predictions(predict_risk, predict_time):
    page_header(
        "ML Predictions",
        "Estimate late-delivery risk and expected delivery "
        "duration using trained models.",
    )

    st.info(
        "Predictions are estimates based on training data, "
        "not delivery guarantees."
    )

    with st.expander(
        "How to use the prediction inputs"
    ):
        st.markdown(
            "- **Purchase hour:** 0–23\n"
            "- **Purchase weekday:** Monday = 0 through Sunday = 6\n"
            "- **Purchase month:** 1–12\n"
            "- **Days to estimated delivery:** estimated date minus purchase date\n"
            "- **Item count, total price, freight:** order-level values"
        )

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Late Delivery Risk")

        with st.form("late_risk_form"):
            hour = st.number_input(
                "Purchase hour",
                0, 23, 12,
                key="risk_hour",
            )

            weekday = st.selectbox(
                "Purchase weekday (Monday = 0)",
                options=list(range(7)),
                index=2,
                key="risk_weekday",
            )

            month = st.selectbox(
                "Purchase month",
                options=list(range(1, 13)),
                index=5,
                key="risk_month",
            )

            days = st.number_input(
                "Days to estimated delivery",
                min_value=0.0,
                value=15.0,
                step=1.0,
                key="risk_est_days",
            )

            items = st.number_input(
                "Item count",
                min_value=1,
                value=1,
                step=1,
                key="risk_items",
            )

            price = st.number_input(
                "Total item price",
                min_value=0.0,
                value=100.0,
                step=10.0,
                key="risk_price",
            )

            freight = st.number_input(
                "Total freight",
                min_value=0.0,
                value=20.0,
                step=5.0,
                key="risk_freight",
            )

            submitted = st.form_submit_button(
                "Estimate Late Risk",
                use_container_width=True,
            )

        if submitted:
            try:
                features = build_risk_features(
                    purchase_hour=hour,
                    purchase_day_of_week=weekday,
                    purchase_month=month,
                    days_to_estimated_delivery=days,
                    item_count=items,
                    total_price=price,
                    total_freight=freight,
                )

                with st.status(
                    "Evaluating delivery risk...",
                    expanded=True,
                ) as status:

                    result = predict_risk(features)

                    probability, flag = (
                        read_late_risk_result(result)
                    )

                    status.update(
                        label="Risk estimate ready",
                        state="complete",
                        expanded=False,
                    )

                st.metric(
                    "Late Delivery Probability",
                    f"{probability:.1%}",
                )

                if flag:
                    st.warning(
                        "Model classification: "
                        "Higher late-delivery risk"
                    )
                else:
                    st.success(
                        "Model classification: "
                        "Not classified as late"
                    )

            except Exception as exc:
                st.error(
                    "Could not generate late-delivery estimate: "
                    f"{exc}"
                )

    with right:
        st.subheader("Estimated Delivery Duration")

        with st.form("delivery_time_form"):
            hour = st.number_input(
                "Purchase hour",
                0, 23, 12,
                key="time_hour",
            )

            weekday = st.selectbox(
                "Purchase weekday (Monday = 0)",
                options=list(range(7)),
                index=2,
                key="time_weekday",
            )

            month = st.selectbox(
                "Purchase month",
                options=list(range(1, 13)),
                index=5,
                key="time_month",
            )

            days = st.number_input(
                "Estimated delivery days",
                min_value=0.0,
                value=15.0,
                step=1.0,
                key="time_est_days",
            )

            items = st.number_input(
                "Item count",
                min_value=1,
                value=1,
                step=1,
                key="time_items",
            )

            price = st.number_input(
                "Total item price",
                min_value=0.0,
                value=100.0,
                step=10.0,
                key="time_price",
            )

            freight = st.number_input(
                "Total freight",
                min_value=0.0,
                value=20.0,
                step=5.0,
                key="time_freight",
            )

            submitted = st.form_submit_button(
                "Estimate Delivery Time",
                use_container_width=True,
            )

        if submitted:
            try:
                features = build_duration_features(
                    purchase_hour=hour,
                    purchase_day_of_week=weekday,
                    purchase_month=month,
                    estimated_delivery_days=days,
                    item_count=items,
                    total_price=price,
                    total_freight=freight,
                )

                with st.status(
                    "Estimating delivery duration...",
                    expanded=True,
                ) as status:

                    result = predict_time(features)

                    duration = read_duration_result(
                        result
                    )

                    status.update(
                        label="Duration estimate ready",
                        state="complete",
                        expanded=False,
                    )

                st.metric(
                    "Predicted Delivery Duration",
                    f"{duration:.2f} days",
                )

            except Exception as exc:
                st.error(
                    "Could not generate delivery-time estimate: "
                    f"{exc}"
                )

    st.divider()

    st.caption(
        "Prediction inputs must match the feature definitions "
        "used during model training."
    )