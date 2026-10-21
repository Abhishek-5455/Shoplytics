
import pandas as pd
import streamlit as st

from ecommerce_ml.db import get_connection
from ecommerce_ml.models.predict_delivery_time import predict_delivery_time
from ecommerce_ml.models.predict_late_delivery import predict_delivery_risk

st.set_page_config(
    page_title="Shoplytics | E-commerce Intelligence",
    layout="wide",
)


st.sidebar.markdown(
    """
    <div style="padding: 12px 4px 18px 4px;">
        <div style="font-size: 26px; font-weight: 750; letter-spacing: -1px;">
            Shoplytics
        </div>
        <div style="font-size: 12px; opacity: 0.65; margin-top: 4px;">
            E-COMMERCE INTELLIGENCE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

st.sidebar.caption("WORKSPACE")

page = st.sidebar.radio(
    "Navigate",
    [
        "Business Overview",
        "Delivery Analytics",
        "ML Predictions",
    ],
    format_func=lambda item: {
        "Business Overview": "Business Overview",
        "Delivery Analytics": "Delivery Analytics",
        "ML Predictions": "ML Predictions",
    }[item],
    label_visibility="collapsed",
    key="shoplytics_navigation",
)

st.sidebar.divider()

st.sidebar.caption("DATA")
st.sidebar.caption("PostgreSQL · Read-only analytics")
st.sidebar.caption("Models · Local inference")

st.sidebar.divider()

if st.sidebar.button("Refresh dashboard", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Shoplytics | Phase 7")


@st.cache_data(ttl=60)
def read_query(query: str) -> pd.DataFrame:
    """Execute a read-only query and return its results."""
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]

        return pd.DataFrame(rows, columns=columns)
    finally:
        connection.close()


def load_overview():
    metrics = read_query(
        """
        SELECT
            COUNT(*) AS total_orders,
            COUNT(*) FILTER (
                WHERE order_status = 'delivered'
            ) AS delivered_orders,
            COUNT(*) FILTER (
                WHERE order_status = 'canceled'
            ) AS canceled_orders,
            COUNT(*) FILTER (
                WHERE order_status = 'delivered'
                  AND order_delivered_customer_date IS NOT NULL
                  AND order_estimated_delivery_date IS NOT NULL
                  AND order_delivered_customer_date >
                      order_estimated_delivery_date
            ) AS late_orders,
            COUNT(*) FILTER (
                WHERE order_status = 'delivered'
                  AND order_delivered_customer_date IS NOT NULL
                  AND order_estimated_delivery_date IS NOT NULL
            ) AS comparable_deliveries
        FROM public.orders
        """
    )

    payments = read_query(
        """
        SELECT COALESCE(SUM(payment_value), 0) AS total_payment_value
        FROM public.payments
        """
    )

    monthly_orders = read_query(
        """
        SELECT
            DATE_TRUNC('month', order_purchase_timestamp) AS month,
            COUNT(*) AS order_count
        FROM public.orders
        WHERE order_purchase_timestamp IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    )

    order_status = read_query(
        """
        SELECT order_status, COUNT(*) AS order_count
        FROM public.orders
        GROUP BY order_status
        ORDER BY order_count DESC
        """
    )

    return metrics.iloc[0], payments.iloc[0], monthly_orders, order_status

if st.button("Refresh dashboard"):
    st.cache_data.clear()
    st.rerun()

try:
    if page == 'Business Overview':
        st.title("Shoplytics")
        st.subheader("Business Overview")

        with st.spinner("Loading data from PostgreSQL..."):
            metrics, payments, monthly_orders, order_status = load_overview()
    
        total_orders = int(metrics["total_orders"])
        delivered_orders = int(metrics["delivered_orders"])
        canceled_orders = int(metrics["canceled_orders"])
        late_orders = int(metrics["late_orders"])
        comparable = int(metrics["comparable_deliveries"])
        total_payment = float(payments["total_payment_value"])
    
        late_rate = late_orders / comparable * 100 if comparable else 0.0
    
        st.subheader("Business Overview")
    
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Orders", f"{total_orders:,}")
        col2.metric("Delivered Orders", f"{delivered_orders:,}")
        col3.metric("Canceled Orders", f"{canceled_orders:,}")
        col4.metric(
            "Late Delivery Rate",
            f"{late_rate:.2f}%",
            help="Among delivered orders with both actual and estimated delivery dates.",
        )
    
        st.metric("Total Payment Value", f"{total_payment:,.2f}")
    
        left, right = st.columns(2)
    
        with left:
            st.subheader("Monthly Order Volume")
            if not monthly_orders.empty:
                chart_data = monthly_orders.set_index("month")
                st.line_chart(chart_data["order_count"])
            else:
                st.info("No order data available.")
    
        with right:
            st.subheader("Orders by Status")
            if not order_status.empty:
                chart_data = order_status.set_index("order_status")
                st.bar_chart(chart_data["order_count"])
            else:
                st.info("No order-status data available.")
    
        st.caption(
            f"{late_orders:,} late deliveries out of "
            f"{comparable:,} deliveries with comparable dates."
        )

    elif page == 'Delivery Analytics':
        st.title("Shoplytics")
        st.subheader("Delivery Analytics")

        delivery_data = read_query(
            """
            SELECT
                DATE_TRUNC('month', order_purchase_timestamp) AS month,
                EXTRACT(
                    EPOCH FROM (
                        order_delivered_customer_date
                        - order_purchase_timestamp
                    )
                ) / 86400.0 AS actual_delivery_days,
                EXTRACT(
                    EPOCH FROM (
                        order_estimated_delivery_date
                        - order_purchase_timestamp
                    )
                ) / 86400.0 AS estimated_delivery_days,
                CASE
                    WHEN order_delivered_customer_date >
                        order_estimated_delivery_date
                    THEN 'Late'
                    ELSE 'On time'
                END AS delivery_status
            FROM public.orders
            WHERE order_status = 'delivered'
            AND order_purchase_timestamp IS NOT NULL
            AND order_delivered_customer_date IS NOT NULL
            AND order_estimated_delivery_date IS NOT NULL
            AND order_delivered_customer_date >= order_purchase_timestamp
            ORDER BY month
            """
        )

        if delivery_data.empty:
            st.info("No completed deliveries with valid dates were found.")
        else:
            actual_days = delivery_data["actual_delivery_days"]
            estimated_days = delivery_data["estimated_delivery_days"]

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Average Actual Delivery",
                f"{actual_days.mean():.2f} days",
            )
            metric2.metric(
                "Average Estimated Delivery",
                f"{estimated_days.mean():.2f} days",
            )
            metric3.metric(
                "Average Delivery Variance",
                f"{(actual_days - estimated_days).mean():.2f} days",
            )

            monthly_delivery = (
                delivery_data.groupby("month", as_index=False)
                .agg(
                    actual_days=("actual_delivery_days", "mean"),
                    estimated_days=("estimated_delivery_days", "mean"),
                )
                .set_index("month")
            )

            st.write("Actual vs Estimated Delivery Duration")
            st.line_chart(monthly_delivery)

            status_counts = (
                delivery_data["delivery_status"]
                .value_counts()
                .rename_axis("delivery_status")
                .to_frame("orders")
            )

            st.write("Completed Delivery Status")
            st.bar_chart(status_counts)

            st.write("Delivery Records")
            st.dataframe(
                delivery_data,
                use_container_width=True,
                hide_index=True,
            )


    
    # st.divider()
    elif page == 'ML Predictions':
        st.title("Shoplytics")
        st.subheader("Machine Learning Predictions")

        prediction_tab1, prediction_tab2 = st.tabs(
            ["Late-Delivery Risk", "Delivery-Time Estimate"]
        )

        with prediction_tab1:
            st.write("Estimate the probability that an order will arrive late.")

            with st.form("late_delivery_form"):
                purchase_hour = st.number_input(
                    "Purchase hour", min_value=0, max_value=23, value=10
                )
                purchase_day = st.number_input(
                    "Purchase day of week (Monday = 0)",
                    min_value=0,
                    max_value=6,
                    value=2,
                )
                purchase_month = st.number_input(
                    "Purchase month", min_value=1, max_value=12, value=6
                )
                days_to_estimated = st.number_input(
                    "Days until estimated delivery",
                    min_value=0.0,
                    value=15.0,
                )
                item_count = st.number_input(
                    "Number of items", min_value=1, value=1
                )
                total_price = st.number_input(
                    "Total item price", min_value=0.0, value=100.0
                )
                total_freight = st.number_input(
                    "Total freight cost", min_value=0.0, value=20.0
                )

                late_submitted = st.form_submit_button("Predict late-delivery risk")

            
            if late_submitted:
                features = pd.DataFrame(
                    [
                        {
                            "purchase_hour": purchase_hour,
                            "purchase_day_of_week": purchase_day,
                            "purchase_month": purchase_month,
                            "days_to_estimated_delivery": days_to_estimated,
                            "item_count": item_count,
                            "total_price": total_price,
                            "total_freight": total_freight,
                        }
                    ]
                )
                try:
                    with st.spinner("Analyzing delivery risk..."):
                        result = predict_delivery_risk(features).iloc[0]
                        probability = float(result["late_delivery_probability"])
                        prediction = int(result["delivery_prediction"])

                    st.markdown("### Prediction Result")

                    with st.container(border=True):
                        left, right = st.columns([2, 1])

                        with left:
                            st.caption("LATE-DELIVERY PROBABILITY")
                            st.metric(
                                "Risk score",
                                f"{probability:.1%}",
                            )
                            st.progress(
                                max(0.0, min(1.0, probability)),
                                text=f"{probability:.1%} estimated risk",
                            )

                        with right:
                            st.caption("MODEL CLASSIFICATION")
                            if prediction == 1:
                                st.error("Late delivery")
                                st.write(
                                    "The model classifies this order as "
                                    "at risk of arriving late."
                                )
                            else:
                                st.success("Not classified as late")
                                st.write(
                                    "The model does not classify this order "
                                    "as likely to arrive late."
                                )

                        st.caption(
                            "This result is a model estimate, not a guarantee "
                            "of actual delivery performance."
                        )

                except Exception as exc:
                    st.error(f"Late-delivery prediction failed: {exc}")


        with prediction_tab2:
            st.write("Estimate delivery duration in days.")

            with st.form("delivery_time_form"):
                time_purchase_hour = st.number_input(
                    "Purchase hour", min_value=0, max_value=23, value=10,
                    key="time_purchase_hour",
                )
                time_purchase_day = st.number_input(
                    "Purchase day of week (Monday = 0)",
                    min_value=0,
                    max_value=6,
                    value=2,
                    key="time_purchase_day",
                )
                time_purchase_month = st.number_input(
                    "Purchase month", min_value=1, max_value=12, value=6,
                    key="time_purchase_month",
                )
                estimated_days = st.number_input(
                    "Estimated delivery days",
                    min_value=0.0,
                    value=15.0,
                    key="time_estimated_days",
                )
                time_item_count = st.number_input(
                    "Number of items", min_value=1, value=1,
                    key="time_item_count",
                )
                time_total_price = st.number_input(
                    "Total item price", min_value=0.0, value=100.0,
                    key="time_total_price",
                )
                time_total_freight = st.number_input(
                    "Total freight cost", min_value=0.0, value=20.0,
                    key="time_total_freight",
                )

                time_submitted = st.form_submit_button("Predict delivery time")

            
            if time_submitted:
                features = pd.DataFrame(
                    [
                        {
                            "purchase_hour": time_purchase_hour,
                            "purchase_day_of_week": time_purchase_day,
                            "purchase_month": time_purchase_month,
                            "estimated_delivery_days": estimated_days,
                            "item_count": time_item_count,
                            "total_price": time_total_price,
                            "total_freight": time_total_freight,
                        }
                    ]
                )
                try:
                    with st.spinner("Estimating delivery duration..."):
                        result = predict_delivery_time(features).iloc[0]
                        predicted_days = float(
                            result["predicted_delivery_days"]
                        )

                    st.markdown("### Prediction Result")

                    with st.container(border=True):
                        st.caption("ESTIMATED DELIVERY DURATION")

                        st.metric(
                            "Predicted delivery time",
                            f"{predicted_days:.2f} days",
                        )

                        if predicted_days >= 0:
                            st.progress(
                                min(predicted_days / 30.0, 1.0),
                                text="Duration scale: 0–30 days",
                            )

                        st.info(
                            f"The model estimates approximately "
                            f"{predicted_days:.1f} days from purchase "
                            f"to customer delivery."
                        )

                        st.caption(
                            "The result is a model estimate, not a guaranteed "
                            "delivery date."
                        )

                except Exception as exc:
                    st.error(f"Delivery-time prediction failed: {exc}")

except Exception as exc:
    st.error("Could not load dashboard data from PostgreSQL.")
    st.code(f"{type(exc).__name__}: {exc}")
