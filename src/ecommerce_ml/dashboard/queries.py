"""SQL query builders. Column names are fixed by application code."""

from __future__ import annotations

from datetime import date

from ecommerce_ml.dashboard.utils import date_params, date_predicate


ORDERS_DATE = date_predicate("order_purchase_timestamp")
ALIASED_ORDERS_DATE = date_predicate("o.order_purchase_timestamp")


def overview_queries():
    return {
        "overview": f"""
            SELECT COUNT(*) AS total_orders,
                   COUNT(*) FILTER (
                       WHERE order_status = 'delivered'
                   ) AS delivered_orders,
                   COUNT(*) FILTER (
                       WHERE order_status = 'canceled'
                   ) AS canceled_orders
            FROM public.orders
            WHERE {ORDERS_DATE};
        """,

        "payment_total": f"""
            SELECT COALESCE(
                SUM(p.payment_value), 0
            ) AS total_payment_value
            FROM public.payments p
            JOIN public.orders o
                ON o.order_id = p.order_id
            WHERE {ALIASED_ORDERS_DATE};
        """,

        "monthly": f"""
            SELECT DATE_TRUNC(
                       'month', order_purchase_timestamp
                   ) AS month,
                   COUNT(*) AS total_orders
            FROM public.orders
            WHERE {ORDERS_DATE}
            GROUP BY 1
            ORDER BY 1;
        """,

        "statuses": f"""
            SELECT order_status,
                   COUNT(*) AS orders
            FROM public.orders
            WHERE {ORDERS_DATE}
            GROUP BY order_status
            ORDER BY orders DESC;
        """,
    }


def delivery_query():
    return f"""
        SELECT
            order_id,
            order_status,
            order_purchase_timestamp,
            order_delivered_customer_date,
            order_estimated_delivery_date,

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
                WHEN order_delivered_customer_date IS NULL
                     OR order_estimated_delivery_date IS NULL
                    THEN 'Unknown'

                WHEN order_delivered_customer_date
                     > order_estimated_delivery_date
                    THEN 'Late'

                ELSE 'On time'
            END AS delivery_result

        FROM public.orders
        WHERE {ORDERS_DATE}
        ORDER BY order_purchase_timestamp DESC;
    """


def sales_queries():
    return {
        "monthly": f"""
            SELECT
                DATE_TRUNC(
                    'month', o.order_purchase_timestamp
                ) AS month,

                COALESCE(
                    SUM(p.payment_value), 0
                ) AS recorded_payment_value,

                COUNT(DISTINCT o.order_id) AS total_orders

            FROM public.orders o

            LEFT JOIN public.payments p
                ON p.order_id = o.order_id

            WHERE {ALIASED_ORDERS_DATE}

            GROUP BY 1
            ORDER BY 1;
        """,

        "category": f"""
            SELECT
                COALESCE(
                    t.product_category_name_english,
                    p.product_category_name,
                    'Unknown'
                ) AS category,

                COUNT(*) AS total_items,

                COALESCE(
                    SUM(oi.price), 0
                ) AS item_price_total

            FROM public.order_items oi

            JOIN public.orders o
                ON o.order_id = oi.order_id

            JOIN public.products p
                ON p.product_id = oi.product_id

            LEFT JOIN public.product_category_translation t
                ON p.product_category_name =
                   t.product_category_name

            WHERE {ALIASED_ORDERS_DATE}

            GROUP BY 1
            ORDER BY item_price_total DESC
            LIMIT 15;
        """,

        "products": f"""
            SELECT
                oi.product_id,
                COUNT(*) AS total_items,

                COALESCE(
                    SUM(oi.price), 0
                ) AS item_price_total,

                AVG(oi.price) AS average_item_price

            FROM public.order_items oi

            JOIN public.orders o
                ON o.order_id = oi.order_id

            WHERE {ALIASED_ORDERS_DATE}

            GROUP BY oi.product_id
            ORDER BY item_price_total DESC
            LIMIT 10;
        """,
    }


def customer_query():
    return f"""
        SELECT
            customer_unique_id,
            COUNT(DISTINCT o.order_id) AS order_count

        FROM public.customers c

        JOIN public.orders o
            ON o.customer_id = c.customer_id

        WHERE {ALIASED_ORDERS_DATE}

        GROUP BY customer_unique_id;
    """


def payment_queries():
    return {
        "methods": f"""
            SELECT
                COALESCE(
                    p.payment_type, 'Unknown'
                ) AS payment_type,

                COUNT(*) AS payment_records,

                COUNT(DISTINCT p.order_id) AS distinct_orders,

                COALESCE(
                    SUM(p.payment_value), 0
                ) AS total_payment_value,

                AVG(
                    p.payment_value
                ) AS average_payment_value,

                AVG(
                    p.payment_installments
                ) AS average_installments

            FROM public.payments p

            JOIN public.orders o
                ON o.order_id = p.order_id

            WHERE {ALIASED_ORDERS_DATE}

            GROUP BY 1
            ORDER BY total_payment_value DESC;
        """,

        "orders": f"""
            SELECT
                COUNT(DISTINCT p.order_id)
                    AS orders_with_payment

            FROM public.payments p

            JOIN public.orders o
                ON o.order_id = p.order_id

            WHERE {ALIASED_ORDERS_DATE};
        """,
    }


def query_params(start_date: date, end_date: date) -> tuple:
    return date_params(start_date, end_date)