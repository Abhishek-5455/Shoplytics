from ecommerce_ml.analysis.loader import load_csv

import pandas as pd

def build_delivery_features():
    orders = load_csv("olist_orders_dataset.csv")
    items = load_csv("olist_order_items_dataset.csv")

    date_columns = [
        'order_purchase_timestamp',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for column in date_columns:
        orders[column] = pd.to_datetime(orders[column], errors='coerce')

    delivered = orders.loc[
        orders['order_status'].eq('delivered')
    ].dropna(subset=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]).copy()

    item_features = (
        items.groupby('order_id')
        .agg(
            item_count=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum")
        )
        .reset_index()
    )

    data = delivered.merge(
        item_features,
        on='order_id',
        how='inner',
        validate='one_to_one'
    )

    data['purchase_hour'] = data['order_purchase_timestamp'].dt.hour
    data['purchase_day_of_week'] = data['order_purchase_timestamp'].dt.day_of_week
    data['purchase_month'] = data['order_purchase_timestamp'].dt.month

    data['days_to_estimated_delivery'] = (
        data['order_estimated_delivery_date'] - data['order_purchase_timestamp']
    ).dt.total_seconds() / 86400

    data['is_late'] = (
        data['order_estimated_delivery_date'] < data['order_delivered_customer_date']
    ).astype('int8')

    feature_columns = [
        "order_id",
        "purchase_hour",
        "purchase_day_of_week",
        "purchase_month",
        "days_to_estimated_delivery",
        "item_count",
        "total_price",
        "total_freight",
        "is_late"
    ]

    return data[feature_columns].reset_index(drop=True)


