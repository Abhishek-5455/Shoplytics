import pandas as pd

from ecommerce_ml.analysis.loader import load_csv

def build_delivery_time_features() -> pd.DataFrame:
    orders = load_csv('olist_orders_dataset.csv')
    items = load_csv('olist_order_items_dataset.csv')

    date_columns = [
        'order_purchase_timestamp',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for col in date_columns:
        orders[col] = pd.to_datetime(
            orders[col], errors='coerce'
        )

    delivered = orders.loc[
        orders['order_status'].eq('delivered')
    ].dropna(
        subset=[
            'order_purchase_timestamp',
            'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]
    ).copy()

    item_features = (
        items.groupby('order_id')
        .agg(
            item_count=('order_item_id', 'count'),
            total_price=('price', 'sum'),
            total_freight=('freight_value', 'sum')
        )
        .reset_index()
    )

    data = delivered.merge(
        item_features,
        how='inner',
        on='order_id',
        validate='one_to_one'
    )

    data['purchase_hour'] = data['order_purchase_timestamp'].dt.hour
    data['purchase_day_of_week'] = data['order_purchase_timestamp'].dt.day_of_week
    data['purchase_month'] = data['order_purchase_timestamp'].dt.month

    data['estimated_delivery_days'] = (
        data['order_estimated_delivery_date'] - data['order_purchase_timestamp']
    ).dt.total_seconds() / 86400

    data['delivery_time_days'] = (
        data['order_delivered_customer_date'] - data['order_purchase_timestamp']
    ).dt.total_seconds() / 86400

    data = data.loc[
        data['delivery_time_days'].ge(0) 
        & data['estimated_delivery_days'].ge(0)
    ].copy()

    features = [
        'order_id',
        'order_purchase_timestamp',
        'purchase_hour',
        'purchase_day_of_week',
        'purchase_month',
        'estimated_delivery_days',
        'item_count',
        'total_price',
        'total_freight',
        'delivery_time_days'
    ]

    return data[features]

if __name__ == "__main__":
    data = build_delivery_time_features()

    print("Shape: ", data.shape)
    print("Columns: ", data.columns.tolist())
    print(data.head(10).to_string(index=False))