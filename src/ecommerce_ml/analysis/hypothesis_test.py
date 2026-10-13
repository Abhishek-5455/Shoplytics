import pandas as pd
from scipy.stats import mannwhitneyu

from .loader import load_csv

def compare_deliverd_and_canceled_price():
    orders = load_csv("olist_orders_dataset.csv")
    items = load_csv("olist_order_items_dataset.csv")

    order_value = (
        items.groupby('order_id')['price']
        .sum()
        .rename('order_value')
        .reset_index()
    )

    data = orders[['order_id', 'order_status']].merge(
        order_value, 
        on='order_id', 
        how='inner',
        validate='one_to_one'
    )

    delivered = data.loc[
        data['order_status'] == 'delivered', 'order_value'
    ]

    canceled = data.loc[
        data['order_status'] == 'canceled', 'order_value'
    ]

    if delivered.empty or canceled.empty:
        raise ValueError("Delivered and Canceled must contain data")

    statistics, p_value = mannwhitneyu(
        delivered,
        canceled,
        alternative='two-sided'
    )

    return {
        'delivered_count': len(delivered),
        'canceled_count': len(canceled),
        'delivered_median': delivered.median(),
        'canceled_median': canceled.median(),
        'u_statistics': statistics,
        'p_value': p_value,
        'significant_at_0.05': p_value < 0.05
    }

if __name__ == '__main__':
    for key, value in compare_deliverd_and_canceled_price().items():
        print(f"{key}: {value}")