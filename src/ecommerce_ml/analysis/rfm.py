import pandas as pd

from .loader import load_csv

def build_rfm(reference_date: str | None = None) -> pd.DataFrame:
    orders = load_csv("olist_orders_dataset.csv")
    items = load_csv("olist_order_items_dataset.csv")
    customers = load_csv("olist_customers_dataset.csv")

    orders['order_purchase_timestamp'] = pd.to_datetime(
        orders['order_purchase_timestamp'],
        errors='coerce'
    )

    order_values = (
        items.groupby('order_id')['price']
        .sum()
        .rename('order_values')
    )

    delivered = orders.loc[
        orders['order_status'] == 'delivered'
    ].dropna(subset=['order_purchase_timestamp'])

    data = delivered.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="inner",
        validate="many_to_one",
    )

    data = data.merge(
        order_values,
        on="order_id",
        how="inner",
        validate="one_to_one",
    )

    if data.empty:
        raise ValueError("No delivered orders found")

    if reference_date is None:
        reference = orders['order_purchase_timestamp'].max().normalize() + pd.Timedelta(days=1)
    else:
        reference = pd.Timestamp(reference_date)

    rfm = (
        data.groupby("customer_unique_id")
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_values", "sum"),
        )
    )

    rfm['recency'] = (
        reference - rfm['last_purchase'].dt.normalize()
    ).dt.days

    return rfm[['recency', 'frequency', 'monetary']].reset_index()

if __name__ == '__main__':
    result = build_rfm()
    print(result.head(10).to_string(index=False))

    print("rfm summary")
    print(result[['recency', 'frequency', 'monetary']].describe().to_string())
