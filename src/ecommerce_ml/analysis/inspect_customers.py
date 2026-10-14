from .loader import load_csv


def customer_order_summary():
    orders = load_csv("olist_orders_dataset.csv")

    customer_orders = (
        orders.groupby('customer_id')['order_id']
        .nunique()
        .reset_index(name='order_count')
    )

    return customer_orders.describe().T

if __name__ == '__main__':
    print(customer_order_summary().to_string())