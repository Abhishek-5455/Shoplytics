from .loader import load_csv
import pandas as pd

def order_purchase_date_summary():
	orders = load_csv('olist_orders_dataset.csv')

	timestamps = pd.to_datetime(orders['order_purchase_timestamp'], errors='coerce')

	return pd.DataFrame([{
		'total_orders': len(orders),
		'missing_timestamps': int(timestamps.isna().sum()),
		'earliest_order': timestamps.min(),
		'latest_order': timestamps.max(),
		'date_range_day': (timestamps.max() - timestamps.min()).days
	}])

if __name__ == '__main__':
	for key, value in order_purchase_date_summary().items():
		print(f"{key}: {value}")