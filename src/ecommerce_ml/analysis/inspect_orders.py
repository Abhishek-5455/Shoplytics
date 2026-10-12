from .loader import load_csv

def order_status_distribution():
	orders = load_csv("olist_orders_dataset.csv")

	result = (orders['order_status']
				.value_counts(dropna=False)
				.rename_axis('order_status')
				.reset_index(name='order_count'))

	result['percentage'] = ((result['order_count'] / len(orders)) * 100).round(2)

	return result.sort_values(by="order_count", ascending=False)

if __name__ == '__main__':
	print(order_status_distribution().to_string(index=False))


