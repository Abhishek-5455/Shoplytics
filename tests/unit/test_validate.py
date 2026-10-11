import pandas as pd

from ecommerce_ml.data.validate import validate_csv


def test_valid_order_data(tmp_path):
	csv_path = tmp_path / "olist_customers_dataset.csv"

	pd.DataFrame({
		"customer_id": ['A', 'B', 'C'],
		"customer_unique_id": ['a', 'b', 'c']
	}).to_csv(csv_path, index=False)


	assert validate_csv(csv_path) == []

def test_duplicate_primary_key_entries(tmp_path):
	csv_path = tmp_path / 'olist_customers_dataset.csv'

	pd.DataFrame({
		'customer_id': ['A','A','B'],
		"customer_unique_id": ['a', 'b', 'c']
	}).to_csv(csv_path, index=False)

	assert "Duplicate primary key values: ['customer_id']" in validate_csv(csv_path) 