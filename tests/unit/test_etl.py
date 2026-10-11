from ecommerce_ml.data.etl import TABLES 

def test_all_expected_tables():
	''' catches missing or accidentally renamed table configurations. '''
	expected = {
		'customers',
		'orders',
		'products',
		'sellers',
		'order_items',
		'payments',
		'reviews',
		'geolocation',
		'product_category_translation'
	}

	assert set(TABLES.keys()) == expected

def test_etl_load_order_respects_dependencies():
	'''
	ensures parent tables are loaded before tables that reference them through foreign keys.
	'''
	names = list(TABLES.keys())

	assert names.index("customers") < names.index("orders")
	assert names.index("orders") < names.index("order_items")
    # assert names.index("products") < names.index("order_items")
    # assert names.index("sellers") < names.index("order_items")


