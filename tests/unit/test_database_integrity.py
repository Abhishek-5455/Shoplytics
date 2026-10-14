from ecommerce_ml.db import get_connection


def test_required_tables_exist():
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

	with get_connection() as conn, conn.cursor() as cur:
		sql = """
					SELECT table_name
					FROM information_schema.tables
					WHERE table_schema = 'public'
					"""

		cur.execute(sql)
		actual = {row[0] for row in cur.fetchall()}

	assert expected.issubset(actual)

def test_no_orphaned_orders():
	with get_connection() as conn, conn.cursor() as cur:
		sql = """
					SELECT count(*)
					FROM orders o
					LEFT JOIN customers c
					ON o.customer_id = c.customer_id
					WHERE c.customer_id IS NULL
					"""

		cur.execute(sql)

		assert cur.fetchone()[0] == 0

def test_no_orphaned_order_items():
	with get_connection() as conn:
		with conn.cursor() as cur:
			sql = """
					SELECT count(*)
					FROM order_items oi
					LEFT JOIN orders o ON oi.order_id = o.order_id
					LEFT JOIN products p ON oi.product_id = p.product_id
					LEFT JOIN sellers s ON oi.seller_id = s.seller_id
					WHERE o.order_id IS NULL OR p.product_id is NULL OR s.seller_id IS NULL
				"""

			cur.execute(sql)
			assert cur.fetchone()[0] == 0
