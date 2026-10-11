from ecommerce_ml.db import get_connection

def test_db_connection():
	with get_connection() as conn:
		with conn.cursor() as cur:
			cur.execute("SELECT 1")

			assert cur.fetchone() == (1, )
			