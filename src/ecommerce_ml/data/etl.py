import logging
import time
from pathlib import Path

import pandas as pd

from ecommerce_ml.db import get_connection

ROOT_DIR = Path(__file__).parent.parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

TABLES = {
	'customers': (
		"olist_customers_dataset.csv",
		[
			"customer_id",
			"customer_unique_id",
			"customer_zip_code_prefix",
			"customer_city",
			"customer_state"
		],
		["customer_id"]
	),
	"orders": (
        "olist_orders_dataset.csv",
        [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        ["order_id"],
    ),
    "products": (
        "olist_products_dataset.csv",
        [
            "product_id",
            "product_category_name",
            "product_name_length",
            "product_description_length",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ],
        ["product_id"],
    ),
    "sellers": (
        "olist_sellers_dataset.csv",
        [
            "seller_id",
            "seller_zip_code_prefix",
            "seller_city",
            "seller_state",
        ],
        ["seller_id"],
    ),
    "order_items": (
        "olist_order_items_dataset.csv",
        [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "shipping_limit_date",
            "price",
            "freight_value",
        ],
        ["order_id", "order_item_id"],
    ),
    "payments": (
        "olist_order_payments_dataset.csv",
        [
            "order_id",
            "payment_sequential",
            "payment_type",
            "payment_installments",
            "payment_value",
        ],
        ["order_id", "payment_sequential"],
    ),
    "reviews": (
        "olist_order_reviews_dataset.csv",
        [
            "review_id",
            "order_id",
            "review_score",
            "review_comment_title",
            "review_comment_message",
            "review_creation_date",
            "review_answer_timestamp",
        ],
        ["review_id"],
    ),
    "geolocation": (
        "olist_geolocation_dataset.csv",
        [
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
            "geolocation_city",
            "geolocation_state",
        ],
        None,
    ),
    "product_category_translation": (
        "product_category_name_translation.csv",
        [
            "product_category_name",
            "product_category_name_english",
        ],
        ["product_category_name"],
    )
}

def load_data(table_name: str) -> int:
    filename, columns, primary_key = TABLES[table_name]
    path = RAW_DIR / filename

    if not Path(path).is_file():
        raise FileNotFoundError(f"CSV not found {path}")

    batch_size = 3000
    total_rows = 0
    columns_sql = ", ".join(f'"{c}"' for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))

    sql = (
        f'INSERT INTO "{table_name}" ({columns_sql}) VALUES ({placeholders})'
    )

    if primary_key:
        conflict_columns = ", ".join(f'"{c}"' for c in primary_key)
        sql += f"ON CONFLICT ({conflict_columns}) DO NOTHING"
    elif table_name == "geolocation":
        sql += " ON CONFLICT DO NOTHING"

    with get_connection() as conn, conn.cursor() as cur:
        for chunk in pd.read_csv(path, chunksize=batch_size):
            missing_columns = set(columns) - set(chunk.columns)

            if missing_columns:
                raise ValueError(
                    f"{filename} has missing columns: {sorted(missing_columns)}"
                )

            chunk = chunk[columns].astype(object).where(
                pd.notna(chunk), None
            )

            rows = list(chunk.itertuples(index=False, name=None))

            cur.executemany(sql, rows)
            total_rows += len(rows)

    return total_rows

def run_etl() -> None:
    start = time.perf_counter()
    logger.info("ETL started")

    for table_name in TABLES:
        table_start = time.perf_counter()
        logger.info("Loading table : %s", table_name)

        try:
            count = load_data(table_name)
            elapsed = time.perf_counter() - table_start

            logger.info(
                "Loaded %s: %s rows processed in %.2f seconds",
                table_name,
                count,
                elapsed
            )
        except Exception:
            logger.exception("Failed to load table: %s", table_name)
            raise

    logger.info(
        "ETL completed in %.2f seconds",
        time.perf_counter() - start,
    )

if __name__ == '__main__':
	run_etl()
