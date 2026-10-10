from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'raw'

PRIMARY_KEY = {
    'olist_customers_dataset.csv': ['customer_id'],
    'olist_order_items_dataset.csv': ['order_id', 'order_item_id'],
    'olist_order_payments_dataset.csv': ['order_id','payment_sequential'],
    'olist_orders_dataset.csv': ['order_id'],
    'olist_products_dataset.csv': ['product_id'],
    'olist_sellers_dataset.csv': ['seller_id'],
}

REQUIRED_COLUMNS = {
    "olist_customers_dataset.csv": ["customer_id", "customer_unique_id"],
    "olist_orders_dataset.csv": ["order_id", "customer_id", "order_status"],
    "olist_order_items_dataset.csv": [
        "order_id", "order_item_id", "product_id", "seller_id"
    ],
    "olist_order_payments_dataset.csv": [
        "order_id", "payment_sequential", "payment_value"
    ],
    "olist_products_dataset.csv": ["product_id", "product_category_name"],
    "olist_sellers_dataset.csv": ["seller_id"],
}

def validate_csv(file_path: str) -> list[str]:
    errors = []

    if not Path(file_path).is_file():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    required_columns = REQUIRED_COLUMNS.get(Path(file_path).name)
    if required_columns is None:
        return errors

    df = pd.read_csv(file_path)

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"Missing columns: {', '.join(missing_columns)}")

    key = PRIMARY_KEY.get(Path(file_path).name)

    if key:
        if df[key].isna().any().any():
            errors.append(f"Null in primary key values: {key}")

        if df.duplicated(subset=key).any():
            errors.append(f"Duplicate primary key values: {key}")

    return errors

def main() -> None:
    failed = False
    for file_name in REQUIRED_COLUMNS:
        errors = validate_csv(RAW_DATA_DIR / file_name)

        if errors:
            failed = True
            print(f"Failed -> {file_name}: {errors}")
        else:
            print(f"Passed -> {file_name}")

    if failed:
        raise SystemExit()

if __name__ == '__main__':
    main()
