# Shoplytics: E-commerce Intelligence

Shoplytics analyzes the Olist Brazilian E-Commerce dataset with a PostgreSQL-
compatible Neon database, builds customer and delivery analytics, trains
delivery prediction models, and serves the results through a Streamlit
dashboard and a FastAPI service.

Neon is the hosted database provider; the database itself is PostgreSQL.
The project connects with `psycopg` using Neon's PostgreSQL connection string
and requires SSL.

## Technology

- Python 3.11+
- Neon (hosted PostgreSQL)
- pandas, scikit-learn, psycopg
- Streamlit dashboard
- FastAPI prediction API
- pytest and Ruff


## Repository layout

- `src/ecommerce_ml/analysis/`: RFM analysis and customer clustering
- `src/ecommerce_ml/data/`: CSV validation, profiling, and Neon ETL
- `src/ecommerce_ml/features/`: Feature builders for delivery models
- `src/ecommerce_ml/models/`: Model training, evaluation, and prediction
- `src/ecommerce_ml/dashboard/`: Streamlit data, views, and UI
- `sql/001-create-tables.sql`: Neon database schema
- `data/raw/`: Downloaded Olist CSV files (not tracked by Git)
- `data/processed/`: Generated reports (not tracked by Git)
- `models/`: Serialized prediction models
- `tests/`: Unit tests

## End-to-end setup

Run commands from the repository root.

### 1. Create the local environment and data directories

Install [uv](https://docs.astral.sh/uv/) if needed, then run:

```bash
uv sync
uv run python -m ecommerce_ml.scripts.initial_setup
```

The setup command creates `data/raw`, `data/processed` and report output directories. To use pip instead of uv:

```bash
python -m venv .venv
Activate venv -> .venv\Scripts\Activate
python -m pip install -e ".[dev]"
python -m ecommerce_ml.scripts.initial_setup
```

### 2. Download and place the Olist data

Download the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
from Kaggle and extract its CSV files into `data/raw/`.

The pipeline expects these files in `data/raw/`:

- `olist_customers_dataset.csv`
- `olist_geolocation_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `product_category_name_translation.csv`

### 3. Configure Neon

Create a Neon project and database. In the Neon dashboard, copy the connection
string for the database (use the pooled connection string for serverless apps
if recommended by Neon). Keep `sslmode=require` in the URL.

Create .env in your root folder
```dotenv
DATABASE_URL=connection_string
```
Paste your connection string in the placeholder.


Keep `.env` private; it is ignored by Git. The application reads `DATABASE_URL`
from the environment / `.env`. 

### 4. Create the Neon schema and load the CSV data

Open Neon's SQL Editor, connect to the target database, and run the contents
of `sql/001-create-tables.sql`. Then, from the repository root, validate the
CSV files and load the data:

```bash
uv run python -m ecommerce_ml.data.audit
uv run python -m ecommerce_ml.data.validate
uv run python -m ecommerce_ml.data.etl
```

The ETL inserts the customers, orders, products, sellers, order items,
payments, reviews, geolocation, and category translation tables. Existing
primary-key rows are skipped, so rerunning the load does not duplicate them.

### 5. Train the delivery prediction models

The two training modules build their features from the raw order and item CSVs,
then save model artifacts under `models/`:

```bash
uv run python -m ecommerce_ml.models.random_forest_classifier
uv run python -m ecommerce_ml.models.extra_tree_regressor
```

This generates:

- `models/late_delivery_random_forest.joblib`
- `models/delivery_time_extra_trees.joblib`

Training may take several minutes depending on your machine. The dashboard's
prediction forms and API endpoints need these model files to exist.

### 6. Generate customer segmentation reports

The RFM analysis and clustering also use the raw CSVs:

```bash
uv run python -m ecommerce_ml.analysis.cluster_customers
```

This writes `customer_cluster_report.csv` (per-customer cluster assignments)
and `customer_segment_report.csv` (cluster aggregates) into
`data/processed/`. The Customer Analytics page displays the full-dataset
cluster profiles and filtered customer records; this report is not affected by
the dashboard purchase-date filter.

### 7. Run the dashboard and API

Start the Streamlit dashboard:

```bash
uv run streamlit run src/ecommerce_ml/dashboard/app.py
```

## Tests and lint

```bash
uv run pytest
uv run ruff check .
```

The database integration tests require a reachable Neon database and a valid
`DATABASE_URL`. CSV-based analysis and model training do not require a database
connection.