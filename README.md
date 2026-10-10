# E-Commerce Intelligence Platform

An end-to-end analytics and machine learning platform built using the Olist Brazilian E-Commerce dataset.

## Objectives

- Data engineering with Python, SQL, and PostgreSQL
- Customer segmentation and retention analysis
- Classification, regression, and recommendation systems
- Sales forecasting and anomaly detection
- Production APIs, dashboards, and monitoring

## Technology Stack

- Python 3.11+
- PostgreSQL
- pandas, NumPy, scikit-learn
- FastAPI, Streamlit
- Docker, pytest, Ruff

## Project Structure

- `src/ecommerce_ml/`: Application code
- `data/raw/`: Original datasets
- `data/processed/`: Transformed datasets
- `configs/`: Application configuration
- `notebooks/`: Exploratory analysis
- `reports/`: Generated reports
- `tests/`: Automated tests

## Dataset

Brazilian E-Commerce Public Dataset by Olist.

Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

The dataset is downloaded research data, not a live production connection.

## Setup

Create and activate a virtual environment, then install the project:

```bash
python -m pip install -e ".[dev]"
```

## Development Principles

- Validate data before transforming it.
- Prevent data leakage in machine learning.
- Test assumptions and edge cases.
- Keep credentials and raw datasets out of Git.
- Make experiments reproducible.
