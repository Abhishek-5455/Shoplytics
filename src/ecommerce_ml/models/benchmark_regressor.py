import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecommerce_ml.features.delivery_time import build_delivery_time_features

FEATURES = [
    "purchase_hour",
    "purchase_day_of_week",
    "purchase_month",
    "estimated_delivery_days",
    "item_count",
    "total_price",
    "total_freight",
]

TARGET = "delivery_time_days"


def evaluate_model(name, model, y_test, predictions) -> dict:

    return {
        "model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
        'Median AE': median_absolute_error(y_test, predictions)
    }


def train_delivery_time_models():
    data = build_delivery_time_features()

    # Sort by purchase time before splitting.
    data = data.sort_values("order_purchase_timestamp").reset_index(drop=True)

    split_index = int(len(data) * 0.8)

    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    models = {
        'Mean Baseline': DummyRegressor(strategy='mean'),
        'Median Baseline': DummyRegressor(strategy='median'),
        'Ridge Regression': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append(
            evaluate_model(
                name,
                model,
                y_test,
                predictions
            )
        )

    report = pd.DataFrame(results).sort_values("MAE")

    return report
    


if __name__ == "__main__":
    report = train_delivery_time_models()
    
    print("\nModel comparison:")
    print(report.round(4).to_string(index=False))