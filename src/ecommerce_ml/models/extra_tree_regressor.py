import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ecommerce_ml.features.delivery_time import build_delivery_time_features
from .benchmark_regressor import FEATURES, TARGET, evaluate_model

from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'

def train_regressor():
    data = build_delivery_time_features()
    data = data.sort_values("order_purchase_timestamp").reset_index(drop=True)
    
    split_index = int(len(data) * 0.8)

    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    model = ExtraTreesRegressor(
        n_estimators=350,
        min_samples_leaf=20,
        min_samples_split=2,
        max_features=0.6,
        max_depth=30,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / 'delivery_time_extra_trees.joblib'

    joblib.dump(
        {
            'model': model,
            'features': FEATURES,
            'target': TARGET,
            'prediction_unit': 'days'
        },
        model_path
    )

    print(f"Model Saved to {model_path}")

    predictions = model.predict(X_test)

    return evaluate_model('Extra Tree', model, y_test, predictions)

if __name__ == '__main__':
    for key, value in train_regressor().items():
        print(f"{key}: {value}")