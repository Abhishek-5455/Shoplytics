from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'delivery_time_extra_trees.joblib'

def load_delivery_time_model():
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(f"File not exist {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    return artifact['model'], artifact['features']

def predict_delivery_time(input: pd.DataFrame):

    model, features = load_delivery_time_model()

    missing_columns = set(features) - set(input.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = input[features]

    if X.isna().any().any():
        raise ValueError("Input features cannot be empty")

    predictions = model.predict(X)

    return pd.DataFrame(
        {'predicted_delivery_days': predictions},
        index=X.index
    )

if __name__ == '__main__':
    X = pd.DataFrame([
        {
            "purchase_hour": 10,
            "purchase_day_of_week": 2,
            "purchase_month": 6,
            "estimated_delivery_days": 15.0,
            "item_count": 1,
            "total_price": 100.0,
            "total_freight": 20.0,
        },
        {
            "purchase_hour": 18,
            "purchase_day_of_week": 5,
            "purchase_month": 11,
            "estimated_delivery_days": 25.0,
            "item_count": 3,
            "total_price": 450.0,
            "total_freight": 65.0,
        },
    ])

    results = predict_delivery_time(X)
    print("Delivery-time predictions:")
    print(results.round(2).to_string(index=False))  
