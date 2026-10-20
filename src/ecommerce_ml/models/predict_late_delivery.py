from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'late_delivery_random_forest.joblib'


def load_model():
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(f"No file found in {MODEL_PATH}, run models/random_forest_classifier.py first")

    artifact = joblib.load(MODEL_PATH)
    return (
        artifact['model'],
        artifact['features'],
        artifact['threshold']
    )


def predict_delivery_risk(data: pd.DataFrame):
    model, features, threshold = load_model()

    missing_columns = set(features) - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing required features: {missing_columns}")

    X = data[features]
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame({
        'late_delivery_probability': probabilities,
        'delivery_prediction': predictions
    })

if __name__ == '__main__':
    X = pd.DataFrame([
        {
            "purchase_hour": 9,
            "purchase_day_of_week": 1,
            "purchase_month": 3,
            "days_to_estimated_delivery": 7.0,
            "item_count": 1,
            "total_price": 50.0,
            "total_freight": 12.0,
        },
        {
            "purchase_hour": 14,
            "purchase_day_of_week": 5,
            "purchase_month": 11,
            "days_to_estimated_delivery": 25.0,
            "item_count": 4,
            "total_price": 450.0,
            "total_freight": 65.0,
        },
        {
            "purchase_hour": 20,
            "purchase_day_of_week": 6,
            "purchase_month": 12,
            "days_to_estimated_delivery": 5.0,
            "item_count": 2,
            "total_price": 180.0,
            "total_freight": 30.0,
        },
        {
            "purchase_hour": 11,
            "purchase_day_of_week": 2,
            "purchase_month": 6,
            "days_to_estimated_delivery": 15.0,
            "item_count": 8,
            "total_price": 1200.0,
            "total_freight": 150.0,
        },
        {
            "purchase_hour": 16,
            "purchase_day_of_week": 0,
            "purchase_month": 8,
            "days_to_estimated_delivery": 10.0,
            "item_count": 1,
            "total_price": 90.0,
            "total_freight": 18.0,
        },
        {
            "purchase_hour": 7,
            "purchase_day_of_week": 4,
            "purchase_month": 1,
            "days_to_estimated_delivery": 30.0,
            "item_count": 3,
            "total_price": 700.0,
            "total_freight": 85.0,
        },
        {
            "purchase_hour": 22,
            "purchase_day_of_week": 3,
            "purchase_month": 9,
            "days_to_estimated_delivery": 3.0,
            "item_count": 5,
            "total_price": 320.0,
            "total_freight": 45.0,
        },
        {
            "purchase_hour": 13,
            "purchase_day_of_week": 6,
            "purchase_month": 4,
            "days_to_estimated_delivery": 12.0,
            "item_count": 2,
            "total_price": 250.0,
            "total_freight": 35.0,
        },
    ])

    result = predict_delivery_risk(X)

    print(result.round(4).to_string(index=False))

    
