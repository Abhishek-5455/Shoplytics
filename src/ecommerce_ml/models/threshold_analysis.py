import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from .random_forest_classifier import train_delivery_classifier
from ecommerce_ml.features.engineering import build_delivery_features

def analyze_thresholds(data: pd.DataFrame):
    model, X_test, y_test = train_delivery_classifier(data)

    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    result = []
    for threshold in thresholds:
        print(f"Result calculated for threshold {threshold}")
        predictions = (probabilities >= threshold).astype(int)

        result.append({
            'threshold': threshold,
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0)
        })


    result = pd.DataFrame(result)

    print(result.round(4).to_string(index=False))

if __name__ == '__main__':
    data = build_delivery_features()
    analyze_thresholds(data)