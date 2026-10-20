from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ecommerce_ml.features.late_delivery import build_delivery_features

from .benchmark_classifiers import FEATURES, TARGET

ROOT_DIR = Path(__name__).parent.parent.parent.parent
MODEL_DIR = ROOT_DIR / 'models'

def train_delivery_classifier(data: pd.DataFrame, test_size=0.2, random_state = 42):
    if data.empty:
        raise ValueError("Training data is empty!!")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=250,
            min_samples_split=2,
            min_samples_leaf=20,
            max_features=0.7,
            class_weight='balanced_subsample',
            random_state=random_state,
            max_depth=None,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_classifier(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        'precisison': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'accuracy': accuracy_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'roc_auc': roc_auc_score(y_test, probabilities)
    }

    print("Classification Report")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))

if __name__ == '__main__':
    data = build_delivery_features()
    
    print("Dataset shape:", data.shape)
    print("\nLate-delivery distribution:")

    print(data["is_late"].value_counts(normalize=True).round(4))

    model, X_test, y_test = train_delivery_classifier(data)

    evaluate_classifier(model, X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / 'late_delivery_random_forest.joblib'

    joblib.dump(
        {
            'model': model,
            'features': FEATURES,
            'target': TARGET,
            'threshold': 0.5
        },
        model_path
    )

    print(f"Model saved to {model_path}")
