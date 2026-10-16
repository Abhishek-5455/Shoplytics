import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ecommerce_ml.features.engineering import build_delivery_features

FEATURES = [
    "purchase_hour",
    "purchase_day_of_week",
    "purchase_month",
    "days_to_estimated_delivery",
    "item_count",
    "total_price",
    "total_freight",
]

TARGET = "is_late"

RANDOM_STATE = 42

def get_classifier():
    models = {
        "Dummy Baseline": DummyClassifier(strategy='most_frequent'),
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=RANDOM_STATE
            ))  
        ]),
        "Decision Tree": DecisionTreeClassifier(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            min_samples_leaf=20
        ),
        "Random Forest": RandomForestClassifier(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            min_samples_leaf=5,
            n_estimators=200,
            max_depth=12,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            min_samples_leaf=5,
            n_estimators=200,
            max_depth=12,
            n_jobs=-1
        )
    }

    return models


def calculate_metrics(model, X, y) -> dict:
    """Calculate classification metrics at the default 0.5 threshold."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1_score": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
    }

def benchmark_classifier(data: pd.DataFrame):
    if data.empty:
        raise ValueError("Data is empty!!")

    if data[FEATURES].isna().any().any():
        raise ValueError('Feature must not contain any null values')

    if data[TARGET].nunique() != 2:
        raise ValueError("Target must contain two classes: 0 and 1")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        random_state=RANDOM_STATE,
        test_size=0.4,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        random_state=RANDOM_STATE,
        test_size=0.2,
        stratify=y_temp
    )

    validation_result = []

    for name, model in get_classifier().items():
        print(f"training {name}")

        model.fit(X_test, y_test)
        metrics = calculate_metrics(model, X_val, y_val)

        validation_result.append({
            'model': name,
            **metrics
        })

    print("All models trained")
    validation_report = (
        pd.DataFrame(validation_result)
        .sort_values('pr_auc', ascending=False)
        .reset_index(drop=True)
    )

    best_name = validation_report.loc[0, 'model']
    best_model = get_classifier()[best_name]

    print("Refitting on train + validation data")
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    best_model.fit(X_train_full, y_train_full)
    test_metrics = calculate_metrics(best_model, X_test, y_test)

    return validation_report, best_name, best_model, test_metrics


if __name__ == "__main__":
    data = build_delivery_features()

    validation_report, best_name, _, test_metrics = benchmark_classifier(data)

    print("Validation Report")
    print(validation_report)

    print(f"Best model: {best_name}")

    print("Test Metrics")
    print(test_metrics)





