from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

    return metrics, confusion_matrix(y_test, predictions)