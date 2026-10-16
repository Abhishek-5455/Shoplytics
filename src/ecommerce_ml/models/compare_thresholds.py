from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

import pandas as pd

def compare_thresholds(
    model, 
    X_test, 
    y_test, 
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)
):
    probabilities = model.predict_proba(X_test)[:, 1]
    result = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        result.append({
            'threshold': threshold,
            'precisioon': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions)
        })

    return pd.DataFrame(result)


