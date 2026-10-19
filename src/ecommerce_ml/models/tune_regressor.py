import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ecommerce_ml.features.delivery_time import build_delivery_time_features
from .benchmark_regressor import FEATURES, TARGET, evaluate_model

def tune_delivery_time_model():
    data = build_delivery_time_features()
    data = data.sort_values('order_purchase_timestamp').reset_index(drop=True)

    test_index = int(len(data) * 0.8)
    train_data = data.iloc[:test_index]

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    validation_index = int(len(X_train) * 0.8)

    X_fit = X_train.iloc[:validation_index]
    y_fit = y_train.iloc[:validation_index]

    parameter_distribution = {
        "n_estimators": [150, 250, 350],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [2, 5, 10, 20],
        'max_features': [0.6, 0.8, 1.0]
    }

    search = RandomizedSearchCV(
        estimator=ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1
        ),
        param_distributions=parameter_distribution,
        n_iter=15,
        scoring='neg_mean_absolute_error',
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_fit, y_fit)

    print("Best parameters: ")
    for name, value in search.best_params_.items():
        print(f"{name}: {value}")

    print("cross validation MAE: ", round(-search.best_score_, 4))

if __name__ == '__main__':
    tune_delivery_time_model()

