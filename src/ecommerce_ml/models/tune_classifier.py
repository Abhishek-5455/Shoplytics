from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from ecommerce_ml.features.late_delivery import build_delivery_features

from .benchmark_classifiers import FEATURES, TARGET


def tune_best_classifier():
    """Tunes the random forest hyperparameters"""
    data = build_delivery_features()

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        random_state=42,
        stratify=y,
        test_size=0.2
    )

    parameter_distributions = {
        "n_estimators": [150, 250, 350],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [5, 10, 20],
        "max_features": ["sqrt", 0.7],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            random_state=42,
            n_jobs=1,
        ),
        param_distributions=parameter_distributions,
        n_iter=12,
        scoring="average_precision",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print("Best parameters:")
    print(search.best_params_)

    print(f"\nBest cross-validation PR-AUC: {search.best_score_:.4f}")

    best_model = search.best_estimator_
    probabilities = best_model.predict_proba(X_test)[:, 1]
    test_pr_auc = average_precision_score(y_test, probabilities)

    print(f"Test PR-AUC: {test_pr_auc:.4f}")


if __name__ == "__main__":
    tune_best_classifier()