from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd

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

def train_delivery_classifier(data: pd.DataFrame, test_size=0.2, random_state = 42):
    if data.empty:
        raise ValueError("Training data is empty!!")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    model.fit(X_train, y_train)
    return model, X_test, y_test