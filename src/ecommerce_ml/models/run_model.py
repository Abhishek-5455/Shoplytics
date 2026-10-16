from ecommerce_ml.features.engineering import build_delivery_features
from .classifier import train_delivery_classifier
from .evaluation import evaluate_classifier
from .compare_thresholds import compare_thresholds

def main():
    data = build_delivery_features()

    print("Dataset shape:", data.shape)
    print("\nLate-delivery distribution:")
    print(data["is_late"].value_counts(normalize=True).round(4))

    model, X_test, y_test = train_delivery_classifier(data)

    metrics, confusion_matrix = evaluate_classifier(model, X_test, y_test)

    print("Classification metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix)

    print("Threshold Comparison")
    print(compare_thresholds(model, X_test, y_test))


if __name__ == '__main__':
    main()