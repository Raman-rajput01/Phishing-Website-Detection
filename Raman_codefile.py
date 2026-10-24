"""
Project: Phishing Website Detection using Machine Learning
Author: Raman Kumar

This program trains a machine-learning model to classify website records
as phishing or legitimate using the UCI Phishing Websites dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import joblib
from ucimlrepo import fetch_ucirepo


def load_dataset():
    """Load the Phishing Websites dataset from the UCI repository."""
    print("Loading the UCI Phishing Websites dataset...")
    dataset = fetch_ucirepo(id=327)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    # Convert feature values to numeric values where possible.
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True))

    target = targets.iloc[:, 0]
    # UCI labels are commonly -1 (legitimate) and 1 (phishing).
    # Encode labels consistently as 0 and 1 for model training.
    target = pd.to_numeric(target, errors="coerce")
    valid_rows = target.notna()
    features = features.loc[valid_rows]
    target = target.loc[valid_rows].astype(int)

    # Keep original target values for clarity: -1 = legitimate, 1 = phishing.
    return features, target


def main():
    X, y = load_dataset()

    print(f"Dataset rows: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nModel accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=sorted(y.unique()),
        target_names=[
            "Legitimate (-1)" if label == -1 else "Phishing (1)"
            for label in sorted(y.unique())
        ],
        zero_division=0
    ))

    # Save the trained model and the feature order used during training.
    joblib.dump(
        {"model": model, "feature_names": list(X.columns)},
        "phishing_detection_model.joblib"
    )
    print("\nSaved model to phishing_detection_model.joblib")

    # Save a confusion matrix image for the project report.
    cm = confusion_matrix(y_test, predictions, labels=sorted(y.unique()))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Legitimate (-1)" if label == -1 else "Phishing (1)"
            for label in sorted(y.unique())
        ]
    )
    display.plot(values_format="d")
    plt.title("Phishing Website Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

    # Show the most useful features according to the trained model.
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False).head(10)

    print("\nTop 10 features by model importance:")
    print(importance.to_string())

    importance.sort_values().plot(kind="barh", figsize=(8, 5))
    plt.title("Top 10 Important Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()

    print(
        "\nNote: This is an educational model. Do not rely on it as the only "
        "check before opening a website."
    )


if __name__ == "__main__":
    main()
