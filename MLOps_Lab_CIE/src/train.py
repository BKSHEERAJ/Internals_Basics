import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.data_loader import load_data


def train_model():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Initialize model
    model = RandomForestClassifier()

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # Save results
    os.makedirs("results", exist_ok=True)
    results = {"accuracy": float(accuracy)}

    with open("results/step1_s1.json", "w") as f:
        json.dump(results, f)

    print("Training complete. Accuracy:", accuracy)


if __name__ == "__main__":
    train_model()