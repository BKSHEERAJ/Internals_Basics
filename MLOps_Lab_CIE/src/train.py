import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn


def train():
    mlflow.set_experiment("genomeflow-sequencing-hours")

    df = pd.read_csv("data/training_data.csv")

    X = df.drop("sequencing_hours", axis=1)
    y = df["sequencing_hours"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0)
    }

    best_model_name = None
    best_rmse = float("inf")
    best_model = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            # Train
            model.fit(X_train, y_train)

            # 🔥 CRITICAL (for Task 4)
            mlflow.sklearn.log_model(model, "model")

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Params
            mlflow.log_param("model", name)
            if name == "Ridge":
                mlflow.log_param("alpha", 1.0)

            # Metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            # Tag
            mlflow.set_tag("domain", "dna___genomics")

            results.append({
                "name": name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_model = model

    # Save best model locally
    os.makedirs("models", exist_ok=True)
    import joblib
    joblib.dump(best_model, "models/model.pkl")

    # Save JSON
    os.makedirs("results", exist_ok=True)

    output = {
        "experiment_name": "genomeflow-sequencing-hours",
        "models": results,
        "best_model": best_model_name,
        "best_metric_name": "rmse",
        "best_metric_value": best_rmse
    }

    with open("results/step1_s1.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Task 1 completed (with MLflow model logging)")


if __name__ == "__main__":
    train()