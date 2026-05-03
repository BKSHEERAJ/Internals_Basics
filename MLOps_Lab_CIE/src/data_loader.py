import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    # Create a simple dummy dataset
    data = {
        "feature1": [1, 2, 3, 4, 5, 6],
        "feature2": [10, 20, 30, 40, 50, 60],
        "target":   [0, 1, 0, 1, 0, 1]
    }

    df = pd.DataFrame(data)

    X = df[["feature1", "feature2"]]
    y = df["target"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test