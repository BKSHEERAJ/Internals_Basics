import argparse
import joblib
import numpy as np

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--read_length_bp", type=float, required=True)
    parser.add_argument("--sample_quality_score", type=float, required=True)
    parser.add_argument("--coverage_depth", type=float, required=True)
    parser.add_argument("--is_whole_genome", type=int, required=True)

    args = parser.parse_args()

    # Load trained model
    model = joblib.load("models/model.pkl")

    # Prepare input
    features = np.array([[ 
        args.read_length_bp,
        args.sample_quality_score,
        args.coverage_depth,
        args.is_whole_genome
    ]])

    # Predict
    prediction = model.predict(features)[0]

    print(float(prediction))


if __name__ == "__main__":
    main()