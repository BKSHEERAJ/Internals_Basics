from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model once
model = joblib.load("models/model.pkl")


class InputData(BaseModel):
    read_length_bp: float
    sample_quality_score: float
    coverage_depth: float
    is_whole_genome: int


@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array([[
            data.read_length_bp,
            data.sample_quality_score,
            data.coverage_depth,
            data.is_whole_genome
        ]])

        prediction = model.predict(features)[0]

        return {
            "prediction": float(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))