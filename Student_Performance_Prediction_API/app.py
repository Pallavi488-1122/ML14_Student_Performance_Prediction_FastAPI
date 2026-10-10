from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from train_model import FEATURES, MODEL_PATH, train_and_save_model

if not MODEL_PATH.exists():
    train_and_save_model()
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict an approximate student score from study habits and academic indicators.",
    version="1.0.0",
)

class StudentInput(BaseModel):
    study_hours: float = Field(..., ge=0, le=24, description="Study hours per day")
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage")
    previous_score: float = Field(..., ge=0, le=100, description="Previous exam score")
    assignments_completed: float = Field(..., ge=0, le=100, description="Assignment completion percentage")
    sleep_hours: float = Field(..., ge=0, le=24, description="Sleep hours per day")

@app.get("/")
def home() -> dict[str, Any]:
    return {
        "message": "Student Performance Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "required_fields": FEATURES,
    }

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": "true"}

@app.post("/predict")
def predict(student: StudentInput) -> dict[str, Any]:
    try:
        input_df = pd.DataFrame([student.model_dump()], columns=FEATURES)
        predicted_score = float(model.predict(input_df)[0])
        predicted_score = round(max(0.0, min(100.0, predicted_score)), 2)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.") from exc

    if predicted_score >= 75:
        level = "Excellent"
    elif predicted_score >= 60:
        level = "Good"
    elif predicted_score >= 40:
        level = "Average"
    else:
        level = "Needs Improvement"

    return {
        "predicted_score": predicted_score,
        "performance_level": level,
        "note": "Demo estimate trained on synthetic data; not a validated academic assessment.",
    }
