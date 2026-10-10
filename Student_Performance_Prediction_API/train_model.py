from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_performance_model.joblib"
FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignments_completed",
    "sleep_hours",
]

def train_and_save_model():
    """Train a demo regression model on generated synthetic data."""
    rng = np.random.default_rng(42)
    n = 1200
    study_hours = rng.uniform(0, 10, n)
    attendance = rng.uniform(40, 100, n)
    previous_score = rng.uniform(20, 100, n)
    assignments_completed = rng.uniform(0, 100, n)
    sleep_hours = rng.uniform(3, 10, n)

    noise = rng.normal(0, 5, n)
    score = (
        3.0 * study_hours
        + 0.22 * attendance
        + 0.35 * previous_score
        + 0.12 * assignments_completed
        + 1.0 * sleep_hours
        + noise - 10
    )
    score = np.clip(score, 0, 100)

    x = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "assignments_completed": assignments_completed,
        "sleep_hours": sleep_hours,
    })
    model = RandomForestRegressor(n_estimators=160, random_state=42)
    model.fit(x[FEATURES], score)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    return model

if __name__ == "__main__":
    train_and_save_model()
