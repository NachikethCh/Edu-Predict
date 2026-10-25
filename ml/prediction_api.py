from pathlib import Path
from typing import Literal

import traceback
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "ml-models"

app = FastAPI(
    title="EduPredict ML API",
    description="Experimental Early and Updated student grade prediction.",
    version="1.0.0",
)

# Load models once when the service starts.
try:
    MODELS = {
        mode: joblib.load(MODEL_DIR / f"{mode}_model.joblib")
        for mode in ("early", "updated")
    }
except Exception as exc:
    raise RuntimeError(
        f"Could not load trained models from {MODEL_DIR}: {exc}"
    ) from exc


class StudentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    g1: float | None = Field(default=None, ge=0, le=20)
    age: int = Field(ge=15, le=22)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=4)
    absences: int = Field(ge=0, le=93)

    schoolsup: bool
    famsup: bool
    paid: bool
    higher: bool
    internet: bool

    goout: int = Field(ge=1, le=5)
    traveltime: int = Field(ge=1, le=4)
    health: int = Field(ge=1, le=5)


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["early", "updated"]
    student: StudentInput

    @model_validator(mode="after")
    def validate_mode_features(self):
        if self.mode == "early" and self.student.g1 is not None:
            raise ValueError(
                "Early Prediction must not include G1."
            )

        if self.mode == "updated" and self.student.g1 is None:
            raise ValueError(
                "Updated Prediction requires G1."
            )

        return self


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": sorted(MODELS.keys()),
    }


@app.get("/models")
def model_info():
    return {
        mode: {
            "features": bundle["features"],
            "target": bundle["target"],
            "target_scale": bundle["target_scale"],
            "dataset_rows": bundle["dataset_rows"],
            "cross_validation_metrics": (
                bundle["cross_validation"]["metrics"]
            ),
            "validated_for_kv": bundle["validated_for_kv"],
            "validated_for_cbse": bundle["validated_for_cbse"],
        }
        for mode, bundle in MODELS.items()
    }


@app.post("/predict")
def predict(request: PredictionInput):
    bundle = MODELS[request.mode]

    # Convert the API field names to the model's training column names.
    student = request.student.model_dump(exclude_none=True)
    
    #Convert Booleans to 0/1, matching the training dataset.
    binary_features = [
        "schoolsup",
        "famsup",
        "paid",
        "higher",
        "internet",
    ]
    
    for feature in binary_features:
        if feature in student:
            student[feature] = int(student[feature])

    if "g1" in student:
        student["G1"] = student.pop("g1")

    features = bundle["features"]

    missing = [name for name in features if name not in student]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing model features: {missing}",
        )

    # Supply features in precisely the order used for training.
    row = pd.DataFrame(
        [[student[name] for name in features]],
        columns=features,
    )

    try:
        prediction = float(bundle["pipeline"].predict(row)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

    # The source dataset's target scale is 0–20.
    prediction = max(0.0, min(20.0, prediction))
    prediction = round(prediction, 1)

    return {
        "mode": request.mode,
        "predictedGrade": prediction,
        "targetScale": "0-20",
        "model": "RandomForestRegressor",
        "featureCount": len(features),
        "featuresUsed": features,
        "dataset": "UCI Student Performance - Mathematics",
        "datasetRows": bundle["dataset_rows"],
        "crossValidationMetrics": (
            bundle["cross_validation"]["metrics"]
        ),
        "validatedForKV": False,
        "validatedForCBSE": False,
        "warning": (
            "Experimental prediction based on Portuguese student data. "
            "Not validated for KV or CBSE students. "
            "This is an estimate, not a guaranteed grade."
        ),
    }
