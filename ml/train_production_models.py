from pathlib import Path
import json

import joblib
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "artifacts/api-server/data/student-mat.csv"
OUTPUT_DIR = ROOT / "artifacts/ml-models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Feature definitions
# G2 is deliberately excluded from BOTH models.
# ---------------------------------------------------------

BINARY_FEATURES = [
    "schoolsup",
    "famsup",
    "paid",
    "higher",
    "internet",
]

BASE_FEATURES = [
    "age",
    "studytime",
    "failures",
    "absences",
    "schoolsup",
    "famsup",
    "paid",
    "higher",
    "internet",
    "goout",
    "traveltime",
    "health",
]

SCENARIOS = {
    "early": {
        "features": BASE_FEATURES,
        "model_params": {
            "n_estimators": 600,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },
    },
    "updated": {
        "features": ["G1"] + BASE_FEATURES,
        "model_params": {
            "n_estimators": 400,
            "max_depth": 16,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 0.8,
        },
    },
}


# ---------------------------------------------------------
# Preprocessing and model pipeline
# ---------------------------------------------------------

def build_pipeline(features, model_params):
    numeric_features = [
        feature
        for feature in features
        if feature not in BINARY_FEATURES
    ]

    binary_features = [
        feature
        for feature in features
        if feature in BINARY_FEATURES
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("binary", binary_pipeline, binary_features),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        **model_params,
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])


# ---------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------

def load_dataset():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    df = pd.read_csv(DATA, sep=";")

    required = set(BASE_FEATURES + ["G1", "G2", "G3"])
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("The dataset contains no rows.")

    for column in BINARY_FEATURES:
        mapped = df[column].map({"yes": 1, "no": 0})

        invalid = df[column].notna() & mapped.isna()
        if invalid.any():
            values = df.loc[invalid, column].unique().tolist()
            raise ValueError(
                f"Unexpected values in {column}: {values}"
            )

        df[column] = mapped

    df["G1"] = pd.to_numeric(df["G1"], errors="raise")
    df["G3"] = pd.to_numeric(df["G3"], errors="raise")

    if df["G3"].isna().any():
        raise ValueError("Target G3 contains missing values.")

    return df


# ---------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------

def main():
    df = load_dataset()
    target = df["G3"].astype(float)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    metrics = {
        "dataset": "UCI Student Performance - Mathematics",
        "dataset_rows": int(len(df)),
        "target": "G3",
        "target_scale": "0-20",
        "cross_validation": "5-fold shuffled KFold, random_state=42",
        "sklearn_version": sklearn.__version__,
        "warning": (
            "Research results on Portuguese students; "
            "not validated for KV or CBSE students."
        ),
        "models": {},
    }

    print("scikit-learn:", sklearn.__version__)
    print("Dataset rows:", len(df))
    print("Target: G3")
    print("Cross-validation: 5-fold")
    print()

    for mode, config in SCENARIOS.items():
        features = config["features"]

        # Guard against accidental target leakage.
        assert "G2" not in features
        assert "G3" not in features

        if mode == "early":
            assert "G1" not in features
        else:
            assert "G1" in features

        X = df[features].copy()

        pipeline = build_pipeline(
            features,
            config["model_params"],
        )

        # Each fold independently fits preprocessing and the model.
        predictions = cross_val_predict(
            pipeline,
            X,
            target,
            cv=cv,
            n_jobs=1,
            method="predict",
        )

        scores = {
            "mae": float(mean_absolute_error(target, predictions)),
            "rmse": float(
                root_mean_squared_error(target, predictions)
            ),
            "r2": float(r2_score(target, predictions)),
        }

        # Fit the final model on all available rows.
        # Cross-validation results above are reported separately.
        pipeline.fit(X, target)

        bundle = {
            "mode": mode,
            "pipeline": pipeline,
            "features": features,
            "target": "G3",
            "target_scale": [0, 20],
            "dataset": "UCI Student Performance - Mathematics",
            "dataset_rows": int(len(df)),
            "model_type": "RandomForestRegressor",
            "model_params": config["model_params"],
            "cross_validation": {
                "method": "5-fold shuffled KFold",
                "random_state": 42,
                "metrics": scores,
            },
            "validated_for_kv": False,
            "validated_for_cbse": False,
        }

        model_path = OUTPUT_DIR / f"{mode}_model.joblib"
        joblib.dump(bundle, model_path)

        metrics["models"][mode] = {
            "features": features,
            "model_params": config["model_params"],
            "cross_validation_metrics": scores,
            "model_file": str(model_path.relative_to(ROOT)),
            "validated_for_kv": False,
            "validated_for_cbse": False,
        }

        print("=" * 65)
        print(f"MODE: {mode.upper()}")
        print("Feature count:", len(features))
        print("Features:", ", ".join(features))
        print(f"MAE:  {scores['mae']:.3f}")
        print(f"RMSE: {scores['rmse']:.3f}")
        print(f"R²:   {scores['r2']:.3f}")
        print("Saved:", model_path)

    metrics_path = OUTPUT_DIR / "evaluation.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 65)
    print("TRAINING COMPLETED")
    print("Metrics:", metrics_path)
    print("Models:", OUTPUT_DIR)
    print()
    print("IMPORTANT: These are research models, not KV-validated models.")


if __name__ == "__main__":
    main()
