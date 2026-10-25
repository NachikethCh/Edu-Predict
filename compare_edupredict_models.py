#!/usr/bin/env python3
"""Research-only model comparison for EduPredict's UCI Mathematics dataset."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
csv_path = ROOT / "artifacts" / "api-server" / "data" / "student-mat.csv"
if not csv_path.exists():
    print(f"ERROR: Dataset not found at {csv_path}")
    print("Place this script in the EduPredict-KV-Ready project root and run it there.")
    sys.exit(2)

df = pd.read_csv(csv_path, sep=";")
if "G3" not in df.columns:
    print(f"ERROR: Expected target column G3. Found: {list(df.columns)}")
    sys.exit(2)

y = pd.to_numeric(df["G3"], errors="raise")
X = df.drop(columns=["G3"])

def make_preprocessor(features):
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop")

feature_sets = {
    "all_features": X,
    "without_G1": X.drop(columns=["G1"]) if "G1" in X.columns else X,
    "without_G1_G2": X.drop(columns=[c for c in ("G1", "G2") if c in X.columns]),
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
models = [
    ("DummyMean", lambda prep: Pipeline([("prep", prep), ("model", DummyRegressor(strategy="mean"))])),
    ("KNN_k3_uniform", lambda prep: Pipeline([("prep", prep), ("model", KNeighborsRegressor(n_neighbors=3, weights="uniform"))])),
    ("KNN_k5_uniform", lambda prep: Pipeline([("prep", prep), ("model", KNeighborsRegressor(n_neighbors=5, weights="uniform"))])),
    ("KNN_k9_uniform", lambda prep: Pipeline([("prep", prep), ("model", KNeighborsRegressor(n_neighbors=9, weights="uniform"))])),
    ("KNN_k5_distance", lambda prep: Pipeline([("prep", prep), ("model", KNeighborsRegressor(n_neighbors=5, weights="distance"))])),
    ("KNN_k9_distance", lambda prep: Pipeline([("prep", prep), ("model", KNeighborsRegressor(n_neighbors=9, weights="distance"))])),
    ("RandomForest", lambda prep: Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1))])),
    ("ExtraTrees", lambda prep: Pipeline([("prep", prep), ("model", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1))])),
    ("HistGradientBoosting", lambda prep: Pipeline([("prep", prep), ("model", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, l2_regularization=1.0, random_state=42))])),
]

results = []
print(f"scikit-learn: {sklearn.__version__}")
print(f"Dataset: {csv_path}")
print(f"Rows: {len(df)} | predictors: {X.shape[1]} | CV: 5-fold shuffled, seed=42")
print("Research results only: this does not validate performance on Indian/KV students.")
for set_name, features in feature_sets.items():
    for model_name, factory in models:
        estimator = factory(make_preprocessor(features))
        scores = cross_validate(
            estimator, features, y, cv=cv, n_jobs=1,
            scoring={"mae": "neg_mean_absolute_error",
                     "rmse": "neg_root_mean_squared_error", "r2": "r2"},
            error_score="raise",
        )
        maes = -scores["test_mae"]
        rmses = -scores["test_rmse"]
        r2s = scores["test_r2"]
        results.append({
            "feature_set": set_name, "model": model_name,
            "mae_mean": maes.mean(), "mae_sd": maes.std(ddof=1),
            "rmse_mean": rmses.mean(), "r2_mean": r2s.mean(),
        })
        print(f"{set_name:16} {model_name:22} MAE={maes.mean():.3f}±{maes.std(ddof=1):.3f} "
              f"RMSE={rmses.mean():.3f} R2={r2s.mean():.3f}")

out = pd.DataFrame(results).sort_values(["feature_set", "mae_mean"])
output_path = ROOT / "model-comparison-results.csv"
out.to_csv(output_path, index=False)
print("\nBEST PER FEATURE SET (lowest mean CV MAE):")
for set_name, group in out.groupby("feature_set", sort=False):
    row = group.iloc[0]
    print(f"  {set_name}: {row['model']} | MAE={row['mae_mean']:.3f}±{row['mae_sd']:.3f} "
          f"| RMSE={row['rmse_mean']:.3f} | R2={row['r2_mean']:.3f}")
print(f"\nSaved results to: {output_path}")
