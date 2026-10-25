#!/usr/bin/env python3
"""Compare candidate regressors using ONLY EduPredict production feature columns."""
from pathlib import Path
import sys
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
csv_path = ROOT / "artifacts" / "api-server" / "data" / "student-mat.csv"
if not csv_path.exists():
    sys.exit(f"ERROR: Dataset not found: {csv_path}")

df = pd.read_csv(csv_path, sep=";")
# These are the exact source columns corresponding to FEATURE_KEYS in the
# existing TypeScript model. In particular, G2 and all other unused columns
# are excluded so this comparison matches production inputs.
columns = [
    "G1", "age", "studytime", "failures", "absences", "schoolsup",
    "famsup", "paid", "higher", "internet", "goout", "traveltime", "health",
]
missing = [c for c in ["G3", *columns] if c not in df.columns]
if missing:
    sys.exit(f"ERROR: Missing expected dataset columns: {missing}")

X = df[columns].copy()
y = pd.to_numeric(df["G3"], errors="raise")

binary = ["schoolsup", "famsup", "paid", "higher", "internet"]
for c in binary:
    X[c] = X[c].map({"yes": 1, "no": 0})
numeric = [c for c in columns if c not in binary]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric),
    ("binary", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]), binary),
])
cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = [
    ("KNN_k3_uniform", KNeighborsRegressor(n_neighbors=3, weights="uniform")),
    ("KNN_k5_uniform", KNeighborsRegressor(n_neighbors=5, weights="uniform")),
    ("KNN_k9_uniform", KNeighborsRegressor(n_neighbors=9, weights="uniform")),
    ("KNN_k5_distance", KNeighborsRegressor(n_neighbors=5, weights="distance")),
    ("KNN_k9_distance", KNeighborsRegressor(n_neighbors=9, weights="distance")),
    ("RandomForest", RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, max_features=0.8,
        random_state=42, n_jobs=-1)),
    ("ExtraTrees", ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=2, max_features=0.8,
        random_state=42, n_jobs=-1)),
    ("HistGradientBoosting", HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.06, l2_regularization=1.0,
        random_state=42)),
]

print(f"scikit-learn: {sklearn.__version__}")
print(f"Dataset: {csv_path}")
print(f"Rows: {len(df)} | predictors: {len(columns)}")
print("FEATURES: " + ", ".join(columns))
print("Five-fold shuffled CV, seed=42. G2 and all non-production features excluded.")
print("Research results only; not validation for Indian/KV students.")
results = []
for name, model in models:
    estimator = Pipeline([("prep", preprocessor), ("model", model)])
    scores = cross_validate(
        estimator, X, y, cv=cv, n_jobs=1,
        scoring={"mae": "neg_mean_absolute_error",
                 "rmse": "neg_root_mean_squared_error", "r2": "r2"},
        error_score="raise",
    )
    mae = -scores["test_mae"]
    rmse = -scores["test_rmse"]
    r2 = scores["test_r2"]
    row = {
        "model": name,
        "mae_mean": mae.mean(),
        "mae_sd": mae.std(ddof=1),
        "rmse_mean": rmse.mean(),
        "r2_mean": r2.mean(),
    }
    results.append(row)
    print(f"{name:22} MAE={row['mae_mean']:.3f}±{row['mae_sd']:.3f} "
          f"RMSE={row['rmse_mean']:.3f} R2={row['r2_mean']:.3f}")

result_df = pd.DataFrame(results).sort_values("mae_mean")
output = ROOT / "production-features-comparison.csv"
result_df.to_csv(output, index=False)
best = result_df.iloc[0]
print("\nBEST WITH PRODUCTION FEATURES:")
print(f"{best['model']} | MAE={best['mae_mean']:.3f}±{best['mae_sd']:.3f} "
      f"| RMSE={best['rmse_mean']:.3f} | R2={best['r2_mean']:.3f}")
print(f"Saved results to: {output}")
