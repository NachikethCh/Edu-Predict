from pathlib import Path

import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("artifacts/api-server/data/student-mat.csv")
df = pd.read_csv(DATA, sep=";")

binary = ["schoolsup", "famsup", "paid", "higher", "internet"]
base_features = [
    "age", "studytime", "failures", "absences",
    "schoolsup", "famsup", "paid", "higher",
    "internet", "goout", "traveltime", "health",
]

for col in binary:
    df[col] = df[col].map({"yes": 1, "no": 0})

y = df["G3"].astype(float)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "RandomForest": (
        RandomForestRegressor(random_state=42, n_jobs=1),
        {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 5, 8, 12, 16],
            "model__min_samples_leaf": [1, 2, 4, 6],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": [0.6, 0.8, 1.0],
        },
    ),
    "ExtraTrees": (
        ExtraTreesRegressor(random_state=42, n_jobs=1),
        {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 5, 8, 12, 16],
            "model__min_samples_leaf": [1, 2, 4, 6],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": [0.6, 0.8, 1.0],
        },
    ),
    "HistGradientBoosting": (
        HistGradientBoostingRegressor(random_state=42),
        {
            "model__max_iter": [100, 150, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.07, 0.1],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__l2_regularization": [0.0, 0.5, 1.0, 2.0],
        },
    ),
}

scenarios = {
    "early_no_grades": base_features,
    "updated_with_G1": ["G1"] + base_features,
}

all_results = []

print("scikit-learn:", sklearn.__version__)
print("Dataset rows:", len(df))
print("Evaluation: 5-fold CV, shuffled, random_state=42")
print("Tuning: randomized search with MAE as the selection metric")
print("Research results only; not validated for Indian/KV students.\n")

for scenario, features in scenarios.items():
    X = df[features].copy()

    numeric = [c for c in features if c not in binary]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("binary", SimpleImputer(strategy="most_frequent"), binary),
    ])

    print("=" * 72)
    print("SCENARIO:", scenario)
    print("Features:", ", ".join(features))

    for model_name, (model, params) in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=15,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X, y)

        # Evaluate the complete tuning procedure on separate outer folds.
        # This is slower but gives a less optimistic estimate than reporting
        # the same CV scores used to select the hyperparameters.
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_score

        outer_scores = []
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=123)

        for train_idx, test_idx in outer_cv.split(X, y):
            inner_search = clone(search)
            inner_search.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = inner_search.predict(X.iloc[test_idx])
            actual = y.iloc[test_idx]

            from sklearn.metrics import (
                mean_absolute_error,
                root_mean_squared_error,
                r2_score,
            )

            outer_scores.append({
                "mae": mean_absolute_error(actual, pred),
                "rmse": root_mean_squared_error(actual, pred),
                "r2": r2_score(actual, pred),
            })

        outer_df = pd.DataFrame(outer_scores)
        row = {
            "scenario": scenario,
            "model": model_name,
            "mae_mean": outer_df["mae"].mean(),
            "mae_sd": outer_df["mae"].std(ddof=1),
            "rmse_mean": outer_df["rmse"].mean(),
            "r2_mean": outer_df["r2"].mean(),
            "best_inner_cv_mae": -search.best_score_,
            "best_params": str(search.best_params_),
        }
        all_results.append(row)

        print(
            f"{model_name:22} "
            f"MAE={row['mae_mean']:.3f} +/- {row['mae_sd']:.3f}  "
            f"RMSE={row['rmse_mean']:.3f}  "
            f"R2={row['r2_mean']:.3f}"
        )
        print("  Best parameters:", search.best_params_)

results = pd.DataFrame(all_results)
output = Path("tuned-model-results.csv")
results.to_csv(output, index=False)

print("\n" + "=" * 72)
print("RESULTS SUMMARY (sorted by outer-fold MAE)")
print(results.sort_values(["scenario", "mae_mean"]).to_string(index=False))
print("\nSaved:", output.resolve())
