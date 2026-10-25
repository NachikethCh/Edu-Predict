# EduPredict AI — Student Performance Predictor

EduPredict AI is a student-support prototype that is being adapted for Kendriya Vidyalaya (KV)/CBSE curriculum options and estimates a Mathematics grade only where the existing research model is available. Other listed subjects remain unsupported until separately trained and evaluated models exist. Students and educators can review the prediction context, inspect held-out model evaluation, and test one or more records from a CSV file.

## Start the app

Run the `artifacts/api-server: API Server` and `artifacts/edupredict-ai: web` workflows. No account, external API key, paid API, or database is required.

## Model in brief

- **Dataset:** UCI Student Performance — Mathematics subset only (395 records from the original study of two Portuguese secondary schools).
- **Target:** G3, the final-period grade on a 0–20 scale.
- **Prediction timing:** after period one. G1 is available; G2 and G3 are excluded from model inputs.
- **Algorithm:** standardized inverse-distance weighted k-nearest-neighbors regression, with k=9.
- **Evaluation:** a seeded 80/20 train/test split for the Mathematics subset only. The app reports Mathematics test-set MAE, RMSE, R², pass-threshold metrics at 10/20, and a training-mean baseline. Portuguese is not a supported app subject.
- **Scope:** this public dataset is not a CBSE dataset, and its results do not establish performance for other schools or countries.

## KV/CBSE curriculum scope

The interface offers configurable subject lists by Classes I–XII and Science, Commerce, or Humanities stream for Classes XI–XII. Subject offerings vary by school; administrators should confirm their local KV subject combinations. The current model is a research demonstration for Mathematics in Classes IX–XII only and is not validated for KV students. Other subjects must not receive predictions until suitable data and independent evaluation are available.

## Project documentation

See [`docs/project-report.md`](docs/project-report.md) for the problem statement, SDG 4 alignment, users, data provenance and privacy, feature descriptions, model evaluation, prototype capture checklist, test methodology, reflection templates, limitations, and a short pitch script.

## Attribution

Paulo Cortez and Alice Silva, “Using data mining to predict secondary school student performance,” 2008. UCI Machine Learning Repository, [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), DOI [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T). Data licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
