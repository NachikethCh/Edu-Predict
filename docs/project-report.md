# EduPredict AI — Project Report

## 1. Problem statement and SDG alignment

### Problem statement

Students and educators often receive a final course result after there is little time left to respond. The project explores whether information available after the first grading period can provide an early, understandable estimate of a student's final course grade and prompt supportive follow-up.

### Alignment with SDG 4: Quality Education

The prototype aligns with the goal of inclusive, equitable quality education by making an early-warning workflow available for students and educators to explore. It is designed to support timely discussion and learning support. This project has not measured or demonstrated an improvement in academic outcomes, and it does not claim to do so.

### Intended measurable benefit

If evaluated with a school and its consent, the measurable goal would be to improve the timeliness of support conversations. Possible measures include the time between a first-period risk signal and a teacher check-in, participation in an agreed support plan, and later course outcomes compared with an appropriate baseline. No such school evaluation has been performed for this prototype.

## 2. Intended users and needs

- **Students:** want an understandable view of how early course signals relate to a possible final grade, along with practical steps they can choose to try.
- **Educators:** want a quick screening aid that can help prioritize supportive conversations, without replacing professional judgment.
- **Project reviewers:** need to inspect the data source, model choices, evaluation method, limitations, and tests.

## 3. Proposed solution and design rationale

EduPredict AI accepts a small set of first-period academic and study-context fields, predicts the final-period grade on the dataset's 0–20 scale, and shows the estimate alongside similar-record context and input signals. A separate evaluation view reports results on held-out records and compares the model with a training-mean baseline. A CSV workflow validates rows individually and makes valid predictions available for export.

The project uses a regression model rather than directly assigning a student to a permanent category. Grade bands are presentation aids: below 10 is “needs support,” 10 to under 14 is “developing,” and 14 or higher is “on track.” The 10/20 cutoff is used for the pass-threshold evaluation; the additional display bands are prototype heuristics, not official grading categories.

### User workflow / storyboard

1. A student or educator opens the dashboard and reads what the tool can and cannot say.
2. They select a class, an applicable stream, and a subject from the configurable KV/CBSE curriculum, then enter first-period grade and available context. Only supported model combinations can generate estimates. The curriculum settings page allows a user to edit the subject list in the current browser; it is not an authenticated or centrally managed school administration system.
3. The app returns a rounded estimate, the grade band, the grade range of similar training records, and the share of those neighbors whose final grade was at least 10/20.
4. The user reviews the input signals and practical suggestions, then decides whether a supportive conversation or study adjustment would be useful.
5. A reviewer can open model evaluation to inspect held-out metrics, baseline comparisons, and feature importance.
6. An educator can upload a CSV, review valid and invalid rows, and export the valid predictions.

## 4. Dataset, collection method, features, and privacy

### Source and collection

- **Dataset:** UCI Machine Learning Repository, *Student Performance*, dataset 320.
- **Source page:** https://archive.ics.uci.edu/dataset/320/student+performance
- **DOI:** 10.24432/C5TG7T
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Original study:** Paulo Cortez and Alice Silva, “Using data mining to predict secondary school student performance,” 2008.
- **Collection described by UCI:** school reports and questionnaires from two Portuguese secondary schools.
- **Rows used by this application:** 395 Mathematics records. The original UCI repository also contains a separate Portuguese-language subset, but it is excluded from this application.
- **Target:** `G3`, final-period grade, 0–20.

The data is public, historical, and specific to two Portuguese schools. This application uses the Mathematics subset only (395 records); the Portuguese subset is not used or offered as a prediction subject. It is not a CBSE dataset and should not be treated as representative of students in India or of any school not represented in the source.

### Prediction-time feature choice

Prediction is framed as occurring after period one. `G1` is available at that point. `G2` and `G3` are excluded from model inputs because they occur later; `G3` is the target. This avoids using later-period information to predict an earlier outcome.

| Model input | Source meaning / representation |
|---|---|
| First-period grade (`G1`) | Grade from 0 to 20 |
| Age | Student age, 15–22 |
| Weekly study time | Four ordered categories, 1–4 |
| Past class failures | Encoded from 0–4; 4 represents three or more |
| Absences | Recorded count, 0–93 |
| School support | Whether extra educational support is recorded |
| Family support | Whether family educational support is recorded |
| Paid subject classes | Whether paid subject classes are recorded |
| Higher education intent | Whether the student reports wanting higher education |
| Internet at home | Yes/no |
| Going out | Ordered scale, 1–5 |
| Travel time | Ordered scale, 1–4 |
| Health | Self-reported ordered scale, 1–5 |

Sex, school identity, address, family composition, parent education, occupations, and relationship status are not model inputs. The selected feature list is a prototype scope choice, not a fairness certification.

### Privacy

Student entries and uploaded CSV contents are processed for the current request and are not saved to an account or database by this prototype. Do not upload names, student IDs, contact details, or other direct identifiers. Public-source data can still be sensitive when combined with school records; any real school use would need consent, access controls, retention rules, and a privacy review.

## 5. Model selection, training, and evaluation

### Algorithm and preprocessing

EduPredict uses standardized, inverse-distance weighted k-nearest-neighbors regression with `k=9`. Feature means and standard deviations are learned from the training subset only. Boolean inputs are represented as 0/1. A neighbor closer in standardized feature space receives more weight when averaging its final grade.

This is a genuine supervised model trained from the UCI records. The fixed seed is 42. The Mathematics subset has a deterministic 80/20 train/test split. The training-mean predictor is reported as a simple baseline. Feature importance is calculated by permuting each test feature and reporting the resulting increase in test MAE; it is a model sensitivity measure, not evidence that a feature causes an outcome.

### Metrics and results

Regression errors are measured in grade points on the 0–20 scale. Pass-threshold metrics classify grades at 10/20. These results are computed on held-out rows; they are not evidence of expected performance at another school.

| Subject | Rows | Train / test | Model MAE | Model RMSE | R² | Baseline MAE | Baseline RMSE | Pass accuracy | Pass precision | Pass recall | Pass F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mathematics | 395 | 316 / 79 | 2.07 | 2.78 | 0.536 | 3.02 | 4.09 | 0.722 | 0.870 | 0.714 | 0.784 |

The model's MAE and RMSE are lower than the training-mean baseline on this Mathematics test split. The feature-importance view ranks first-period grade highest in this run. This single holdout is not external validation and does not establish performance in KV schools.

### Evaluation caveats

- The split is one deterministic holdout, not cross-validation or an external validation.
- `G1` is strongly associated with `G3` in the source dataset, as noted by UCI; including it makes this a post-first-period prediction, not a prediction made before any grade exists.
- Grade estimates and neighbor summaries are approximate associations, not calibrated probabilities or causal explanations.
- A well-performing result on these records does not validate use for a different curriculum, country, student group, or high-stakes decision.

## 6. Prototype description and capture checklist

The prototype includes a dashboard, individual prediction form and result, model evaluation page, and CSV upload/validation/export flow.

### Screenshots or screen-capture checklist

- [ ] Dashboard with project scope and dataset notice visible.
- [ ] Prediction form with feature definitions and validation.
- [ ] A clearly labeled sample prediction and its supporting signals.
- [ ] Model evaluation page showing the Mathematics held-out split, baseline metrics, explicit KV validation limits, and supported-model scope.
- [ ] CSV upload showing a mix of valid and invalid rows with row-level messages.
- [ ] CSV export opened in a spreadsheet to confirm the output columns.
- [ ] Narrow mobile viewport showing navigation and forms without horizontal overflow.

Capture screenshots only after opening the running prototype. Do not present a test/demo record as a real student outcome.

## 7. Testing methodology and results

The API test suite covers deterministic training and held-out evaluation, predictions for new records, prediction output bounds, CSV row parsing, invalid CSV rows, unsupported subject combinations, missing feature columns, and API request validation. The current UI revision still needs a full build and browser smoke test in Replit before it should be described as release-verified.

### Release verification status

The source includes API/model tests, but this edited ZIP could not be dependency-installed or fully built in the current environment because the package registry was unreachable. Run the project typecheck, API tests, production build, and browser smoke tests in Replit before presenting the revision as fully verified. User feedback has not been collected.

### Feedback record

No interviews or user feedback have been collected for this prototype. If feedback is gathered later, record the date, participant role, consent status, questions asked, and verbatim or accurately summarized responses here. Do not add invented quotes or activities.

## 8. Reflection templates

Complete these only with real observations.

### Team reflection (complete if a team worked on the project)

- What did the team plan and complete?
- Which responsibilities did each member actually take on?
- What changed during development, and why?
- What evidence from tests or user feedback informed decisions?
- What would the team do differently next time?

### Individual reflection

- What did I contribute myself?
- What technical or subject-matter idea did I learn?
- What was difficult, and how did I respond?
- Which result surprised me, and what evidence supports that?
- What would I improve before using this with real students?

## 9. Limitations, ethics, and future work

### Limitations and ethical considerations

- The dataset covers only two Portuguese schools and is not representative of all students.
- Grade distributions, subject structures, support systems, and school practices differ across contexts.
- Predictions may reproduce patterns in historical data and may be less useful for groups or circumstances poorly represented there.
- Several input fields describe personal circumstances. The interface should be used for support, not punishment, ranking, admission, or labeling.
- The model provides no calibrated uncertainty interval. The range shown is a descriptive range of nearby training examples, not a confidence interval.
- The result may not reflect a student's current circumstances or potential. Human review and student context matter.
- No study has shown that using this app improves grades, retention, equity, or learning.

### Future improvements

- Validate with an appropriately consented, locally relevant dataset before making any local performance claim.
- Compare the current model with regularized regression and tree-based models using nested or repeated evaluation.
- Add calibration and subgroup error analysis only where data, consent, and privacy protections support it.
- Conduct usability sessions with students and educators and document feedback accurately.
- Co-design support recommendations with educators and students; evaluate accessibility and localization.
- Define a school-level privacy, retention, and governance process before any real deployment.

## 10. Video-pitch script (about 2 minutes)

> Students can receive clear signs that they need help only after much of a course has passed. EduPredict AI explores whether first-period information can help students and educators start a supportive conversation earlier.
>
> This project aligns with United Nations Sustainable Development Goal 4, Quality Education. It is a prototype intended to support timely attention, not to claim that it has already improved student outcomes.
>
> The application uses only the Mathematics subset of the UCI Student Performance dataset, collected from two Portuguese secondary schools. The interface provides configurable KV/CBSE curriculum choices for Classes I–XII, but a prediction is enabled only for Mathematics in Classes IX–XII because that is the only available evaluated model. Other subjects are marked unsupported rather than receiving an unvalidated estimate. It predicts the final-period grade from the first-period grade and selected early academic and study-context fields. The second-period grade and final grade are excluded from the inputs to avoid looking ahead.
>
> EduPredict trains a standardized, weighted nearest-neighbors regression model and compares it with a training-mean baseline using a fixed 80/20 held-out split. On this dataset, the Mathematics model had a mean absolute error of 2.07 grade points. These are results on one historical test split; they do not show how the model would perform in a CBSE school or any other setting.
>
> In the demonstration, I select a class and subject, enter first-period information, review an estimate only if the combination is supported, then open the evaluation page to see the Mathematics held-out metrics. Educators can also upload a CSV, inspect validation errors, and export valid predictions.
>
> The tool does not store student entries. Its predictions are not official grades, probabilities, or decisions about a student. Before any real use, it would need local validation, student and educator feedback, and a strong privacy and fairness review.
>
> Acknowledgement: Paulo Cortez and Alice Silva, the UCI Machine Learning Repository, and the original Student Performance dataset authors.
