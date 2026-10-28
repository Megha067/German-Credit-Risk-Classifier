# German Credit Risk Classifier (ML Assignment 2)

## a. Problem statement

A bank has to decide whether to approve a loan applicant. Approving a customer who later defaults costs far more than turning away a good customer. This project builds and compares five classifiers that predict whether an applicant is a **good (0)** or **bad (1)** credit risk from their financial and personal profile. All five models are then served through an interactive Streamlit web app.

Bad credit risk is treated as the **positive class** (1), so Precision, Recall and F1 measure how well the model finds defaulters.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Statlog (German Credit Data) |
| Source | UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| Instances | 1,000 (minimum required: 500) |
| Features | 20 raw features: 7 numeric, 13 categorical (minimum required: 12) |
| Features after encoding | 61 (7 scaled numeric + 54 one-hot columns) |
| Target | `credit_risk`: 0 = good (700 rows), 1 = bad (300 rows) |
| Missing values | None |
| Class balance | 70% good / 30% bad (imbalanced) |

**Numeric features:** `duration_months`, `credit_amount`, `installment_rate`, `residence_since`, `age`, `existing_credits`, `liable_people`.

**Categorical features (coded A11, A12, ...):** `checking_status`, `credit_history`, `purpose`, `savings_status`, `employment_since`, `personal_status_sex`, `other_debtors`, `property`, `other_installment_plans`, `housing`, `job`, `telephone`, `foreign_worker`.

**Preprocessing:** the data is split 80/20 with stratification (`random_state=26`), giving 800 train rows and 200 test rows. A `ColumnTransformer` standardises the numeric columns and one-hot encodes the categorical ones. It is fitted on the training split only and saved as `model/preprocessor.joblib`. `test_data.csv` holds the 200 held-out rows in raw form (13 of the 20 feature columns are text codes), so the app applies the saved preprocessor to the uploaded file and never refits.

**Model settings:**

| Model | Settings |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `max_depth=5` |
| KNN | `n_neighbors=7` |
| Naive Bayes | `GaussianNB` defaults |
| SVM | RBF kernel; AUC is computed from `decision_function` |

## c. GitHub repository link

`<PASTE YOUR GITHUB REPO LINK HERE>`

Live Streamlit app: `(https://german-credit-risk-classifier.streamlit.app/)`

```
german-credit-classifier/
|-- app.py                  Streamlit app
|-- train.py                trains all 5 models, saves artifacts and test_data.csv
|-- requirements.txt
|-- README.md
|-- test_data.csv           200 held-out rows (raw features + credit_risk)
|-- raw/german.data         original UCI file
`-- model/
    |-- preprocessor.joblib
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- support_vector_machine.joblib
    `-- metrics.csv
```

To reproduce: `pip install -r requirements.txt`, then `python train.py`, then `streamlit run app.py`.

## d. Models used

Metrics on the 200-row held-out test set (positive class = bad credit):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7300 | 0.8052 | 0.5556 | 0.5000 | 0.5263 | 0.3392 |
| Decision Tree | 0.7300 | 0.7490 | 0.5789 | 0.3667 | 0.4490 | 0.2948 |
| KNN | 0.7500 | 0.6907 | 0.6667 | 0.3333 | 0.4444 | 0.3361 |
| Naive Bayes | 0.7250 | 0.7795 | 0.5281 | 0.7833 | 0.6309 | 0.4457 |
| Support Vector Machine | 0.7600 | 0.8068 | 0.6579 | 0.4167 | 0.5102 | 0.3783 |

Extra metric shown in the app, using the dataset's official cost matrix (approving a bad customer costs 5, rejecting a good one costs 1): bank cost = 5 x FN + 1 x FP, lower is better.

| Model | Bank cost |
|---|---|
| Logistic Regression | 174 |
| Decision Tree | 206 |
| KNN | 210 |
| Naive Bayes | 107 |
| Support Vector Machine | 188 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second-best AUC (0.805) and a balanced precision/recall (0.56 / 0.50). It is stable and gives well-calibrated probabilities, which suits credit scoring. Accuracy of 0.73 is barely above the 0.70 obtained by predicting "good" for everyone, so accuracy alone is misleading here. |
| Decision Tree | Depth was limited to 5 to curb overfitting. It has the lowest MCC (0.295) and a low recall of 0.37, so it misses most defaulters. A single tree is unstable on 800 training rows and split choices change a lot between samples. |
| KNN | Second-highest accuracy (0.75) and the highest precision (0.67), but the lowest recall (0.33) and the lowest AUC (0.691). Distance in the 61-dimensional one-hot space is noisy, and the majority good class dominates the neighbours, so it rarely flags a customer as bad. |
| Naive Bayes | Lowest accuracy (0.725) but by far the best recall (0.78), F1 (0.63) and MCC (0.446). Its independence assumption is violated by the one-hot columns, which pushes it to over-predict "bad". That cautious behaviour is useful when missing a defaulter is 5 times costlier: it has the lowest bank cost (107). |
| Support Vector Machine | Best accuracy (0.76) and best AUC (0.807), with good precision (0.66) and the second-best MCC (0.378). Recall is only 0.42, so it is conservative about labelling bad risk at the default threshold. The best at ranking applicants, but not tuned for the cost of missing defaulters. |

**Overall winner for this dataset:**

- **By standard metrics:** the **Support Vector Machine** has the best accuracy and AUC. Logistic Regression is within 0.002 AUC of it, which is well inside the noise of a 200-row test set.
- **By the metric that suits this problem:** **Naive Bayes** is the best choice. It leads on Recall, F1 and MCC and has the lowest bank cost, because missing a bad customer is 5 times more expensive than rejecting a good one. This is the model I would deploy.
- **Caveat:** with only 200 test rows (60 bad cases), one extra correct prediction changes recall by about 1.7 points. Differences of a few points between models are not statistically solid.

## Streamlit app features

- CSV upload of test data, with checks for missing columns and invalid target values
- Model selection dropdown for all five models
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC and bank cost
- Confusion matrix and classification report
- A "Compare all models" tab showing all five models side by side on the uploaded file
- Saved models are loaded from `model/`, and nothing is retrained
