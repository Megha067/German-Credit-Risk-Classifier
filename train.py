"""Train 5 classifiers on the Statlog German Credit dataset and save artifacts.

Outputs
-------
model/preprocessor.joblib   fitted ColumnTransformer (one-hot + scaling)
model/<name>.joblib         5 trained classifiers
test_data.csv               held-out test split (raw features + target)
model/metrics.csv           test-set metrics for the README table
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "liable_people", "telephone",
    "foreign_worker",
]
TARGET = "credit_risk"  # 1 = bad credit risk (positive class), 0 = good
NUMERIC = ["duration_months", "credit_amount", "installment_rate",
           "residence_since", "age", "existing_credits", "liable_people"]
CATEGORICAL = [c for c in COLUMNS if c not in NUMERIC]
SEED = 26


def load_data() -> pd.DataFrame:
    df = pd.read_csv(BASE / "raw" / "german.data", sep=" ", header=None,
                     names=COLUMNS + [TARGET])
    # original coding: 1 = good, 2 = bad  ->  0 = good, 1 = bad
    df[TARGET] = (df[TARGET] == 2).astype(int)
    return df


def main() -> None:
    df = load_data()
    X, y = df[COLUMNS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    pre = ColumnTransformer(
        [("num", StandardScaler(), NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
          CATEGORICAL)])
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                  random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=5,
                                                random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", random_state=SEED),
    }

    rows = []
    for name, clf in models.items():
        clf.fit(Xtr, y_train)
        pred = clf.predict(Xte)
        # SVC has no predict_proba here; its decision_function is a valid
        # ranking score for AUC
        score = (clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba")
                 else clf.decision_function(Xte))
        rows.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, score),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "MCC": matthews_corrcoef(y_test, pred),
        })
        joblib.dump(clf, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    joblib.dump(pre, MODEL_DIR / "preprocessor.joblib")
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(BASE / "test_data.csv", index=False)

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print(f"train={len(X_train)} test={len(X_test)} "
          f"features after encoding={Xtr.shape[1]}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
