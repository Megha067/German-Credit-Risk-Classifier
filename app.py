"""Streamlit app: evaluate pre-trained credit-risk classifiers on an uploaded test CSV.

Models are loaded from model/*.joblib; nothing is retrained on uploaded data.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
TARGET = "credit_risk"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
CLASS_NAMES = ["Good credit (0)", "Bad credit (1)"]
FN_COST, FP_COST = 5, 1  # dataset's official cost matrix

st.set_page_config(page_title="German Credit Risk Classifier", layout="wide")


@st.cache_resource
def load_artifacts():
    pre = joblib.load(MODEL_DIR / "preprocessor.joblib")
    models = {n: joblib.load(MODEL_DIR / f) for n, f in MODEL_FILES.items()}
    return pre, models


def scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def evaluate(clf, X, y):
    pred = clf.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, scores(clf, X)),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "Bank cost (5:1)": FN_COST * fn + FP_COST * fp,
    }


st.title("German Credit Risk - Model Evaluation Dashboard")
st.caption("Statlog (German Credit Data), UCI. Positive class = bad credit "
           "risk. Pre-trained models are applied to your test CSV; "
           "no retraining happens here.")

pre, models = load_artifacts()
expected = list(pre.feature_names_in_)

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Select model", list(MODEL_FILES))
    st.markdown(f"Required columns: {len(expected)} features + `{TARGET}` "
                "(0 = good, 1 = bad).")
    with open(BASE / "test_data.csv", "rb") as f:
        st.download_button("Download sample test_data.csv", f,
                           "test_data.csv", "text/csv")

if uploaded is None:
    st.info("Upload a test CSV in the sidebar (or download the sample test "
            "file from there and upload it) to begin.")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in expected + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing columns: {missing}")
    st.stop()
if not set(df[TARGET].unique()) <= {0, 1}:
    st.error(f"`{TARGET}` must contain only 0 (good) and 1 (bad).")
    st.stop()
if df[TARGET].nunique() < 2:
    st.error("The target column needs both classes to compute AUC.")
    st.stop()

X = pre.transform(df[expected])
y = df[TARGET].astype(int)

tab_sel, tab_all, tab_data = st.tabs(
    [f"Selected model: {model_name}", "Compare all models", "Uploaded data"])

with tab_sel:
    pred, m = evaluate(models[model_name], X, y)
    st.subheader("Evaluation metrics")
    cols = st.columns(len(m))
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.0f}" if k.startswith("Bank") else f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(confusion_matrix(y, pred, labels=[0, 1]), annot=True,
                    fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        rep = classification_report(y, pred, target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3))
    st.caption("Bank cost = 5 x (bad customers approved) + 1 x (good "
               "customers rejected); lower is better.")

with tab_all:
    st.subheader("All five models on the uploaded test data")
    table = pd.DataFrame(
        {n: evaluate(c, X, y)[1] for n, c in models.items()}).T
    table.index.name = "ML Model Name"
    st.dataframe(table.style.format("{:.4f}", subset=table.columns[:-1])
                 .format("{:.0f}", subset=["Bank cost (5:1)"])
                 .highlight_max(subset=table.columns[:-2], color="#c6f6d5")
                 .highlight_min(subset=["Bank cost (5:1)"], color="#c6f6d5"),
                 width="stretch")
    st.caption("Green = best value in each column.")

with tab_data:
    st.write(f"{df.shape[0]} rows x {df.shape[1]} columns; "
             f"bad-credit share = {y.mean():.1%}")
    st.dataframe(df, width="stretch")
