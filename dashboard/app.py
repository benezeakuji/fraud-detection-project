import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Detection Analyst Dashboard", layout="wide")

@st.cache_resource
def load_everything():
    df = pd.read_csv("data/creditcard.csv").drop_duplicates()
    X = df.drop(columns=["Class"])
    y = df["Class"]

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    model = joblib.load("models/xgb_cost.joblib")
    explainer = shap.TreeExplainer(model)
    return model, explainer, X_test, y_test

model, explainer, X_test, y_test = load_everything()

scores = model.predict_proba(X_test)[:, 1]
results_df = X_test.copy()
results_df["true_label"] = y_test.values
results_df["risk_score"] = scores
results_df["flagged"] = results_df["risk_score"] >= 0.5
results_df["transaction_id"] = results_df.index

st.title("Fraud Detection Analyst Dashboard")
st.caption(f"Model: XGBoost (cost-sensitive) | Scoring {len(results_df):,} held-out test transactions")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions scored", f"{len(results_df):,}")
col2.metric("Flagged as fraud", f"{int(results_df['flagged'].sum()):,}")
col3.metric("Actual fraud in set", f"{int(results_df['true_label'].sum()):,}")
col4.metric("Flag rate", f"{results_df['flagged'].mean():.3%}")

st.subheader("Flagged Transactions (Risk Queue)")
flagged = results_df[results_df["flagged"]].sort_values("risk_score", ascending=False)
st.dataframe(
    flagged[["transaction_id", "Amount", "risk_score", "true_label"]].head(50),
    use_container_width=True,
)

st.subheader("Explain a Flagged Transaction")
if len(flagged) > 0:
    selected_id = st.selectbox("Select transaction ID", flagged["transaction_id"].head(50).tolist())
    row = X_test.loc[[selected_id]]
    sv = np.array(explainer.shap_values(row))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    contrib = pd.DataFrame({
        "feature": row.columns,
        "shap_value": sv.flatten()
    }).sort_values("shap_value", key=abs, ascending=False).head(10)
    st.write(f"**Risk score:** {results_df.loc[selected_id, 'risk_score']:.4f}")
    st.bar_chart(contrib.set_index("feature")["shap_value"])
    st.caption("Top features driving this transaction's fraud score. Positive = pushes toward fraud.")
else:
    st.info("No flagged transactions in this test slice.")