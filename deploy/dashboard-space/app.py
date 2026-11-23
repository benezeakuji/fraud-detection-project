import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st

st.set_page_config(page_title="Fraud Detection Analyst Dashboard", layout="wide")

# Resolve paths relative to this script's own location, not the process's
# current working directory — Streamlit Cloud runs apps with the repo root
# as cwd, not the app's own folder, so a bare "models/..." path breaks there.
MODELS_DIR = Path(__file__).resolve().parent / "models"


@st.cache_resource
def load_everything():
    # Load the SAME held-out test split used to produce Table 5.1 in the report
    # (models/X_test.csv, models/y_test.csv — 42,559 transactions, 71 fraud,
    # already scaled by preprocess.py), rather than re-downloading and
    # re-splitting the 150MB primary dataset on every deploy / cold start.
    X_test = pd.read_csv(MODELS_DIR / "X_test.csv")
    y_test = pd.read_csv(MODELS_DIR / "y_test.csv")["Class"]

    model = joblib.load(MODELS_DIR / "xgb_cost.joblib")
    explainer = shap.TreeExplainer(model)
    return model, explainer, X_test, y_test


@st.cache_resource
def load_paysim_summary():
    # Precomputed on the full 6.36M-row PaySim dataset (paysim_validate.py's
    # methodology: a fresh XGBoost model trained on PaySim's own feature set,
    # since it doesn't share V1-V28 with the credit-card data). Only the
    # small summary + a 50-row flagged sample are shipped here, not the
    # 493MB source file.
    path = MODELS_DIR / "paysim_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


model, explainer, X_test, y_test = load_everything()

scores = model.predict_proba(X_test)[:, 1]
results_df = X_test.copy()
results_df["true_label"] = y_test.values
results_df["risk_score"] = scores
results_df["flagged"] = results_df["risk_score"] >= 0.5
results_df["transaction_id"] = results_df.index

st.title("Fraud Detection Analyst Dashboard")

tab_primary, tab_paysim = st.tabs([
    "Primary Dataset — Credit Card (Kaggle/ULB)",
    "Secondary Dataset — PaySim (cross-validation)",
])

with tab_primary:
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

with tab_paysim:
    paysim = load_paysim_summary()
    if paysim is None:
        st.warning("PaySim summary not found — run the export script and redeploy to populate this tab.")
    else:
        rows, cols = paysim["dataset_shape"]
        st.caption(
            f"Model: XGBoost (fresh fit, PaySim's own features — step, type, amount, balances) "
            f"| Cross-validated on {rows:,} real mobile-money transactions "
            f"({paysim['total_fraud']:,} fraud, {paysim['fraud_rate']:.3%} rate)"
        )
        m = paysim["metrics"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precision", f"{m['precision']:.3f}")
        c2.metric("Recall", f"{m['recall']:.3f}")
        c3.metric("F1-score", f"{m['f1']:.3f}")
        c4.metric("AUC-ROC", f"{m['auc_roc']:.3f}")
        c5.metric("AUC-PR", f"{m['auc_pr']:.3f}")

        st.subheader("Confusion Matrix (held-out PaySim test set)")
        tn, fp = m["confusion_matrix"][0]
        fn, tp = m["confusion_matrix"][1]
        cm_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual: Legit", "Actual: Fraud"],
            columns=["Predicted: Legit", "Predicted: Fraud"],
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption(
            f"Test set: {paysim['test_rows']:,} transactions "
            f"(trained on {paysim['train_rows']:,}, class imbalance weight {paysim['scale_pos_weight']:.1f}x)"
        )

        st.subheader("Flagged Transactions (Risk Queue)")
        sample_df = pd.DataFrame(paysim["flagged_sample"])
        st.dataframe(
            sample_df[["transaction_id", "type", "amount", "risk_score", "true_label", "isFlaggedFraud"]],
            use_container_width=True,
        )
        st.caption(
            "Top 50 flagged transactions by risk score, precomputed from the full PaySim run. "
            "'isFlaggedFraud' is PaySim's own naive built-in flag, shown for comparison with the model's risk score."
        )
