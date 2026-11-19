import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

# --- Rebuild the exact same split as before (same random_state=42 => identical result) ---
raw_df = pd.read_csv("data/creditcard.csv")
df = raw_df.drop_duplicates()

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

# --- Rule-based baseline: flag top 1% of transactions by RAW (unscaled) amount ---
amount_threshold = raw_df["Amount"].quantile(0.99)
raw_amount_test = raw_df.loc[X_test.index, "Amount"]
baseline_pred = (raw_amount_test > amount_threshold).astype(int)

print(f"Baseline rule: flag any transaction over {amount_threshold:.2f}\n")

def evaluate(y_true, y_pred, y_score, name):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)
    print(f"{name}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  AUC-ROC={auc_roc:.3f}  AUC-PR={auc_pr:.3f}")
    print(f"  Confusion matrix [[TN FP] [FN TP]]: {cm.tolist()}\n")

# --- Baseline first ---
evaluate(y_test, baseline_pred, baseline_pred.astype(float), "Rule-based baseline (top 1% by amount)")

# --- All 6 trained models ---
model_names = ["logreg_smote", "rf_smote", "xgb_smote", "logreg_cost", "rf_cost", "xgb_cost"]

for name in model_names:
    model = joblib.load(f"models/{name}.joblib")
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_pred, y_score, name)