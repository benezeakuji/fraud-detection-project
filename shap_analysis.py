import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Same preprocessing and split as before ---
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

# --- Load the winning model from Stage 5 ---
model = joblib.load("models/xgb_cost.joblib")
explainer = shap.TreeExplainer(model)

# --- Global importance: explain a 3,000-row sample ---
sample = X_test.sample(n=3000, random_state=42)
shap_values = explainer.shap_values(sample)

# (Handle the fact that different shap/xgboost versions return slightly
# different array shapes for binary classification.)
sv = np.array(shap_values)
if sv.ndim == 3:
    sv = sv[:, :, 1]

mean_abs_shap = np.abs(sv).mean(axis=0)
importance = pd.DataFrame({
    "feature": sample.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("Top 10 globally important features:")
print(importance.head(10).to_string(index=False))

# --- Worked example: the fraud case the model is MOST confident about ---
proba = model.predict_proba(X_test)[:, 1]
proba_series = pd.Series(proba, index=X_test.index)
fraud_idx = y_test[y_test == 1].index
best_idx = proba_series.loc[fraud_idx].idxmax()

print(f"\nWorked example: transaction index {best_idx}, risk score {proba_series.loc[best_idx]:.5f}")

row = X_test.loc[[best_idx]]
sv_one = explainer.shap_values(row)
sv_one = np.array(sv_one)
if sv_one.ndim == 3:
    sv_one = sv_one[:, :, 1]

contrib = pd.DataFrame({
    "feature": row.columns,
    "shap_value": sv_one.flatten()
}).sort_values("shap_value", key=abs, ascending=False)

print("\nTop 8 features driving this specific prediction:")
print(contrib.head(8).to_string(index=False))