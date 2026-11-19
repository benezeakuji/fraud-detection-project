import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import os

os.makedirs("models", exist_ok=True)

# --- Same preprocessing as Stages 2 and 3 ---
df = pd.read_csv("data/creditcard.csv")
df = df.drop_duplicates()

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

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# --- Save these so later stages don't need to redo preprocessing ---
joblib.dump(scaler, "models/scaler.joblib")
X_val.to_csv("models/X_val.csv", index=False)
y_val.to_csv("models/y_val.csv", index=False)
X_test.to_csv("models/X_test.csv", index=False)
y_test.to_csv("models/y_test.csv", index=False)

# --- Define the six models ---
# n_jobs=-1 tells scikit-learn/XGBoost to use all CPU cores on YOUR machine.
# (My own build used n_jobs=2 to simulate constrained hardware — on your
# laptop, -1 just trains faster. The results themselves won't change.)
models_smote = {
    "logreg_smote": LogisticRegression(max_iter=1000, random_state=42),
    "rf_smote": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "xgb_smote": xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.15,
                                     eval_metric="aucpr", random_state=42, n_jobs=-1, tree_method="hist"),
}

models_cost = {
    "logreg_cost": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "rf_cost": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
    "xgb_cost": xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.15,
                                    scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                                    random_state=42, n_jobs=-1, tree_method="hist"),
}

print("Training SMOTE-resampled models...")
for name, model in models_smote.items():
    t0 = time.time()
    model.fit(X_train_smote, y_train_smote)
    elapsed = time.time() - t0
    joblib.dump(model, f"models/{name}.joblib")
    print(f"  {name}: trained in {elapsed:.1f}s, saved to models/{name}.joblib")

print("\nTraining cost-sensitive models...")
for name, model in models_cost.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    joblib.dump(model, f"models/{name}.joblib")
    print(f"  {name}: trained in {elapsed:.1f}s, saved to models/{name}.joblib")

print("\nAll 6 models trained and saved.")