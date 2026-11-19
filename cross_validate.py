import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# --- Same preprocessing and split as before ---
df = pd.read_csv("data/creditcard.csv").drop_duplicates()
X = df.drop(columns=["Class"])
y = df["Class"]

scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# --- Same SMOTE resampling as Stage 3 ---
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# --- Take a 60,000-row stratified subsample, purely so this runs in a
#     reasonable amount of time on a laptop ---
sub_idx, _ = train_test_split(
    X_train_smote.index, train_size=60000, stratify=y_train_smote, random_state=42
)
X_cv = X_train_smote.loc[sub_idx]
y_cv = y_train_smote.loc[sub_idx]

print(f"Cross-validation subsample: {len(X_cv)} rows, {y_cv.mean():.2%} fraud\n")

models = {
    "logreg_smote": LogisticRegression(max_iter=1000, random_state=42),
    "rf_smote": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "xgb_smote": xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.15,
                                     eval_metric="aucpr", random_state=42, n_jobs=-1, tree_method="hist"),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X_cv, y_cv, cv=skf, scoring="f1", n_jobs=1)
    print(f"{name}: mean F1 = {scores.mean():.3f}, std = {scores.std():.4f}")