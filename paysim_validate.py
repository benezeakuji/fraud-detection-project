import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb

df = pd.read_csv("data/paysim.csv")
print("Shape:", df.shape)
print("Fraud cases: {}, Rate: {:.4%}".format(df["isFraud"].sum(), df["isFraud"].mean()))

# Convert the text 'type' column into numbers
le = LabelEncoder()
df["type_enc"] = le.fit_transform(df["type"])

features = ["step", "type_enc", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"]
X = df[features].copy()
y = df["isFraud"]

num_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Training set: {len(X_train)} rows, scale_pos_weight = {scale_pos_weight:.1f}")

model = xgb.XGBClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.15,
    scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
    random_state=42, n_jobs=-1, tree_method="hist"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]

print("\n--- Results on real PaySim test set ---")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_score):.3f}")
print(f"AUC-PR:    {average_precision_score(y_test, y_score):.3f}")
print("Confusion matrix [[TN FP] [FN TP]]:", confusion_matrix(y_test, y_pred).tolist())