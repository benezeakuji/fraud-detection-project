import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# --- Same preprocessing as Stage 2 ---
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

print("Training set before any imbalance handling: {} rows, {} fraud ({:.4%})".format(
    len(X_train), y_train.sum(), y_train.mean()))

# --- Technique 1: SMOTE ---
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print()
print("After SMOTE: {} rows, {} fraud ({:.2%})".format(
    len(X_train_smote), y_train_smote.sum(), y_train_smote.mean()))

# --- Technique 2: cost-sensitive weight ---
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print()
print("Cost-sensitive scale_pos_weight: {:.1f}".format(scale_pos_weight))