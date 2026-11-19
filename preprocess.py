import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Step 1: Load the data ---
df = pd.read_csv("data/creditcard.csv")
print("Original shape:", df.shape)

# --- Step 2: Remove duplicate rows ---
duplicate_count = df.duplicated().sum()
print("Duplicate rows found:", duplicate_count)

df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)
print("Fraud cases after dedup:", df["Class"].sum())

# --- Step 3: Separate features (X) from the label (y) ---
X = df.drop(columns=["Class"])
y = df["Class"]

# --- Step 4: Scale Time and Amount ---
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

# --- Step 5: Split into train / validation / test (70% / 15% / 15%) ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print()
print("Train set: {} rows, {} fraud".format(len(X_train), y_train.sum()))
print("Validation set: {} rows, {} fraud".format(len(X_val), y_val.sum()))
print("Test set: {} rows, {} fraud".format(len(X_test), y_test.sum()))