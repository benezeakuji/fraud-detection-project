import time
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Fraud Scoring API")

# --- Load the trained model and rebuild the fitted scaler ---
model = joblib.load("models/xgb_cost.joblib")

# Rebuild the exact same scaler fit as before (same data, same random_state
# isn't even needed here since StandardScaler's fit doesn't depend on the
# train/test split — it's fit on the full cleaned dataset before splitting).
df = pd.read_csv("data/creditcard.csv").drop_duplicates()
scaler = StandardScaler()
scaler.fit(df[["Time", "Amount"]])

explainer = shap.TreeExplainer(model)

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class Transaction(BaseModel):
    time: float = Field(..., description="Seconds since the first transaction")
    amount: float = Field(..., ge=0, description="Transaction amount")
    v: list[float] = Field(..., min_length=28, max_length=28, description="V1 through V28")


class ScoreResponse(BaseModel):
    risk_score: float
    flagged: bool
    top_contributing_features: list[dict]
    response_time_ms: float


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
async def score_transaction(txn: Transaction):
    t0 = time.time()

    row = {"Time": txn.time, "Amount": txn.amount}
    for i, v in enumerate(txn.v, start=1):
        row[f"V{i}"] = v
    X = pd.DataFrame([row])[FEATURE_ORDER]
    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

    proba = float(model.predict_proba(X)[0, 1])
    flagged = proba >= 0.5

    sv = np.array(explainer.shap_values(X))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    contrib = sorted(
        zip(FEATURE_ORDER, sv.flatten().tolist()),
        key=lambda pair: abs(pair[1]), reverse=True
    )[:5]
    top_features = [{"feature": f, "shap_contribution": round(v, 4)} for f, v in contrib]

    elapsed_ms = (time.time() - t0) * 1000
    return ScoreResponse(
        risk_score=round(proba, 4),
        flagged=flagged,
        top_contributing_features=top_features,
        response_time_ms=round(elapsed_ms, 2),
    )