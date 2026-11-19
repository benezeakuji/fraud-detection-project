# Fraud Detection Prototype — MIT Professional Master's Project

A machine learning-based system for detecting fraudulent electronic payment
transactions, built as the practical component of a Master of Information
Technology (MIT) Professional Master's Project at Miva Open University.

## What this is

Six trained model variants (Logistic Regression, Random Forest, XGBoost —
each under SMOTE and cost-sensitive class-imbalance handling), a SHAP-based
explainability module, a FastAPI fraud-scoring service, and a Streamlit
analyst dashboard. Validated on the Kaggle/ULB European credit card dataset
and cross-checked on the real PaySim mobile-money dataset.

## Headline results

Best model: XGBoost (cost-sensitive) — Precision 0.917, Recall 0.775,
F1-score 0.840, AUC-ROC 0.963, AUC-PR 0.802, against a rule-based baseline
of F1 0.004. Full results and discussion are in the project report.

## Project structure

```
verify_data.py        # validates the raw dataset
preprocess.py          # dedup, scaling, stratified 70/15/15 split
imbalance.py           # SMOTE + cost-sensitive weight computation
train_models.py        # trains and saves all 6 model variants
evaluate_models.py     # scores all models on the held-out test set
cross_validate.py      # 5-fold stability check on SMOTE-balanced data
shap_analysis.py       # global + local SHAP explainability
paysim_validate.py     # secondary validation on the real PaySim dataset
api/main.py            # FastAPI fraud-scoring service
dashboard/app.py       # Streamlit analyst dashboard
```

## Setup

1. Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate

2. Install dependencies:

pip install -r requirements.txt

3. Download the dataset (not included in this repo — see `.gitignore`):
   - Primary dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud), save as `data/creditcard.csv`
   - Secondary dataset: [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1), save as `data/paysim.csv`

## Running the pipeline

Run in order: `verify_data.py` → `preprocess.py` → `imbalance.py` →
`train_models.py` → `evaluate_models.py` → `cross_validate.py` →
`shap_analysis.py` → `paysim_validate.py`.

To run the API: `uvicorn api.main:app --reload`
To run the dashboard: `streamlit run dashboard/app.py`

## Author

Benjamin Ezeakuji — MIT, Miva Open University