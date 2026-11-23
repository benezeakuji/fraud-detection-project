---
title: Fraud Scoring API
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Fraud Scoring API

FastAPI service for the MIT Professional Master's Project *"Design and
Implementation of a Machine Learning-Based Fraud Detection System for
Electronic Transactions in Nigeria"* (Benjamin Ezeakuji, Miva Open
University).

Loads the trained XGBoost (cost-sensitive) model and its fitted scaler,
and exposes:

- `GET /health` — liveness check
- `POST /score` — score a transaction and return its risk score, flagged
  status, and top SHAP-contributing features
- `GET /docs` — interactive Swagger UI for testing requests live

Source: https://github.com/benezeakuji/fraud-detection-project
