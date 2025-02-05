# feature_importance.py
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load your model
model = joblib.load("rf_model_sp500.pkl")
# Define features in the same order as used in training
features = ["RSI", "SMA50", "SMA20", "MACD", "MACD_signal"]
importance = model.feature_importances_

plt.figure(figsize=(8, 6))
plt.bar(features, importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()