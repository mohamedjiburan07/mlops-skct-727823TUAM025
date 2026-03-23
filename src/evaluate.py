# src/evaluate.py

# Roll No: 727823TUAM025
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

print("Roll No: 727823TUAM025")
print("Timestamp:", datetime.now())

# Load split data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

# Train model again (same logic)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train.values.ravel())

preds = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print("R2:", r2)
print("RMSE:", rmse)
print("MAE:", mae)