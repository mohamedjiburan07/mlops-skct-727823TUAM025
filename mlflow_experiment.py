# mlflow_experiment.py

import mlflow
import pandas as pd
import numpy as np
import time
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# A1
mlflow.set_experiment("SKCT_727823TUAM025_SalesForecasting")

df = pd.read_csv("data/sales.csv")

X = df.drop("Sales", axis=1)
y = df["Sales"]

best_r2 = -1

for seed in range(12):  # A3

    for model_name in ["lr", "rf"]:  # A4 (2 algorithms)

        with mlflow.start_run():

            # A2
            mlflow.set_tags({
                "student_name": "MOHAMED JIBURAN S",
                "roll_number": "727823TUAM025",
                "dataset": "synthetic_sales"
            })

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=seed
            )

            if model_name == "lr":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(random_state=seed)

            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            preds = model.predict(X_test)

            # A5
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)

            mlflow.log_metric("r2", r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            # A6
            mlflow.log_metric("training_time_seconds", train_time)
            mlflow.log_param("random_seed", seed)

            size_mb = os.path.getsize(__file__) / (1024 * 1024)
            mlflow.log_metric("model_size_mb", size_mb)

            # Save preds for evaluation
            pd.DataFrame(preds).to_csv("data/preds.csv", index=False)

            # A7
            if r2 > best_r2:
                best_r2 = r2
                mlflow.sklearn.log_model(model, "best_model")

print("Best R2:", best_r2)