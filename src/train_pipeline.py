# src/train_pipeline.py

# Roll No: 727823TUAM025
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

print("Roll No: 727823TUAM025")
print("Timestamp:", datetime.now())

df = pd.read_csv("data/sales.csv")

X = df.drop("Sales", axis=1)
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Train-test split done")