# src/data_prep.py

# Roll No: 727823TUAM025
import pandas as pd
import numpy as np
from datetime import datetime

print("Roll No: 727823TUAM025")
print("Timestamp:", datetime.now())

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "Day": np.arange(n),
    "Promo": np.random.randint(0, 2, n),
    "Price": np.random.uniform(10, 50, n),
})

df["Sales"] = (
    200
    + 15 * df["Promo"]
    - 2.5 * df["Price"]
    + np.random.normal(0, 15, n)
)

df.to_csv("data/sales.csv", index=False)
print("Dataset created")