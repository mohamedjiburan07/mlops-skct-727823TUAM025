# 📊 Sales Forecasting using MLOps (MLflow + Azure Pipeline)

## 👤 Student Details

* **Name:** MOHAMED JIBURAN S
* **Roll Number:** 727823TUAM025

---

## 📌 Project Title

Sales Forecasting using Synthetic Dataset with MLflow Experiment Tracking and Azure ML Pipeline

---

## 📁 Project Structure

```
mlops-skct-727823TUAM025/
│
├── data/
│   └── sales.csv
│
├── src/
│   ├── data_prep.py
│   ├── train_pipeline.py
│   ├── evaluate.py
│
├── mlflow_experiment.py
├── pipeline_727823TUAM025.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* MLflow
* Azure ML Pipeline (YAML)

---

## 📊 Dataset

A synthetic dataset is generated with the following features:

* **Day** → Sequential time feature
* **Promo** → Promotion indicator (0 or 1)
* **Price** → Product price
* **Sales** → Target variable

---

## 🔄 Workflow

### 1. Data Preparation

* Synthetic dataset generation using NumPy
* Saved as `data/sales.csv`

### 2. Training Pipeline

* Train-test split (80:20)
* Data saved as:

  * X_train.csv
  * X_test.csv
  * y_train.csv
  * y_test.csv

### 3. Model Training (MLflow)

* Algorithms used:

  * Linear Regression
  * Random Forest Regressor
* 12+ runs executed
* Metrics logged:

  * R² Score
  * RMSE
  * MAE

### 4. Evaluation

* Model evaluated on test data
* Metrics calculated using sklearn

### 5. Pipeline

* Implemented using YAML with 3 stages:

  * Data Preparation
  * Training
  * Evaluation

---

## 📈 Results

| Metric   | Value  |
| -------- | ------ |
| R² Score | ~0.69  |
| RMSE     | ~17.78 |
| MAE      | ~13.30 |

---

## ⚠️ Challenges Faced

Initially, the model produced:

* R² = 1.0
* RMSE = 0.0

This was due to **data leakage**, caused by using the same dataset for both training and testing.

### ✅ Solution

* Implemented proper `train_test_split`
* Ensured predictions were made on unseen data

---

## 🚀 How to Run

```bash
pip install pandas numpy scikit-learn mlflow

python src/data_prep.py
python src/train_pipeline.py
python mlflow_experiment.py
mlflow ui
python src/evaluate.py
```

---

## 📊 MLflow Tracking

* **Experiment Name:**
  `SKCT_727823TUAM025_SalesForecasting`

* Tracks:

  * Metrics
  * Parameters
  * Model artifacts

---

## ✅ Conclusion

This project demonstrates:

* End-to-end machine learning pipeline
* Experiment tracking using MLflow
* Model evaluation and comparison
* Deployment-ready pipeline structure
