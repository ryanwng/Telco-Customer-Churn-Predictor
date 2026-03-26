# Telco Customer Churn Prediction

This project implements a machine learning pipeline to predict customer churn for a telecommunications company. By identifying at-risk customers, businesses can proactively develop retention strategies to reduce turnover and increase Customer Lifetime Value (CLV).

## 🚀 Project Overview
The goal of this analysis is to build a classification model that predicts whether a customer will leave the service (Churn) based on their demographic information, account details, and usage patterns.

### Key Business Questions:
* Which services or contract types are associated with higher churn?
* Can we accurately identify customers likely to leave before they cancel?
* What are the primary financial drivers of churn (Monthly vs. Total Charges)?

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`

## 📊 Dataset
The project uses the **WA_Fn-UseC_-Telco-Customer-Churn** dataset (commonly found on Kaggle). 
* **Size:** 7,043 rows, 21 columns.
* **Target Variable:** `Churn` (Yes/No).

## 📉 Workflow & Methodology

### 1. Data Cleaning & Preprocessing
* **Type Conversion:** Converted `TotalCharges` to numeric, handling missing values introduced by empty strings.
* **Handling Missingness:** Dropped a negligible number of rows (11) with null values in `TotalCharges`.
* **Feature Engineering:** * Dropped unique identifiers (`customerID`) that provide no predictive power.
    * Binary encoding for the target variable (`Churn`).
    * One-Hot Encoding for categorical features (e.g., Contract type, Payment method).
* **Feature Scaling:** Applied `MinMaxScaler` to `tenure`, `MonthlyCharges`, and `TotalCharges` to ensure features are on a comparable scale (0 to 1).

### 2. Modeling
* **Algorithm:** Logistic Regression.
* **Data Split:** 80% Training / 20% Testing.
* **Random State:** 42 (for reproducibility).

### 3. Evaluation
The model's performance was evaluated using:
* **Accuracy Score:** The percentage of total correct predictions.
* **Confusion Matrix:** To visualize True Positives, True Negatives, False Positives, and False Negatives.
* **Classification Report:** Precision, Recall, and F1-score to assess performance specifically for the "Churn" class.

## 🏆 Results
* **Overall Accuracy:** ~78.75%
* **Analysis:** The model performs strongly in identifying customers who stay (high precision/recall for class 0). Further tuning (e.g., handling class imbalance or testing Random Forests) could be explored to improve the recall of customers who actually churn.

## ⚙️ How to Run
1. Clone this repository.
2. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
