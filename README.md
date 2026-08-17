# UPI Transactions 2024 — Data Analytics & Fraud Detection

Exploratory data analysis of UPI (Unified Payments Interface) transaction data for 2024, followed by a baseline machine learning model to flag potentially fraudulent transactions.

## 📌 Overview

This project analyzes UPI transaction data to uncover patterns in transaction timing, spending behavior, merchant activity, payment failures, and geographic distribution. It concludes with a logistic regression model that predicts the likelihood of fraud based on transaction attributes.

## 📂 Dataset

- **Source:** [UPI Transactions 2024 - Kaggle](https://www.kaggle.com/datasets/skullagos5246/upi-transactions-2024-dataset)

## 🛠️ Tech Stack

- Python 3
- pandas — data manipulation
- seaborn / matplotlib — visualization
- scikit-learn — model building and evaluation
- scipy — sparse matrix operations
- numpy — numerical operations

## 📊 Project Structure

The notebook (`UPI_DATA_ANALYTICS_organized.ipynb`) is organized into the following sections:

1. **Setup** — Import libraries
2. **Load Data** — Read and clean the raw CSV
3. **Preprocess Timestamps** — Convert timestamp column, check data types
4. **Preview Data** — Initial look at the dataset
5. **Transaction Volume by Day & Hour** — Heatmap of activity patterns
6. **Amount Distribution by Age Group** — Spending behavior across age groups
7. **Amount Distribution by Merchant Category** — Spending behavior across categories
8. **Total Spend by Merchant Category** — Ranked merchant category spend
9. **Total Amount by Transaction Type** — Value moved by transaction type
10. **In-Bank vs Cross-Bank Transactions** — Comparison of transaction routing
11. **Payment Failures by Device Type** — Failure trends by device
12. **Payment Failures by Network Type** — Failure trends by network
13. **Transaction Amount by State** — Geographic breakdown of spend
14. **Fraud Detection Model** — Logistic regression with threshold tuning
15. **Summary** — Key takeaways from the analysis

## 🔍 Key Findings

- Transaction activity peaks during specific day/hour windows.
- Spending patterns vary noticeably by age group and merchant category.
- A small number of merchant categories and transaction types account for most of the total transaction value.
- Cross-bank transactions outnumber in-bank transactions.
- Payment failures are concentrated in specific device and network types.
- Certain states contribute disproportionately to total transaction value.

## 🤖 Fraud Detection Model

- **Model:** Logistic Regression
- **Features:** transaction amount, sender age group, hour of day, day of week
- **Approach:** One-hot encoding for categorical features, stratified train/test split, threshold tuning using precision-recall-F1 curves instead of a fixed 0.5 cutoff
- **Output:** Confusion matrix and precision/recall/F1 scores at both default and optimal thresholds

## ⚠️ Limitations

- Fraud model does not account for class imbalance in `fraud_flag`, which may inflate accuracy while hurting recall on actual fraud cases.
- Only a single model (logistic regression) is tested.
