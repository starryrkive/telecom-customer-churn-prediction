# Telecom Customer Churn Prediction Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Framework](https://img.shields.io/badge/Data_Science-Pipeline-green.svg)]()

A machine learning classification pipeline designed to proactively identify subscription accounts at high risk of customer attrition (churn). This project transitions from a baseline statistical implementation to an optimized, class-balanced framework engineered specifically to prevent revenue leakage from undetected flight risks.

---

## Technical Summary and Performance

* **The Problem:** Subscription-based business models experience heavy financial deficits from customer attrition. Identifying at-risk profiles early allows customer success teams to offer strategic loyalty incentives.
* **The Solution:** Developed a Logistic Regression classifier capable of predicting consumer churn with 82.04% baseline accuracy.
* **The Optimization:** Real-world data is highly imbalanced (fewer users leave than stay), causing standard models to suffer from low recall. By injecting specialized feature engineering and tuning model class weights, undetected churn events were slashed by approximately 60%, dropping missed opportunities from 150 down to just 61 accounts.

---

## Tech Stack and Environment

* **Language:** Python 3.9+
* **Environment:** VS Code and Jupyter Notebook Kernels
* **Core Libraries:** pandas, scikit-learn, matplotlib, seaborn, numpy

---

## Core Features and Pipeline Engineering

### 1. Data Cleansing and Data Type Correction
* Isolated structural noise by removing non-predictive features (customerID).
* Standardized database entry errors within billing structures (TotalCharges) using a robust median imputation strategy.
* Remapped the target objective (Churn) into categorical binary vector space $y \in \{0, 1\}$.

### 2. High-Impact Feature Engineering
Introduced a high-variance derivative ratio called ChargeRatio:
$$\text{ChargeRatio} = \frac{\text{MonthlyCharges}}{\text{TotalCharges} + 1}$$
This isolates billing velocity, instantly surfacing early-tenure accounts facing disproportionately steep operational costs—which the model flagged as the single highest indicator of churn probability.

### 3. Categorical Transformation
Utilized One-Hot Encoding via `pd.get_dummies(drop_first=True)` to convert multi-class qualitative indices (e.g., contract types, internet frameworks, and payment platforms) into high-fidelity numerical arrays.

---

## Model Scorecard Comparison

### Baseline Model (Standard Optimization)
High accuracy, but highly vulnerable to missing true customer flight patterns.
* **Overall Accuracy:** 82.04%
* **Churn Class Recall:** 0.60 (Missed 150 real churn cases)

### Optimized Model (class_weight='balanced' + Engineered Features)
Strategic adjustments penalize the algorithm exponentially harder for failing to intercept true churn patterns.
* **Overall Accuracy:** ~76.5%
* **Churn Structural Performance:** Missed churn events dropped from 150 to 61.
* **The Precision-Recall Trade-off:** Setting a highly sensitive predictive alarm captures vastly more flight risks (High Recall), but introduces minor increases in false alerts for loyal clients (Lower Precision). In subscription business dynamics, a false alert is vastly cheaper than a lost customer.

---

## Top Business Drivers Explored
Interrogating the model's coefficients revealed the core operational pain points and retention anchors:

* **Major Risk Factors (Drives Churn):** Steep ChargeRatio margins and subscription to InternetService_Fiber optic infrastructures.
* **Safe Anchors (Drives Retention):** Commitment to Contract_Two year or Contract_One year agreements, alongside active user utilization of dedicated TechSupport_Yes lines.

---

## Installation and Reproduction

1. **Clone the repository:**
```bash
   git clone [https://github.com/starryrkive/telecom-customer-churn-prediction.git](https://github.com/starryrkive/telecom-customer-churn-prediction.git)
   cd telecom-customer-churn-prediction
   python -m venv ml_env
   source ml_env/bin/activate
   pip install pandas scikit-learn matplotlib seaborn notebook ipykernel
