# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using a highly imbalanced dataset.

## Overview

This project builds and compares multiple classification models to identify fraudulent transactions from legitimate ones. The dataset is extremely imbalanced (~0.17% fraud), making it a strong exercise in handling class imbalance, choosing the right evaluation metrics, and comparing model performance beyond simple accuracy.

## Dataset

- **Source:** [Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) on Kaggle
- **Rows:** ~284,807 transactions
- **Features:**
  - `Time` — seconds elapsed since the first transaction in the dataset
  - `Amount` — transaction amount
  - `V1`–`V28` — anonymized features from a PCA transformation (original features withheld for confidentiality)
  - `Class` — target variable (`1` = fraud, `0` = legitimate)

## Project Workflow

1. **Data loading & inspection** — check shape, types, and missing values
2. **Exploratory Data Analysis (EDA)**
   - Class imbalance visualization
   - Transaction amount distribution by class
   - Correlation of `V1`–`V28` with `Class`
3. **Preprocessing**
   - Scale `Amount` and `Time` (the only non-PCA features)
   - Train/test split with stratification
4. **Handling class imbalance**
   - Compare `class_weight='balanced'` vs SMOTE oversampling
5. **Model comparison**
   - Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, XGBoost
   - Evaluated via cross-validation
6. **Evaluation metrics**
   - Precision, Recall, F1 Score
   - Precision-Recall AUC (more informative than accuracy or ROC-AUC given the extreme imbalance)
7. **Hyperparameter tuning**
   - GridSearchCV / RandomizedSearchCV on top-performing models
8. **Final model evaluation**
   - Confusion matrix, classification report, feature importance

## Why Not Just Use Accuracy?

Because fraud makes up only ~0.17% of transactions, a model that predicts "legitimate" for every transaction would already score ~99.8% accuracy while catching zero fraud. This project prioritizes **Recall** (catching actual fraud) and **Precision** (minimizing false alarms), balanced through the **F1 Score** and **Precision-Recall curve**.

## Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
seaborn
matplotlib
```

Install with:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost seaborn matplotlib
```

## How to Run

1. Download `creditcard.csv` from the Kaggle link above
2. Place it in the project directory
3. Open and run the notebook cells in order (data loading → EDA → preprocessing → modeling → evaluation)

## Results

_To be filled in after model comparison and tuning are complete — e.g. best model, its F1/Precision/Recall scores, and key features driving fraud predictions._

## Notes

- `V1`–`V28` are PCA components and are not individually interpretable — feature importance from these should be discussed in terms of "component contribution," not real-world meaning.
- Always use **stratified** train/test splits given the severity of the class imbalance.