# Loan Approval Prediction System

A machine learning project that predicts whether a loan application should be **Approved** or **Rejected** using applicant details and financial information. Built as part of the Infobharat Interns Machine Learning Internship Program.

## Project Overview
The project covers the full ML workflow: synthetic data creation, cleaning, exploratory data analysis, visualization, model training and comparison, evaluation, and a console-based prediction system with input validation.

## Dataset
- Synthetic dataset of **1,800 applicants** (plus 15 deliberate duplicate rows and 60 missing values to practise cleaning).
- Columns: ApplicantID, Gender, Age, MonthlyIncome, LoanAmount, CreditScore, EmploymentStatus, ExistingLoans, LoanTerm, PropertyArea, LoanApproved (target: 0 = Rejected, 1 = Approved).
- Approval is generated from realistic rules: higher credit score and income increase approval chances; a heavy repayment burden, more existing loans and unemployment reduce them.

## Libraries and Tools
Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib, VS Code

## Project Files
| File | Purpose |
|---|---|
| `generate_data.py` | Creates the synthetic dataset (`loan_data.csv`) |
| `preprocess.py` | Removes duplicates, fills missing values, validates data, saves `loan_data_cleaned.csv` |
| `eda.py` | Prints insights and saves 4 charts |
| `train_model.py` | Trains and compares 3 models, saves the best as `loan_model.joblib` |
| `predict_cli.py` | Console-based prediction system with error handling |

## Steps to Run
```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
python generate_data.py
python preprocess.py
python eda.py
python train_model.py
python predict_cli.py
```

## Data Preprocessing
- Removed 15 duplicate rows
- Filled missing numeric values with the median and missing Gender with the most common value
- Validated ranges (age, credit score, income, loan amount, categories)
- Scaled numeric features (StandardScaler) and one-hot encoded text columns
- Gender is **not** used for prediction, to keep the model fair

## Key Insights
- Credit score has the strongest effect on approval: about **20%** approval for scores 300-600, rising to about **88%** for scores 801-900.
- **Unemployed** applicants are approved only about **14%** of the time, versus about 63-66% for employed and self-employed applicants.
- Approved applicants earn more on average (about Rs 57,000/month) than rejected ones (about Rs 41,700/month).
- Top predictive features: Credit Score, Monthly Income, Loan Amount.

## Visualizations
Bar chart (approvals by employment), correlation heatmap, scatter plot (income vs loan amount), pie chart (approved vs rejected), confusion matrix, and feature importance. Saved as `chart1` to `chart6` PNG files.

## Model Performance (20% test set, 360 applicants)
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** (best) | 0.794 | 0.803 | 0.842 | 0.822 |
| Decision Tree | 0.761 | 0.772 | 0.818 | 0.794 |
| Random Forest | 0.794 | 0.812 | 0.828 | 0.820 |

Confusion matrix (best model): 115 rejected correctly, 171 approved correctly, 42 wrongly approved, 32 wrongly rejected.

## Console Prediction System
Asks for age, monthly income, loan amount, credit score, employment status, existing loans, loan term and property area, then prints **APPROVED** or **REJECTED** with the approval probability. Invalid, empty or out-of-range input is rejected with a clear message and the question is asked again, so the program never crashes.

## Note
The data is synthetic and for learning purposes only. It must not be used for real lending decisions.