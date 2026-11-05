import numpy as np
import pandas as pd

# Same random numbers every run, so results are repeatable
np.random.seed(42)
n = 1800

# ---------- Basic details ----------
applicant_id = np.arange(1001, 1001 + n)
gender = np.random.choice(["Male", "Female"], n)
age = np.random.randint(21, 66, n)
employment = np.random.choice(
    ["Employed", "Self-Employed", "Unemployed"], n, p=[0.60, 0.25, 0.15]
)
property_area = np.random.choice(
    ["Urban", "Semi-Urban", "Rural"], n, p=[0.40, 0.35, 0.25]
)

# ---------- Income depends on employment ----------
income = np.where(
    employment == "Employed", np.random.normal(55000, 15000, n),
    np.where(employment == "Self-Employed", np.random.normal(60000, 25000, n),
             np.random.normal(15000, 6000, n)),
)
income = np.clip(income, 8000, 200000).round(-2)

# ---------- Credit score, loans, term, loan amount ----------
credit_score = np.clip(np.random.normal(690, 85, n), 300, 900).round().astype(int)
existing_loans = np.random.choice([0, 1, 2, 3, 4], n, p=[0.35, 0.30, 0.20, 0.10, 0.05])
loan_term = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], n)
loan_amount = np.clip(income * np.random.uniform(3, 20, n), 50000, 5000000).round(-3)

# ---------- Approval rule (realistic relationships) ----------
emi_ratio = (loan_amount / loan_term) / income  # monthly payment vs income

score = (
    (credit_score - 650) / 50          # better credit -> more approval
    + (income - 40000) / 30000         # higher income -> more approval
    - 2.5 * emi_ratio                  # heavy repayment burden -> less approval
    - 0.6 * existing_loans             # more running loans -> less approval
    + np.where(employment == "Employed", 0.8,
               np.where(employment == "Self-Employed", 0.3, -2.0))
    + np.where(property_area == "Urban", 0.3,
               np.where(property_area == "Semi-Urban", 0.1, -0.2))
    + np.random.normal(0, 0.8, n)      # some randomness
    + 0.3
)
probability = 1 / (1 + np.exp(-score))
loan_approved = (np.random.rand(n) < probability).astype(int)

df = pd.DataFrame({
    "ApplicantID": applicant_id,
    "Gender": gender,
    "Age": age,
    "MonthlyIncome": income,
    "LoanAmount": loan_amount,
    "CreditScore": credit_score,
    "EmploymentStatus": employment,
    "ExistingLoans": existing_loans,
    "LoanTerm": loan_term,
    "PropertyArea": property_area,
    "LoanApproved": loan_approved,
})

# ---------- Add a little "dirty" data on purpose ----------
# (so the cleaning step has something real to fix)
for col in ["MonthlyIncome", "CreditScore", "Gender"]:
    idx = np.random.choice(df.index, 20, replace=False)
    df.loc[idx, col] = np.nan
duplicates = df.sample(15, random_state=1)
df = pd.concat([df, duplicates], ignore_index=True)

df.to_csv("loan_data.csv", index=False)

print("Dataset created: loan_data.csv")
print("Rows and columns:", df.shape)
print("Approval rate:", round(df["LoanApproved"].mean() * 100, 1), "%")
print()
print(df.head())