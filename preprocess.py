import pandas as pd

# ---------- 1. Load the dataset ----------
df = pd.read_csv("loan_data.csv")
print("Original shape:", df.shape)

# ---------- 2. Check problems ----------
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ---------- 3. Remove duplicates ----------
df = df.drop_duplicates()

# ---------- 4. Handle missing values ----------
# Numbers -> fill with median (middle value, not affected by extremes)
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["CreditScore"] = df["CreditScore"].fillna(df["CreditScore"].median())
# Text -> fill with most common value (mode)
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["CreditScore"] = df["CreditScore"].astype(int)

# ---------- 5. Data validation ----------
# Keep only rows where values make sense
before = len(df)
df = df[
    df["Age"].between(18, 70)
    & df["CreditScore"].between(300, 900)
    & (df["MonthlyIncome"] > 0)
    & (df["LoanAmount"] > 0)
    & (df["LoanTerm"] > 0)
    & (df["ExistingLoans"] >= 0)
    & df["EmploymentStatus"].isin(["Employed", "Self-Employed", "Unemployed"])
    & df["PropertyArea"].isin(["Urban", "Semi-Urban", "Rural"])
    & df["Gender"].isin(["Male", "Female"])
    & df["LoanApproved"].isin([0, 1])
]
print("\nInvalid rows removed:", before - len(df))

# ---------- 6. Final check ----------
print("Missing values left:", int(df.isnull().sum().sum()))
print("Duplicates left:", int(df.duplicated().sum()))
print("Cleaned shape:", df.shape)

# ---------- 7. Save cleaned data ----------
df.to_csv("loan_data_cleaned.csv", index=False)
print("\nSaved: loan_data_cleaned.csv")

# ---------- 8. Encoding preview ----------
# Machine learning needs numbers, so text columns are converted to 0/1 columns.
# (The final encoding + scaling happens inside train_model.py)
encoded = pd.get_dummies(
    df.drop(columns=["ApplicantID"]),
    columns=["Gender", "EmploymentStatus", "PropertyArea"],
    dtype=int,
)
print("\nColumns after encoding:")
print(list(encoded.columns))