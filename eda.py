import matplotlib
matplotlib.use("Agg")  # save charts as image files (no pop-up windows)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")
df = pd.read_csv("loan_data_cleaned.csv")

# =====================================================
# PART 1: EDA (find patterns with numbers)
# =====================================================
print("Overall approval rate: %.1f%%" % (df["LoanApproved"].mean() * 100))

print("\n--- Average values: Approved (1) vs Rejected (0) ---")
print(df.groupby("LoanApproved")[["MonthlyIncome", "CreditScore", "LoanAmount", "ExistingLoans"]]
      .mean().round(0))

print("\n--- Approval rate (%) by employment status ---")
print((df.groupby("EmploymentStatus")["LoanApproved"].mean() * 100).round(1))

print("\n--- Approval rate (%) by property area ---")
print((df.groupby("PropertyArea")["LoanApproved"].mean() * 100).round(1))

print("\n--- Approval rate (%) by credit score range ---")
bands = pd.cut(df["CreditScore"], [299, 600, 700, 800, 900],
               labels=["300-600", "601-700", "701-800", "801-900"])
print((df.groupby(bands, observed=True)["LoanApproved"].mean() * 100).round(1))

# =====================================================
# PART 2: VISUALIZATIONS (saved as PNG files)
# =====================================================

# Chart 1: Bar chart - approvals by employment status
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df, x="EmploymentStatus", hue="LoanApproved",
                   palette={0: "#e07b7b", 1: "#5aa66b"})
ax.set_title("Loan Approvals by Employment Status")
ax.set_xlabel("Employment Status")
ax.set_ylabel("Number of Applicants")
ax.legend(title="Result", labels=["Rejected", "Approved"])
plt.tight_layout()
plt.savefig("chart1_approval_by_employment.png", dpi=150)
plt.close()

# Chart 2: Correlation heatmap
plt.figure(figsize=(8, 6))
numeric = df.drop(columns=["ApplicantID"]).select_dtypes("number")
sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig("chart2_correlation_heatmap.png", dpi=150)
plt.close()

# Chart 3: Scatter plot - income vs loan amount
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="MonthlyIncome", y="LoanAmount", hue="LoanApproved",
                palette={0: "#e07b7b", 1: "#5aa66b"}, alpha=0.6)
plt.title("Monthly Income vs Loan Amount")
plt.xlabel("Monthly Income")
plt.ylabel("Loan Amount")
plt.legend(title="Result", labels=["Rejected", "Approved"])
plt.tight_layout()
plt.savefig("chart3_income_vs_loan.png", dpi=150)
plt.close()

# Chart 4: Pie chart - approved vs rejected
plt.figure(figsize=(5.5, 5.5))
counts = df["LoanApproved"].value_counts().sort_index()
plt.pie(counts, labels=["Rejected", "Approved"], autopct="%1.1f%%",
        colors=["#e07b7b", "#5aa66b"], startangle=90)
plt.title("Approved vs Rejected Applications")
plt.tight_layout()
plt.savefig("chart4_approval_pie.png", dpi=150)
plt.close()

print("\nSaved 4 charts: chart1 ... chart4 (.png files in your folder)")