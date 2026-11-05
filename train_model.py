import matplotlib
matplotlib.use("Agg")  # save charts as image files (no pop-up windows)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ---------- 1. Load cleaned data ----------
df = pd.read_csv("loan_data_cleaned.csv")
X = df.drop(columns=["ApplicantID", "LoanApproved", "Gender"])  # inputs
y = df["LoanApproved"]                                  # answer to predict

# ---------- 2. Split: 80% train, 20% test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training rows:", len(X_train), "| Testing rows:", len(X_test))

# ---------- 3. Preprocessing (scale numbers, encode text) ----------
numeric_cols = ["Age", "MonthlyIncome", "LoanAmount", "CreditScore",
                "ExistingLoans", "LoanTerm"]
text_cols = ["EmploymentStatus", "PropertyArea"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), text_cols),
])

# ---------- 4. Models to compare ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

# ---------- 5. Train and evaluate each model ----------
results = []
trained = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    trained[name] = (pipe, pred)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
    })

results_df = pd.DataFrame(results).round(3)
print("\n===== Model Comparison =====")
print(results_df.to_string(index=False))

# ---------- 6. Pick the best model (highest F1-Score) ----------
best_name = results_df.sort_values("F1-Score", ascending=False).iloc[0]["Model"]
best_pipe, best_pred = trained[best_name]
print("\nBest model:", best_name)

print("\nConfusion Matrix (rows = actual, columns = predicted):")
cm = confusion_matrix(y_test, best_pred)
print(cm)
print("\nDetailed report:")
print(classification_report(y_test, best_pred, target_names=["Rejected", "Approved"]))

# Confusion matrix picture
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"])
plt.title("Confusion Matrix - " + best_name)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("chart5_confusion_matrix.png", dpi=150)
plt.close()

# ---------- 7. Feature importance (Random Forest) ----------
rf_pipe, _ = trained["Random Forest"]
names = rf_pipe.named_steps["prep"].get_feature_names_out()
importance = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=names)
importance = importance.sort_values().tail(10)

plt.figure(figsize=(8, 5))
importance.plot(kind="barh", color="#4a7fb5")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("chart6_feature_importance.png", dpi=150)
plt.close()
print("\nTop 5 important features:")
print(importance.sort_values(ascending=False).head(5).round(3))

# ---------- 8. Save the best model for the prediction app ----------
joblib.dump(best_pipe, "loan_model.joblib")
print("\nSaved: loan_model.joblib")