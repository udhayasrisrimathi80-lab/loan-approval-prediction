import joblib
import pandas as pd

# ---------- Load the trained model ----------
try:
    model = joblib.load("loan_model.joblib")
except FileNotFoundError:
    print("Model file not found. Please run train_model.py first.")
    raise SystemExit


def ask_number(prompt, low, high, is_int=True):
    """Keep asking until the user types a valid number in the allowed range."""
    while True:
        text = input(prompt).strip()
        if text == "":
            print("  Input cannot be empty. Please try again.")
            continue
        try:
            value = int(text) if is_int else float(text)
        except ValueError:
            print("  Please enter a valid number (digits only).")
            continue
        if not (low <= value <= high):
            print(f"  Value must be between {low} and {high}.")
            continue
        return value


def ask_choice(prompt, options):
    """Show a numbered menu and return the chosen option."""
    print(prompt)
    for i, option in enumerate(options, start=1):
        print(f"   {i}. {option}")
    while True:
        text = input("  Enter option number: ").strip()
        if text.isdigit() and 1 <= int(text) <= len(options):
            return options[int(text) - 1]
        print(f"  Please type a number from 1 to {len(options)}.")


def get_applicant():
    """Collect and validate all inputs."""
    print("\nEnter applicant details")
    print("-" * 30)
    age = ask_number("Age (18-70): ", 18, 70)
    income = ask_number("Monthly income in Rs (1000-1000000): ", 1000, 1_000_000, is_int=False)
    loan = ask_number("Loan amount in Rs (10000-10000000): ", 10_000, 10_000_000, is_int=False)
    credit = ask_number("Credit score (300-900): ", 300, 900)
    employment = ask_choice("Employment status:", ["Employed", "Self-Employed", "Unemployed"])
    existing = ask_number("Number of existing loans (0-10): ", 0, 10)
    term = ask_number("Loan term in months (6-360): ", 6, 360)
    area = ask_choice("Property area:", ["Urban", "Semi-Urban", "Rural"])

    # Column names must match the training data
    return pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": income,
        "LoanAmount": loan,
        "CreditScore": credit,
        "EmploymentStatus": employment,
        "ExistingLoans": existing,
        "LoanTerm": term,
        "PropertyArea": area,
    }])


def main():
    print("=" * 40)
    print("   LOAN APPROVAL PREDICTION SYSTEM")
    print("=" * 40)
    while True:
        try:
            applicant = get_applicant()
            result = model.predict(applicant)[0]
            chance = model.predict_proba(applicant)[0][1] * 100
            print("\n" + "=" * 40)
            print("  Prediction:", "APPROVED" if result == 1 else "REJECTED")
            print(f"  Approval probability: {chance:.1f}%")
            print("=" * 40)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break
        except Exception as error:  # last safety net so the program never crashes
            print("Something went wrong:", error)

        try:
            again = input("\nCheck another applicant? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            again = "n"
        if again != "y":
            print("Thank you! Goodbye!")
            break


if __name__ == "__main__":
    main()