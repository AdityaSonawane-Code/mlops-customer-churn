import os
import pandas as pd


LOG_PATH = "monitoring/predictions.csv"


def load_predictions():
    if not os.path.exists(LOG_PATH):
        print("No prediction log found.")
        return None

    df = pd.read_csv(LOG_PATH)

    if df.empty:
        print("Prediction log is empty.")
        return None

    return df


def generate_monitoring_report():
    df = load_predictions()

    if df is None:
        return

    total_predictions = len(df)

    high_risk = (df["risk_level"] == "HIGH").sum()
    medium_risk = (df["risk_level"] == "MEDIUM").sum()
    low_risk = (df["risk_level"] == "LOW").sum()

    average_probability = df["churn_probability"].mean()

    churn_predictions = (df["prediction"] == "Yes").sum()
    non_churn_predictions = (df["prediction"] == "No").sum()

    print("\n" + "=" * 50)
    print("CUSTOMER CHURN MONITORING REPORT")
    print("=" * 50)

    print(f"Total predictions       : {total_predictions}")
    print(f"Predicted churn         : {churn_predictions}")
    print(f"Predicted non-churn     : {non_churn_predictions}")

    print("\nRisk Distribution")
    print("-" * 30)
    print(f"High risk               : {high_risk}")
    print(f"Medium risk             : {medium_risk}")
    print(f"Low risk                : {low_risk}")

    print("\nModel Statistics")
    print("-" * 30)
    print(f"Average churn probability: {average_probability:.4f}")
    print(f"Average churn percentage : {average_probability * 100:.2f}%")

    print("=" * 50)


if __name__ == "__main__":
    generate_monitoring_report()