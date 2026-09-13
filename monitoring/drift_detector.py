import pandas as pd
from scipy.stats import ks_2samp


TRAIN_PATH = "data/telco_churn.csv"
PRODUCTION_PATH = "monitoring/predictions.csv"

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen"
]


def detect_numeric_drift():
    train_df = pd.read_csv(TRAIN_PATH)
    production_df = pd.read_csv(PRODUCTION_PATH)

    # Clean TotalCharges in both datasets
    train_df["TotalCharges"] = pd.to_numeric(
        train_df["TotalCharges"],
        errors="coerce"
    )

    production_df["TotalCharges"] = pd.to_numeric(
        production_df["TotalCharges"],
        errors="coerce"
    )

    print("\n" + "=" * 75)
    print("CUSTOMER CHURN - DATA DRIFT DETECTION")
    print("=" * 75)

    drift_found = False

    for feature in NUMERIC_FEATURES:

        train_values = train_df[feature].dropna()
        production_values = production_df[feature].dropna()

        if len(production_values) < 5:
            print(
                f"{feature:<20} "
                f"NEED MORE DATA "
                f"({len(production_values)} production samples)"
            )
            continue

        statistic, p_value = ks_2samp(
            train_values,
            production_values
        )

        if p_value < 0.05:
            status = "DRIFT DETECTED"
            drift_found = True
        else:
            status = "NORMAL"

        print(
            f"{feature:<20} "
            f"KS={statistic:.4f}  "
            f"p-value={p_value:.4f}  "
            f"{status}"
        )

    print("-" * 75)

    if drift_found:
        print("Overall Drift Status: DRIFT DETECTED")
    elif len(production_df) < 5:
        print("Overall Drift Status: NEED MORE PRODUCTION DATA")
    else:
        print("Overall Drift Status: NO SIGNIFICANT DRIFT")

    print("=" * 75)


if __name__ == "__main__":
    detect_numeric_drift()