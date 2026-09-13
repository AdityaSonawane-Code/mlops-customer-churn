import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/telco_churn.csv"
MODEL_DIR = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# Custom transformer
# ============================================================

class TotalChargesConverter(BaseEstimator, TransformerMixin):
    """
    Converts TotalCharges from text to numeric.
    Invalid or blank values become NaN.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(
                X["TotalCharges"],
                errors="coerce"
            )

        return X


# ============================================================
# Load and clean dataset
# ============================================================

def load_data():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Original dataset shape: {df.shape}")

    # Remove unnecessary whitespace from column names
    df.columns = df.columns.str.strip()

    # Remove whitespace from string values
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Convert target
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # Remove customer ID because it is only an identifier
    df = df.drop(columns=["customerID"])

    # Remove rows where target is missing
    df = df.dropna(subset=["Churn"])

    print(f"Cleaned dataset shape: {df.shape}")

    return df


# ============================================================
# Create preprocessing pipeline
# ============================================================

def create_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumeric features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# Main preprocessing
# ============================================================

def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load dataset
    df = load_data()

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nTarget percentage:")
    print(y.value_counts(normalize=True).mul(100).round(2))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # Create preprocessing pipeline
    preprocessor = create_preprocessor(X_train)

    # Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using the fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    print("\nProcessed train shape:", X_train_processed.shape)
    print("Processed test shape:", X_test_processed.shape)

    # Save preprocessing pipeline
    preprocessor_path = os.path.join(
        MODEL_DIR,
        "preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    print(f"\nPreprocessor saved to: {preprocessor_path}")

    # Save processed datasets
    train_data = pd.DataFrame(
        X_train_processed
    )

    train_data["Churn"] = y_train.reset_index(drop=True)

    test_data = pd.DataFrame(
        X_test_processed
    )

    test_data["Churn"] = y_test.reset_index(drop=True)

    train_data.to_csv(
        "data/train_processed.csv",
        index=False
    )

    test_data.to_csv(
        "data/test_processed.csv",
        index=False
    )

    print("Processed training data saved.")
    print("Processed testing data saved.")

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()