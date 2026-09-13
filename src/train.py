import os
import json
import joblib
import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = "data/train_processed.csv"
TEST_PATH = "data/test_processed.csv"

MODEL_DIR = "models"

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
COMPARISON_PATH = os.path.join(MODEL_DIR, "model_comparison.csv")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# ============================================================
# MLflow
# ============================================================

mlflow.set_experiment("customer-churn-prediction")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading processed datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")

    # ========================================================
    # MODELS
    # ========================================================

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    }

    results = []

    best_model = None
    best_model_name = None
    best_roc_auc = -1

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for model_name, model in models.items():

        print()
        print("=" * 60)
        print(f"Training: {model_name}")
        print("=" * 60)

        metrics = evaluate_model(
            model.fit(X_train, y_train),
            X_test,
            y_test
        )

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1_score']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

        # ====================================================
        # MLflow
        # ====================================================

        with mlflow.start_run(run_name=model_name):

            mlflow.log_param("model_name", model_name)

            for parameter, value in model.get_params().items():

                try:
                    mlflow.log_param(parameter, value)
                except Exception:
                    pass

            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("f1_score", metrics["f1_score"])
            mlflow.log_metric("roc_auc", metrics["roc_auc"])

            # Logistic Regression / Random Forest can use
            # MLflow sklearn flavor normally.
            #
            # XGBoost is logged as a generic artifact because
            # newer MLflow versions may reject XGBoost through
            # the skops security validation.

            if model_name != "XGBoost":

                mlflow.sklearn.log_model(
                    model,
                    name="model"
                )

            else:

                xgb_path = os.path.join(
                    MODEL_DIR,
                    "xgboost_model.json"
                )

                model.save_model(xgb_path)

                mlflow.log_artifact(
                    xgb_path,
                    artifact_path="xgboost_model"
                )

        # ====================================================
        # SAVE RESULTS
        # ====================================================

        result = {
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"]
        }

        results.append(result)

        # ====================================================
        # BEST MODEL
        # ====================================================

        if metrics["roc_auc"] > best_roc_auc:

            best_roc_auc = metrics["roc_auc"]
            best_model = model
            best_model_name = model_name

    # ========================================================
    # SAVE MODEL COMPARISON
    # ========================================================

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df.sort_values(
        by="roc_auc",
        ascending=False
    )

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    joblib.dump(
        best_model,
        BEST_MODEL_PATH
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {
        "best_model": best_model_name,
        "best_roc_auc": best_roc_auc,
        "training_rows": len(train_df),
        "testing_rows": len(test_df),
        "features": X_train.shape[1]
    }

    with open(METADATA_PATH, "w") as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(f"Best model : {best_model_name}")
    print(f"Best ROC-AUC: {best_roc_auc:.4f}")

    print()
    print("Saved files:")

    print(f" - {BEST_MODEL_PATH}")
    print(f" - {COMPARISON_PATH}")
    print(f" - {METADATA_PATH}")


if __name__ == "__main__":
    main()