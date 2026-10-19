#!/usr/bin/env python3

"""
Runs sequentially:
load → validate → preprocess → feature engineering
→ train → evaluate → log model
"""

import os
import sys
import time
import argparse
import json

import pandas as pd
import mlflow
import joblib
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# Fix import path for local modules
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


# ============================================================
# Local modules
# ============================================================

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_feature import build_features


# ============================================================
# Main pipeline
# ============================================================

def main(args):

    # ========================================================
    # Project root
    # ========================================================

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )


    # ========================================================
    # MLflow Setup
    # ========================================================

    # Use SQLite database instead of old file-based mlruns
    db_path = os.path.join(
        project_root,
        "mlflow.db"
    )

    mlflow_uri = (
        args.mlflow_uri
        or f"sqlite:///{db_path.replace(os.sep, '/')}"
    )

    mlflow.set_tracking_uri(
        mlflow_uri
    )

    mlflow.set_experiment(
        args.experiment
    )


    # ========================================================
    # Start MLflow Run
    # ========================================================

    with mlflow.start_run() as run:

        # MLflow automatically generates this ID
        run_id = run.info.run_id

        print(
            f"\n📌 MLflow Run ID: {run_id}"
        )


        # ====================================================
        # Create unique local artifact folder
        # ====================================================

        artifacts_dir = os.path.join(
            project_root,
            "artifacts",
            run_id
        )

        os.makedirs(
            artifacts_dir,
            exist_ok=True
        )

        print(
            f"📁 Local artifact directory: "
            f"{artifacts_dir}"
        )


        # ====================================================
        # Log parameters
        # ====================================================

        mlflow.log_param(
            "model",
            "xgboost"
        )

        mlflow.log_param(
            "threshold",
            args.threshold
        )

        mlflow.log_param(
            "test_size",
            args.test_size
        )


        # ====================================================
        # STAGE 1: Load data
        # ====================================================

        print("\n🔄 Loading data...")

        df = load_data(
            args.input
        )

        print(
            f"✅ Data loaded: "
            f"{df.shape[0]} rows, "
            f"{df.shape[1]} columns"
        )


        # ====================================================
        # STAGE 2: Preprocessing
        # ====================================================

        print("\n🔧 Preprocessing data...")

        df = preprocess_data(
            df
        )


        # Save processed dataset
        processed_path = os.path.join(
            project_root,
            "data",
            "processed",
            "telco_churn_processed.csv"
        )

        os.makedirs(
            os.path.dirname(processed_path),
            exist_ok=True
        )

        df.to_csv(
            processed_path,
            index=False
        )

        print(
            f"✅ Processed dataset saved to "
            f"{processed_path} | "
            f"Shape: {df.shape}"
        )


        # ====================================================
        # STAGE 3: Feature Engineering
        # ====================================================

        print("\n🛠️ Building features...")

        target = args.target

        if target not in df.columns:

            raise ValueError(
                f"Target column '{target}' "
                f"not found in data"
            )


        df_enc = build_features(
            df,
            target_col=target
        )


        # Convert boolean columns to integers
        for column in df_enc.select_dtypes(
            include=["bool"]
        ).columns:

            df_enc[column] = (
                df_enc[column]
                .astype(int)
            )


        print(
            f"✅ Feature engineering completed: "
            f"{df_enc.shape[1]} features"
        )


        # ====================================================
        # Feature Metadata
        # ====================================================

        feature_cols = list(
            df_enc
            .drop(columns=[target])
            .columns
        )


        # ----------------------------------------------------
        # Save feature_columns.json
        # ----------------------------------------------------

        feature_columns_path = os.path.join(
            artifacts_dir,
            "feature_columns.json"
        )

        with open(
            feature_columns_path,
            "w"
        ) as file:

            json.dump(
                feature_cols,
                file,
                indent=2
            )


        # Also log feature columns to MLflow
        mlflow.log_text(
            "\n".join(feature_cols),
            artifact_file="feature_columns.txt"
        )


        # ====================================================
        # Preprocessing Artifact
        # ====================================================

        preprocessing_artifact = {
            "feature_columns": feature_cols,
            "target": target
        }


        preprocessing_path = os.path.join(
            artifacts_dir,
            "preprocessing.pkl"
        )


        joblib.dump(
            preprocessing_artifact,
            preprocessing_path
        )


        # Log preprocessing artifact to MLflow
        mlflow.log_artifact(
            preprocessing_path
        )


        print(
            f"✅ Saved {len(feature_cols)} "
            f"feature columns for serving"
        )


        # ====================================================
        # STAGE 4: Train/Test Split
        # ====================================================

        print("\n📊 Splitting data...")

        X = df_enc.drop(
            columns=[target]
        )

        y = df_enc[target]


        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=args.test_size,

            stratify=y,

            random_state=42
        )


        print(
            f"✅ Train: {X_train.shape[0]} samples | "
            f"Test: {X_test.shape[0]} samples"
        )


        # ====================================================
        # Class Imbalance
        # ====================================================

        scale_pos_weight = (
            (y_train == 0).sum()
            /
            (y_train == 1).sum()
        )


        print(
            f"📈 Class imbalance ratio: "
            f"{scale_pos_weight:.2f}"
        )


        # ====================================================
        # STAGE 5: XGBoost Model
        # ====================================================

        print("\n🤖 Training XGBoost model...")


        model = XGBClassifier(

            n_estimators=301,

            learning_rate=0.034,

            max_depth=7,

            subsample=0.95,

            colsample_bytree=0.98,

            n_jobs=-1,

            random_state=42,

            eval_metric="logloss",

            scale_pos_weight=scale_pos_weight
        )


        # ====================================================
        # Train Model
        # ====================================================

        start_time = time.time()


        model.fit(
            X_train,
            y_train
        )


        train_time = (
            time.time()
            - start_time
        )


        # ====================================================
        # Save Model to This Run's Artifact Folder
        # ====================================================

        model_path = os.path.join(
            artifacts_dir,
            "model.pkl"
        )


        joblib.dump(
            model,
            model_path
        )


        print(
            f"✅ Model saved to "
            f"{model_path}"
        )


        # Log training time
        mlflow.log_metric(
            "train_time",
            train_time
        )


        print(
            f"🤖 Model trained in "
            f"{train_time:.2f} seconds"
        )


        # ====================================================
        # STAGE 6: Model Evaluation
        # ====================================================

        print("\n📊 Evaluating model...")


        start_prediction = time.time()


        # Probability of class 1 = Churn
        proba = model.predict_proba(
            X_test
        )[:, 1]


        # Apply threshold
        y_pred = (
            proba >= args.threshold
        ).astype(int)


        pred_time = (
            time.time()
            - start_prediction
        )


        # ====================================================
        # Calculate Metrics
        # ====================================================

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            proba
        )


        # ====================================================
        # Log Metrics to MLflow
        # ====================================================

        mlflow.log_metric(
            "pred_time",
            pred_time
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )


        # ====================================================
        # Print Performance
        # ====================================================

        print(
            "\n🎯 Model Performance:"
        )

        print(
            f"   Precision: {precision:.3f}"
        )

        print(
            f"   Recall:    {recall:.3f}"
        )

        print(
            f"   F1 Score:  {f1:.3f}"
        )

        print(
            f"   ROC AUC:   {roc_auc:.3f}"
        )


        # ====================================================
        # STAGE 7: Log Model to MLflow
        # ====================================================

        print(
            "\n💾 Saving model to MLflow..."
        )


        mlflow.sklearn.log_model(
            model,
            artifact_path="model",

            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]
        )


        # ====================================================
        # Final Summary
        # ====================================================

        print(
            "\n⏱️ Performance Summary:"
        )

        print(
            f"   Training time: "
            f"{train_time:.2f}s"
        )

        print(
            f"   Inference time: "
            f"{pred_time:.4f}s"
        )

        print(
            f"   Samples per second: "
            f"{len(X_test) / pred_time:.0f}"
        )


        # ====================================================
        # Classification Report
        # ====================================================

        print(
            "\n📈 Detailed Classification Report:"
        )

        print(
            classification_report(
                y_test,
                y_pred,
                digits=3
            )
        )


        # ====================================================
        # Artifact Summary
        # ====================================================

        print(
            "\n📦 Artifacts created:"
        )

        print(
            f"   {model_path}"
        )

        print(
            f"   {preprocessing_path}"
        )

        print(
            f"   {feature_columns_path}"
        )

        print(
            f"\n✅ MLflow Run ID: {run_id}"
        )


# ============================================================
# Command-line arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run churn pipeline "
            "with XGBoost + MLflow"
        )
    )


    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV"
    )


    parser.add_argument(
        "--target",
        type=str,
        default="Churn"
    )


    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35
    )


    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2
    )


    parser.add_argument(
        "--experiment",
        type=str,
        default="Telco Churn"
    )


    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help="Override MLflow tracking URI"
    )


    args = parser.parse_args()

    main(args)


# ============================================================
# Run:
#
# python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv
#
# ============================================================