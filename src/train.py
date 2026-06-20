"""
train.py
--------
PURPOSE : Train 7 classifiers, track experiments with MLflow,
          save best model as best_model.pkl
INPUT   : data/processed/utility_features.parquet
OUTPUT  : outputs/models/best_model.pkl
          outputs/models/sensitivity_results.csv
          MLflow experiment logs (viewable at http://localhost:5000)

WHY MLFLOW:
    Without MLflow you train a model, get a number, and forget it.
    With MLflow every single run is logged — parameters, metrics,
    the model itself. You can compare 50 runs in a table and see
    exactly which model + feature set + threshold combination won.
    This is how data scientists work in production.
    It also gives you a screenshot for your portfolio.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.impute          import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.utils     import ensure_dir, log_section
from src.features  import get_feature_sets, LEAKAGE_COLS, IDENTITY_COLS

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

INPUT_PATH   = "data/processed/utility_features.parquet"
OUTPUT_DIR   = "outputs/models"
MODEL_PATH   = os.path.join(OUTPUT_DIR, "best_model.pkl")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "sensitivity_results.csv")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 80% train, 20% test

# Target column — what we are predicting
TARGET_COL = "high_risk"

# Label thresholds to test in sensitivity analysis
# WHY TWO THRESHOLDS:
#   Top 10% = only the very worst utilities (stricter, fewer positives)
#   Top 20% = broader definition (more positives, better class balance)
#   Your capstone showed Top 20% gives PR-AUC 0.6185 vs 0.4184 for Top 10%
LABEL_THRESHOLDS = [0.10, 0.20]

# Categorical columns for OneHotEncoder
CAT_COLS = ["Ownership", "NERC Region"]

# MLflow experiment name
MLFLOW_EXPERIMENT = "power-outage-risk-classifier"


# ══════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════

def get_model_factories() -> dict:
    """
    Returns all 7 classifiers with their configurations.

    WHY THESE 7:
        We cover the full spectrum from simple to complex:
        - Logistic Regression: baseline linear model
        - Decision Tree: interpretable, shows feature splits
        - Random Forest: ensemble of trees, robust
        - Extra Trees: faster Random Forest variant
        - HistGradientBoosting: sklearn's fast gradient boosting
        - XGBoost: industry standard gradient boosting
        - LightGBM: fastest gradient boosting, great on tabular data

    WHY class_weight='balanced':
        High risk = 336 utilities (20%)
        Low risk  = 1341 utilities (80%)
        Without balancing, models learn to always predict Low Risk
        and get 80% accuracy while being useless.
        balanced makes the model pay equal attention to both classes.

    WHY scale_pos_weight in XGBoost/LightGBM:
        These libraries use a different parameter for class balancing.
        scale_pos_weight = negative_count / positive_count
        = 1341 / 336 ≈ 4.0
        This tells the model: each high-risk utility counts as 4.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=8,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,       # use all CPU cores
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            scale_pos_weight=4.0,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            scale_pos_weight=4.0,
            random_state=RANDOM_STATE,
            verbose=-1,      # suppress LightGBM output
        ),
    }


# ══════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════

def run_training() -> pd.DataFrame:
    """
    Runs full sensitivity analysis:
        7 models × 2 feature sets × 2 label thresholds = 28 experiments

    Each experiment is logged to MLflow so you can compare them all
    in a visual table at http://localhost:5000
    """
    log_section("STAGE 3 — MODEL TRAINING")

    # ── Load features ──────────────────────────────────────────────
    print(f"  Loading: {INPUT_PATH}")
    utility_df = pd.read_parquet(INPUT_PATH)
    print(f"  Loaded {len(utility_df):,} utilities, {utility_df.shape[1]} columns")

    ensure_dir(OUTPUT_DIR)

    # ── Set up MLflow ──────────────────────────────────────────────
    # MLflow stores experiment data in ./mlruns/ folder
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"  MLflow experiment: {MLFLOW_EXPERIMENT}")
    print(f"  View results: run 'mlflow ui' then open http://localhost:5000")

    # ── Get feature sets ───────────────────────────────────────────
    feature_sets = get_feature_sets(utility_df)

    # ── Get model factories ────────────────────────────────────────
    models = get_model_factories()

    # ── Run sensitivity analysis ───────────────────────────────────
    results = []
    best_pr_auc   = -1
    best_pipeline = None
    best_run_info = {}

    total_runs = len(models) * len(feature_sets) * len(LABEL_THRESHOLDS)
    run_number = 0

    print(f"\n  Running {total_runs} experiments...")
    print(f"  {'Run':>3}  {'Model':<25} {'Features':<20} {'Threshold':>9} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>6}")
    print(f"  {'─'*3}  {'─'*25} {'─'*20} {'─'*9} {'─'*8} {'─'*8} {'─'*6}")

    for pct in LABEL_THRESHOLDS:

        # Create label for this threshold
        # WHY CREATE LABELS HERE not in features.py:
        #   Sensitivity analysis means testing different label definitions.
        #   We recompute labels fresh for each threshold so they are
        #   independent of whatever threshold features.py used.
        threshold_val = utility_df["risk_score"].quantile(1 - pct)
        y = (utility_df["risk_score"] >= threshold_val).astype(int)

        label_name = f"Top {int(pct*100)}%"

        for feature_set_name, feature_cols in feature_sets.items():

            # Get only the columns that exist in the dataframe
            available_features = [
                c for c in feature_cols
                if c in utility_df.columns
            ]

            X = utility_df[available_features]

            # Identify which of our available features are categorical
            # WHY: ColumnTransformer needs to know which columns get
            # OneHotEncoder vs StandardScaler
            cat_in_X = [
                c for c in CAT_COLS
                if c in available_features
            ]
            num_in_X = [
                c for c in available_features
                if c not in cat_in_X
            ]

            # Build preprocessing pipeline
            # WHY A PIPELINE:
            #   Without Pipeline, you risk fitting the scaler on ALL data
            #   including test data — that is data leakage.
            #   Pipeline guarantees scaler/encoder fit ONLY on train data
            #   and transforms test data using train statistics.
            preprocessor = _build_preprocessor(num_in_X, cat_in_X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                stratify=y,       # keeps class ratio same in train/test
                random_state=RANDOM_STATE,
            )

            for model_name, model in models.items():
                run_number += 1

                # Build full pipeline: preprocessor → model
                pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier",   model),
                ])

                # ── MLflow run ─────────────────────────────────────
                run_name = f"{model_name}_{feature_set_name}_{label_name}"

                with mlflow.start_run(run_name=run_name):

                    # Log parameters — what settings we used
                    mlflow.log_param("model",          model_name)
                    mlflow.log_param("feature_set",    feature_set_name)
                    mlflow.log_param("label_threshold", label_name)
                    mlflow.log_param("n_features",     len(available_features))
                    mlflow.log_param("n_train",        len(X_train))
                    mlflow.log_param("n_test",         len(X_test))
                    mlflow.log_param("positive_rate",  y.mean().round(3))

                    try:
                        # Train
                        pipe.fit(X_train, y_train)

                        # Predict
                        y_pred  = pipe.predict(X_test)
                        y_proba = pipe.predict_proba(X_test)[:, 1]

                        # Calculate metrics
                        roc_auc   = roc_auc_score(y_test, y_proba)
                        pr_auc    = average_precision_score(y_test, y_proba)
                        f1        = f1_score(y_test, y_pred, zero_division=0)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall    = recall_score(y_test, y_pred, zero_division=0)

                        # Log metrics to MLflow
                        mlflow.log_metric("roc_auc",   roc_auc)
                        mlflow.log_metric("pr_auc",    pr_auc)
                        mlflow.log_metric("f1",        f1)
                        mlflow.log_metric("precision", precision)
                        mlflow.log_metric("recall",    recall)

                        # Log model to MLflow
                        mlflow.sklearn.log_model(pipe, "model")

                        # Print progress line
                        print(
                            f"  {run_number:>3}  {model_name:<25} "
                            f"{feature_set_name:<20} {label_name:>9} "
                            f"{roc_auc:>8.4f} {pr_auc:>8.4f} {f1:>6.4f}"
                        )

                        # Track best model
                        if pr_auc > best_pr_auc:
                            best_pr_auc   = pr_auc
                            best_pipeline = pipe
                            best_run_info = {
                                "model":         model_name,
                                "feature_set":   feature_set_name,
                                "threshold":     label_name,
                                "roc_auc":       round(roc_auc, 4),
                                "pr_auc":        round(pr_auc, 4),
                                "f1":            round(f1, 4),
                                "precision":     round(precision, 4),
                                "recall":        round(recall, 4),
                                "n_features":    len(available_features),
                            }

                        # Store result
                        results.append({
                            "model":         model_name,
                            "feature_set":   feature_set_name,
                            "threshold":     label_name,
                            "roc_auc":       round(roc_auc, 4),
                            "pr_auc":        round(pr_auc, 4),
                            "f1":            round(f1, 4),
                            "precision":     round(precision, 4),
                            "recall":        round(recall, 4),
                        })

                    except Exception as e:
                        print(f"  {run_number:>3}  {model_name:<25} FAILED: {e}")
                        mlflow.log_param("error", str(e))

    # ── Save best model ────────────────────────────────────────────
    if best_pipeline is not None:
        joblib.dump(best_pipeline, MODEL_PATH)
        print(f"\n  ✓ Best model saved → {MODEL_PATH}")
        print(f"  Best: {best_run_info['model']} | "
              f"{best_run_info['feature_set']} | "
              f"PR-AUC={best_run_info['pr_auc']}")

    # ── Save all results ───────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"  ✓ All results saved → {RESULTS_PATH}")

    # ── Save metrics for DVC ───────────────────────────────────────
    with open(METRICS_PATH, "w") as f:
        json.dump(best_run_info, f, indent=2)
    print(f"  ✓ Best metrics saved → {METRICS_PATH}")

    return results_df


# ══════════════════════════════════════════════════════════════════
# HELPER — build preprocessing pipeline
# ══════════════════════════════════════════════════════════════════

def _build_preprocessor(
    numeric_cols: list,
    categorical_cols: list
) -> ColumnTransformer:
    """
    Builds sklearn ColumnTransformer for mixed feature types.

    NUMERIC PIPELINE:
        Step 1: SimpleImputer(strategy='median')
            Fills missing values with the column median.
            WHY MEDIAN not MEAN: Median is robust to outliers.
            A utility with SAIDI = 10,000 (data error) would
            skew the mean drastically but not the median.

        Step 2: StandardScaler()
            Scales values to mean=0, std=1.
            WHY: Logistic Regression and other linear models are
            sensitive to feature scale. A feature ranging 0-10,000
            will dominate one ranging 0-1 without scaling.
            Tree models (RF, XGBoost) do not need this but it
            does not hurt them either.

    CATEGORICAL PIPELINE:
        Step 1: SimpleImputer(strategy='most_frequent')
            Fills missing text values with the most common value.

        Step 2: OneHotEncoder(handle_unknown='ignore')
            Converts 'Investor Owned' → [1, 0, 0, 0]
                     'Cooperative'   → [0, 1, 0, 0]
            WHY handle_unknown='ignore':
                If a new utility type appears in test data that
                was not in training data, ignore it instead of crash.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,   # return dense array not sparse matrix
        )),
    ])

    transformers = []

    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",   # drop any columns not specified
    )


# ══════════════════════════════════════════════════════════════════
# RUN AS SCRIPT
# Usage: python -m src.train
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results_df = run_training()

    print(f"\n{'═'*60}")
    print("  TRAINING COMPLETE — TOP 10 RESULTS BY PR-AUC")
    print(f"{'═'*60}")
    print(results_df.head(10).to_string(index=False))
    print(f"\n  To view MLflow UI: mlflow ui")
    print(f"  Then open: http://localhost:5000")
    print(f"{'═'*60}\n")