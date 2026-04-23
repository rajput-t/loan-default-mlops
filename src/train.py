import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, classification_report,
    average_precision_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────
df = pd.read_csv('../data/loan_data.csv')
df = df.drop(columns=['Unnamed: 0'])
df = df.rename(columns={'SeriousDlqin2yrs': 'default'})

# Clip outliers at 99th percentile
clip_cols = [
    'RevolvingUtilizationOfUnsecuredLines',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfTimes90DaysLate'
]
for col in clip_cols:
    df[col] = df[col].clip(upper=df[col].quantile(0.99))

# ─────────────────────────────────────────────
# 2. FEATURES & TARGET
# ─────────────────────────────────────────────
X = df.drop(columns=['default'])
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 3. PIPELINE (Impute → Scale → Model)
# ─────────────────────────────────────────────
def make_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# ─────────────────────────────────────────────
# 4. MODELS TO EXPERIMENT WITH
# ─────────────────────────────────────────────
experiments = [
    {
        "name": "LogisticRegression_baseline",
        "model": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=0.1
        ),
        "params": {"C": 0.1, "max_iter": 1000}
    },
    {
        "name": "RandomForest_v1",
        "model": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "params": {"n_estimators": 100, "max_depth": 8}
    },
    {
        "name": "RandomForest_v2_deeper",
        "model": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "params": {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5}
    },
    {
        "name": "GradientBoosting_v1",
        "model": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ),
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4}
    },
]

# ─────────────────────────────────────────────
# 5. MLFLOW EXPERIMENT LOOP
# ─────────────────────────────────────────────
mlflow.set_tracking_uri("../mlruns")  # always save to project root
mlflow.set_experiment("loan_default_prediction")

for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):

        pipeline = make_pipeline(exp["model"])
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        auc      = roc_auc_score(y_test, y_proba)
        pr_auc   = average_precision_score(y_test, y_proba)
        f1       = f1_score(y_test, y_pred, average='weighted')

        # Cross-val AUC (more reliable than single split)
        cv_auc = cross_val_score(
            pipeline, X_train, y_train,
            cv=StratifiedKFold(n_splits=5),
            scoring='roc_auc', n_jobs=-1
        ).mean()

        # Log to MLflow
        mlflow.log_params(exp["params"])
        mlflow.log_metrics({
            "test_auc":    round(auc, 4),
            "pr_auc":      round(pr_auc, 4),
            "f1_weighted": round(f1, 4),
            "cv_auc_mean": round(cv_auc, 4)
        })

        # Log the entire pipeline as a model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=exp["name"]
        )

        print(f"\n{'='*50}")
        print(f"Run: {exp['name']}")
        print(f"  Test AUC  : {auc:.4f}")
        print(f"  PR-AUC    : {pr_auc:.4f}")
        print(f"  CV AUC    : {cv_auc:.4f}")
        print(f"  F1        : {f1:.4f}")

print("\n✅ All runs logged. Launch UI with: mlflow ui")