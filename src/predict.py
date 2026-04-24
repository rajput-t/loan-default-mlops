import mlflow.sklearn
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

# ─────────────────────────────────────────────
# 1. LOAD MODEL FROM REGISTRY
# ─────────────────────────────────────────────
# Point to same mlruns folder as train.py
mlflow.set_tracking_uri("../mlruns")

# Load by name + stage — no file paths needed
model = mlflow.sklearn.load_model(
    model_uri="models:/GradientBoosting_v1@champion"
)

print("✅ Model loaded from MLflow Registry")
print(f"   Model type: {type(model.named_steps['model']).__name__}")

# ─────────────────────────────────────────────
# 2. DEFINE FEATURE SCHEMA
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents'
]

# ─────────────────────────────────────────────
# 3. PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_default(input_data: dict) -> dict:
    """
    Takes a dict of feature values, returns
    default probability and risk classification.
    """
    df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

    proba = model.predict_proba(df)[0][1]  # probability of default

    # Risk bucketing — useful for business logic
    if proba < 0.15:
        risk = "LOW"
    elif proba < 0.40:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "default_probability": round(float(proba), 4),
        "risk_category":       risk,
        "approve_loan":        proba < 0.40
    }

# ─────────────────────────────────────────────
# 4. TEST WITH SAMPLE APPLICANTS
# ─────────────────────────────────────────────
test_cases = [
    {
        "label": "Low Risk Applicant",
        "data": {
            'RevolvingUtilizationOfUnsecuredLines': 0.1,
            'age': 45,
            'NumberOfTime30-59DaysPastDueNotWorse': 0,
            'DebtRatio': 0.2,
            'MonthlyIncome': 8000,
            'NumberOfOpenCreditLinesAndLoans': 6,
            'NumberOfTimes90DaysLate': 0,
            'NumberRealEstateLoansOrLines': 1,
            'NumberOfTime60-89DaysPastDueNotWorse': 0,
            'NumberOfDependents': 2
        }
    },
    {
        "label": "High Risk Applicant",
        "data": {
            'RevolvingUtilizationOfUnsecuredLines': 0.95,
            'age': 28,
            'NumberOfTime30-59DaysPastDueNotWorse': 4,
            'DebtRatio': 0.85,
            'MonthlyIncome': 2500,
            'NumberOfOpenCreditLinesAndLoans': 12,
            'NumberOfTimes90DaysLate': 3,
            'NumberRealEstateLoansOrLines': 0,
            'NumberOfTime60-89DaysPastDueNotWorse': 2,
            'NumberOfDependents': 3
        }
    },
    {
        "label": "Medium Risk Applicant",
        "data": {
            'RevolvingUtilizationOfUnsecuredLines': 0.45,
            'age': 35,
            'NumberOfTime30-59DaysPastDueNotWorse': 1,
            'DebtRatio': 0.50,
            'MonthlyIncome': 4500,
            'NumberOfOpenCreditLinesAndLoans': 8,
            'NumberOfTimes90DaysLate': 0,
            'NumberRealEstateLoansOrLines': 1,
            'NumberOfTime60-89DaysPastDueNotWorse': 1,
            'NumberOfDependents': 1
        }
    }
]

print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

for case in test_cases:
    result = predict_default(case["data"])
    print(f"\n📋 {case['label']}")
    print(f"   Default Probability : {result['default_probability']:.1%}")
    print(f"   Risk Category       : {result['risk_category']}")
    print(f"   Loan Decision       : {'✅ APPROVE' if result['approve_loan'] else '❌ DECLINE'}")