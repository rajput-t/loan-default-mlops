import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ─────────────────────────────────────────────
# 1. LOAD MODEL AT STARTUP
# ─────────────────────────────────────────────
mlflow.set_tracking_uri("../mlruns")
mlflow.set_registry_uri("../mlruns")

print("Loading model from MLflow registry...")
model = mlflow.sklearn.load_model("models:/GradientBoosting_v1@champion")
print("Model loaded successfully")

# ─────────────────────────────────────────────
# 2. FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts probability of loan default using a GradientBoosting model tracked via MLflow.",
    version="1.0.0"
)

# ─────────────────────────────────────────────
# 3. REQUEST & RESPONSE SCHEMAS
# ─────────────────────────────────────────────
class ApplicantData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, le=1,
        example=0.1,
        description="Total balance on credit cards / credit limits (0 to 1)")
    age: int = Field(..., ge=18, le=100,
        example=45,
        description="Age of the applicant in years")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0,
        example=0,
        description="Number of times 30-59 days past due in last 2 years")
    DebtRatio: float = Field(..., ge=0,
        example=0.2,
        description="Monthly debt payments / monthly gross income")
    MonthlyIncome: float = Field(..., ge=0,
        example=8000,
        description="Monthly income in USD")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0,
        example=6,
        description="Number of open loans and lines of credit")
    NumberOfTimes90DaysLate: int = Field(..., ge=0,
        example=0,
        description="Number of times 90+ days past due")
    NumberRealEstateLoansOrLines: int = Field(..., ge=0,
        example=1,
        description="Number of mortgage and real estate loans")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0,
        example=0,
        description="Number of times 60-89 days past due in last 2 years")
    NumberOfDependents: int = Field(..., ge=0,
        example=2,
        description="Number of dependents in family")

class PredictionResponse(BaseModel):
    default_probability: float
    risk_category: str
    approve_loan: bool
    model_version: str = "GradientBoosting_v1@champion"

# ─────────────────────────────────────────────
# 4. ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Loan Default Prediction API is running",
        "docs": "Visit /docs for interactive API documentation"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantData):
    try:
        # Map field names back to original dataset column names
        input_data = {
            'RevolvingUtilizationOfUnsecuredLines': applicant.RevolvingUtilizationOfUnsecuredLines,
            'age':                                  applicant.age,
            'NumberOfTime30-59DaysPastDueNotWorse': applicant.NumberOfTime30_59DaysPastDueNotWorse,
            'DebtRatio':                            applicant.DebtRatio,
            'MonthlyIncome':                        applicant.MonthlyIncome,
            'NumberOfOpenCreditLinesAndLoans':       applicant.NumberOfOpenCreditLinesAndLoans,
            'NumberOfTimes90DaysLate':              applicant.NumberOfTimes90DaysLate,
            'NumberRealEstateLoansOrLines':          applicant.NumberRealEstateLoansOrLines,
            'NumberOfTime60-89DaysPastDueNotWorse': applicant.NumberOfTime60_89DaysPastDueNotWorse,
            'NumberOfDependents':                   applicant.NumberOfDependents
        }

        df = pd.DataFrame([input_data])
        proba = model.predict_proba(df)[0][1]

        if proba < 0.15:
            risk = "LOW"
        elif proba < 0.40:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return PredictionResponse(
            default_probability=round(float(proba), 4),
            risk_category=risk,
            approve_loan=proba < 0.40
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────
# 5. RUN SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)