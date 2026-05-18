"""
Risk Router — ML Patient Risk Scoring
POST /api/risk/score
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from services.llm_service import LLMService

router = APIRouter()
llm    = LLMService()


class PatientVitals(BaseModel):
    age:               int   = Field(..., ge=0,    le=120,  description="Age in years")
    systolic_bp:       int   = Field(..., ge=60,   le=250,  description="Systolic BP mmHg")
    diastolic_bp:      int   = Field(..., ge=40,   le=150,  description="Diastolic BP mmHg")
    heart_rate:        int   = Field(..., ge=20,   le=250,  description="Heart rate bpm")
    spo2:              float = Field(..., ge=70.0, le=100.0,description="SpO2 %")
    temperature:       float = Field(..., ge=33.0, le=43.0, description="Temperature °C")
    respiratory_rate:  int   = Field(..., ge=4,    le=60,   description="Resp rate /min")
    blood_glucose:     float = Field(..., ge=1.0,  le=40.0, description="Blood glucose mmol/L")
    consciousness:     int   = Field(0,  ge=0,    le=1,    description="0=alert, 1=altered")
    patient_name:      Optional[str] = None


@router.post("/score")
async def score_patient_risk(request: Request, vitals: PatientVitals):
    """
    Run ML risk classifier on patient vitals.
    Returns risk level + probability + clinical flags + LLM explanation.
    """
    ml_service = request.app.state.ml
    patient_dict = vitals.model_dump()

    # ML prediction
    risk_result = ml_service.predict(patient_dict)

    # LLM explanation of the risk score
    explanation = await llm.explain_risk(risk_result, patient_dict)
    risk_result["explanation"] = explanation

    return {
        "patient": vitals.patient_name or "Anonymous",
        "vitals":  patient_dict,
        "result":  risk_result,
    }
