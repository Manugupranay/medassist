"""
Triage Router — POST /api/triage/analyze
Exposes the LangGraph Clinical Triage Agent via FastAPI
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class TriageRequest(BaseModel):
    patient_input: str
    session_id: Optional[str] = "default"


class TriageResponse(BaseModel):
    synthesis: str
    risk_level: str
    risk_probability: float
    risk_flags: list
    soap_note: Optional[str]
    appointment_booked: bool
    trace: list
    high_risk_actions_taken: bool


@router.post("/analyze", response_model=TriageResponse)
async def analyze_patient(body: TriageRequest, request: Request):
    """
    Clinical Triage Agent — LangGraph orchestrated multi-step reasoning.

    Takes a free-text clinical description and:
    1. Extracts vitals and plans tool usage
    2. Runs ML risk scoring + RAG retrieval in parallel
    3. Synthesizes a clinical recommendation
    4. If HIGH risk: generates SOAP note + books follow-up
    """
    if not body.patient_input.strip():
        raise HTTPException(status_code=400, detail="Patient input cannot be empty")

    try:
        from services.triage_agent import run_triage
        result = await run_triage(body.patient_input)

        risk = result.get("risk_result") or {}
        return TriageResponse(
            synthesis=result.get("synthesis") or "Clinical assessment complete.",
            risk_level=risk.get("risk_level", "unknown").upper(),
            risk_probability=round(float(risk.get("probability", 0.5)), 3),
            risk_flags=risk.get("flags", []),
            soap_note=result.get("soap_note"),
            appointment_booked=result.get("appointment_result") is not None,
            trace=result.get("trace", []),
            high_risk_actions_taken=result.get("needs_action", False),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Triage agent error: {str(e)}"
        )


@router.get("/health")
async def triage_health():
    """Check if LangGraph is available."""
    try:
        import langgraph
        return {"status": "ok", "langgraph": True, "version": langgraph.__version__}
    except ImportError:
        return {"status": "ok", "langgraph": False, "mode": "fallback"}
