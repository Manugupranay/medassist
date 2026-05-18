"""
Notes Router — Clinical Note Generation (LLM)
POST /api/notes/soap
POST /api/notes/referral
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from services.llm_service import LLMService

router = APIRouter()
llm    = LLMService()


class SOAPRequest(BaseModel):
    name:            Optional[str] = "Anonymous"
    age:             Optional[int] = None
    chief_complaint: str
    history:         str
    bp:              Optional[str] = None
    hr:              Optional[str] = None
    spo2:            Optional[str] = None
    temp:            Optional[str] = None
    rr:              Optional[str] = None
    examination:     Optional[str] = ""
    investigations:  Optional[str] = ""
    assessment:      Optional[str] = ""


class ReferralRequest(BaseModel):
    patient_name:     str
    dob:              Optional[str] = None
    referring_doctor: str
    referral_to:      str
    reason:           str
    history:          Optional[str] = ""
    medications:      Optional[str] = ""
    investigations:   Optional[str] = ""
    urgency:          Optional[str] = "Routine"


@router.post("/soap")
async def generate_soap_note(data: SOAPRequest):
    """Generate a structured SOAP clinical note using Claude."""
    note = await llm.generate_soap_note(data.model_dump())
    return {"note": note, "type": "SOAP", "patient": data.name}


@router.post("/referral")
async def generate_referral(data: ReferralRequest):
    """Generate a formal referral letter using Claude."""
    letter = await llm.generate_referral_letter(data.model_dump())
    return {"letter": letter, "type": "Referral", "patient": data.patient_name}
