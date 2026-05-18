"""
Appointments Router — MCP Google Calendar + Gmail
POST /api/appointments/book
GET  /api/appointments/schedule
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional

from services.mcp_service import MCPService

router = APIRouter()
mcp    = MCPService()


class AppointmentRequest(BaseModel):
    patient_name:     str
    patient_email:    Optional[str] = ""
    doctor_name:      str
    type:             str = "General Consultation"
    date:             str   # YYYY-MM-DD
    time:             str   # HH:MM
    duration:         int  = 30
    notes:            Optional[str] = ""


@router.post("/book")
async def book_appointment(appt: AppointmentRequest):
    """
    Book appointment via MCP:
    1. Creates Google Calendar event
    2. Sends Gmail confirmation to patient
    """
    result = await mcp.book_appointment(appt.model_dump())
    if not result["success"]:
        raise HTTPException(500, result.get("message", "Booking failed"))
    return result


@router.get("/schedule/{doctor_name}")
async def get_schedule(doctor_name: str):
    """Fetch today's calendar for a doctor via MCP."""
    return await mcp.get_todays_schedule(doctor_name)
