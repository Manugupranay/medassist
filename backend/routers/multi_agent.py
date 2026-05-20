"""
Multi-Agent Router — POST /api/multiagent/analyze
Exposes the Multi-Agent Clinical System via FastAPI
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List

router = APIRouter()

class MultiAgentRequest(BaseModel):
    patient_input: str
    vitals: Optional[Dict] = None
    risk: Optional[Dict] = None

class AgentOutput(BaseModel):
    agent: str
    icon: str
    output: str
    time_s: float
    cost_usd: float
    status: str

class MultiAgentResponse(BaseModel):
    trace: List[str]
    diagnosis: AgentOutput
    pharmacy: AgentOutput
    care_plan: AgentOutput
    consensus: AgentOutput
    total_cost: float
    total_time: float
    total_tokens: int
    agents_run: int
    parallel_agents: int

@router.post("/analyze", response_model=MultiAgentResponse)
async def multi_agent_analyze(body: MultiAgentRequest):
    """
    Multi-Agent Clinical System.
    Runs 3 specialist agents in parallel then a consensus generator.
    """
    if not body.patient_input.strip():
        raise HTTPException(status_code=400, detail="Patient input required")
    try:
        from services.multi_agent_service import run_multi_agent
        result = await run_multi_agent(
            patient_input=body.patient_input,
            vitals=body.vitals or {},
            risk=body.risk or {}
        )
        return MultiAgentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-agent error: {str(e)}")

@router.get("/health")
async def health():
    return {"status": "ok", "agents": ["diagnosis", "pharmacy", "care_plan", "consensus"]}
