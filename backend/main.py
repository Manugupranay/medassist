"""
MedAssist AI — FastAPI Backend
Production-grade medical AI with RAG + LLM + ML + MCP
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os

import uvicorn

from routers import chat, risk, appointments, notes, triage, multi_agent
from services.rag_service import RAGService
from services.llm_service import LLMServiceError
from services.ml_service import MLService

# ── Startup / Shutdown ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 MedAssist AI starting up...")
    app.state.rag = RAGService()
    await app.state.rag.initialize()
    app.state.ml  = MLService()
    app.state.ml.load_model()
    print("✅ All services ready.")
    yield
    print("🛑 MedAssist AI shutting down.")

# ── App ─────────────────────────────────────────────────────────
app = FastAPI(
    title="MedAssist AI",
    description="Clinical Decision Support System with RAG + LLM + ML + MCP",
    version="1.0.0",
    lifespan=lifespan,
)

# Comma-separated allowlist; "*" stays the default for local development but
# should be pinned to real origins in any deployed environment.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────
app.include_router(chat.router,         prefix="/api/chat",         tags=["Clinical Q&A (RAG + LLM)"])
app.include_router(risk.router,         prefix="/api/risk",         tags=["Risk Scoring (ML)"])
app.include_router(appointments.router, prefix="/api/appointments", tags=["Appointments (MCP)"])
app.include_router(multi_agent.router, prefix="/api/multiagent", tags=["Multi-Agent System"])
app.include_router(triage.router, prefix="/api/triage", tags=["Clinical Triage Agent (LangGraph)"])
app.include_router(notes.router,        prefix="/api/notes",        tags=["Clinical Notes (LLM)"])


@app.exception_handler(LLMServiceError)
async def llm_service_error_handler(request: Request, exc: LLMServiceError):
    """Surface upstream LLM failures as 502 with a readable reason.

    Without this, a timeout or rate limit from the Anthropic API reaches the
    caller as a bare 500 and the actual cause is only visible in the logs.
    """
    return JSONResponse(
        status_code=502,
        content={"detail": str(exc), "service": "anthropic"},
    )


@app.get("/")
