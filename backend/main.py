"""
MedAssist AI — FastAPI Backend
Production-grade medical AI with RAG + LLM + ML + MCP
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

import uvicorn

from routers import chat, risk, appointments, notes, triage, multi_agent
from services.rag_service import RAGService
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

@app.get("/")
async def root():
    return {"status": "ok", "service": "MedAssist AI", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Loopback by default. Set HOST=0.0.0.0 explicitly when the server needs
    # to be reachable from outside the machine (containers, LAN testing).
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "true").strip().lower() == "true"
    uvicorn.run("main:app", host=host, port=port, reload=reload_enabled)






