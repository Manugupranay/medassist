"""
MedAssist AI — FastAPI Backend
Production-grade medical AI with RAG + LLM + ML + MCP
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from routers import chat, risk, appointments, notes
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────
app.include_router(chat.router,         prefix="/api/chat",         tags=["Clinical Q&A (RAG + LLM)"])
app.include_router(risk.router,         prefix="/api/risk",         tags=["Risk Scoring (ML)"])
app.include_router(appointments.router, prefix="/api/appointments", tags=["Appointments (MCP)"])
app.include_router(notes.router,        prefix="/api/notes",        tags=["Clinical Notes (LLM)"])

@app.get("/")
async def root():
    return {"status": "ok", "service": "MedAssist AI", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


