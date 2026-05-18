"""
Chat Router — RAG + LLM Clinical Q&A
POST /api/chat/ask
POST /api/chat/history (for conversation context)
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

from services.llm_service import LLMService

router = APIRouter()
llm    = LLMService()

# In production: per-session conversation history (use Redis)
_conversation_store: Dict[str, List] = {}


class ChatRequest(BaseModel):
    query:      str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer:  str
    sources: List[Dict]
    query:   str


@router.post("/ask", response_model=ChatResponse)
async def ask_clinical_question(request: Request, body: ChatRequest):
    """
    Clinical Q&A endpoint.
    1. Retrieves relevant medical knowledge via RAG
    2. Passes context + question to Claude
    3. Returns cited answer
    """
    if not body.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    # Step 1 — RAG retrieval
    rag_service = request.app.state.rag
    chunks = rag_service.retrieve(body.query, top_k=4)

    # Step 2 — Conversation history
    history = _conversation_store.get(body.session_id, [])

    # Step 3 — LLM generation
    result = await llm.chat_with_rag(body.query, chunks, history)

    # Step 4 — Update conversation history
    history.append({"role": "user",      "content": body.query})
    history.append({"role": "assistant", "content": result["answer"]})
    _conversation_store[body.session_id] = history[-20:]  # keep last 10 turns

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        query=body.query,
    )


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    _conversation_store.pop(session_id, None)
    return {"cleared": True, "session_id": session_id}
