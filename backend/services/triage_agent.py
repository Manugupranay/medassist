"""
Clinical Triage Agent — LangGraph Orchestration
================================================
Orchestrates existing MedAssist modules as tools:
  - risk_score     → ML patient risk scoring
  - rag_search     → RAG medical knowledge retrieval
  - generate_soap  → SOAP note generation
  - book_followup  → MCP appointment booking

Flow:
  User input → Plan → [Tools in parallel] → Synthesize → Conditional action → Response
"""

import os
import json
import httpx
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-sonnet-4-20250514"
API_URL           = "https://api.anthropic.com/v1/messages"

# ── LangGraph imports ──────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️  LangGraph not installed — triage agent will use fallback mode")


# ── Agent State ────────────────────────────────────────────────────
class TriageState(TypedDict):
    # Input
    patient_input: str          # Raw clinician input
    extracted_vitals: Dict      # Parsed vitals from input
    # Tool outputs
    risk_result: Optional[Dict] # ML risk score result
    rag_chunks: Optional[List]  # Retrieved medical knowledge
    soap_note: Optional[str]    # Generated SOAP note
    appointment_result: Optional[Dict]  # Booking result
    # Reasoning
    plan: Optional[str]         # Claude's initial plan
    synthesis: Optional[str]    # Final synthesized response
    # Control
    needs_action: bool          # Whether HIGH risk action needed
    trace: List[str]            # Execution trace for transparency


# ── Claude helper ──────────────────────────────────────────────────
async def call_claude(messages: List[Dict], system: str, max_tokens: int = 1500) -> str:
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


# ══════════════════════════════════════════════════════════════════
# NODE 1 — PLAN
# Claude reads the raw input and extracts vitals + decides what tools to call
# ══════════════════════════════════════════════════════════════════
async def plan_node(state: TriageState) -> TriageState:
    """Claude analyzes the clinical input and extracts structured data."""

    system = """You are a clinical triage AI. Your job is to:
1. Extract patient vitals from the clinical description
2. Identify the key clinical concern
3. Decide which tools are needed

Always respond in this exact JSON format:
{
  "vitals": {
    "age": <number or 65>,
    "systolic_bp": <number or 120>,
    "diastolic_bp": <number or 80>,
    "heart_rate": <number or 80>,
    "spo2": <number or 98>,
    "temperature": <number or 37.0>,
    "respiratory_rate": <number or 16>,
    "blood_glucose": <number or 5.5>,
    "consciousness": <0 for alert, 1 for altered>
  },
  "chief_complaint": "<one sentence summary>",
  "rag_query": "<best search query for the medical knowledge base>",
  "plan_summary": "<2 sentence clinical plan>",
  "needs_risk_score": true,
  "needs_rag": true
}"""

    response = await call_claude(
        [{"role": "user", "content": f"Clinical input: {state['patient_input']}"}],
        system=system,
        max_tokens=800
    )

    try:
        clean = response.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        parsed = json.loads(clean)
    except Exception:
        parsed = {
            "vitals": {"age": 65, "systolic_bp": 120, "diastolic_bp": 80,
                       "heart_rate": 80, "spo2": 98, "temperature": 37.0,
                       "respiratory_rate": 16, "blood_glucose": 5.5, "consciousness": 0},
            "chief_complaint": state["patient_input"][:100],
            "rag_query": state["patient_input"],
            "plan_summary": "Clinical triage initiated.",
            "needs_risk_score": True,
            "needs_rag": True
        }

    state["extracted_vitals"] = parsed.get("vitals", {})
    state["plan"] = parsed.get("plan_summary", "")
    state["trace"].append(f"✅ PLAN: {parsed.get('plan_summary', 'Plan created')}")
    state["trace"].append(f"📋 Chief complaint: {parsed.get('chief_complaint', '')}")
    state["_rag_query"] = parsed.get("rag_query", state["patient_input"])
    return state


# ══════════════════════════════════════════════════════════════════
# NODE 2a — RISK SCORE TOOL
# Calls the existing ML risk scoring service directly
# ══════════════════════════════════════════════════════════════════
async def risk_score_node(state: TriageState) -> TriageState:
    """Calls the existing ML risk service to score patient vitals."""
    from services.ml_service import MLService
    from fastapi import Request

    try:
        ml = MLService()
        ml.load_model()
        vitals = state["extracted_vitals"]

        import numpy as np
        features = np.array([[
            vitals.get("age", 65),
            vitals.get("systolic_bp", 120),
            vitals.get("diastolic_bp", 80),
            vitals.get("heart_rate", 80),
            vitals.get("spo2", 98),
            vitals.get("temperature", 37.0),
            vitals.get("respiratory_rate", 16),
            vitals.get("blood_glucose", 5.5),
            vitals.get("consciousness", 0)
        ]])
        prob = ml.pipeline.predict_proba(features)[0][1]
        risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
        flags = []
        v = vitals
        if v.get("systolic_bp", 120) < 90: flags.append("Hypotension (SBP < 90 mmHg)")
        if v.get("heart_rate", 80) > 120: flags.append("Tachycardia (HR > 120 bpm)")
        if v.get("spo2", 98) < 94: flags.append("Low SpO2 (< 94%)")
        if v.get("temperature", 37) > 38.5: flags.append("Fever (Temp > 38.5 C)")
        if v.get("respiratory_rate", 16) > 22: flags.append("Tachypnoea (RR > 22)")
        if v.get("consciousness", 0) == 1: flags.append("Altered consciousness")
        result = {"risk_level": risk_level, "probability": round(float(prob), 3), "flags": flags}
        state["risk_result"] = result
        state["needs_action"] = result.get("risk_level", "").upper() == "HIGH"
        state["trace"].append(
            f"⚠️  RISK SCORE: {result.get('risk_level', 'Unknown').upper()} "
            f"({round(result.get('probability', 0)*100)}% probability)"
        )
        if result.get("flags"):
            state["trace"].append(f"🚩 Flags: {', '.join(result['flags'])}")
    except Exception as e:
        state["risk_result"] = {"risk_level": "unknown", "probability": 0.5, "flags": []}
        state["needs_action"] = False
        state["trace"].append(f"⚠️  Risk score error: {str(e)[:80]}")

    return state


# ══════════════════════════════════════════════════════════════════
# NODE 2b — RAG SEARCH TOOL
# Calls the existing RAG service directly
# ══════════════════════════════════════════════════════════════════
async def rag_search_node(state: TriageState) -> TriageState:
    """Retrieves relevant clinical knowledge using the existing RAG service."""
    from services.rag_service import RAGService

    try:
        rag = RAGService()
        await rag.initialize()
        query = state.get("_rag_query", state["patient_input"])
        chunks = rag.retrieve(query, top_k=3)
        state["rag_chunks"] = chunks
        sources = [c["source"] for c in chunks]
        state["trace"].append(f"📚 RAG RETRIEVED: {len(chunks)} chunks from {', '.join(set(sources))}")
    except Exception as e:
        state["rag_chunks"] = []
        state["trace"].append(f"⚠️  RAG error: {str(e)[:80]}")

    return state


# ══════════════════════════════════════════════════════════════════
# NODE 3 — SYNTHESIZE
# Claude combines risk score + RAG chunks into a clinical recommendation
# ══════════════════════════════════════════════════════════════════
async def synthesize_node(state: TriageState) -> TriageState:
    """Claude synthesizes all tool outputs into a structured clinical response."""

    risk = state.get("risk_result", {})
    chunks = state.get("rag_chunks", [])

    context = "\n\n".join([
        f"[{c['source']}]: {c['content']}"
        for c in chunks
    ]) if chunks else "No specific protocols retrieved."

    risk_summary = (
        f"Risk Level: {risk.get('risk_level', 'Unknown').upper()}\n"
        f"Probability: {round(risk.get('probability', 0)*100)}%\n"
        f"Clinical Flags: {', '.join(risk.get('flags', [])) or 'None'}"
    )

    system = """You are a senior clinical decision support AI.
Synthesize the risk assessment and clinical knowledge into a structured recommendation.
Format your response with these exact sections:
CLINICAL ASSESSMENT:
RISK STRATIFICATION:
IMMEDIATE ACTIONS:
CLINICAL REASONING:
⚕️ This supports — not replaces — clinical judgement."""

    prompt = f"""Patient presentation: {state['patient_input']}

RISK ASSESSMENT:
{risk_summary}

RETRIEVED CLINICAL KNOWLEDGE:
{context}

Provide a structured clinical recommendation."""

    synthesis = await call_claude(
        [{"role": "user", "content": prompt}],
        system=system,
        max_tokens=1200
    )

    state["synthesis"] = synthesis
    state["trace"].append("🧠 SYNTHESIZE: Clinical recommendation generated")
    return state


# ══════════════════════════════════════════════════════════════════
# NODE 4a — SOAP NOTE (conditional — HIGH risk only)
# Calls the existing LLM service to generate a SOAP note
# ══════════════════════════════════════════════════════════════════
async def soap_note_node(state: TriageState) -> TriageState:
    """Generates SOAP note for high-risk patients using existing LLM service."""
    from services.llm_service import LLMService

    try:
        llm = LLMService()
        vitals = state.get("extracted_vitals", {})
        patient_data = {
            "name": "Patient",
            "age": vitals.get("age", "Unknown"),
            "chief_complaint": state["patient_input"][:200],
            "history": state["patient_input"],
            "bp": f"{vitals.get('systolic_bp', 'N/A')}/{vitals.get('diastolic_bp', 'N/A')}",
            "hr": vitals.get("heart_rate", "N/A"),
            "spo2": vitals.get("spo2", "N/A"),
            "temp": vitals.get("temperature", "N/A"),
            "rr": vitals.get("respiratory_rate", "N/A"),
            "assessment": f"HIGH RISK — {', '.join(state.get('risk_result', {}).get('flags', []))}"
        }
        soap = await llm.generate_soap_note(patient_data)
        state["soap_note"] = soap
        state["trace"].append("📝 SOAP NOTE: Generated for high-risk patient")
    except Exception as e:
        state["soap_note"] = None
        state["trace"].append(f"⚠️  SOAP note error: {str(e)[:80]}")

    return state


# ══════════════════════════════════════════════════════════════════
# NODE 4b — BOOK FOLLOWUP (conditional — HIGH risk only)
# Uses MCP service to schedule urgent follow-up
# ══════════════════════════════════════════════════════════════════
async def book_followup_node(state: TriageState) -> TriageState:
    """Books urgent follow-up appointment for high-risk patients via MCP."""
    from services.mcp_service import MCPService
    from datetime import datetime, timedelta

    try:
        mcp = MCPService()
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        result = await mcp.book_appointment(
            patient_name="High-Risk Patient (Auto-Triage)",
            doctor_name="On-Call Physician",
            date=tomorrow,
            time="09:00",
            reason=f"URGENT: High-risk triage — {state['patient_input'][:100]}"
        )
        state["appointment_result"] = result
        state["trace"].append(f"📅 APPOINTMENT: Urgent follow-up scheduled for {tomorrow}")
    except Exception as e:
        state["appointment_result"] = {"status": "mcp_unavailable", "message": str(e)[:80]}
        state["trace"].append("📅 APPOINTMENT: MCP booking attempted (requires Google credentials)")

    return state


# ══════════════════════════════════════════════════════════════════
# ROUTING FUNCTION
# Decides whether to trigger HIGH-RISK actions
# ══════════════════════════════════════════════════════════════════
def route_after_synthesis(state: TriageState) -> str:
    if state.get("needs_action", False):
        return "high_risk_actions"
    return "end"


# ══════════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ══════════════════════════════════════════════════════════════════
def build_triage_graph():
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(TriageState)

    graph.add_node("plan",      plan_node)
    graph.add_node("risk_score", risk_score_node)
    graph.add_node("rag_search", rag_search_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("soap_note",  soap_note_node)
    graph.add_node("book_followup", book_followup_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "risk_score")
    graph.add_edge("risk_score", "rag_search")
    graph.add_edge("rag_search", "synthesize")

    graph.add_conditional_edges(
        "synthesize",
        route_after_synthesis,
        {
            "high_risk_actions": "soap_note",
            "end": END
        }
    )
    graph.add_edge("soap_note", "book_followup")
    graph.add_edge("book_followup", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════
# FALLBACK — runs without LangGraph if not installed
# ══════════════════════════════════════════════════════════════════
async def run_triage_fallback(patient_input: str) -> Dict:
    """Simplified triage without LangGraph — for environments without the package."""
    state = TriageState(
        patient_input=patient_input,
        extracted_vitals={},
        risk_result=None,
        rag_chunks=None,
        soap_note=None,
        appointment_result=None,
        plan=None,
        synthesis=None,
        needs_action=False,
        trace=["⚠️ Running in fallback mode (LangGraph not available)"]
    )
    state = await plan_node(state)
    state = await risk_score_node(state)
    state = await rag_search_node(state)
    state = await synthesize_node(state)
    if state["needs_action"]:
        state = await soap_note_node(state)
        state = await book_followup_node(state)
    return dict(state)


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════
async def run_triage(patient_input: str) -> Dict:
    """Run the full clinical triage agent."""
    initial_state = {
        "patient_input": patient_input,
        "extracted_vitals": {},
        "risk_result": None,
        "rag_chunks": None,
        "soap_note": None,
        "appointment_result": None,
        "plan": None,
        "synthesis": None,
        "needs_action": False,
        "trace": ["🚀 Clinical Triage Agent started"],
        "_rag_query": patient_input
    }

    if not LANGGRAPH_AVAILABLE:
        return await run_triage_fallback(patient_input)

    graph = build_triage_graph()
    result = await graph.ainvoke(initial_state)
    result["trace"].append("✅ Triage complete")
    return result
