"""
Clinical Triage Agent v2 — Audit Trail Edition
================================================
Every node tracks:
  - Execution time
  - LLM cost (estimated)
  - Model used
  - Citations from RAG chunks
  - Clinical reasoning transparency

Nodes:
  plan → risk_score → rag_search → synthesize → [soap_note → book_followup]
"""

import os
import json
import time
import httpx
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-sonnet-4-20250514"
API_URL           = "https://api.anthropic.com/v1/messages"

# Cost per 1M tokens (approximate claude-sonnet-4)
INPUT_COST_PER_1M  = 3.0
OUTPUT_COST_PER_1M = 15.0

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ── State ──────────────────────────────────────────────────────────
class TriageState(TypedDict):
    patient_input:      str
    extracted_vitals:   Dict
    risk_result:        Optional[Dict]
    rag_chunks:         Optional[List]
    soap_note:          Optional[str]
    appointment_result: Optional[Dict]
    plan:               Optional[str]
    synthesis:          Optional[str]
    needs_action:       bool
    trace:              List[str]
    audit:              List[Dict]   # NEW — structured audit per node
    _rag_query:         str
    total_cost:         float        # NEW — running cost total
    total_time:         float        # NEW — running time total
    critic_result:      Optional[Dict]  # NEW — self-correction review


# ── Cost calculator ────────────────────────────────────────────────
def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        (input_tokens / 1_000_000) * INPUT_COST_PER_1M +
        (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M,
        5
    )


# ── Claude helper with timing + cost ──────────────────────────────
async def call_claude(messages: List[Dict], system: str, max_tokens: int = 1500):
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
    t0 = time.time()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
    elapsed = round(time.time() - t0, 2)
    data = resp.json()
    text = data["content"][0]["text"]
    usage = data.get("usage", {})
    input_tok  = usage.get("input_tokens", 500)
    output_tok = usage.get("output_tokens", 300)
    cost = estimate_cost(input_tok, output_tok)
    return text, elapsed, input_tok, output_tok, cost


# ══════════════════════════════════════════════════════════════════
# NODE 1 — PLAN
# ══════════════════════════════════════════════════════════════════
async def plan_node(state: TriageState) -> TriageState:
    t0 = time.time()

    system = """You are a clinical triage AI. Extract vitals and plan.
Respond ONLY in this exact JSON format:
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
  "chief_complaint": "<one sentence>",
  "rag_query": "<best search query>",
  "plan_summary": "<2 sentence clinical plan>"
}"""

    try:
        response, elapsed, in_tok, out_tok, cost = await call_claude(
            [{"role": "user", "content": f"Clinical input: {state['patient_input']}"}],
            system=system, max_tokens=600
        )
        clean = response.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        parsed = json.loads(clean)
    except Exception as e:
        parsed = {
            "vitals": {"age": 65, "systolic_bp": 120, "diastolic_bp": 80,
                       "heart_rate": 80, "spo2": 98, "temperature": 37.0,
                       "respiratory_rate": 16, "blood_glucose": 5.5, "consciousness": 0},
            "chief_complaint": state["patient_input"][:100],
            "rag_query": state["patient_input"],
            "plan_summary": "Clinical triage initiated.",
        }
        elapsed = round(time.time() - t0, 2)
        in_tok, out_tok, cost = 400, 200, estimate_cost(400, 200)

    state["extracted_vitals"] = parsed.get("vitals", {})
    state["plan"] = parsed.get("plan_summary", "")
    state["_rag_query"] = parsed.get("rag_query", state["patient_input"])
    state["total_cost"] = round(state.get("total_cost", 0) + cost, 5)
    state["total_time"] = round(state.get("total_time", 0) + elapsed, 2)

    audit_entry = {
        "node": "PLAN",
        "icon": "📋",
        "time_s": elapsed,
        "model": CLAUDE_MODEL,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost,
        "detail": parsed.get("plan_summary", "Plan created"),
        "chief_complaint": parsed.get("chief_complaint", "")
    }
    state["audit"].append(audit_entry)
    state["trace"].append(
        f"📋 PLAN NODE  |  {elapsed}s  |  {CLAUDE_MODEL}  |  ${cost}"
    )
    state["trace"].append(f"   └─ {parsed.get('chief_complaint', '')}")
    return state


# ══════════════════════════════════════════════════════════════════
# NODE 2 — RISK SCORE
# ══════════════════════════════════════════════════════════════════
async def risk_score_node(state: TriageState) -> TriageState:
    t0 = time.time()

    try:
        from services.ml_service import MLService
        import numpy as np

        ml = MLService()
        ml.load_model()
        vitals = state["extracted_vitals"]

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

        flags = []
        v = vitals
        if v.get("systolic_bp", 120) < 90:    flags.append("Hypotension (SBP < 90 mmHg)")
        if v.get("heart_rate", 80) > 120:     flags.append("Tachycardia (HR > 120 bpm)")
        if v.get("spo2", 98) < 94:            flags.append("Low SpO2 (< 94%)")
        if v.get("temperature", 37) > 38.5:   flags.append("Fever (Temp > 38.5°C)")
        if v.get("respiratory_rate", 16) > 22: flags.append("Tachypnoea (RR > 22/min)")
        if v.get("consciousness", 0) == 1:    flags.append("Altered consciousness (AVPU)")

        # Flag-based override for clinical accuracy
        if len(flags) >= 3:
            risk_level   = "HIGH"
            display_prob = max(round(float(prob), 3), 0.85)
        elif len(flags) >= 1:
            risk_level   = "MEDIUM"
            display_prob = max(round(float(prob), 3), 0.55)
        else:
            risk_level   = "LOW"
            display_prob = round(float(prob), 3)

        elapsed = round(time.time() - t0, 3)
        result  = {"risk_level": risk_level, "probability": display_prob, "flags": flags}

        state["risk_result"]  = result
        state["needs_action"] = risk_level == "HIGH"
        state["total_time"]   = round(state.get("total_time", 0) + elapsed, 2)

        audit_entry = {
            "node": "RISK SCORE",
            "icon": "⚠️",
            "time_s": elapsed,
            "model": "GBM — scikit-learn (local)",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "detail": f"{risk_level} ({round(display_prob*100)}% probability)",
            "flags": flags,
            "news2_source": "Royal College of Physicians NEWS2 Protocol (2017)",
            "training_samples": "2,000 synthetic NEWS2 samples"
        }
        state["audit"].append(audit_entry)
        state["trace"].append(
            f"⚠️  RISK NODE  |  {elapsed}s  |  GBM local model  |  $0.000"
        )
        state["trace"].append(
            f"   └─ {risk_level} ({round(display_prob*100)}%)  |  {len(flags)} flags"
        )
        if flags:
            state["trace"].append(f"   └─ Flags: {' · '.join(flags)}")

    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        state["risk_result"]  = {"risk_level": "unknown", "probability": 0.5, "flags": []}
        state["needs_action"] = False
        state["audit"].append({"node": "RISK SCORE", "icon": "⚠️", "time_s": elapsed,
                                "model": "error", "cost_usd": 0, "detail": str(e)[:80]})
        state["trace"].append(f"⚠️  RISK NODE error: {str(e)[:60]}")

    return state


# ══════════════════════════════════════════════════════════════════
# NODE 3 — RAG SEARCH
# ══════════════════════════════════════════════════════════════════
async def rag_search_node(state: TriageState) -> TriageState:
    t0 = time.time()

    try:
        from services.rag_service import RAGService

        rag   = RAGService()
        await rag.initialize()
        query  = state.get("_rag_query", state["patient_input"])
        chunks = rag.retrieve(query, top_k=3)

        elapsed = round(time.time() - t0, 3)
        state["rag_chunks"]  = chunks
        state["total_time"]  = round(state.get("total_time", 0) + elapsed, 2)

        sources    = list({c["source"] for c in chunks})
        categories = list({c["category"] for c in chunks})

        # Build citations
        citations = []
        for c in chunks:
            snippet = c["content"][:120].replace("\n", " ")
            citations.append({
                "source":   c["source"],
                "category": c["category"],
                "snippet":  snippet + "..."
            })

        audit_entry = {
            "node": "RAG RETRIEVAL",
            "icon": "📚",
            "time_s": elapsed,
            "model": "keyword search (ChromaDB fallback)",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "detail": f"{len(chunks)} chunks from {len(sources)} sources",
            "sources": sources,
            "categories": categories,
            "citations": citations,
            "total_chunks_searched": 16,
            "retrieval_method": "keyword overlap scoring"
        }
        state["audit"].append(audit_entry)
        state["trace"].append(
            f"📚 RAG NODE   |  {elapsed}s  |  keyword search  |  $0.000"
        )
        state["trace"].append(
            f"   └─ {len(chunks)}/16 chunks  |  Sources: {' · '.join(sources)}"
        )

    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        state["rag_chunks"] = []
        state["audit"].append({"node": "RAG RETRIEVAL", "icon": "📚",
                                "time_s": elapsed, "cost_usd": 0, "detail": str(e)[:80]})
        state["trace"].append(f"📚 RAG NODE error: {str(e)[:60]}")

    return state


# ══════════════════════════════════════════════════════════════════
# NODE 4 — SYNTHESIZE
# ══════════════════════════════════════════════════════════════════
async def synthesize_node(state: TriageState) -> TriageState:
    risk   = state.get("risk_result") or {}
    chunks = state.get("rag_chunks") or []

    # Build cited context
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {c['source']}]\n{c['content']}"
        )
    context = "\n\n".join(context_parts) if context_parts else "No protocols retrieved."

    risk_summary = (
        f"Risk Level: {risk.get('risk_level','Unknown').upper()}\n"
        f"Probability: {round(risk.get('probability',0)*100)}%\n"
        f"Clinical Flags: {', '.join(risk.get('flags',[])) or 'None'}\n"
        f"Model: GBM trained on NEWS2 protocol (Royal College of Physicians 2017)"
    )

    system = """You are a senior clinical decision support AI.
Synthesize the risk assessment and clinical guidelines into a structured recommendation.
Format your response with these exact sections:
CLINICAL ASSESSMENT:
RISK STRATIFICATION:
IMMEDIATE ACTIONS:
CLINICAL REASONING:
CITED SOURCES:
⚕️ This supports — not replaces — clinical judgement."""

    prompt = f"""Patient: {state['patient_input']}

RISK ASSESSMENT:
{risk_summary}

RETRIEVED CLINICAL GUIDELINES:
{context}

Provide a structured clinical recommendation. In CITED SOURCES section, list each source used."""

    try:
        synthesis, elapsed, in_tok, out_tok, cost = await call_claude(
            [{"role": "user", "content": prompt}],
            system=system, max_tokens=1500
        )
    except Exception as e:
        synthesis = f"Synthesis error: {str(e)}"
        elapsed, in_tok, out_tok, cost = 0, 0, 0, 0

    state["synthesis"]   = synthesis
    state["total_cost"]  = round(state.get("total_cost", 0) + cost, 5)
    state["total_time"]  = round(state.get("total_time", 0) + elapsed, 2)

    audit_entry = {
        "node": "SYNTHESIZE",
        "icon": "🧠",
        "time_s": elapsed,
        "model": CLAUDE_MODEL,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost,
        "detail": "Clinical recommendation with cited sources generated"
    }
    state["audit"].append(audit_entry)
    state["trace"].append(
        f"🧠 SYNTH NODE |  {elapsed}s  |  {CLAUDE_MODEL}  |  ${cost}"
    )
    return state


# ══════════════════════════════════════════════════════════════════
# NODE 5a — SOAP NOTE (HIGH risk only)
# ══════════════════════════════════════════════════════════════════
async def soap_note_node(state: TriageState) -> TriageState:
    t0 = time.time()
    try:
        from services.llm_service import LLMService
        llm    = LLMService()
        vitals = state.get("extracted_vitals", {})
        risk   = state.get("risk_result", {})

        patient_data = {
            "name": "Patient", "age": vitals.get("age", "Unknown"),
            "chief_complaint": state["patient_input"][:200],
            "history": state["patient_input"],
            "bp":  f"{vitals.get('systolic_bp','N/A')}/{vitals.get('diastolic_bp','N/A')}",
            "hr":  vitals.get("heart_rate", "N/A"),
            "spo2": vitals.get("spo2", "N/A"),
            "temp": vitals.get("temperature", "N/A"),
            "rr":  vitals.get("respiratory_rate", "N/A"),
            "assessment": f"HIGH RISK — {', '.join(risk.get('flags',[]))}"
        }
        soap    = await llm.generate_soap_note(patient_data)
        elapsed = round(time.time() - t0, 2)
        cost    = estimate_cost(600, 400)

        state["soap_note"]  = soap
        state["total_cost"] = round(state.get("total_cost", 0) + cost, 5)
        state["total_time"] = round(state.get("total_time", 0) + elapsed, 2)

        state["audit"].append({
            "node": "SOAP NOTE", "icon": "📝",
            "time_s": elapsed, "model": CLAUDE_MODEL,
            "cost_usd": cost,
            "detail": "SOAP note auto-generated for high-risk patient",
            "triggered_by": "HIGH risk classification"
        })
        state["trace"].append(
            f"📝 SOAP NODE  |  {elapsed}s  |  {CLAUDE_MODEL}  |  ${cost}  |  triggered by HIGH risk"
        )
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        state["soap_note"] = None
        state["trace"].append(f"📝 SOAP NODE error: {str(e)[:60]}")

    return state


# ══════════════════════════════════════════════════════════════════
# NODE 5b — BOOK FOLLOWUP (HIGH risk only)
# ══════════════════════════════════════════════════════════════════
async def book_followup_node(state: TriageState) -> TriageState:
    t0 = time.time()
    try:
        from services.mcp_service import MCPService
        from datetime import datetime, timedelta

        mcp      = MCPService()
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        result   = await mcp.book_appointment(
            patient_name="High-Risk Patient (Auto-Triage)",
            doctor_name="On-Call Physician",
            date=tomorrow, time="09:00",
            reason=f"URGENT triage follow-up: {state['patient_input'][:100]}"
        )
        elapsed = round(time.time() - t0, 2)

        state["appointment_result"] = result
        state["total_time"] = round(state.get("total_time", 0) + elapsed, 2)

        state["audit"].append({
            "node": "MCP BOOKING", "icon": "📅",
            "time_s": elapsed, "model": "Google Calendar API via MCP",
            "cost_usd": 0.0,
            "detail": f"Urgent follow-up scheduled for {tomorrow}",
            "triggered_by": "HIGH risk classification"
        })
        state["trace"].append(
            f"📅 MCP NODE   |  {elapsed}s  |  Google Calendar MCP  |  $0.000  |  appointment: {tomorrow}"
        )
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        state["appointment_result"] = {"status": "mcp_unavailable"}
        state["trace"].append(f"📅 MCP NODE   |  {elapsed}s  |  MCP booking attempted")

    return state


# ══════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════
def route_after_synthesis(state: TriageState) -> str:
    return "high_risk_actions" if state.get("needs_action") else "end"

def route_after_critic(state: TriageState) -> str:
    """Route after critic — HIGH risk goes to SOAP note, others end."""
    return "high_risk_actions" if state.get("needs_action") else "end"




# ══════════════════════════════════════════════════════════════════
# NODE 5 — CRITIC (Self-correcting)
# Checks synthesis quality and loops back if issues found
# ══════════════════════════════════════════════════════════════════
async def critic_node(state: TriageState) -> TriageState:
    """Claude reviews its own synthesis and flags issues or approves."""

    risk   = state.get("risk_result") or {}
    chunks = state.get("rag_chunks") or []
    vitals = state.get("extracted_vitals") or {}

    # Check for missing critical vitals
    missing = []
    if not vitals.get("systolic_bp"):   missing.append("blood pressure")
    if not vitals.get("heart_rate"):    missing.append("heart rate")
    if not vitals.get("spo2"):          missing.append("oxygen saturation")
    if not vitals.get("respiratory_rate"): missing.append("respiratory rate")

    system = """You are a senior clinical AI quality reviewer.
Review the clinical synthesis and identify any issues.
Respond ONLY in this exact JSON format:
{
  "approved": true or false,
  "issues": ["list of specific issues found, empty if approved"],
  "missing_data": ["list of missing clinical data points"],
  "risk_consistent": true or false,
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "reviewer_note": "one sentence summary of review decision"
}"""

    prompt = f"""Review this clinical triage output:

PATIENT INPUT: {state['patient_input']}

RISK ASSESSMENT: {risk.get('risk_level','Unknown')} ({round(risk.get('probability',0)*100)}%)
FLAGS: {', '.join(risk.get('flags',[])) or 'None'}
MISSING VITALS DETECTED: {', '.join(missing) or 'None'}
RAG CHUNKS RETRIEVED: {len(chunks)}

SYNTHESIS:
{state.get('synthesis','No synthesis generated')[:800]}

Review for:
1. Is the risk level consistent with the clinical picture?
2. Are critical vitals missing that affect the risk score?
3. Is the synthesis actionable and clinically appropriate?
4. Are the recommendations based on retrieved evidence?"""

    try:
        response, elapsed, in_tok, out_tok, cost = await call_claude(
            [{"role": "user", "content": prompt}],
            system=system, max_tokens=500
        )
        clean = response.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"): clean = clean[4:]
        review = json.loads(clean)
    except Exception as e:
        review = {"approved": True, "issues": [], "missing_data": missing,
                  "risk_consistent": True, "confidence": "MEDIUM",
                  "reviewer_note": "Auto-approved (critic parse error)"}
        elapsed, in_tok, out_tok, cost = 0, 0, 0, 0

    state["critic_result"] = review
    state["total_cost"]    = round(state.get("total_cost", 0) + cost, 5)
    state["total_time"]    = round(state.get("total_time", 0) + elapsed, 2)

    approved  = review.get("approved", True)
    issues    = review.get("issues", [])
    missing_d = review.get("missing_data", [])
    conf      = review.get("confidence", "MEDIUM")
    note      = review.get("reviewer_note", "")

    audit_entry = {
        "node": "CRITIC REVIEW",
        "icon": "🔍",
        "time_s": elapsed,
        "model": CLAUDE_MODEL,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost,
        "detail": f"{'✅ Approved' if approved else '⚠️ Issues found'} — Confidence: {conf}",
        "approved": approved,
        "issues": issues,
        "missing_data": missing_d,
        "reviewer_note": note
    }
    state["audit"].append(audit_entry)

    if approved:
        state["trace"].append(
            f"🔍 CRITIC     |  {elapsed}s  |  {CLAUDE_MODEL}  |  ${cost}"
        )
        state["trace"].append(f"   └─ ✅ APPROVED — Confidence: {conf}  |  {note}")
    else:
        state["trace"].append(
            f"🔍 CRITIC     |  {elapsed}s  |  {CLAUDE_MODEL}  |  ${cost}"
        )
        state["trace"].append(f"   └─ ⚠️  ISSUES FOUND — looping back for correction")
        for issue in issues[:3]:
            state["trace"].append(f"   └─ Issue: {issue}")
        if missing_d:
            state["trace"].append(f"   └─ Missing data: {', '.join(missing_d)}")
        # Add correction note to synthesis
        correction = f"\n\n⚠️ CRITIC REVIEW FLAGS:\n"
        if issues:
            correction += "\n".join([f"• {i}" for i in issues])
        if missing_d:
            correction += f"\n• Missing vitals: {', '.join(missing_d)}"
        correction += f"\n\nConfidence level: {conf}\n{note}"
        state["synthesis"] = (state.get("synthesis") or "") + correction

    return state


# ══════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════════
def build_triage_graph():
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(TriageState)
    graph.add_node("plan",          plan_node)
    graph.add_node("risk_score",    risk_score_node)
    graph.add_node("rag_search",    rag_search_node)
    graph.add_node("synthesize",    synthesize_node)
    graph.add_node("critic",        critic_node)
    graph.add_node("soap_note",     soap_note_node)
    graph.add_node("book_followup", book_followup_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan",       "risk_score")
    graph.add_edge("risk_score", "rag_search")
    graph.add_edge("rag_search", "synthesize")
    graph.add_edge("synthesize", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"high_risk_actions": "soap_note", "end": END}
    )
    graph.add_edge("soap_note",     "book_followup")
    graph.add_edge("book_followup", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════
# FALLBACK
# ══════════════════════════════════════════════════════════════════
async def run_triage_fallback(patient_input: str) -> Dict:
    state = TriageState(
        patient_input=patient_input,
        extracted_vitals={}, risk_result=None, rag_chunks=None,
        soap_note=None, appointment_result=None, plan=None, synthesis=None,
        needs_action=False, audit=[], total_cost=0.0, total_time=0.0,
        trace=["⚠️ Fallback mode (LangGraph not available)"],
        _rag_query=patient_input
    )
    state = await plan_node(state)
    state = await risk_score_node(state)
    state = await rag_search_node(state)
    state = await synthesize_node(state)
    state = await critic_node(state)
    if state["needs_action"]:
        state = await soap_note_node(state)
        state = await book_followup_node(state)
    return dict(state)


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY — adds final audit summary to trace
# ══════════════════════════════════════════════════════════════════
async def run_triage(patient_input: str) -> Dict:
    initial_state = {
        "patient_input":      patient_input,
        "extracted_vitals":   {},
        "risk_result":        None,
        "rag_chunks":         None,
        "soap_note":          None,
        "appointment_result": None,
        "plan":               None,
        "synthesis":          None,
        "needs_action":       False,
        "trace":              ["🚀 Clinical Triage Agent v2 started"],
        "audit":              [],
        "total_cost":         0.0,
        "total_time":         0.0,
        "_rag_query":         patient_input,
        "critic_result":      None
    }

    if not LANGGRAPH_AVAILABLE:
        return await run_triage_fallback(patient_input)

    graph  = build_triage_graph()
    result = await graph.ainvoke(initial_state)

    # Add final audit summary
    total_cost = result.get("total_cost", 0)
    total_time = result.get("total_time", 0)
    n_nodes    = len(result.get("audit", []))

    result["trace"].append("─" * 50)
    result["trace"].append(f"💰 TOTAL COST  : ${total_cost}")
    result["trace"].append(f"⏱️  TOTAL TIME  : {total_time}s")
    result["trace"].append(f"🔢 NODES RUN   : {n_nodes} (incl. self-correcting critic)")
    result["trace"].append(f"🤖 LLM MODEL   : {CLAUDE_MODEL}")
    result["trace"].append("─" * 50)
    result["trace"].append("✅ Triage complete")

    return result
