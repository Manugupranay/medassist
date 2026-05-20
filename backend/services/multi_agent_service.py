"""
MedAssist Multi-Agent System — Angle 3
========================================
Three specialist agents consult in parallel, then a
Consensus Generator synthesizes a unified recommendation.

Agents:
  1. Diagnosis Specialist   — symptom analysis, differential diagnosis
  2. Pharmacy Specialist    — drug interactions, contraindications, dosing
  3. Care Plan Specialist   — care pathway, admission, escalation, follow-up

  4. Consensus Generator    — reads all three, produces unified output
"""

import os, json, time, asyncio, httpx
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-sonnet-4-20250514"
API_URL           = "https://api.anthropic.com/v1/messages"

INPUT_COST_PER_1M  = 3.0
OUTPUT_COST_PER_1M = 15.0

def estimate_cost(in_tok: int, out_tok: int) -> float:
    return round((in_tok/1_000_000)*INPUT_COST_PER_1M + (out_tok/1_000_000)*OUTPUT_COST_PER_1M, 5)

async def call_claude(messages: List[Dict], system: str, max_tokens: int = 1200):
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {"model": CLAUDE_MODEL, "max_tokens": max_tokens, "system": system, "messages": messages}
    t0 = time.time()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
    elapsed = round(time.time()-t0, 2)
    data    = resp.json()
    text    = data["content"][0]["text"]
    usage   = data.get("usage", {})
    in_tok  = usage.get("input_tokens", 400)
    out_tok = usage.get("output_tokens", 300)
    cost    = estimate_cost(in_tok, out_tok)
    return text, elapsed, in_tok, out_tok, cost


# ══════════════════════════════════════════════════════════════════
# AGENT 1 — DIAGNOSIS SPECIALIST
# ══════════════════════════════════════════════════════════════════
async def diagnosis_agent(patient_input: str, vitals: Dict, risk: Dict) -> Dict:
    """Specialist in symptom analysis and differential diagnosis."""

    system = """You are a specialist Diagnosis AI agent in a multi-agent clinical system.
Your ONLY job is to analyze symptoms and produce a differential diagnosis.
Be precise, use medical terminology, and cite clinical reasoning.
Format your response with these exact sections:
PRIMARY DIAGNOSIS:
DIFFERENTIAL DIAGNOSES:
KEY CLINICAL INDICATORS:
DIAGNOSTIC CONFIDENCE:
RECOMMENDED INVESTIGATIONS:"""

    flags = ', '.join(risk.get('flags', [])) or 'None detected'
    prompt = f"""Patient presentation: {patient_input}

Extracted vitals:
- BP: {vitals.get('systolic_bp','?')}/{vitals.get('diastolic_bp','?')} mmHg
- HR: {vitals.get('heart_rate','?')} bpm
- SpO2: {vitals.get('spo2','?')}%
- Temp: {vitals.get('temperature','?')}°C
- RR: {vitals.get('respiratory_rate','?')}/min
- Consciousness: {'Alert' if vitals.get('consciousness',0)==0 else 'Altered'}

Risk level: {risk.get('risk_level','Unknown')} ({round(risk.get('probability',0)*100)}%)
Clinical flags: {flags}

Provide your specialist diagnosis assessment."""

    try:
        text, elapsed, in_tok, out_tok, cost = await call_claude(
            [{"role":"user","content":prompt}], system=system, max_tokens=800
        )
        return {"agent": "Diagnosis Specialist", "icon": "🩺", "output": text,
                "time_s": elapsed, "cost_usd": cost, "tokens": in_tok+out_tok,
                "model": CLAUDE_MODEL, "status": "success"}
    except Exception as e:
        return {"agent": "Diagnosis Specialist", "icon": "🩺",
                "output": f"Error: {str(e)}", "time_s": 0, "cost_usd": 0,
                "tokens": 0, "model": CLAUDE_MODEL, "status": "error"}


# ══════════════════════════════════════════════════════════════════
# AGENT 2 — PHARMACY SPECIALIST
# ══════════════════════════════════════════════════════════════════
async def pharmacy_agent(patient_input: str, vitals: Dict, risk: Dict) -> Dict:
    """Specialist in drug interactions, contraindications, and dosing."""

    system = """You are a specialist Clinical Pharmacy AI agent in a multi-agent clinical system.
Your ONLY job is to assess medication safety, drug interactions, and recommend appropriate pharmacological treatment.
Be precise about dosing, contraindications, and monitoring requirements.
Format your response with these exact sections:
RECOMMENDED MEDICATIONS:
DOSING GUIDANCE:
CONTRAINDICATIONS & CAUTIONS:
DRUG INTERACTION FLAGS:
MONITORING REQUIREMENTS:"""

    prompt = f"""Patient presentation: {patient_input}

Key clinical parameters:
- Age: {vitals.get('age','?')} years
- BP: {vitals.get('systolic_bp','?')}/{vitals.get('diastolic_bp','?')} mmHg (hypotension risk)
- HR: {vitals.get('heart_rate','?')} bpm
- Temp: {vitals.get('temperature','?')}°C
- SpO2: {vitals.get('spo2','?')}%
- Consciousness: {'Alert' if vitals.get('consciousness',0)==0 else 'Altered — check drug clearance'}

Risk level: {risk.get('risk_level','Unknown')}
Clinical flags: {', '.join(risk.get('flags',[])) or 'None'}

Provide your specialist pharmacy assessment including drug recommendations and safety checks."""

    try:
        text, elapsed, in_tok, out_tok, cost = await call_claude(
            [{"role":"user","content":prompt}], system=system, max_tokens=800
        )
        return {"agent": "Pharmacy Specialist", "icon": "💊", "output": text,
                "time_s": elapsed, "cost_usd": cost, "tokens": in_tok+out_tok,
                "model": CLAUDE_MODEL, "status": "success"}
    except Exception as e:
        return {"agent": "Pharmacy Specialist", "icon": "💊",
                "output": f"Error: {str(e)}", "time_s": 0, "cost_usd": 0,
                "tokens": 0, "model": CLAUDE_MODEL, "status": "error"}


# ══════════════════════════════════════════════════════════════════
# AGENT 3 — CARE PLAN SPECIALIST
# ══════════════════════════════════════════════════════════════════
async def care_plan_agent(patient_input: str, vitals: Dict, risk: Dict) -> Dict:
    """Specialist in care pathways, admission decisions, and escalation."""

    system = """You are a specialist Care Plan AI agent in a multi-agent clinical system.
Your ONLY job is to design the clinical care pathway — admission decisions, escalation protocols, monitoring plans, and follow-up.
Be specific about timelines, escalation triggers, and disposition decisions.
Format your response with these exact sections:
DISPOSITION DECISION:
IMMEDIATE CARE PATHWAY:
ESCALATION TRIGGERS:
MONITORING PLAN:
FOLLOW-UP REQUIREMENTS:"""

    prompt = f"""Patient presentation: {patient_input}

Clinical risk assessment:
- Risk Level: {risk.get('risk_level','Unknown')} ({round(risk.get('probability',0)*100)}%)
- Flags: {', '.join(risk.get('flags',[])) or 'None'}
- Age: {vitals.get('age','?')} | BP: {vitals.get('systolic_bp','?')}/{vitals.get('diastolic_bp','?')} | HR: {vitals.get('heart_rate','?')}
- SpO2: {vitals.get('spo2','?')}% | Temp: {vitals.get('temperature','?')}°C | RR: {vitals.get('respiratory_rate','?')}
- Consciousness: {'Alert' if vitals.get('consciousness',0)==0 else 'ALTERED — high acuity'}

Provide your specialist care plan including admission decision and full pathway."""

    try:
        text, elapsed, in_tok, out_tok, cost = await call_claude(
            [{"role":"user","content":prompt}], system=system, max_tokens=800
        )
        return {"agent": "Care Plan Specialist", "icon": "📋", "output": text,
                "time_s": elapsed, "cost_usd": cost, "tokens": in_tok+out_tok,
                "model": CLAUDE_MODEL, "status": "success"}
    except Exception as e:
        return {"agent": "Care Plan Specialist", "icon": "📋",
                "output": f"Error: {str(e)}", "time_s": 0, "cost_usd": 0,
                "tokens": 0, "model": CLAUDE_MODEL, "status": "error"}


# ══════════════════════════════════════════════════════════════════
# AGENT 4 — CONSENSUS GENERATOR
# ══════════════════════════════════════════════════════════════════
async def consensus_agent(patient_input: str, diagnosis: Dict, pharmacy: Dict, care_plan: Dict) -> Dict:
    """Synthesizes all three specialist outputs into unified recommendation."""

    system = """You are the Consensus Generator in a multi-agent clinical AI system.
You receive independent assessments from three specialist agents and synthesize them into one unified, coherent clinical recommendation.
Identify agreements, resolve conflicts, and produce the final actionable plan.
Format your response with these exact sections:
CONSENSUS DIAGNOSIS:
UNIFIED TREATMENT PLAN:
PHARMACY CONSENSUS:
CARE PATHWAY:
AGENT CONFLICTS RESOLVED:
FINAL RECOMMENDATION:
⚕️ This multi-agent consensus supports — not replaces — clinical judgement."""

    prompt = f"""Patient: {patient_input}

═══ DIAGNOSIS SPECIALIST OUTPUT ═══
{diagnosis.get('output','No output')}

═══ PHARMACY SPECIALIST OUTPUT ═══
{pharmacy.get('output','No output')}

═══ CARE PLAN SPECIALIST OUTPUT ═══
{care_plan.get('output','No output')}

Synthesize these three specialist assessments into a unified clinical recommendation.
Explicitly note any conflicts between agents and how you resolved them."""

    try:
        text, elapsed, in_tok, out_tok, cost = await call_claude(
            [{"role":"user","content":prompt}], system=system, max_tokens=1500
        )
        return {"agent": "Consensus Generator", "icon": "🤝", "output": text,
                "time_s": elapsed, "cost_usd": cost, "tokens": in_tok+out_tok,
                "model": CLAUDE_MODEL, "status": "success"}
    except Exception as e:
        return {"agent": "Consensus Generator", "icon": "🤝",
                "output": f"Error: {str(e)}", "time_s": 0, "cost_usd": 0,
                "tokens": 0, "model": CLAUDE_MODEL, "status": "error"}


# ══════════════════════════════════════════════════════════════════
# ORCHESTRATOR — runs all agents
# ══════════════════════════════════════════════════════════════════
async def run_multi_agent(patient_input: str, vitals: Dict = None, risk: Dict = None) -> Dict:
    """
    Main entry point for multi-agent system.
    Runs 3 specialists in parallel, then consensus generator.
    """
    if vitals is None: vitals = {}
    if risk   is None: risk   = {"risk_level": "UNKNOWN", "probability": 0.5, "flags": []}

    trace = ["🚀 Multi-Agent System started", f"👥 Dispatching 3 specialist agents in parallel..."]
    t_start = time.time()

    # Run 3 specialists IN PARALLEL
    diagnosis_task  = diagnosis_agent(patient_input, vitals, risk)
    pharmacy_task   = pharmacy_agent(patient_input, vitals, risk)
    care_plan_task  = care_plan_agent(patient_input, vitals, risk)

    diagnosis_result, pharmacy_result, care_plan_result = await asyncio.gather(
        diagnosis_task, pharmacy_task, care_plan_task
    )

    trace.append(f"🩺 DIAGNOSIS   |  {diagnosis_result['time_s']}s  |  {CLAUDE_MODEL}  |  ${diagnosis_result['cost_usd']}")
    trace.append(f"   └─ Primary assessment complete")
    trace.append(f"💊 PHARMACY    |  {pharmacy_result['time_s']}s  |  {CLAUDE_MODEL}  |  ${pharmacy_result['cost_usd']}")
    trace.append(f"   └─ Drug safety assessment complete")
    trace.append(f"📋 CARE PLAN   |  {care_plan_result['time_s']}s  |  {CLAUDE_MODEL}  |  ${care_plan_result['cost_usd']}")
    trace.append(f"   └─ Care pathway assessment complete")
    trace.append(f"🤝 Sending all 3 outputs to Consensus Generator...")

    # Run consensus generator
    consensus_result = await consensus_agent(
        patient_input, diagnosis_result, pharmacy_result, care_plan_result
    )

    trace.append(f"🤝 CONSENSUS   |  {consensus_result['time_s']}s  |  {CLAUDE_MODEL}  |  ${consensus_result['cost_usd']}")
    trace.append(f"   └─ Unified recommendation generated")

    # Build audit
    total_time = round(time.time()-t_start, 2)
    total_cost = round(sum(r['cost_usd'] for r in [diagnosis_result, pharmacy_result, care_plan_result, consensus_result]), 5)
    total_tokens = sum(r['tokens'] for r in [diagnosis_result, pharmacy_result, care_plan_result, consensus_result])

    trace.append("─"*50)
    trace.append(f"💰 TOTAL COST   : ${total_cost}")
    trace.append(f"⏱️  TOTAL TIME   : {total_time}s")
    trace.append(f"🔢 AGENTS RUN   : 4 (3 parallel + 1 consensus)")
    trace.append(f"🔤 TOTAL TOKENS : {total_tokens}")
    trace.append("─"*50)
    trace.append("✅ Multi-agent consensus complete")

    return {
        "trace":            trace,
        "diagnosis":        diagnosis_result,
        "pharmacy":         pharmacy_result,
        "care_plan":        care_plan_result,
        "consensus":        consensus_result,
        "total_cost":       total_cost,
        "total_time":       total_time,
        "total_tokens":     total_tokens,
        "agents_run":       4,
        "parallel_agents":  3,
    }
