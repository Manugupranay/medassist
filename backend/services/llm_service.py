"""
LLM Service — Claude API wrapper
Handles all generative AI tasks:
  - Clinical Q&A with RAG context
  - SOAP note generation
  - Referral letter writing
  - Drug interaction explanations
"""

import os
from dotenv import load_dotenv
load_dotenv()
import httpx
import json
from typing import List, Dict, Any, AsyncIterator

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-sonnet-4-20250514"
API_URL           = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = """You are MedAssist AI, a clinical decision support system designed to assist qualified healthcare professionals. 

IMPORTANT GUIDELINES:
- You assist trained clinicians — not the general public
- Always recommend clinical judgement takes precedence over AI suggestions
- Cite sources when answering from retrieved context
- Flag drug interactions, contraindications, and red flags clearly
- Use structured clinical language (SBAR, SOAP when relevant)
- Never provide a definitive diagnosis — support clinical reasoning
- If a query is outside medical scope, redirect appropriately

Your capabilities:
1. Answer clinical questions using retrieved medical knowledge (RAG)
2. Explain drug interactions and dosing
3. Summarise treatment protocols and guidelines
4. Generate SOAP notes and referral letters from clinical data
5. Risk-stratify patients based on vitals and history

Always end clinical answers with: "⚕️ This information supports — not replaces — clinical judgement."
"""


class LLMService:
    def __init__(self):
        self.headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def chat_with_rag(
        self, query: str, rag_chunks: List[Dict], conversation_history: List[Dict]
    ) -> Dict[str, Any]:
        """Answer a clinical question using retrieved context."""
        context_text = self._format_context(rag_chunks)

        user_message = f"""RETRIEVED MEDICAL KNOWLEDGE:
{context_text}

CLINICAL QUESTION:
{query}

Please answer based on the retrieved knowledge above. Cite the sources. Be precise and clinically accurate."""

        messages = conversation_history + [{"role": "user", "content": user_message}]

        response = await self._call_api(messages)
        return {
            "answer": response,
            "sources": [{"source": c["source"], "category": c["category"]} for c in rag_chunks],
        }

    async def generate_soap_note(self, patient_data: Dict) -> str:
        """Generate a structured SOAP note from patient data."""
        prompt = f"""Generate a professional SOAP note for the following patient encounter:

Patient: {patient_data.get('name', 'Anonymous')}, Age: {patient_data.get('age', 'Unknown')}
Chief Complaint: {patient_data.get('chief_complaint', '')}
History: {patient_data.get('history', '')}
Vitals: BP {patient_data.get('bp', 'N/A')}, HR {patient_data.get('hr', 'N/A')}, 
        SpO2 {patient_data.get('spo2', 'N/A')}%, Temp {patient_data.get('temp', 'N/A')}°C, 
        RR {patient_data.get('rr', 'N/A')}/min
Examination Findings: {patient_data.get('examination', '')}
Investigations: {patient_data.get('investigations', 'None documented')}
Assessment: {patient_data.get('assessment', '')}

Format as a proper SOAP note with clear S / O / A / P sections.
Be concise, clinical, and use appropriate medical terminology."""

        return await self._call_api([{"role": "user", "content": prompt}])

    async def generate_referral_letter(self, patient_data: Dict) -> str:
        """Generate a formal referral letter."""
        prompt = f"""Write a formal medical referral letter with the following details:

From: Dr {patient_data.get('referring_doctor', 'Referring Physician')}
To: {patient_data.get('referral_to', 'Specialist')} Department
Patient: {patient_data.get('patient_name', 'Patient')}, DOB: {patient_data.get('dob', 'Unknown')}
Reason for referral: {patient_data.get('reason', '')}
Relevant history: {patient_data.get('history', '')}
Current medications: {patient_data.get('medications', 'None documented')}
Relevant investigations: {patient_data.get('investigations', 'Attached')}
Urgency: {patient_data.get('urgency', 'Routine')}

Write a professional, concise referral letter in standard medical format.
Include: patient details, reason for referral, relevant history, examination findings, investigations, current management, and specific question/request for the specialist."""

        return await self._call_api([{"role": "user", "content": prompt}])

    async def explain_risk(self, risk_result: Dict, patient_data: Dict) -> str:
        """Generate clinical explanation of ML risk score."""
        prompt = f"""A patient has been assessed with the following risk score:

Risk Level: {risk_result['risk_level'].upper()}
Risk Score: {risk_result['risk_score']}%
Clinical Flags: {', '.join(risk_result['flags']) if risk_result['flags'] else 'None identified'}

Patient Vitals:
- Age: {patient_data.get('age')} years
- BP: {patient_data.get('systolic_bp')}/{patient_data.get('diastolic_bp')} mmHg
- HR: {patient_data.get('heart_rate')} bpm
- SpO2: {patient_data.get('spo2')}%
- Temp: {patient_data.get('temperature')}°C
- RR: {patient_data.get('respiratory_rate')}/min
- Blood Glucose: {patient_data.get('blood_glucose')} mmol/L

Provide a brief (3-4 sentence) clinical interpretation of this risk assessment. 
Mention which parameters are driving the risk score and suggest immediate clinical actions."""

        return await self._call_api([{"role": "user", "content": prompt}])

    def _format_context(self, chunks: List[Dict]) -> str:
        return "\n\n".join([
            f"[Source: {c['source']} | Category: {c['category']} | Relevance: {c.get('score', 'N/A')}]\n{c['content']}"
            for i, c in enumerate(chunks, 1)
        ])

    async def _call_api(self, messages: List[Dict]) -> str:
        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": 1500,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(API_URL, headers=self.headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

