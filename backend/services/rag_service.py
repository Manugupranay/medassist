"""
RAG Service
- Embeds medical knowledge base into ChromaDB
- Retrieves top-k relevant chunks for any query
- Returns cited sources alongside content
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

CHROMA_AVAILABLE = False

# ── Medical Knowledge Base ───────────────────────────────────────
# In production: ingest real PDFs, drug databases, clinical guidelines.
# Here we seed with high-quality structured medical knowledge.
MEDICAL_KNOWLEDGE = [
    {
        "id": "drug_001",
        "source": "Drug Interaction Database",
        "category": "drug_interactions",
        "content": "Warfarin and NSAIDs (ibuprofen, naproxen, aspirin) have a major interaction. NSAIDs inhibit platelet function and damage gastric mucosa, significantly increasing bleeding risk in patients on warfarin. Monitor INR closely. Consider alternative analgesics like paracetamol.",
    },
    {
        "id": "drug_002",
        "source": "Drug Interaction Database",
        "category": "drug_interactions",
        "content": "Metformin should be used with caution in patients with eGFR < 30 mL/min/1.73m². Risk of lactic acidosis increases with renal impairment. Dose reduction required for eGFR 30-45. Contraindicated below eGFR 30. Monitor renal function every 3-6 months.",
    },
    {
        "id": "drug_003",
        "source": "Drug Interaction Database",
        "category": "drug_interactions",
        "content": "ACE inhibitors (lisinopril, enalapril) combined with potassium-sparing diuretics (spironolactone, eplerenone) can cause dangerous hyperkalemia. Monitor serum potassium within 1-2 weeks of starting combination therapy. Target K+ < 5.5 mmol/L.",
    },
    {
        "id": "symptom_001",
        "source": "Clinical Symptom Guidelines",
        "category": "symptoms",
        "content": "Chest pain differential diagnosis: (1) Cardiac: ACS/MI — crushing, radiating to arm/jaw, diaphoresis, nausea. STEMI needs immediate PCI. (2) Pulmonary embolism — pleuritic pain, dyspnoea, tachycardia, risk factors (DVT, immobility). (3) Aortic dissection — tearing pain, radiating to back, BP differential between arms. (4) Pneumothorax — sudden onset, unilateral. (5) GERD — burning, worse lying flat, relieved by antacids.",
    },
    {
        "id": "symptom_002",
        "source": "Clinical Symptom Guidelines",
        "category": "symptoms",
        "content": "Red flag symptoms for headache requiring urgent investigation: thunderclap headache (worst of life), headache with fever + neck stiffness (meningitis), new headache in patient >50 (temporal arteritis, malignancy), headache with focal neurological deficit, headache with papilloedema (raised ICP), progressive headache worse in morning. These require immediate CT head +/- LP.",
    },
    {
        "id": "symptom_003",
        "source": "Clinical Symptom Guidelines",
        "category": "symptoms",
        "content": "Type 2 Diabetes diagnostic criteria: Fasting plasma glucose ≥ 7.0 mmol/L (126 mg/dL), or 2-hour plasma glucose ≥ 11.1 mmol/L during OGTT, or HbA1c ≥ 48 mmol/mol (6.5%), or random plasma glucose ≥ 11.1 mmol/L with symptoms. Must be confirmed on two separate occasions unless symptomatic hyperglycaemia.",
    },
    {
        "id": "protocol_001",
        "source": "Treatment Protocols — WHO/NHS",
        "category": "treatment",
        "content": "Hypertension management: Stage 1 (140-159/90-99): lifestyle modification for 3-6 months before drug therapy if no end-organ damage. Stage 2 (≥160/100): immediate pharmacotherapy. First-line agents: ACE inhibitor or ARB (diabetics/proteinuria), calcium channel blocker (elderly, Afro-Caribbean), thiazide diuretic. Target BP <140/90 general population, <130/80 in diabetes/CKD.",
    },
    {
        "id": "protocol_002",
        "source": "Treatment Protocols — WHO/NHS",
        "category": "treatment",
        "content": "Sepsis 6 bundle (within 1 hour): (1) High-flow oxygen, (2) Blood cultures x2, (3) IV antibiotics — broad spectrum empirical, (4) IV fluid resuscitation 500ml crystalloid bolus, (5) Measure lactate, (6) Urine output monitoring. NEWS2 score ≥ 5 triggers sepsis alert. qSOFA: ≥2 of altered mentation, RR≥22, SBP≤100 indicates high risk.",
    },
    {
        "id": "protocol_003",
        "source": "Treatment Protocols — WHO/NHS",
        "category": "treatment",
        "content": "Acute MI management: STEMI — primary PCI within 90 min (door-to-balloon). Adjunct therapy: aspirin 300mg + P2Y12 inhibitor (ticagrelor preferred), anticoagulation (UFH/LMWH/bivalirudin), beta-blocker (if stable), statin (atorvastatin 80mg). NSTEMI: risk stratify with GRACE score, early invasive strategy if high risk within 24 hrs.",
    },
    {
        "id": "drug_004",
        "source": "Drug Reference — BNF",
        "category": "medications",
        "content": "Amoxicillin dosing: Standard adult dose 250-500mg three times daily for 5-7 days. For severe infections: 500mg-1g three times daily. Penicillin allergy: use clarithromycin 500mg twice daily or doxycycline 100mg twice daily. Common side effects: diarrhoea, nausea, rash. Avoid in penicillin hypersensitivity.",
    },
    {
        "id": "drug_005",
        "source": "Drug Reference — BNF",
        "category": "medications",
        "content": "Paracetamol (acetaminophen) toxicity: Overdose threshold >150 mg/kg or >7.5g in adults. Treatment: N-acetylcysteine (NAC) IV within 8-10 hours of ingestion most effective. Nomogram (Rumack-Matthew) used to guide treatment. Monitor LFTs, PT/INR, renal function. Liver transplant may be needed for fulminant hepatic failure.",
    },
    {
        "id": "vitals_001",
        "source": "Clinical Reference — Normal Ranges",
        "category": "vitals",
        "content": "Normal adult vital signs: Heart rate 60-100 bpm (tachycardia >100, bradycardia <60). Blood pressure 90-120 / 60-80 mmHg. Respiratory rate 12-20 breaths/min. SpO2 >95% (concern <94%, critical <90%). Temperature 36.1-37.2°C. Capillary refill time <2 seconds. GCS 15 (normal). Blood glucose fasting 4-6 mmol/L.",
    },
    {
        "id": "vitals_002",
        "source": "NEWS2 Scoring System",
        "category": "vitals",
        "content": "NEWS2 (National Early Warning Score 2): Scores physiological parameters — Respiration rate, SpO2, systolic BP, pulse rate, consciousness, temperature. Score 0-4 = low risk (routine monitoring). Score 5-6 = medium risk (urgent review). Score ≥7 = high risk (emergency response). CVPU: alert, verbal, pain, unresponsive.",
    },
    {
        "id": "condition_001",
        "source": "Clinical Guidelines — NICE",
        "category": "conditions",
        "content": "Chronic Kidney Disease staging: G1 eGFR≥90 (normal/high), G2 60-89 (mildly decreased), G3a 45-59 (mildly-moderately decreased), G3b 30-44, G4 15-29 (severely decreased), G5 <15 (kidney failure). ACR (albumin:creatinine ratio): A1 <3 mg/mmol normal, A2 3-30 moderately increased, A3 >30 severely increased. Risk of progression = GFR + ACR combined.",
    },
    {
        "id": "condition_002",
        "source": "Clinical Guidelines — NICE",
        "category": "conditions",
        "content": "COPD management by GOLD stage: GOLD 1 (FEV1≥80%): SABA PRN. GOLD 2 (50-79%): LAMA or LABA. GOLD 3 (30-49%): LAMA + LABA. GOLD 4 (<30%): LAMA + LABA + ICS. Pulmonary rehabilitation for MRC grade ≥3. Smoking cessation is the single most effective intervention. Annual influenza vaccine, pneumococcal vaccine.",
    },
    {
        "id": "emergency_001",
        "source": "Emergency Protocols",
        "category": "emergency",
        "content": "Anaphylaxis management: (1) Remove trigger, call for help. (2) Adrenaline (epinephrine) 0.5mg IM (1:1000) to anterolateral thigh — most important step. Repeat every 5 min if no improvement. (3) High-flow oxygen 15L/min. (4) IV access + fluid resuscitation. (5) Antihistamine (chlorphenamine) + corticosteroid (hydrocortisone) — secondary only. Monitor for biphasic reaction for 6-12 hours.",
    },
]


class RAGService:
    """
    Retrieval-Augmented Generation service.
    Embeds medical knowledge into a vector store and retrieves
    the most relevant chunks for any clinical query.
    """

    def __init__(self):
        self.client = None
        self.collection = None
        self.use_chroma = CHROMA_AVAILABLE
        # Fallback: simple keyword index when ChromaDB not installed
        self._keyword_index: List[Dict] = []

    async def initialize(self):
        if self.use_chroma:
            self._init_chroma()
        else:
            print("⚠️  ChromaDB not found — using keyword fallback search")
            self._keyword_index = MEDICAL_KNOWLEDGE
        print(f"📚 RAG service ready — {len(MEDICAL_KNOWLEDGE)} knowledge chunks loaded")

    def _init_chroma(self):
        self.client = chromadb.Client()
        ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.create_collection(
            name="medical_knowledge",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection.add(
            ids=[doc["id"] for doc in MEDICAL_KNOWLEDGE],
            documents=[doc["content"] for doc in MEDICAL_KNOWLEDGE],
            metadatas=[{"source": doc["source"], "category": doc["category"]} for doc in MEDICAL_KNOWLEDGE],
        )

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Return top_k most relevant knowledge chunks for a query."""
        if self.use_chroma and self.collection:
            results = self.collection.query(query_texts=[query], n_results=top_k)
            chunks = []
            for i, doc in enumerate(results["documents"][0]):
                chunks.append({
                    "content": doc,
                    "source": results["metadatas"][0][i]["source"],
                    "category": results["metadatas"][0][i]["category"],
                    "score": round(1 - results["distances"][0][i], 3),
                })
            return chunks
        else:
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based fallback retrieval."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self._keyword_index:
            doc_words = set(doc["content"].lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored.append({
                    "content": doc["content"],
                    "source": doc["source"],
                    "category": doc["category"],
                    "score": round(overlap / max(len(query_words), 1), 3),
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k] if scored else [self._keyword_index[0]]
