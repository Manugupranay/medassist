# MedAssist AI — Clinical Decision Support System

A production-grade healthcare AI bot solving real clinical workflow problems using **RAG + LLM + ML + MCP**.

## What It Solves

Doctors and clinic staff waste hours on:
- Looking up drug interactions manually → **RAG answers in seconds with cited sources**
- Risk-assessing patients from paper vitals → **ML model scores instantly**
- Writing SOAP notes and referral letters → **Claude generates in 10 seconds**
- Booking appointments and emailing patients → **MCP does it automatically**

---

## Architecture

```
medassist/
├── backend/
│   ├── main.py                    # FastAPI app + lifespan
│   ├── requirements.txt
│   ├── .env.example
│   ├── routers/
│   │   ├── chat.py                # POST /api/chat/ask
│   │   ├── risk.py                # POST /api/risk/score
│   │   ├── appointments.py        # POST /api/appointments/book
│   │   └── notes.py               # POST /api/notes/soap|referral
│   ├── services/
│   │   ├── rag_service.py         # ChromaDB + keyword fallback
│   │   ├── llm_service.py         # Claude Sonnet API
│   │   ├── ml_service.py          # scikit-learn GBM classifier
│   │   └── mcp_service.py         # Google Calendar + Gmail via MCP
│   └── models/
│       └── risk_model.pkl         # Auto-generated on first run
└── frontend/
    └── index.html                 # Complete React-style single-page app
```

---

## AI Stack

| Module | Technology | What it does |
|--------|-----------|--------------|
| **RAG** | ChromaDB + sentence-transformers | Embeds 15 medical knowledge chunks, retrieves top-4 relevant passages for any clinical query |
| **LLM** | Claude Sonnet (Anthropic API) | Reasons over retrieved context, generates clinical notes, explains risk scores |
| **ML** | scikit-learn GBM + NEWS2 | Trained on 2,000 synthetic patient vitals, classifies risk as Low/Medium/High with probability scores |
| **MCP** | Google Calendar + Gmail via Anthropic MCP | Books appointment events, sends patient confirmation emails |

---

## Setup

### 1. Clone and configure

```bash
cd backend
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run backend

```bash
uvicorn main:app --reload --port 8000
```

The ML model trains automatically on first run (~5 seconds).
ChromaDB indexes medical knowledge on startup.

### 4. Open frontend

Open `frontend/index.html` in any browser.
Or serve it:
```bash
cd frontend && python -m http.server 3000
```

### 5. Enable MCP (Google Calendar + Gmail)

In Claude.ai Settings → Integrations → connect Google account.
The MCP servers (`gcal.mcp.claude.com`, `gmail.mcp.claude.com`) are then available via the Anthropic API.

---

## API Endpoints

### Clinical Q&A (RAG + LLM)
```
POST /api/chat/ask
{
  "query": "What is the interaction between warfarin and ibuprofen?",
  "session_id": "optional_session"
}
```

### Patient Risk Scoring (ML)
```
POST /api/risk/score
{
  "age": 72, "systolic_bp": 82, "diastolic_bp": 50,
  "heart_rate": 128, "spo2": 88.0, "temperature": 38.9,
  "respiratory_rate": 28, "blood_glucose": 18.5, "consciousness": 1
}
```

### Book Appointment (MCP)
```
POST /api/appointments/book
{
  "patient_name": "John Smith",
  "patient_email": "john@email.com",
  "doctor_name": "Dr. Patel",
  "type": "Follow-up",
  "date": "2025-04-10",
  "time": "10:30",
  "duration": 30
}
```

### Generate SOAP Note (LLM)
```
POST /api/notes/soap
{
  "name": "John Smith", "age": 52,
  "chief_complaint": "Chest pain 2 hours",
  "history": "...", "bp": "138/86", ...
}
```

### Generate Referral Letter (LLM)
```
POST /api/notes/referral
{
  "patient_name": "John Smith",
  "referring_doctor": "Dr. Patel",
  "referral_to": "Cardiology",
  "reason": "...", "urgency": "Urgent (72 hours)"
}
```

---

## Extending the Knowledge Base

Add to `MEDICAL_KNOWLEDGE` in `services/rag_service.py`:

```python
{
    "id": "drug_006",
    "source": "Your Source",
    "category": "drug_interactions",
    "content": "Clinical content here...",
}
```

For production, ingest real PDFs:
```bash
pip install pypdf sentence-transformers
# Then use PyPDFLoader → chunk → embed → add to ChromaDB
```

---

## Production Upgrades

- **Database**: Replace in-memory ChromaDB with hosted Chroma or Pinecone
- **Auth**: Add JWT authentication (FastAPI-Users)
- **PDF Ingestion**: Load real clinical PDFs into the vector store
- **Streaming**: Add SSE streaming for real-time LLM responses
- **Deploy**: Docker → Railway / Render / AWS ECS

---

## Disclaimer

MedAssist AI is a **clinical decision support tool** for qualified healthcare professionals. It does not replace clinical judgement. All AI outputs must be verified against current guidelines before clinical use.
