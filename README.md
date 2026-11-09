# 🎯 Agentic RAG Support Desk

> An AI customer-support ticket resolution platform: tickets are classified, answered from a grounded knowledge base (RAG), and drafted by an agentic LangGraph workflow — with a human approval gate before anything is finalised.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-workflow-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![FAISS](https://img.shields.io/badge/FAISS-retrieval-00A8E8)](https://github.com/facebookresearch/faiss)

---

## ✨ Features

- 🧭 **Automatic triage** — every incoming ticket is classified (billing, account access, subscription, technical issue, order status, general)
- 📚 **Grounded drafting (RAG)** — responses are drafted from retrieved help-centre articles and carry their source citations, instead of being generated from thin air
- 🤖 **Agentic workflow** — a LangGraph state graph runs classify → retrieve → draft as separate, inspectable nodes
- ✋ **Human-in-the-loop approval** — the workflow pauses at an approval gate; a reviewer must approve or reject the draft before it is finalised. Nothing is "sent" unreviewed
- 🔌 **Provider interfaces** — the LLM and embedder sit behind small interfaces, so the whole system runs offline with deterministic fakes for tests and demos, and can be pointed at a real provider with one environment variable
- 🧪 **Offline evaluation** — a small sample ticket set scores drafts on category accuracy, citation correctness and expected key points

---

## 🏗️ Architecture

```
POST /tickets
     │
     ▼
┌─────────────┐   ┌──────────────┐   ┌──────────────┐
│  classify   │──▶│   retrieve   │──▶│    draft     │
│  (LLM node) │   │ (FAISS RAG)  │   │  (LLM node)  │
└─────────────┘   └──────────────┘   └──────────────┘
                                            │
                                            ▼
                                  ⛔ HUMAN APPROVAL GATE
                                  (graph interrupt — a reviewer
                                   approves or rejects the draft)
                                            │ approve
                                            ▼
                                     ┌────────────┐
                                     │  finalise  │
                                     └────────────┘
```

The graph is compiled with LangGraph's `interrupt_before` and a checkpointer: submitting a ticket runs the first three nodes and pauses. `POST /tickets/{id}/approve` resumes the same run to finalise; `POST /tickets/{id}/reject` abandons it and records the reason.

**Retrieval:** help-centre Markdown articles are split into paragraph-aware chunks, embedded, and indexed with FAISS (cosine similarity via inner product over normalised vectors). If FAISS is unavailable, an equivalent NumPy search is used so the app never hard-fails on the optional dependency.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.12, FastAPI, Pydantic, Uvicorn
- **Agent workflow:** LangGraph (state graph + checkpointed interrupt), LangChain provider wrappers (optional)
- **Retrieval:** FAISS, Markdown knowledge base, pluggable embedders
- **Testing / eval:** pytest, FastAPI TestClient, a rubric-based sample-set evaluation
- **Packaging:** Docker (python:3.12-slim), python-dotenv configuration

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Vanoushika/agentic-rag-support-desk.git
cd agentic-rag-support-desk

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No API key is needed: with no key configured, the app uses deterministic offline providers (a rule-based classifier/drafter and a hashing embedder). To use a real LLM/embedder instead, copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

### 2. Run the API

```bash
uvicorn app.main:app --reload
# → http://localhost:8000  (interactive docs at /docs)
```

### 3. Try the flow

```bash
# Submit a ticket — it comes back drafted, waiting for approval
curl -X POST http://localhost:8000/tickets \
  -H 'Content-Type: application/json' \
  -d '{"subject": "Charged twice this month",
       "body": "I was charged twice for my subscription and I want a refund."}'

# Approve the draft (use the id from the response)
curl -X POST http://localhost:8000/tickets/<id>/approve

# …or reject it with a reason
curl -X POST http://localhost:8000/tickets/<id>/reject \
  -H 'Content-Type: application/json' -d '{"reason": "needs a human agent"}'
```

### 4. Run the tests and the evaluation

```bash
pytest            # 18 tests, fully offline
python eval/run_eval.py
python scripts/build_index.py   # optionally persist the FAISS index to index/
```

### 5. Or run with Docker

```bash
docker build -t agentic-rag-support-desk .
docker run -p 8000:8000 agentic-rag-support-desk
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| GET | `/health` | Service health check → `{"status": "ok"}` |
| POST | `/tickets` | Submit a ticket; runs classify → retrieve → draft and returns it at the approval gate (`201`) |
| GET | `/tickets/{ticket_id}` | Fetch one ticket with its draft, status and citations |
| POST | `/tickets/{ticket_id}/approve` | Approve the draft and finalise the response |
| POST | `/tickets/{ticket_id}/reject` | Reject the draft (optional `{"reason": "..."}`) |

Ticket statuses: `awaiting_approval` → `finalised` (approved) or `rejected`.

---

## 📁 Project Structure

```
agentic-rag-support-desk/
├── app/
│   ├── main.py          # FastAPI app and endpoints
│   ├── pipeline.py      # SupportDesk: store + workflow orchestration
│   ├── graph.py         # LangGraph workflow (approval gate via interrupt)
│   ├── kb.py            # Chunking, FAISS index, retrieval
│   ├── llm.py           # LLM interface + deterministic offline provider
│   └── embeddings.py    # Embedder interface + hashing embedder
├── docs/                # Knowledge base: 6 help-centre articles (Markdown)
├── scripts/
│   └── build_index.py   # Persist the FAISS index + chunks to disk
├── eval/
│   ├── sample_tickets.json  # 6 sample tickets with expected outcomes
│   └── run_eval.py          # Rubric scorer (category / citation / key points)
├── tests/               # pytest suite — runs fully offline
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🧪 Evaluation

`eval/run_eval.py` runs the full workflow on six hand-written sample tickets and checks each draft against a simple rubric: correct category, the expected help-centre article among the citations, and the expected key points present in the draft.

Latest run on the sample set (offline providers):

```
Summary over 6 sample tickets:
  category correct : 6/6
  citation correct : 6/6
  key points found : 6/6
  all checks passed: 6/6
```

These are smoke-check results on a tiny sample set, included to show the evaluation pattern — rubric-based scoring of grounded drafts — not production quality figures.

---

## 💡 Why I built this

My day job is building production Generative AI systems — RAG pipelines and LangGraph agentic workflows for support-style use cases, where a wrong answer is worse than no answer. This project distils the patterns that matter most in that work into a small, fully testable system:

- **Grounding before generation** — drafts cite the articles they came from, so a reviewer can verify them in seconds.
- **Humans stay in control** — the approval gate is a hard interrupt in the workflow, not a UI afterthought.
- **Evaluation as a habit** — even a six-ticket rubric catches regressions in classification and retrieval early.
- **Interfaces over lock-in** — swapping the offline providers for a hosted LLM takes one environment variable, because the workflow only depends on two small interfaces.

---

## 📄 License

MIT
