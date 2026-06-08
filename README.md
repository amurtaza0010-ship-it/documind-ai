# DocuMind AI v2 — Advanced Multi-PDF Chatbot

A production-ready RAG chatbot with **Hybrid Search + Cross-Encoder Reranker** for significantly improved accuracy.

## What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| Search | FAISS semantic only | **BM25 + FAISS Hybrid** |
| Reranking | None | **Cross-Encoder (ms-marco-MiniLM-L-6-v2)** |
| Chunk size | 1000 tokens | **900 tokens** |
| Retrieval k | 6 | **8 → reranked to 4** |
| LLM Provider | OpenRouter (paid credits) | **Groq (free & ultra-fast)** |
| Context history | 6 turns | **6 turns** |

## Architecture

```
PDF Upload
  → PyMuPDF + pdfplumber extraction
  → RecursiveCharacterTextSplitter (900 / 200 overlap)
  → BAAI/bge-base-en-v1.5 embeddings
  → FAISS index (persisted)
  → BM25 index (persisted)

Query
  → BM25 keyword search (k=8)
  → FAISS MMR semantic search (k=8)
  → Merge & deduplicate
  → CrossEncoder reranker → top 4
  → Groq LLM (streaming, ultra-fast)
  → Streamlit frontend
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and set your GROQ_API_KEY
# Get a free key at: https://console.groq.com
```

### 3. Start backend

```bash
python -m backend.main
# or
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start frontend (new terminal)

```bash
streamlit run app.py
```

Open: http://localhost:8501

## Available Models (Groq — All Free)

| Model | Context | Notes |
|-------|---------|-------|
| llama-3.1-8b-instant | 128k | **Recommended default — fastest** |
| llama3-8b-8192 | 8k | Fast and reliable |
| gemma2-9b-it | 8k | Good quality |
| mixtral-8x7b-32768 | 32k | Strong reasoning, larger context |

## Project Structure

```
documind_v2/
├── backend/
│   ├── config/
│   │   └── settings.py          # All configuration
│   ├── rag/
│   │   └── pipeline.py          # Hybrid search + reranker
│   ├── services/
│   │   ├── pdf_processor.py     # Dual PDF extraction
│   │   └── llm_service.py       # Groq streaming
│   └── main.py                  # FastAPI app
├── app.py                       # Streamlit frontend
├── requirements.txt
├── .env.example
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Status check |
| POST | /documents/upload | Upload & index PDFs |
| GET | /documents | List indexed documents |
| DELETE | /documents/{filename} | Remove document |
| POST | /documents/{filename}/summarize | Summarize document |
| POST | /sessions | Create chat session |
| GET | /sessions/{id}/history | Get chat history |
| DELETE | /sessions/{id} | Clear session |
| POST | /chat/stream | Streaming chat (SSE) |
| GET | /models | List available models |
| POST | /models/switch | Switch active model |