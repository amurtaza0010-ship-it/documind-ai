# DocuMind AI v2 — Advanced Multi-PDF Chatbot

A production-ready RAG chatbot with **Hybrid Search + Cross-Encoder Reranker** for significantly improved accuracy.

## What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| Search | FAISS semantic only | **BM25 + FAISS Hybrid** |
| Reranking | None | **Cross-Encoder (ms-marco-MiniLM-L-6-v2)** |
| Chunk size | 1000 tokens | **1500 tokens** |
| Retrieval k | 6 | **10 → reranked to 5** |
| LLM models | Mixed (some paid) | **Free models only** |
| Context history | 6 turns | **8 turns** |

## Architecture

```
PDF Upload
  → PyMuPDF + pdfplumber extraction
  → RecursiveCharacterTextSplitter (1500 / 300 overlap)
  → BAAI/bge-small-en-v1.5 embeddings
  → FAISS index (persisted)
  → BM25 index (persisted)

Query
  → BM25 keyword search (k=10)
  → FAISS MMR semantic search (k=10)
  → Merge & deduplicate
  → CrossEncoder reranker → top 5
  → OpenRouter LLM (streaming)
  → Streamlit frontend
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and set your OPENROUTER_API_KEY
# Get a free key at: https://openrouter.ai/keys
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

## Free Models Available

| Model | Provider | Context | Notes |
|-------|----------|---------|-------|
| deepseek/deepseek-chat-v3-0324:free | DeepSeek | 128k | **Recommended default** |
| meta-llama/llama-3.3-70b-instruct:free | Meta | 128k | Very capable |
| google/gemini-2.0-flash-exp:free | Google | 1M | Best for long docs |
| mistralai/mistral-7b-instruct:free | Mistral | 32k | Lightweight |
| qwen/qwen3-8b:free | Alibaba | 32k | Multilingual |

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
│   │   └── llm_service.py       # OpenRouter streaming
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
