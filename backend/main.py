"""
DocuMind AI v2 — FastAPI Backend
"""
import json
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from shared.sse_codec import encode_sse_payload
from shared.stream_assembler import StreamAssembler
from backend.config.settings import settings
from backend.services.pdf_processor import PDFProcessor
from backend.services.llm_service import LLMService
from backend.rag.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global services ───────────────────────────────────────────────────────────
pdf_processor = PDFProcessor()
rag_pipeline  = RAGPipeline()
llm_service   = LLMService()
chat_sessions: dict = {}


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in [settings.upload_dir, settings.vectorstore_dir, "logs"]:
        Path(d).mkdir(exist_ok=True)

    # Verify FAISS is available before accepting indexing requests
    try:
        import faiss  # noqa: F401
        logger.info("FAISS available — vector indexing enabled")
    except ImportError:
        logger.error(
            "FAISS not installed. Indexing will fail. "
            "Run: pip install faiss-cpu"
        )

    logger.info(
        "Vector store path: %s | Upload path: %s | Indexed docs: %d",
        settings.vectorstore_dir,
        settings.upload_dir,
        len(rag_pipeline.get_indexed_documents()),
    )
    logger.info("✅ DocuMind AI v2 started")
    yield
    logger.info("🛑 DocuMind AI v2 shutdown")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocuMind AI",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    query: str
    model: Optional[str] = None
    doc_filter: Optional[List[str]] = None


class SessionCreate(BaseModel):
    session_id: Optional[str] = None


class ModelSwitch(BaseModel):
    model_id: str


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    faiss_ok = True
    try:
        import faiss  # noqa: F401
    except ImportError:
        faiss_ok = False

    return {
        "status":          "ok",
        "version":         settings.app_version,
        "indexed_docs":    len(rag_pipeline.get_indexed_documents()),
        "active_sessions": len(chat_sessions),
        "faiss_available": faiss_ok,
        "vectorstore_dir": settings.vectorstore_dir,
    }


# ── Upload ────────────────────────────────────────────────────────────────────
@app.post("/documents/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    results = []
    logger.info("Upload request received — %d file(s)", len(files))

    for file in files:
        logger.info("Processing upload: %s", file.filename)

        if not file.filename.lower().endswith(".pdf"):
            results.append({"filename": file.filename, "status": "error",
                            "message": "Only PDF files are supported"})
            continue

        save_path = Path(settings.upload_dir) / file.filename
        size = 0

        try:
            with open(save_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_file_size_mb * 1024 * 1024:
                        raise ValueError(f"File exceeds {settings.max_file_size_mb} MB limit")
                    f.write(chunk)

            logger.info("File saved: %s (%d bytes)", save_path, size)

            pdf_doc = pdf_processor.extract(str(save_path), file.filename)
            logger.info(
                "PDF parsed: %s — %d pages, %d chars",
                file.filename,
                pdf_doc.total_pages,
                len(pdf_doc.full_text),
            )

            doc_info = rag_pipeline.index_document(
                pdf_doc,
                file_size=size,
                indexed_at=datetime.now(timezone.utc).isoformat(),
            )
            logger.info(
                "Indexing complete: %s — %d chunks, metadata saved",
                file.filename,
                doc_info.get("chunks"),
            )
            # Root-cause fix: doc_info contains status="indexed" for document metadata.
            # Spreading doc_info AFTER status="success" was overwriting the upload result
            # status, causing the UI to show "Upload failed" on successful indexing.
            results.append({
                "filename": file.filename,
                "status": "success",
                "doc_status": doc_info.get("status", "indexed"),
                "doc_id": doc_info.get("doc_id"),
                "total_pages": doc_info.get("total_pages"),
                "chunks": doc_info.get("chunks"),
                "file_size": doc_info.get("file_size"),
                "indexed_at": doc_info.get("indexed_at"),
            })

        except Exception as exc:
            logger.error(
                "Upload/index error — %s: %s\n%s",
                file.filename,
                exc,
                traceback.format_exc(),
            )
            if save_path.exists():
                save_path.unlink()
            results.append({"filename": file.filename, "status": "error", "message": str(exc)})

    logger.info("Upload batch finished — %s", results)
    return {"results": results}


# ── Documents ─────────────────────────────────────────────────────────────────
@app.get("/documents")
async def list_documents():
    return {"documents": rag_pipeline.get_indexed_documents()}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    removed   = rag_pipeline.remove_document(filename)
    file_path = Path(settings.upload_dir) / filename
    if file_path.exists():
        file_path.unlink()
    if removed:
        return {"status": "success", "message": f"'{filename}' removed"}
    raise HTTPException(404, f"Document '{filename}' not found")


@app.post("/documents/{filename}/summarize")
async def summarize_document(filename: str, model: Optional[str] = None):
    file_path = Path(settings.upload_dir) / filename
    if not file_path.exists():
        raise HTTPException(404, "Document not found")

    pdf_doc   = pdf_processor.extract(str(file_path), filename)
    full_text = "\n".join(p.text for p in pdf_doc.pages[:6])
    summary   = await llm_service.generate_summary(full_text, filename, model)
    return {"filename": filename, "summary": summary}


# ── Sessions ──────────────────────────────────────────────────────────────────
@app.post("/sessions")
async def create_session(body: SessionCreate):
    sid = body.session_id or str(uuid.uuid4())
    if sid not in chat_sessions:
        chat_sessions[sid] = []
    return {"session_id": sid}


@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    return {"history": chat_sessions.get(session_id, [])}


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    chat_sessions[session_id] = []
    return {"status": "cleared"}


# ── Chat stream ───────────────────────────────────────────────────────────────
CASUAL = {
    "hi":          "Hello! 👋 I'm DocuMind AI. Upload a PDF and ask me anything about it.",
    "hello":       "Hey! 👋 Ready to help you explore your documents.",
    "hey":         "Hi there! 😊 What would you like to know?",
    "how are you": "All systems running! 🚀 How can I help you today?",
    "thanks":      "You're welcome! 😊",
    "thank you":   "Happy to help! 🚀",
    "bye":         "Goodbye! 👋 Come back anytime.",
}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):

    q = req.query.lower().strip()

    # Casual greeting handler
    if q in CASUAL:
        async def _casual():
            yield encode_sse_payload(CASUAL[q])
            yield encode_sse_payload("[DONE]")
        return StreamingResponse(_casual(), media_type="text/event-stream")

    if rag_pipeline.is_empty():
        raise HTTPException(400, "No documents indexed. Please upload a PDF first.")

    history = chat_sessions.get(req.session_id, [])

    async def generate():
        retrieved = rag_pipeline.retrieve(
            query=req.query,
            doc_filter=req.doc_filter,
        )

        # Collect sources
        sources: dict = {}
        for doc, _ in retrieved:
            src  = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            sources.setdefault(src, set()).add(page)

        # Stream tokens — preserve whitespace verbatim
        assembler = StreamAssembler()
        chunk_idx = 0
        async for token in llm_service.stream_response(
            query=req.query,
            retrieved_docs=retrieved,
            chat_history=history,
            model=req.model,
        ):
            if chunk_idx < 50:
                logger.info("LLM chunk[%d]: %r", chunk_idx, token)
                chunk_idx += 1
            assembler.append(token)
            yield encode_sse_payload(token)

        full_response = assembler.finalize()

        # Persist history (keep last 20 turns)
        history.extend([
            {"role": "user",      "content": req.query},
            {"role": "assistant", "content": full_response},
        ])
        chat_sessions[req.session_id] = history[-20:]

        # Send sources metadata
        yield encode_sse_payload(
            f"[SOURCES]{json.dumps({'sources': {k: list(v) for k, v in sources.items()}})}"
        )
        yield encode_sse_payload("[DONE]")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Models ────────────────────────────────────────────────────────────────────
@app.get("/models")
async def get_models():
    return {"models": llm_service.get_available_models(), "current": llm_service.model}


@app.post("/models/switch")
async def switch_model(body: ModelSwitch):
    llm_service.set_model(body.model_id)
    return {"status": "ok", "model": body.model_id}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
