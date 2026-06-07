"""
RAG Pipeline v3 — Stable + Fast
--------------------------------
Features:
1. Hybrid Search (BM25 + FAISS)
2. Better chunking
3. Faster retrieval
4. Stable indexing
5. Persistent vector DB
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from backend.config.settings import settings
from backend.services.pdf_processor import PDFDocument

logger = logging.getLogger(__name__)

# =========================================================
# PATHS
# =========================================================

FAISS_PATH = os.path.join(settings.vectorstore_dir, "faiss_index")
CHUNKS_PATH = os.path.join(settings.vectorstore_dir, "chunks.pkl")
META_PATH = os.path.join(settings.vectorstore_dir, "meta.pkl")


# =========================================================
# RAG PIPELINE
# =========================================================

class RAGPipeline:

    def __init__(self):

        logger.info("Loading embedding model...")

        # STABLE EMBEDDINGS
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # BETTER SPLITTING
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.vectorstore: Optional[FAISS] = None
        self.all_chunks: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self.doc_meta: Dict[str, dict] = {}

        self._load()

    # =====================================================
    # INDEX DOCUMENT
    # =====================================================

    def index_document(
        self,
        pdf_doc: PDFDocument,
        file_size: int = 0,
        indexed_at: str = "",
    ) -> dict:

        logger.info("Indexing started: %s", pdf_doc.filename)

        chunks = self._make_chunks(pdf_doc)
        logger.info(
            "Chunks created: %d from %d pages for '%s'",
            len(chunks),
            pdf_doc.total_pages,
            pdf_doc.filename,
        )

        if not chunks:
            raise ValueError(
                f"No text extracted from '{pdf_doc.filename}'"
            )

        # VECTORSTORE
        try:
            if self.vectorstore is None:
                logger.info("Creating new FAISS index at %s", FAISS_PATH)
                self.vectorstore = FAISS.from_documents(
                    chunks,
                    self.embeddings,
                )
            else:
                logger.info("Adding %d chunks to existing FAISS index", len(chunks))
                self.vectorstore.add_documents(chunks)

            logger.info("Embeddings generated and FAISS write successful")

        except Exception as exc:
            logger.error(
                "FAISS indexing failed for '%s': %s",
                pdf_doc.filename,
                exc,
                exc_info=True,
            )
            raise

        # STORE CHUNKS
        self.all_chunks.extend(chunks)

        # BM25
        self._rebuild_bm25()
        logger.info("BM25 index rebuilt — %d total chunks", len(self.all_chunks))

        # META
        self.doc_meta[pdf_doc.filename] = {
            "doc_id": pdf_doc.filename,
            "total_pages": pdf_doc.total_pages,
            "chunks": len(chunks),
            "file_size": file_size,
            "indexed_at": indexed_at,
            "status": "indexed",
        }

        # SAVE
        self._save()
        logger.info(
            "Metadata saved — doc_id=%s, registry size=%d",
            pdf_doc.filename,
            len(self.doc_meta),
        )

        logger.info(
            "Indexed '%s' with %d chunks",
            pdf_doc.filename,
            len(chunks),
        )

        return self.doc_meta[pdf_doc.filename]

    # =====================================================
    # REMOVE DOCUMENT
    # =====================================================

    def remove_document(self, filename: str) -> bool:

        if filename not in self.doc_meta:
            return False

        self.all_chunks = [
            c for c in self.all_chunks
            if c.metadata.get("source") != filename
        ]

        self.doc_meta.pop(filename, None)

        if self.all_chunks:

            self.vectorstore = FAISS.from_documents(
                self.all_chunks,
                self.embeddings
            )

            self._rebuild_bm25()

        else:

            self.vectorstore = None
            self.bm25 = None

        self._save()

        return True

    # =====================================================
    # DOCUMENTS
    # =====================================================

    def get_indexed_documents(self) -> List[dict]:

        return list(self.doc_meta.values())

    def is_empty(self) -> bool:

        return self.vectorstore is None or not self.all_chunks

    # =====================================================
    # RETRIEVE
    # =====================================================

    def retrieve(
        self,
        query: str,
        doc_filter: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:

        if self.is_empty():
            return []

        k = settings.retrieval_k

        # HYBRID SEARCH
        bm25_docs = self._bm25_search(
            query,
            k,
            doc_filter
        )

        faiss_docs = self._faiss_search(
            query,
            k,
            doc_filter
        )

        # DEDUP
        seen = set()
        merged = []

        for doc in bm25_docs + faiss_docs:

            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content[:80],
            )

            if key not in seen:
                seen.add(key)
                merged.append(doc)

        # SIMPLE SCORING
        results = [
            (doc, 1.0)
            for doc in merged[:settings.reranker_top_n]
        ]

        logger.info(
            f"Retrieved {len(results)} chunks"
        )

        return results

    # =====================================================
    # BM25 SEARCH
    # =====================================================

    def _bm25_search(
        self,
        query: str,
        k: int,
        doc_filter: Optional[List[str]]
    ) -> List[Document]:

        if not self.all_chunks:
            return []

        pool = self.all_chunks

        if doc_filter:

            pool = [
                c for c in pool
                if c.metadata.get("source") in doc_filter
            ]

        if not pool:
            return []

        tokenized_pool = [
            c.page_content.lower().split()
            for c in pool
        ]

        local_bm25 = BM25Okapi(tokenized_pool)

        scores = local_bm25.get_scores(
            query.lower().split()
        )

        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return [
            pool[i]
            for i in top_idx
            if scores[i] > 0
        ]

    # =====================================================
    # FAISS SEARCH
    # =====================================================

    def _faiss_search(
        self,
        query: str,
        k: int,
        doc_filter: Optional[List[str]]
    ) -> List[Document]:

        if self.vectorstore is None:
            return []

        try:

            filter_dict = (
                {"source": {"$in": doc_filter}}
                if doc_filter
                else None
            )

            return self.vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=k * 2,
                lambda_mult=0.7,
                filter=filter_dict,
            )

        except Exception as exc:

            logger.warning(
                f"FAISS search failed: {exc}"
            )

            try:

                return self.vectorstore.similarity_search(
                    query,
                    k=k
                )

            except Exception:

                return []

    # =====================================================
    # CHUNKING
    # =====================================================

    def _make_chunks(
        self,
        pdf_doc: PDFDocument
    ) -> List[Document]:

        chunks = []

        for page in pdf_doc.pages:

            if not page.text.strip():
                continue

            splits = self.splitter.split_text(
                page.text
            )

            for split in splits:

                if len(split.strip()) < 30:
                    continue

                chunks.append(
                    Document(
                        page_content=split,
                        metadata={
                            "source": pdf_doc.filename,
                            "page": page.page_num,
                        },
                    )
                )

        return chunks

    # =====================================================
    # BM25
    # =====================================================

    def _rebuild_bm25(self):

        if not self.all_chunks:

            self.bm25 = None

            return

        tokenized = [
            c.page_content.lower().split()
            for c in self.all_chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

    # =====================================================
    # SAVE
    # =====================================================

    def _save(self):

        Path(settings.vectorstore_dir).mkdir(
            exist_ok=True
        )

        if self.vectorstore:
            logger.info("Persisting FAISS index to %s", FAISS_PATH)
            self.vectorstore.save_local(FAISS_PATH)

        logger.info("Persisting %d chunks to %s", len(self.all_chunks), CHUNKS_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.all_chunks, f)

        logger.info("Persisting metadata for %d docs to %s", len(self.doc_meta), META_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.doc_meta, f)

    # =====================================================
    # LOAD
    # =====================================================

    def _load(self):

        try:

            if Path(
                f"{FAISS_PATH}/index.faiss"
            ).exists():

                self.vectorstore = FAISS.load_local(
                    FAISS_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )

                logger.info(
                    "Loaded FAISS index"
                )

            if Path(CHUNKS_PATH).exists():

                with open(CHUNKS_PATH, "rb") as f:

                    self.all_chunks = pickle.load(f)

                self._rebuild_bm25()

                logger.info(
                    f"Loaded {len(self.all_chunks)} chunks"
                )

            if Path(META_PATH).exists():

                with open(META_PATH, "rb") as f:

                    self.doc_meta = pickle.load(f)

        except Exception as exc:

            logger.warning(
                f"Load failed: {exc}"
            )

            self.vectorstore = None
            self.all_chunks = []
            self.bm25 = None
            self.doc_meta = {}