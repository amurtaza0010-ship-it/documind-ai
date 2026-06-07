"""Sidebar component — documents, upload, model, sessions."""

import logging

import streamlit as st

from ui.components.document_card import format_file_size, render_doc_card_html
from ui.utils.api import (
    clear_backend_session,
    create_backend_session,
    delete_document,
    get_health,
    get_models,
    health_check,
    is_upload_success,
    list_documents,
    switch_model,
    upload_document,
)
from ui.utils.state import create_new_session, get_active_messages

logger = logging.getLogger(__name__)

# Upload lifecycle states tracked in session_state.upload_status
UPLOAD_STATE_UPLOADING = "uploading"
UPLOAD_STATE_PROCESSING = "processing"
UPLOAD_STATE_INDEXING = "indexing"
UPLOAD_STATE_COMPLETE = "complete"
UPLOAD_STATE_FAILED = "failed"


def render_sidebar() -> None:
    """Render the full sidebar."""
    with st.sidebar:
        _render_brand()
        _render_new_chat()
        _render_sessions()
        st.divider()
        _render_upload()
        st.divider()
        _render_documents()
        st.divider()
        _render_model_selector()
        st.divider()
        _render_clear_chat()

        if not health_check():
            st.warning("Backend offline — start the API server on port 8000")
        else:
            health = get_health() or {}
            if not health.get("faiss_available", True):
                st.error(
                    "FAISS not installed in backend environment. "
                    "Run `pip install faiss-cpu` in the backend venv, then restart the API."
                )


def _render_brand() -> None:
    st.markdown(
        """
        <div class="dm-sidebar-brand">
            <div class="dm-sidebar-brand-icon">📄</div>
            <div class="dm-sidebar-brand-text">DocuMind</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_new_chat() -> None:
    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        sid = create_new_session()
        create_backend_session(f"session-{sid}")
        st.rerun()


def _render_sessions() -> None:
    sessions = st.session_state.sessions
    if len(sessions) <= 1:
        return

    st.markdown('<p class="dm-sidebar-section-title">Recent Chats</p>', unsafe_allow_html=True)
    for sid, data in reversed(list(sessions.items())):
        label = data.get("title", "Chat")
        is_active = sid == st.session_state.active_session_id
        prefix = "→ " if is_active else ""
        if st.button(
            f"{prefix}{label}",
            key=f"session_{sid}",
            use_container_width=True,
        ):
            st.session_state.active_session_id = sid
            st.rerun()


def _render_upload_flash() -> None:
    """Show one-shot upload result messages, then clear to prevent stale errors."""
    for flash in st.session_state.get("upload_flash", []):
        if flash.get("type") == "success":
            st.success(flash.get("message", ""))
        elif flash.get("type") == "error":
            st.error(flash.get("message", ""))
    st.session_state.upload_flash = []


def _render_upload() -> None:
    """
    Upload + index flow.

    Streamlit pitfall: ``if uploaded_files and st.button(...)`` drops the click
    when ``uploaded_files`` is empty on the button-triggered rerun. We persist
    file bytes in session state and always render the button separately.
    """
    st.markdown('<p class="dm-sidebar-section-title">Upload Documents</p>', unsafe_allow_html=True)

    _render_upload_flash()

    uploaded_files = st.file_uploader(
        "Drop PDF files here or click to browse",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    # Persist selected files whenever the uploader has content
    if uploaded_files:
        st.session_state.pending_index_files = {
            f.name: f.getvalue() for f in uploaded_files
        }
        names = ", ".join(st.session_state.pending_index_files.keys())
        st.caption(f"Ready to index: {names}")

    pending = st.session_state.get("pending_index_files", {})
    process_disabled = not pending

    if st.button(
        "Process & Index",
        use_container_width=True,
        type="primary",
        disabled=process_disabled,
        key="process_index_btn",
    ):
        _process_pending_uploads(pending)


def _process_pending_uploads(pending: dict[str, bytes]) -> None:
    """Send pending PDFs to the backend indexing API."""
    if not pending:
        st.warning("Select PDF files first, then click Process & Index.")
        return

    # Clear stale flash messages and error states from previous upload batches
    st.session_state.upload_flash = []
    for name in pending:
        st.session_state.upload_status.pop(name, None)

    logger.info("Process & Index clicked — %d file(s): %s", len(pending), list(pending))
    total = len(pending)
    progress = st.progress(0, text="Uploading…")

    any_success = False
    for i, (name, file_data) in enumerate(pending.items()):
        # Phase 1: uploading
        st.session_state.upload_status[name] = UPLOAD_STATE_UPLOADING
        progress.progress(
            (i + 0.25) / total,
            text=f"Uploading {name}…",
        )
        logger.info("Upload start: %s", name)

        # Phase 2: processing (PDF extraction on backend)
        st.session_state.upload_status[name] = UPLOAD_STATE_PROCESSING
        progress.progress(
            (i + 0.5) / total,
            text=f"Processing {name}…",
        )
        logger.info("Processing start: %s", name)

        # Phase 3: indexing (chunking + embedding + FAISS)
        st.session_state.upload_status[name] = UPLOAD_STATE_INDEXING
        progress.progress(
            (i + 0.75) / total,
            text=f"Indexing {name}…",
        )
        logger.info("Indexing start: %s", name)

        result = upload_document(name, file_data)

        if is_upload_success(result):
            any_success = True
            st.session_state.upload_status[name] = UPLOAD_STATE_COMPLETE
            msg = (
                f"✓ {name} — {result.get('chunks', '?')} chunks, "
                f"{result.get('total_pages', '?')} pages"
            )
            st.session_state.upload_flash.append({"type": "success", "message": msg})
            logger.info("Indexing success: %s", name)
        else:
            st.session_state.upload_status[name] = UPLOAD_STATE_FAILED
            msg = result.get("message") or "Upload failed"
            st.session_state.upload_flash.append({
                "type": "error",
                "message": f"✗ {name}: {msg}",
            })
            logger.error("Indexing failed for %s: %s", name, msg)

    progress.progress(1.0, text="Complete")

    if any_success:
        st.session_state.pending_index_files = {}
        # Rerun so document list refreshes; flash messages shown on next render
        st.rerun()


def _render_documents() -> None:
    st.markdown('<p class="dm-sidebar-section-title">Indexed Documents</p>', unsafe_allow_html=True)

    search = st.text_input(
        "Search documents",
        placeholder="Filter by name…",
        key="doc_search_input",
        label_visibility="collapsed",
    )

    docs = list_documents()
    logger.debug("Indexed documents from API: %d", len(docs))

    if search:
        docs = [d for d in docs if search.lower() in d.get("doc_id", "").lower()]

    if not docs:
        st.markdown(
            '<div class="dm-empty-state">No documents indexed yet.<br>Upload a PDF to get started.</div>',
            unsafe_allow_html=True,
        )
        return

    for doc in docs:
        doc_id = doc.get("doc_id", "Unknown")
        st.markdown(render_doc_card_html(doc), unsafe_allow_html=True)

        if doc.get("file_size"):
            st.caption(format_file_size(doc["file_size"]))
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🗑", key=f"del_{doc_id}", help=f"Delete {doc_id}"):
                delete_document(doc_id)
                st.session_state.upload_status.pop(doc_id, None)
                st.rerun()


def _render_model_selector() -> None:
    st.markdown('<p class="dm-sidebar-section-title">AI Model</p>', unsafe_allow_html=True)

    models, current = get_models()
    if not models:
        st.caption("Models unavailable")
        return

    model_ids = [m["id"] for m in models]
    model_names = [f"{m['name']} ({m.get('provider', '')})" for m in models]
    default_idx = model_ids.index(current) if current in model_ids else 0

    selected_idx = st.selectbox(
        "Model",
        range(len(model_ids)),
        format_func=lambda i: model_names[i],
        index=default_idx,
        label_visibility="collapsed",
    )

    if st.button("Apply Model", use_container_width=True):
        if switch_model(model_ids[selected_idx]):
            st.toast(f"Switched to {model_names[selected_idx]}", icon="🤖")
        else:
            st.error("Failed to switch model")


def _render_clear_chat() -> None:
    if st.button("Clear Chat", use_container_width=True):
        sid = st.session_state.active_session_id
        messages = get_active_messages()
        messages.clear()
        try:
            clear_backend_session(f"session-{sid}")
        except Exception as exc:
            logger.warning("Failed to clear backend session %s: %s", sid, exc)
        st.rerun()
