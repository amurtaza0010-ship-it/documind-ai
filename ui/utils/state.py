"""Session state initialization."""

import uuid

import streamlit as st


def init_session_state() -> None:
    """Initialize all session state variables."""
    if "sessions" not in st.session_state:
        sid = str(uuid.uuid4())[:8]
        st.session_state.sessions = {
            sid: {"title": "New chat", "messages": []},
        }
        st.session_state.active_session_id = sid

    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = next(iter(st.session_state.sessions))

    if "doc_search" not in st.session_state:
        st.session_state.doc_search = ""

    if "upload_status" not in st.session_state:
        st.session_state.upload_status = {}

    # One-shot flash messages for upload results (survives st.rerun after success)
    if "upload_flash" not in st.session_state:
        st.session_state.upload_flash = []

    if "regenerate_query" not in st.session_state:
        st.session_state.regenerate_query = None

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if "pending_index_files" not in st.session_state:
        st.session_state.pending_index_files = {}


def get_active_messages() -> list:
    """Return messages for the active chat session."""
    sid = st.session_state.active_session_id
    return st.session_state.sessions[sid]["messages"]


def set_active_messages(messages: list) -> None:
    """Update messages for the active chat session."""
    sid = st.session_state.active_session_id
    st.session_state.sessions[sid]["messages"] = messages


def create_new_session() -> str:
    """Create a new chat session and switch to it."""
    sid = str(uuid.uuid4())[:8]
    st.session_state.sessions[sid] = {"title": "New chat", "messages": []}
    st.session_state.active_session_id = sid
    return sid


def update_session_title(session_id: str, first_message: str) -> None:
    """Set session title from the first user message."""
    title = first_message[:40] + ("…" if len(first_message) > 40 else "")
    st.session_state.sessions[session_id]["title"] = title
