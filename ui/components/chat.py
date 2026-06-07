"""Main chat area component."""

import streamlit as st

from ui.components.message import render_chat_history, render_sources
from ui.components.welcome import render_welcome
from ui.utils.api import list_documents
from ui.utils.state import (
    get_active_messages,
    set_active_messages,
    update_session_title,
)
from ui.utils.stream import stream_chat


def render_chat() -> None:
    """Render the chat interface."""
    st.markdown('<div class="dm-chat-area">', unsafe_allow_html=True)

    docs = list_documents()
    has_documents = len(docs) > 0
    messages = get_active_messages()

    if not messages:
        render_welcome(has_documents)

    render_chat_history()

    st.markdown("</div>", unsafe_allow_html=True)

    # Chat input MUST stay outside width-constraining wrappers.
    # Streamlit pins it to the bottom block automatically.
    is_regenerate = False
    prompt = st.session_state.pop("pending_prompt", None)
    if st.session_state.regenerate_query:
        prompt = st.session_state.regenerate_query
        st.session_state.regenerate_query = None
        is_regenerate = True

    if prompt is None:
        prompt = st.chat_input(
            "Ask a question about your documents…",
            disabled=not has_documents,
        )

    if prompt:
        _handle_user_message(prompt, has_documents, is_regenerate=is_regenerate)


def _handle_user_message(prompt: str, has_documents: bool, *, is_regenerate: bool = False) -> None:
    """Process a user message and stream the assistant response."""
    if not has_documents:
        st.warning("Please upload and index a PDF document first.")
        return

    messages = get_active_messages()
    sid = st.session_state.active_session_id

    if not messages and not is_regenerate:
        update_session_title(sid, prompt)

    show_user_bubble = True
    if is_regenerate and messages and messages[-1]["role"] == "user":
        show_user_bubble = False
    else:
        messages.append({"role": "user", "content": prompt})
        set_active_messages(messages)

    if show_user_bubble:
        with st.chat_message("user"):
            st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        typing = st.empty()

        typing.markdown(
            '<div class="dm-typing-dots"><span></span><span></span><span></span></div>',
            unsafe_allow_html=True,
        )

        try:
            def on_token(full: str, _token: str) -> None:
                typing.empty()
                placeholder.markdown(full + '<span class="dm-cursor">▌</span>', unsafe_allow_html=True)

            full_response, sources = stream_chat(
                session_id=f"session-{sid}",
                query=prompt,
                on_token=on_token,
            )

            typing.empty()
            placeholder.markdown(full_response)

            if sources:
                render_sources(sources)

            messages = get_active_messages()
            messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })
            set_active_messages(messages)

        except RuntimeError as exc:
            typing.empty()
            st.error(str(exc))
        except Exception as exc:
            typing.empty()
            st.error(f"Connection error: {exc}")

    from ui.styles import inject_auto_scroll
    inject_auto_scroll()
