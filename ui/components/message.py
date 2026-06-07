"""Chat message rendering components."""

import json

import streamlit as st
import streamlit.components.v1 as components

from ui.utils.state import get_active_messages


def render_sources(sources: dict) -> None:
    """Render source citation cards."""
    if not sources:
        return

    items_html = ""
    for fname, pages in sources.items():
        page_str = ", ".join(str(p) for p in sorted(pages))
        items_html += f"""
        <div class="dm-source-item">
            <span class="dm-source-icon">📄</span>
            <span><strong>{fname}</strong> — pages {page_str}</span>
        </div>
        """

    st.markdown(
        f"""
        <div class="dm-sources">
            <div class="dm-sources-title">Sources</div>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message_actions(content: str, msg_index: int, is_last_assistant: bool) -> None:
    """Render copy and regenerate action buttons."""
    col1, col2, col3 = st.columns([1, 1, 6])

    with col1:
        if st.button("Copy", key=f"copy_{msg_index}", help="Copy response"):
            escaped = json.dumps(content)
            components.html(
                f"""
                <script>
                    navigator.clipboard.writeText({escaped});
                </script>
                """,
                height=0,
            )
            st.toast("Copied to clipboard", icon="✅")

    if is_last_assistant:
        with col2:
            if st.button("Regenerate", key=f"regen_{msg_index}", help="Regenerate response"):
                messages = get_active_messages()
                if len(messages) >= 2:
                    last_user = None
                    for m in reversed(messages):
                        if m["role"] == "user":
                            last_user = m["content"]
                            break
                    if last_user:
                        trimmed = messages[:-1]
                        from ui.utils.state import set_active_messages
                        set_active_messages(trimmed)
                        st.session_state.regenerate_query = last_user
                        st.rerun()


def render_chat_history() -> None:
    """Render all messages in the active session."""
    messages = get_active_messages()
    last_assistant_idx = None
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            last_assistant_idx = i

    for i, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])
            if msg["role"] == "assistant":
                is_last = i == last_assistant_idx
                render_message_actions(msg["content"], i, is_last)
