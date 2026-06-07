"""Welcome screen and suggested prompts."""

import streamlit as st

from ui.theme.tokens import SUGGESTED_PROMPTS


def render_welcome(has_documents: bool) -> None:
    """Render the empty-state welcome screen."""
    if has_documents:
        subtitle = (
            "Your documents are indexed and ready. "
            "Ask anything about their content."
        )
    else:
        subtitle = (
            "Upload PDF documents using the sidebar, then ask questions "
            "to get instant, source-backed answers."
        )

    st.markdown(
        f"""
        <div class="dm-welcome">
            <div class="dm-welcome-icon">📄</div>
            <h2>How can I help you today?</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if has_documents:
        render_suggested_prompts()


def render_suggested_prompts() -> None:
    """Render clickable suggested prompt buttons."""
    cols = st.columns(2)
    for i, prompt in enumerate(SUGGESTED_PROMPTS):
        with cols[i % 2]:
            if st.button(prompt, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()
