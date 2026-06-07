"""Global style injection for DocuMind AI."""

from pathlib import Path

import streamlit as st


_CSS_PATH = Path(__file__).resolve().parent / "theme" / "styles.css"


def inject_global_styles() -> None:
    """Load and inject the design system CSS into the Streamlit app."""
    css = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_app_header() -> None:
    """Render the main app header."""
    st.markdown(
        """
        <div class="dm-app-header">
            <h1>DocuMind AI</h1>
            <p>Intelligent document analysis powered by AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_auto_scroll() -> None:
    """Scroll chat to bottom after each render."""
    st.components.v1.html(
        """
        <script>
            const scrollToBottom = () => {
                const doc = window.parent.document;
                const main = doc.querySelector('section.main');
                if (main) {
                    main.scrollTop = main.scrollHeight;
                }
            };
            scrollToBottom();
            setTimeout(scrollToBottom, 100);
            setTimeout(scrollToBottom, 300);
        </script>
        """,
        height=0,
    )
