"""
DocuMind AI — Streamlit Frontend
Production-grade AI SaaS interface for PDF document chat.
"""

import logging

import streamlit as st

from ui.components.chat import render_chat
from ui.components.sidebar import render_sidebar
from ui.styles import inject_auto_scroll, inject_global_styles, render_app_header
from ui.utils.state import init_session_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state=320,
)

init_session_state()
inject_global_styles()

render_sidebar()

render_app_header()
render_chat()

inject_auto_scroll()
