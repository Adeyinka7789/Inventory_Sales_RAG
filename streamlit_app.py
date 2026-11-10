"""Streamlit entry point — UI wiring only.

Run with:  streamlit run streamlit_app.py

Everything interesting lives in ``core/``. This file reads settings from the
sidebar, builds (or reuses) the index, and hands it to the tab renderers.
"""

from __future__ import annotations

import os

import streamlit as st

st.set_page_config(
    page_title="Acme policy assistant",
    page_icon=":material/policy:",
    layout="wide",
)

# Imports below set_page_config on purpose: Streamlit requires it to be the
# first Streamlit call in the script.
from config import (
    MODE_LABELS,
    groq_api_key,
    load_environment,
)
from core.documents import load_documents
from core.embeddings import get_embeddings
from core.engine import PolicyEngine
from core.vectorstore import build_vector_store
from ui.about import render_about
from ui.chat import render_chat
from ui.evaluation import render_evaluation
from ui.inspector import render_inspector
from ui.sidebar import render_index_stats, render_sidebar

load_environment()


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
# Cache the *index* only, keyed on the chunking settings. The engine itself is
# cheap to build (it holds no network state) so it is recreated per rerun —
# which keeps model/temperature changes instant instead of re-embedding 6
# chunks on every slider tick.
@st.cache_resource(max_entries=4, show_spinner=False)
def load_index(chunk_size: int, chunk_overlap: int):
    documents = load_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = build_vector_store(documents, get_embeddings())
    return store, documents


def _hydrate_secrets() -> None:
    """Allow GROQ_API_KEY from .streamlit/secrets.toml (handy when deploying)."""
    if groq_api_key():
        return
    try:
        value = st.secrets.get("GROQ_API_KEY")
    except Exception:  # no secrets.toml at all — that is fine
        return
    if value:
        os.environ["GROQ_API_KEY"] = str(value)


_hydrate_secrets()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
config = render_sidebar()

st.title("Acme inventory & sales policy assistant", anchor=False)
st.caption(
    "Groq (LLM) · local MiniLM embeddings · FAISS · LangChain. "
    "Every answer is grounded in `data/policies.txt` — and the assistant is "
    "not allowed to approve refunds."
)

try:
    with st.spinner("Loading the embedding model and building the index…"):
        store, documents = load_index(config.chunk_size, config.chunk_overlap)
except Exception as exc:
    st.error(
        f"Could not build the policy index: `{type(exc).__name__}: {exc}`",
        icon=":material/database_off:",
    )
    st.stop()

engine = PolicyEngine(config, store, documents)

with st.sidebar:
    st.divider()
    render_index_stats(engine.stats, config)

chat_tab, inspector_tab, evaluation_tab, about_tab = st.tabs(
    ["Chat", "Retrieval inspector", "Evaluation", "How it works"],
    default="Chat",
)

with chat_tab:
    if groq_api_key():
        render_chat(engine)
    else:
        st.info(
            "Add a **Groq API key** in the sidebar to start chatting. "
            "It is free at console.groq.com.",
            icon=":material/key:",
        )
        st.markdown(
            "Meanwhile, the **Retrieval inspector** tab works without any API "
            "key — retrieval runs entirely on your machine."
        )
        st.caption(f"Current pipeline: {MODE_LABELS[config.mode]} · {config.describe()}")

with inspector_tab:
    render_inspector(engine, config)

with evaluation_tab:
    if groq_api_key():
        render_evaluation(engine, config)
    else:
        st.info("An API key is required to run the evaluation.", icon=":material/key:")

with about_tab:
    render_about(config)
