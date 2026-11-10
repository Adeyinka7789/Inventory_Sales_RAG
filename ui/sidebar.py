"""Sidebar: every knob the instructor wants to turn during a live demo.

Two rules kept this honest:

* Widgets return a plain ``RuntimeConfig``. The rest of the app never reads
  widget state directly, so the same config object can be tested, logged, or
  posted to an API later.
* Expensive settings (chunk size) are grouped and clearly labelled, because
  in Streamlit *any* widget change reruns the whole script.
"""

from __future__ import annotations

import os

import streamlit as st

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MEMORY_TURNS,
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    MAX_CHUNK_SIZE,
    MAX_TEMPERATURE,
    MAX_TOP_K,
    MIN_CHUNK_SIZE,
    MIN_TEMPERATURE,
    MIN_TOP_K,
    MODE_AGENT,
    MODE_DIRECT,
    MODE_LABELS,
    MODEL_CHOICES,
    RuntimeConfig,
    groq_api_key,
)

_MODE_BY_LABEL = {"Agent": MODE_AGENT, "Direct": MODE_DIRECT}
_LABEL_BY_MODE = {v: k for k, v in _MODE_BY_LABEL.items()}


def render_sidebar() -> RuntimeConfig:
    """Draw the sidebar and return the configuration it produced."""
    with st.sidebar:
        st.markdown("#### :material/tune: Settings")

        _render_api_key_block()

        st.markdown("**Pipeline**")
        mode_label = st.segmented_control(
            "Pipeline",
            list(_MODE_BY_LABEL),
            default=_LABEL_BY_MODE[MODE_AGENT],
            key="pipeline_mode",
            label_visibility="collapsed",
            help=(
                "Agent: the model decides when to call search_policy. "
                "Direct: always retrieve first, then answer (classic RAG)."
            ),
        )
        mode = _MODE_BY_LABEL.get(mode_label or "Agent", MODE_AGENT)

        with st.expander("Retrieval", icon=":material/manage_search:", expanded=True):
            top_k = st.slider(
                "Chunks to retrieve (k)",
                min_value=MIN_TOP_K,
                max_value=MAX_TOP_K,
                value=DEFAULT_TOP_K,
                key="top_k",
                help="How many chunks the retriever hands to the model.",
            )

        with st.expander("Chunking", icon=":material/content_cut:"):
            chunk_size = st.slider(
                "Chunk size (characters)",
                min_value=MIN_CHUNK_SIZE,
                max_value=MAX_CHUNK_SIZE,
                value=DEFAULT_CHUNK_SIZE,
                step=50,
                key="chunk_size",
            )
            chunk_overlap = st.slider(
                "Chunk overlap",
                min_value=0,
                max_value=200,
                value=DEFAULT_CHUNK_OVERLAP,
                step=10,
                key="chunk_overlap",
            )
            st.caption("Changing these rebuilds the FAISS index.")

        with st.expander("Model", icon=":material/memory:", expanded=True):
            model = st.selectbox(
                "Groq model",
                MODEL_CHOICES,
                index=MODEL_CHOICES.index(DEFAULT_MODEL),
                key="model",
                help="Free-tier models. Smaller = faster, weaker.",
            )
            temperature = st.slider(
                "Temperature",
                min_value=MIN_TEMPERATURE,
                max_value=MAX_TEMPERATURE,
                value=0.0,
                step=0.1,
                key="temperature",
                help="Keep at 0 for policy answers: less invention.",
            )

        with st.expander("Conversation", icon=":material/forum:"):
            memory_turns = st.slider(
                "Messages replayed as memory",
                min_value=0,
                max_value=12,
                value=DEFAULT_MEMORY_TURNS,
                step=2,
                key="memory_turns",
                help="0 = every question is independent.",
            )

        st.divider()
        left, right = st.columns(2)
        if left.button(
            "Reset chat",
            icon=":material/restart_alt:",
            width="stretch",
            key="reset_chat",
        ):
            st.session_state["chat_messages"] = []
            st.session_state["eval_results"] = None
            st.rerun()
        if right.button(
            "Reload index",
            icon=":material/refresh:",
            width="stretch",
            key="reload_index",
        ):
            st.cache_resource.clear()
            st.rerun()

    return RuntimeConfig(
        model=model,
        temperature=temperature,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        memory_turns=memory_turns,
        mode=mode,
    )


def _render_api_key_block() -> None:
    """Let a student paste a key without editing files — but never log it."""
    if groq_api_key():
        st.badge("Groq key loaded", color="green", icon=":material/key:")
        return

    st.warning(
        "No `GROQ_API_KEY` found. Paste one to try the assistant, or add it to `.env`.",
        icon=":material/key_off:",
    )
    typed = st.text_input(
        "Groq API key",
        type="password",
        key="api_key_input",
        placeholder="gsk_...",
    )
    if typed.strip():
        os.environ["GROQ_API_KEY"] = typed.strip()
        st.rerun()


def render_index_stats(stats: dict[str, int], config: RuntimeConfig) -> None:
    """Sidebar footer: proof that the index matches the sliders."""
    st.caption(
        f"Index: {stats.get('chunks', 0)} chunks · "
        f"{stats.get('sections', 0)} sections · "
        f"{stats.get('dimensions', 0)} dims  \n"
        f"Mode: {MODE_LABELS[config.mode]}"
    )
