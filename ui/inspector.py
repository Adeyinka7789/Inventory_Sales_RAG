"""Retrieval inspector: the part that teaches RAG instead of hiding it.

No LLM runs here. You type a query and see exactly which chunks the vector
store returns, with their scores and the precise text the model would receive.
If the right chunk is missing on this screen, no prompt engineering will save
the answer — that is the lesson.
"""

from __future__ import annotations

import streamlit as st

from config import MAX_TOP_K, RuntimeConfig
from core.engine import PolicyEngine, sources_from_hits
from core.tools import format_for_model
from ui.components import render_source_card

DEFAULT_QUERY = "How many days do I have to return an unopened product?"
WEAK_EVIDENCE = 0.35


def render_inspector(engine: PolicyEngine, config: RuntimeConfig) -> None:
    st.caption(
        "Search the FAISS index directly. No LLM is involved — this is the exact "
        "evidence any answer is built on."
    )

    with st.form("inspector_form", border=False):
        query_col, k_col = st.columns([4, 1], vertical_alignment="bottom")
        query = query_col.text_input(
            "Query",
            value=st.session_state.get("inspect_query", DEFAULT_QUERY),
            key="inspect_query",
        )
        k = k_col.slider(
            "Chunks (k)",
            min_value=1,
            max_value=MAX_TOP_K,
            value=config.top_k,
            key="inspect_k",
        )
        submitted = st.form_submit_button("Search index", icon=":material/search:", width="stretch")

    cache_key = f"{query.strip()}|{k}"
    if submitted or st.session_state.get("inspect_cache_key") != cache_key:
        st.session_state["inspect_hits"] = engine.retrieve(query, k)
        st.session_state["inspect_cache_key"] = cache_key

    hits = st.session_state.get("inspect_hits") or []
    stats = engine.stats

    if not hits:
        st.info("Type a question to search the policy index.", icon=":material/search:")
        return

    metrics = st.container(horizontal=True)
    metrics.metric("Top match", f"{hits[0].similarity:.0%}")
    metrics.metric("Chunks in index", stats.get("chunks", 0))
    metrics.metric("Sections", stats.get("sections", 0))
    metrics.metric("Embedding dims", stats.get("dimensions", 0))

    st.markdown(f"#### Results for “{query}”")

    for source in sources_from_hits(hits):
        render_source_card(source)

    with st.expander("Exact text the model receives", icon=":material/data_object:"):
        st.code(format_for_model(hits), language="text")

    if hits[-1].similarity < WEAK_EVIDENCE:
        st.caption(
            f":gray[The last chunk scores {hits[-1].similarity:.0%} — above zero, "
            "but probably noise. A similarity threshold is a cheap way to stop "
            "weak evidence reaching the prompt.]"
        )
