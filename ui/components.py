"""Small, reusable UI pieces shared by the chat and inspector tabs.

Keeping these here means a source card looks identical everywhere — the kind
of consistency that makes a demo feel finished instead of stitched together.
"""

from __future__ import annotations

import streamlit as st

from config import MODE_LABELS
from core.engine import Answer, Source

# Similarity bands. Not science — a teaching aid so students can see that a
# 0.17 "match" is noise, not evidence, even though FAISS happily returns it.
STRONG_MATCH = 0.55
WEAK_MATCH = 0.35


def match_color(similarity: float) -> str:
    if similarity >= STRONG_MATCH:
        return "green"
    if similarity >= WEAK_MATCH:
        return "orange"
    return "gray"


def render_source_card(source: Source) -> None:
    """One retrieved chunk: where it came from, how close, what it says."""
    with st.container(border=True):
        header = st.container(horizontal=True, vertical_alignment="center")
        header.badge(f"#{source.rank}", color="gray")
        header.markdown(f"**{source.section}**")
        header.badge(
            f"{source.similarity:.0%} match",
            color=match_color(source.similarity),
        )
        header.caption(f"chunk {source.chunk_index}")
        st.markdown(source.text)


def render_sources(sources: tuple[Source, ...], *, expanded: bool = False) -> None:
    label = f"Sources ({len(sources)})" if sources else "Sources"
    with st.expander(label, icon=":material/library_books:", expanded=expanded):
        if not sources:
            st.caption("No policy text was retrieved for this answer. Treat it as ungrounded.")
            return
        for source in sources:
            render_source_card(source)


def render_trace(answer: Answer) -> None:
    """Show what the agent actually did — no more black box."""
    label = "Agent trace" if answer.mode == "agent" else "Retrieval trace"
    with st.expander(label, icon=":material/route:"):
        st.caption(f"Pipeline: {MODE_LABELS.get(answer.mode, answer.mode)}")
        if not answer.tool_calls:
            st.warning(
                "The model answered without calling the retrieval tool. "
                "This is the failure mode to watch for in a live demo.",
                icon=":material/warning:",
            )
        for index, call in enumerate(answer.tool_calls, start=1):
            st.markdown(f"**Step {index} — `{call.name}`**")
            st.json(call.arguments, expanded=False)

        if answer.reasoning:
            with st.expander("Model reasoning", icon=":material/psychology:"):
                st.markdown(answer.reasoning)
        elif answer.mode == "direct":
            st.caption("Single-shot mode: there is no model reasoning step.")


def render_answer_footer(answer: Answer) -> None:
    """The line under every answer: grounding, cost, latency, provenance."""
    latency = f"{answer.latency_s:.1f}s"
    source_count = len(answer.sources)
    if answer.grounded:
        st.caption(f"Grounded in {source_count} retrieved chunk(s) · {latency}")
    else:
        st.caption(f":orange[Not grounded — no retrieval] · {latency}")
    render_sources(answer.sources)
    render_trace(answer)


def render_error(exc: BaseException) -> str:
    """Turn the three exceptions students actually hit into plain English."""
    raw = str(exc)
    lowered = raw.lower()

    if "api key" in lowered or "401" in lowered or "invalid_api_key" in lowered:
        hint = (
            "Groq rejected the API key. Check `GROQ_API_KEY` in `.env` "
            "(or paste a valid key in the sidebar) and restart the app."
        )
    elif "rate limit" in lowered or "429" in lowered:
        hint = (
            "Groq rate limit hit. Wait a few seconds, or switch to a smaller "
            "model (`openai/gpt-oss-20b`) in the sidebar."
        )
    elif "connection" in lowered or "timeout" in lowered or "unreachable" in lowered:
        hint = (
            "Could not reach Groq. Check your internet connection — the "
            "embeddings still run locally, so retrieval keeps working."
        )
    else:
        hint = "Something went wrong while answering."

    st.error(f"{hint}\n\n`{type(exc).__name__}: {raw[:300]}`", icon=":material/error:")
    return hint
