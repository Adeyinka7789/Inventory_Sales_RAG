"""How-it-works tab: the map students should leave with.

Puts the module boundaries, the data flow and the two pipelines on one screen
so the balkanised codebase still reads as a single system.
"""

from __future__ import annotations

import streamlit as st

from config import MODE_LABELS, RuntimeConfig
from core.prompts import DIRECT_PROMPT, SYSTEM_PROMPT

MODULES = [
    ("config.py", "Every default, path and model name. No magic numbers anywhere else."),
    ("core/documents.py", "Loads the .txt, splits section-first, then chunks. Adds section data."),
    ("core/embeddings.py", "Loads the local MiniLM model once per process."),
    ("core/vectorstore.py", "FAISS index; turns L2 distance into a 0-100% match score."),
    ("core/tools.py", "The search_policy tool + a per-request retrieval trace."),
    ("core/prompts.py", "Both system prompts and the context formatter."),
    ("core/engine.py", "Orchestrates retrieval + LLM and emits stream events."),
    ("core/evaluation.py", "Golden-set scoring: keywords, guardrails, section hits."),
    ("ui/*.py", "Streamlit rendering only. No business logic."),
    ("cli.py", "Terminal entry point, reusing the same engine."),
]


def render_about(config: RuntimeConfig) -> None:
    st.caption(
        "One pipeline, four ways to look at it. The code is split by "
        "responsibility so each file can be explained — and tested — on its own."
    )

    st.markdown("#### Data flow")
    st.markdown(
        """
```mermaid
graph LR
    A[data/policies.txt] --> B[documents.py<br/>section split + chunks]
    B --> C[embeddings.py<br/>MiniLM on CPU]
    C --> D[vectorstore.py<br/>FAISS index]
    D --> E[tools.py<br/>search_policy]
    E --> F[engine.py<br/>agent or direct]
    F --> G[ui/*.py<br/>Streamlit]
```
"""
    )

    st.markdown("#### What each file owns")
    st.dataframe(
        [{"File": name, "Responsibility": job} for name, job in MODULES],
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Two pipelines, same index")
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.markdown(f"**{MODE_LABELS['agent']}**")
        st.markdown(
            "The model decides when to call `search_policy`. More flexible "
            "with follow-ups, but it can skip retrieval — which is why the "
            "trace panel shows a warning when that happens."
        )
    with right, st.container(border=True):
        st.markdown(f"**{MODE_LABELS['direct']}**")
        st.markdown(
            "We retrieve first, then hand the excerpts to the model. "
            "Deterministic and usually faster — and the honest baseline "
            "every agentic pipeline should beat."
        )

    with st.expander("System prompt (agent mode)", icon=":material/description:"):
        st.code(SYSTEM_PROMPT, language="text")
        st.caption(
            "Compare with the original single-file demo: 'Always use the tool' "
            "as a polite suggestion. Numbered hard rules are what actually "
            "stopped the model answering from memory."
        )

    with st.expander("System prompt (direct mode)", icon=":material/description:"):
        st.code(DIRECT_PROMPT, language="text")

    st.markdown("#### Try this in class")
    st.markdown(
        f"""
1. Ask *"What are the standard supplier payment terms?"* in **Agent** mode, then
   **Direct** mode. Same index, same model — compare latency and citations.
2. Drop **Chunks to retrieve (k)** to 1 and re-ask the shipping question.
   Watch a two-fact answer become a one-fact answer.
3. Set **Chunk size** to 150. The index rebuilds into more, smaller chunks;
   ask the reorder-formula question again and look at the source cards.
4. Open the **Retrieval inspector** and search *"refund"*. Notice weak matches
   are returned regardless of relevance — retrieval always returns *something*.
5. Run the **Evaluation** tab in both modes. Current mode: {config.mode}.
"""
    )
