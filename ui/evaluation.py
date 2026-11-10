"""Evaluation tab: turn "it looks good" into a number.

Runs the golden question set in ``eval/qa_set.json`` against whatever pipeline
the sidebar is set to, so students can compare Agent vs Direct on the same
data instead of arguing about vibes.
"""

from __future__ import annotations

import csv
import io

import streamlit as st

from config import QA_SET_FILE, RuntimeConfig
from core.engine import PolicyEngine
from core.evaluation import (
    EvalResult,
    load_qa_set,
    run,
    summarise,
)


def render_evaluation(engine: PolicyEngine, config: RuntimeConfig) -> None:
    items = load_qa_set(QA_SET_FILE)

    st.caption(
        "A golden set of questions with expected keywords and guardrails. "
        "Answers are scored by substring match — cheap, deterministic, and "
        "enough to catch hallucinated numbers."
    )

    with st.expander(f"Question set ({len(items)} questions)", icon=":material/checklist:"):
        st.dataframe(
            [
                {
                    "ID": item.id,
                    "Question": item.question,
                    "Must contain": ", ".join(item.expect_keywords),
                    "Section": item.expect_section,
                }
                for item in items
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Change pipelines in the sidebar and re-run to compare. "
            "About a minute per run on the Groq free tier."
        )

    if st.button(
        "Run evaluation",
        icon=":material/play_arrow:",
        type="primary",
        key="run_eval",
    ):
        bar = st.progress(0.0, text="Starting…")

        def progress(index: int, total: int, item) -> None:
            bar.progress((index - 1) / total, text=f"[{index}/{total}] {item.id}")

        results = run(engine, items, on_progress=progress)
        bar.progress(1.0, text="Finished")
        st.session_state["eval_results"] = results
        st.session_state["eval_config"] = config

    results: list[EvalResult] | None = st.session_state.get("eval_results")
    if not results:
        st.info(
            "Run the evaluation to score this pipeline. Nothing is sent anywhere "
            "except the questions themselves.",
            icon=":material/experiment:",
        )
        return

    _render_summary(results, st.session_state.get("eval_config") or config)
    _render_details(results)


def _render_summary(results: list[EvalResult], config: RuntimeConfig) -> None:
    stats = summarise(results)
    st.markdown("#### Summary")
    metrics = st.container(horizontal=True)
    metrics.metric("Pass rate", f"{stats['pass_rate']:.0%}")
    metrics.metric("Grounded", f"{stats['grounded_rate']:.0%}")
    metrics.metric("Retrieval hit", f"{stats['section_hit_rate']:.0%}")
    metrics.metric("Mean latency", f"{stats['mean_latency']:.1f}s")
    st.caption(f"Config: {config.describe()} · {config.mode} mode")

    rows = []
    for result in results:
        rows.append(
            {
                "ID": result.item.id,
                "Passed": result.passed,
                "Grounded": result.grounded,
                "Retrieval hit": result.section_hit,
                "Latency (s)": round(result.latency_s, 1),
                "Missing": ", ".join(result.missing_keywords) or "—",
                "Question": result.item.question,
            }
        )
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Latency (s)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    st.download_button(
        "Download results (CSV)",
        data=_to_csv(results),
        file_name="eval_results.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def _render_details(results: list[EvalResult]) -> None:
    st.markdown("#### Per-question detail")
    for result in results:
        verdict = "pass" if result.passed else "fail"
        with st.expander(
            f"{result.item.id} · {verdict} · {result.item.question}",
            icon=":material/check_circle:" if result.passed else ":material/cancel:",
        ):
            if result.error:
                st.error(result.error, icon=":material/error:")
            st.markdown(result.answer or "_(no answer)_")
            st.caption(f"Why this question exists: {result.item.notes}")
            if result.missing_keywords:
                st.warning(
                    "Missing from the answer: " + ", ".join(result.missing_keywords),
                    icon=":material/search_off:",
                )
            if result.forbidden_hits:
                st.error(
                    "Guardrail breach — the answer contained: " + ", ".join(result.forbidden_hits),
                    icon=":material/gpp_bad:",
                )


def _to_csv(results: list[EvalResult]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["id", "question", "passed", "grounded", "section_hit", "latency_s", "answer", "missing"]
    )
    for result in results:
        writer.writerow(
            [
                result.item.id,
                result.item.question,
                result.passed,
                result.grounded,
                result.section_hit,
                round(result.latency_s, 2),
                result.answer,
                "; ".join(result.missing_keywords),
            ]
        )
    return buffer.getvalue()
