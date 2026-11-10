"""Chat tab: streaming answers, visible sources, visible agent trace.

The interesting part is ``_stream_answer``. It is a plain generator, so
``st.write_stream`` can render tokens as they arrive, while a small collector
dict captures the non-text events (sources, tool calls) that we render once the
answer finishes.
"""

from __future__ import annotations

import streamlit as st

from core.engine import (
    Answer,
    Done,
    PolicyEngine,
    Sources,
    Token,
    ToolCallEvent,
    ToolResultEvent,
)
from ui.components import render_answer_footer, render_error

# Short pill label -> full question. Full sentences make better demo prompts
# than keyword chips, but they are too wide for a pill.
SUGGESTIONS = {
    ":material/assignment_return: Return window": (
        "How many days do I have to return an unopened product?"
    ),
    ":material/sell: Bulk discount": ("What discount do I get if I order 500 units?"),
    ":material/receipt_long: Supplier terms": ("What are the standard supplier payment terms?"),
    ":material/local_shipping: Late delivery": (
        "My delivery arrived 4 days late. Can I get the order value refunded?"
    ),
}


def render_chat(engine: PolicyEngine) -> None:
    """Draw the conversation and handle a new question."""
    messages: list[dict] = st.session_state.setdefault("chat_messages", [])

    for message in messages:
        _render_message(message)

    prompt: str | None = None
    if not messages:
        prompt = _render_suggestions()

    typed = st.chat_input(
        "Ask about returns, discounts, supplier terms, restocking or shipping…",
        submit_mode="disable",
        key="chat_input",
    )
    prompt = typed or prompt
    if prompt:
        _handle_prompt(engine, prompt, messages)


def _render_message(message: dict) -> None:
    role = message.get("role", "assistant")
    if role == "user":
        with st.chat_message("user"):
            st.markdown(message.get("content", ""))
        return

    with st.chat_message("assistant", avatar=":material/policy:"):
        st.markdown(message.get("content", ""))
        answer = message.get("answer")
        if isinstance(answer, Answer):
            render_answer_footer(answer)


def _render_suggestions() -> str | None:
    """First-run pills. Returns a question if one was clicked."""
    placeholder = st.empty()
    selected = placeholder.pills(
        "Try asking",
        list(SUGGESTIONS),
        label_visibility="collapsed",
        key="suggestions",
    )
    if not selected:
        return None
    placeholder.empty()
    return SUGGESTIONS[selected]


def _handle_prompt(engine: PolicyEngine, prompt: str, messages: list[dict]) -> None:
    # History excludes the question we are about to ask.
    history = [
        {"role": message.get("role", ""), "content": message.get("content", "")}
        for message in messages
    ]
    messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    collector: dict = {}
    with st.chat_message("assistant", avatar=":material/policy:"):
        text = st.write_stream(_stream_answer(engine, prompt, history, collector))
        answer = collector.get("answer")
        if isinstance(answer, Answer):
            render_answer_footer(answer)

    messages.append(
        {
            "role": "assistant",
            "content": text if isinstance(text, str) else "".join(text),
            "answer": answer,
        }
    )


def _stream_answer(
    engine: PolicyEngine,
    question: str,
    history: list[dict],
    collector: dict,
):
    """Yield only answer text; stash everything else for the footer."""
    try:
        for event in engine.stream(question, history):
            if isinstance(event, Token):
                yield event.text
            elif isinstance(event, Sources):
                collector["sources"] = event.sources
            elif isinstance(event, ToolCallEvent):
                collector.setdefault("tool_calls", []).append(event.call)
            elif isinstance(event, ToolResultEvent):
                collector["tool_output_chars"] = event.characters
            elif isinstance(event, Done):
                collector["answer"] = event.answer
    except Exception as exc:
        hint = render_error(exc)
        yield f"\n\n> **Answer interrupted.** {hint}"
