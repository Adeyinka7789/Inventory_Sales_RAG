"""The engine: one object that turns a question into a stream of events.

Design notes (worth teaching)
-----------------------------
* **One pipeline, two entry points.** ``answer()`` is just ``stream()`` with
  the tokens glued back together. The CLI, the chat UI and the evaluation
  runner therefore cannot drift apart — a classic source of "it works in the
  notebook but not in the app" bugs.
* **Events, not callbacks.** ``stream()`` yields small dataclasses
  (``Token``, ``Sources``, ``ToolCallEvent``...). The UI decides how to render
  them. Swapping Streamlit for FastAPI+Svelte later would not touch this file.
* **Streaming gotcha.** LangGraph's ``messages`` stream also emits the
  *tool's* output as message chunks. Blindly printing every chunk leaks raw
  policy text into the answer. We filter on ``langgraph_node == "model"`` and
  on the message type — see ``_is_model_token``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_groq import ChatGroq

from config import RuntimeConfig, groq_api_key
from core.documents import chunk_body
from core.prompts import SYSTEM_PROMPT, build_direct_messages, history_messages
from core.tools import RetrievalTrace, format_for_model, make_search_tool
from core.vectorstore import Hit, describe_index, search


class MissingAPIKey(RuntimeError):
    """Raised when GROQ_API_KEY is absent. The UI turns this into a friendly card."""


# ---------------------------------------------------------------------------
# Data returned to the UI
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Source:
    """One retrieved chunk, in display-ready form."""

    rank: int
    section: str
    text: str
    similarity: float
    chunk_index: int


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    name: str
    arguments: dict


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    sources: tuple[Source, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    reasoning: str
    latency_s: float
    mode: str
    model: str

    @property
    def grounded(self) -> bool:
        """True when the answer was produced with retrieved policy text."""
        return bool(self.sources)


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Token:
    text: str


@dataclass(frozen=True, slots=True)
class Sources:
    sources: tuple[Source, ...]


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    call: ToolCallRecord


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    name: str
    output: str
    characters: int


@dataclass(frozen=True, slots=True)
class Done:
    answer: Answer


Event = Token | Sources | ToolCallEvent | ToolResultEvent | Done


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class PolicyEngine:
    """Holds the index + LLM for one configuration; stateless per request."""

    def __init__(
        self,
        config: RuntimeConfig,
        store: FAISS,
        documents: list[Document],
    ) -> None:
        self.config = config
        self.store = store
        self.documents = documents
        self._llm_client: ChatGroq | None = None

    @property
    def llm(self) -> ChatGroq:
        """Created on first use so retrieval works without an API key."""
        if self._llm_client is None:
            key = groq_api_key()
            if not key:
                raise MissingAPIKey(
                    "GROQ_API_KEY is not set. Add it to .env or paste it in the sidebar."
                )
            self._llm_client = ChatGroq(
                model=self.config.model,
                temperature=self.config.temperature,
                api_key=key,
                timeout=60,
                max_retries=2,
            )
        return self._llm_client

    # -- introspection ------------------------------------------------------
    @property
    def stats(self) -> dict[str, int]:
        return describe_index(self.store, self.documents)

    def retrieve(self, query: str, k: int | None = None) -> list[Hit]:
        """Raw retrieval, no LLM. Used by the inspector tab."""
        return search(self.store, query, k=k or self.config.top_k)

    # -- public API ---------------------------------------------------------
    def answer(self, question: str, history: list[dict] | None = None) -> Answer:
        """Non-streaming helper built on top of ``stream`` (CLI, evaluation)."""
        final: Answer | None = None
        for event in self.stream(question, history):
            if isinstance(event, Done):
                final = event.answer
        assert final is not None, "stream() must always end with a Done event"
        return final

    def stream(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> Iterator[Event]:
        """Answer a question, yielding events as they happen."""
        started = time.perf_counter()
        trace = RetrievalTrace()
        parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        emitted_sources = 0
        cleaner = _StripCitations()

        if self.config.is_agent_mode:
            agent = create_agent(
                model=self.llm,
                tools=[make_search_tool(self.store, self.config.top_k, trace)],
                system_prompt=SYSTEM_PROMPT,
            )
            messages = [
                *history_messages(history, self.config.memory_turns),
                HumanMessage(content=question),
            ]

            for mode, payload in agent.stream(
                {"messages": messages},
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    chunk, metadata = payload
                    if not _is_model_token(chunk, metadata):
                        continue
                    reasoning = chunk.additional_kwargs.get("reasoning_content")
                    if reasoning:
                        reasoning_parts.append(str(reasoning))
                    text = _message_text(chunk)
                    if text:
                        clean = cleaner.feed(text)
                        if clean:
                            parts.append(clean)
                            yield Token(clean)
                    continue

                # mode == "updates": tool calls and tool results arrive here
                for node_update in (payload or {}).values():
                    for message in (node_update or {}).get("messages", []) or []:
                        if isinstance(message, ToolMessage):
                            yield ToolResultEvent(
                                name=message.name or "search_policy",
                                output=str(message.content),
                                characters=len(str(message.content)),
                            )
                        elif isinstance(message, AIMessage) and message.tool_calls:
                            for call in message.tool_calls:
                                record = ToolCallRecord(
                                    name=call.get("name", "unknown"),
                                    arguments=dict(call.get("args") or {}),
                                )
                                tool_calls.append(record)
                                yield ToolCallEvent(record)

                if len(trace.hits) > emitted_sources:
                    emitted_sources = len(trace.hits)
                    yield Sources(sources_from_hits(trace.hits))

        else:
            # Classic RAG: always retrieve, then answer with the text in hand.
            hits = search(self.store, question, self.config.top_k)
            trace.record(question, hits)
            call = ToolCallRecord(
                name="retrieval (single-shot)",
                arguments={"query": question, "k": self.config.top_k},
            )
            tool_calls.append(call)
            yield ToolCallEvent(call)
            yield ToolResultEvent(
                name="retrieval (single-shot)",
                output=format_for_model(hits),
                characters=len(format_for_model(hits)),
            )
            yield Sources(sources_from_hits(hits))

            for chunk in self.llm.stream(build_direct_messages(question, hits, history)):
                text = _message_text(chunk)
                if text:
                    clean = cleaner.feed(text)
                    if clean:
                        parts.append(clean)
                        yield Token(clean)

        tail = cleaner.flush()
        if tail:
            parts.append(tail)
            yield Token(tail)

        text = "".join(parts).strip()
        if not text:
            text = (
                "I could not produce an answer from the policy documents. "
                "Please try rephrasing, or contact a human for approval."
            )

        yield Done(
            Answer(
                text=text,
                sources=sources_from_hits(trace.hits),
                tool_calls=tuple(tool_calls),
                reasoning="".join(reasoning_parts).strip(),
                latency_s=time.perf_counter() - started,
                mode=self.config.mode,
                model=self.config.model,
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _StripCitations:
    """Remove ``【1†L1-L2】`` markers that gpt-oss models emit inline.

    They are training artefacts, not policy text, and they look like a bug in a
    demo. Because answers stream token by token, a plain ``re.sub`` on the
    final string would arrive too late — this tiny state machine holds back
    only the unfinished marker and releases everything else immediately.
    """

    _OPEN = "\u3010"  # 【
    _CLOSE = "\u3011"  # 】

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, text: str) -> str:
        self._pending += text
        out: list[str] = []
        index = 0
        while index < len(self._pending):
            char = self._pending[index]
            if char == self._OPEN:
                end = self._pending.find(self._CLOSE, index)
                if end == -1:
                    break  # marker still streaming in — wait for the rest
                index = end + 1
                continue
            out.append(char)
            index += 1
        self._pending = self._pending[index:]
        return "".join(out)

    def flush(self) -> str:
        leftover, self._pending = self._pending, ""
        return "" if leftover.startswith(self._OPEN) else leftover


def sources_from_hits(hits: list[Hit]) -> tuple[Source, ...]:
    return tuple(
        Source(
            rank=rank,
            section=hit.section,
            text=chunk_body(hit.document),
            similarity=hit.similarity,
            chunk_index=int(hit.document.metadata.get("chunk_index", 0)),
        )
        for rank, hit in enumerate(hits, start=1)
    )


def _is_model_token(chunk: object, metadata: dict) -> bool:
    """Only the model's own text, never the tool's output or retrieval noise."""
    return isinstance(chunk, AIMessageChunk) and metadata.get("langgraph_node") == "model"


def _message_text(message: object) -> str:
    """Normalise message content (str or list-of-parts) into plain text."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                pieces.append(str(part.get("text", "")))
        return "".join(pieces)
    return ""
