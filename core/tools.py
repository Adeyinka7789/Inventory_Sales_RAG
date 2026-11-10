"""Stage 6: the tool the agent is allowed to use.

A tool is just a Python function with a docstring — the docstring *is* the
API contract the model reads. Two things matter:

1. The description must say when to use it, not just what it does.
2. We record every call in a ``RetrievalTrace`` so the UI can show exactly
   what was retrieved. Without that, the answer card is a black box.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS

from core.documents import chunk_body
from core.vectorstore import Hit, search


@dataclass
class RetrievalTrace:
    """Mutable per-request log of what retrieval actually did."""

    queries: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)

    def record(self, query: str, hits: list[Hit]) -> None:
        self.queries.append(query)
        # Keep the best score per section so repeated calls do not duplicate cards.
        seen = {id(hit.document) for hit in self.hits}
        for hit in hits:
            if id(hit.document) not in seen:
                self.hits.append(hit)
                seen.add(id(hit.document))

    @property
    def was_used(self) -> bool:
        return bool(self.queries)


def format_for_model(hits: list[Hit]) -> str:
    """The exact string the LLM sees. Also shown in the UI's inspector."""
    if not hits:
        return "No matching policy text was found."
    blocks = []
    for rank, hit in enumerate(hits, start=1):
        # Plain "[1] SECTION NAME" header: the model quotes section names back
        # verbatim, so avoid labels ("SECTION: ...") that read badly in prose.
        blocks.append(f"[{rank}] {hit.section}\n{chunk_body(hit.document)}")
    return "\n\n".join(blocks)


def make_search_tool(
    store: FAISS,
    k: int,
    trace: RetrievalTrace,
) -> Callable[[str], str]:
    """Build a fresh ``search_policy`` tool bound to one request's trace.

    A new closure per request keeps the engine stateless and thread-safe: two
    users cannot overwrite each other's trace.
    """

    def search_policy(question: str) -> str:
        """Search the company's Inventory & Sales policy documents.

        Use this for ANY question about returns, refunds, bulk discounts,
        supplier payment terms, stock reordering, customer complaints, or
        shipping and delivery. Call it before answering, every time.
        """
        hits = search(store, question, k=k)
        trace.record(question, hits)
        return format_for_model(hits)

    return search_policy
