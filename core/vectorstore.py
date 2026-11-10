"""Stage 4: the vector store, and turning FAISS distances into a real score.

FAISS returns an L2 *distance* where lower is better, which is useless to show
a student. Because our embeddings are normalised, distance and cosine
similarity are related by::

    d = 2 - 2 * cos   ->   cos = 1 - d / 2

So we convert once, in one place, and everything upstream talks about
"similarity" (higher is better, 1.0 = identical).
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved chunk plus the score that got it there."""

    document: Document
    distance: float
    similarity: float

    @property
    def section(self) -> str:
        return self.document.metadata.get("section", "GENERAL")


def build_vector_store(documents: list[Document], embeddings: Embeddings) -> FAISS:
    """Embed every chunk and build the in-memory index."""
    if not documents:
        raise ValueError("Cannot build a vector store from zero documents.")
    return FAISS.from_documents(documents, embeddings)


def search(store: FAISS, query: str, k: int = 3) -> list[Hit]:
    """Return the top-k chunks for a query, ranked best-first."""
    if not query.strip():
        return []
    pairs = store.similarity_search_with_score(query, k=k)
    hits: list[Hit] = []
    for document, distance in pairs:
        similarity = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
        hits.append(Hit(document=document, distance=float(distance), similarity=similarity))
    hits.sort(key=lambda hit: hit.distance)
    return hits


def describe_index(store: FAISS, documents: list[Document]) -> dict[str, int]:
    """Small stats block for the sidebar."""
    return {
        "chunks": int(store.index.ntotal),
        "dimensions": int(store.index.d),
        "sections": len({doc.metadata.get("section", "") for doc in documents}),
    }
