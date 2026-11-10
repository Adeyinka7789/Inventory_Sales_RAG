"""Core RAG pipeline: documents -> chunks -> embeddings -> vector store -> LLM.

Nothing in this package imports Streamlit. That is deliberate: the CLI, the
tests and tomorrow's FastAPI version can all reuse it unchanged.
"""

import warnings

# langchain-community prints a sunset DeprecationWarning on import. It is
# expected (the ecosystem is moving to standalone packages) and it is noise in
# a live demo, so we silence it here — once, for every entry point.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)

from core.documents import build_chunks, chunk_body, load_documents
from core.embeddings import get_embeddings
from core.engine import (
    Answer,
    Done,
    MissingAPIKey,
    PolicyEngine,
    Source,
    Sources,
    Token,
    ToolCallEvent,
    ToolCallRecord,
    ToolResultEvent,
    sources_from_hits,
)
from core.tools import RetrievalTrace, format_for_model, make_search_tool
from core.vectorstore import Hit, build_vector_store, describe_index, search

__all__ = [
    "Answer",
    "Done",
    "Hit",
    "MissingAPIKey",
    "PolicyEngine",
    "RetrievalTrace",
    "Source",
    "Sources",
    "Token",
    "ToolCallEvent",
    "ToolCallRecord",
    "ToolResultEvent",
    "build_chunks",
    "build_vector_store",
    "chunk_body",
    "describe_index",
    "format_for_model",
    "get_embeddings",
    "load_documents",
    "make_search_tool",
    "search",
    "sources_from_hits",
]
