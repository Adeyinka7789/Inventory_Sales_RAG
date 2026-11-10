"""Stage 3: turn text into vectors.

We use a small sentence-transformer on the local CPU. That was the right call
for the original demo (no API key, no network flakiness) and it is still the
right call here — but it costs ~90 MB of weights and a few seconds on first
load. Loading it once per process is the difference between a snappy app and a
30-second rerun, so ``get_embeddings`` is cached.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


@lru_cache(maxsize=2)
def get_embeddings(
    model_name: str = EMBEDDING_MODEL,
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """Load (once) the embedding model and return it.

    ``normalize_embeddings=True`` means every vector has unit length, so FAISS
    L2 distance and cosine similarity become interchangeable — that is what
    lets us show an honest 0–100% "match" score in the UI.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
