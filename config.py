"""Central configuration for the Inventory & Sales RAG assistant.

Why this file exists
--------------------
The original single-file demo hard-coded every tunable inside the script:
the model name, chunk size, file path, API key handling. Here they live in
one place so the CLI, the Streamlit app and the evaluation runner all agree
on the same defaults.

Rule for students: if you find yourself typing a magic number in a UI file,
it belongs here instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "policies.txt"
QA_SET_FILE = PROJECT_ROOT / "eval" / "qa_set.json"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Checked against GET https://api.groq.com/openai/v1/models — if a model is
# retired, that endpoint is the fastest place to see what replaced it.

DEFAULT_MODEL = "openai/gpt-oss-120b"
MODEL_CHOICES = (
    "openai/gpt-oss-120b",  # best quality on the free tier
    "openai/gpt-oss-20b",  # much faster, slightly weaker
    "qwen/qwen3.8-27b",  # useful contrast: different tokenizer + style
)

# Runs locally on CPU. No API key, no network after the first download.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 60
DEFAULT_TOP_K = 3
DEFAULT_MEMORY_TURNS = 6  # how many previous messages are replayed to the LLM

MIN_CHUNK_SIZE, MAX_CHUNK_SIZE = 150, 900
MIN_TOP_K, MAX_TOP_K = 1, 6
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.0, 1.0

# Two ways to answer a question, side by side. This is the core teaching
# contrast of the project, not an implementation detail.
MODE_AGENT = "agent"  # the model decides when to call search_policy
MODE_DIRECT = "direct"  # we always retrieve first, then answer (classic RAG)

MODE_LABELS = {
    MODE_AGENT: "Agent (model decides)",
    MODE_DIRECT: "Direct (always retrieve)",
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def load_environment() -> None:
    """Load .env once, and set the Hugging Face / tokenizer env flags.

    These flags silence the first-run warnings students always ask about and
    make sure a slow model download does not fail after 10 seconds.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def groq_api_key() -> str | None:
    """Return the Groq key from the environment, or None if it is missing."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    return key or None


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """One immutable snapshot of every knob that changes an answer.

    Frozen + hashable on purpose: Streamlit's ``st.cache_resource`` can use it
    directly as a cache key, and nothing can mutate settings mid-request.
    """

    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    memory_turns: int = DEFAULT_MEMORY_TURNS
    mode: str = MODE_AGENT

    def __post_init__(self) -> None:
        if self.mode not in MODE_LABELS:
            raise ValueError(f"Unknown mode: {self.mode!r}")
        if self.model not in MODEL_CHOICES:
            object.__setattr__(self, "model", DEFAULT_MODEL)
        object.__setattr__(
            self, "temperature", _clamp(self.temperature, MIN_TEMPERATURE, MAX_TEMPERATURE)
        )
        object.__setattr__(self, "top_k", int(_clamp(self.top_k, MIN_TOP_K, MAX_TOP_K)))
        object.__setattr__(
            self, "chunk_size", int(_clamp(self.chunk_size, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE))
        )
        max_overlap = max(0, self.chunk_size // 2)
        object.__setattr__(self, "chunk_overlap", int(_clamp(self.chunk_overlap, 0, max_overlap)))
        object.__setattr__(self, "memory_turns", max(0, int(self.memory_turns)))

    @property
    def is_agent_mode(self) -> bool:
        return self.mode == MODE_AGENT

    def describe(self) -> str:
        """Short human-readable summary, handy in the UI footer."""
        return (
            f"{self.model} · temp {self.temperature:g} · k={self.top_k} · "
            f"chunk {self.chunk_size}/{self.chunk_overlap}"
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
