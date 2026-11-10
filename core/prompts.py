"""All prompt text in one place, so an instructor can diff them like code.

Two prompts because the project teaches two pipelines:

* ``SYSTEM_PROMPT``  — agentic. The model must *decide* to call the tool.
* ``DIRECT_PROMPT``  — classic RAG. We retrieve first; the model only writes.

Both keep the business guardrails from the original demo (never promise a
refund; say "I don't know" instead of guessing).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from config import DEFAULT_MEMORY_TURNS
from core.documents import chunk_body
from core.vectorstore import Hit

# ---------------------------------------------------------------------------
# Agentic mode
# ---------------------------------------------------------------------------
# The original demo's system prompt said "always use the tool" — in testing it
# still answered from general knowledge about a third of the time, because
# "always" reads as advice rather than a hard rule. Numbered, imperative rules
# fixed it. That difference is worth showing students live.
SYSTEM_PROMPT = """You are the Inventory & Sales policy assistant for Acme Inventory & Sales Co.

RULES — follow them in order:
1. Before answering ANY question about company policy, you MUST call the `search_policy` tool.
   Never answer a policy question from memory, even if you think you know the answer.
2. Answer ONLY using the text returned by `search_policy`. Do not add outside facts,
   do not generalise, and do not soften the numbers.
3. Quote figures exactly as written (days, percentages, currency amounts).
4. If the tool output does not contain the answer, say clearly that you could not
   find it in the policy documents and suggest contacting a human. Do not guess.
5. NEVER promise a refund, a discount, or an exception. Refunds and bulk discounts
   require human approval — say that explicitly.
6. Keep answers short: 1–3 sentences, or a compact list when the policy is a list.
   Mention which policy section you used."""


# ---------------------------------------------------------------------------
# Direct mode
# ---------------------------------------------------------------------------
DIRECT_PROMPT = """You answer questions about Acme Inventory & Sales Co. policies.

You will be given POLICY EXCERPTS that were retrieved for the user's question.
RULES:
1. Answer ONLY from the excerpts. If they do not contain the answer, say so.
2. Quote figures exactly (days, percentages, currency amounts).
3. NEVER promise a refund, discount, or exception — those need human approval.
4. Be concise: 1–3 sentences, or a short list.
5. If the excerpts disagree, say so instead of picking one silently."""


def format_context(hits: list[Hit]) -> str:
    """Numbered excerpts, the same shape the agent's tool returns."""
    if not hits:
        return "(no policy excerpts were retrieved)"
    blocks = []
    for rank, hit in enumerate(hits, start=1):
        blocks.append(f"[{rank}] {hit.section}\n{chunk_body(hit.document)}")
    return "\n\n".join(blocks)


def build_direct_messages(question: str, hits: list[Hit], history: list[dict] | None = None):
    """Message list for single-shot RAG: system, history, question + context."""
    messages = [("system", DIRECT_PROMPT), *history_messages(history)]
    user_block = f"POLICY EXCERPTS:\n{format_context(hits)}\n\nQUESTION: {question}"
    messages.append(("user", user_block))
    return messages


def history_messages(history: list[dict] | None, max_turns: int = DEFAULT_MEMORY_TURNS):
    """Convert the UI's simple dict history into LangChain messages.

    Trimming happens here so a long demo chat cannot blow the context window.
    """
    if not history:
        return []
    recent = history[-max_turns:] if max_turns > 0 else []
    messages = []
    for item in recent:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages
