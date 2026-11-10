"""Terminal version of the same assistant.

This is the file to open first when teaching: no Streamlit, no widgets, just
the pipeline printing to a terminal. It shares ``core/`` with the web app, so
anything you fix here is fixed in the demo too.

Usage:
    python cli.py                 # interactive chat (agent mode)
    python cli.py --demo          # run three sample questions, then chat
    python cli.py --mode direct   # classic single-shot RAG
    python cli.py --k 5           # retrieve more chunks
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    # Windows consoles default to cp1252 and choke on '₦' / '–'.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (
    DATA_FILE,
    DEFAULT_TOP_K,
    MODE_AGENT,
    MODE_DIRECT,
    RuntimeConfig,
    groq_api_key,
    load_environment,
)
from core.documents import load_documents
from core.embeddings import get_embeddings
from core.engine import (
    Done,
    MissingAPIKey,
    PolicyEngine,
    Token,
    ToolCallEvent,
)
from core.vectorstore import build_vector_store

DEMO_QUESTIONS = (
    "How many units do I need to order to get a bulk discount?",
    "My order arrived late, can I get a refund on the whole order?",
    "What are the supplier payment terms?",
)


def build_engine(config: RuntimeConfig) -> PolicyEngine:
    print("Loading documents…", flush=True)
    documents = load_documents(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    print(f"  {len(documents)} chunks from {DATA_FILE.name}")

    print("Loading embeddings (first run downloads ~90 MB)…", flush=True)
    embeddings = get_embeddings()

    print("Building FAISS index…", flush=True)
    store = build_vector_store(documents, embeddings)
    return PolicyEngine(config, store, documents)


def ask(engine: PolicyEngine, question: str, history: list[dict]) -> str:
    """Stream one answer to the terminal and return the final text."""
    print("=" * 68)
    print("Q:", question)
    print("-" * 68, flush=True)

    answer_text = ""
    for event in engine.stream(question, history):
        if isinstance(event, Token):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolCallEvent):
            print(f"\n[step] {event.call.name}({event.call.arguments})", flush=True)
        elif isinstance(event, Done):
            answer_text = event.answer.text
            answer = event.answer
            print(f"\n[grounded in {len(answer.sources)} chunk(s) · {answer.latency_s:.1f}s]")
            for source in answer.sources:
                print(f"  [{source.rank}] {source.section} ({source.similarity:.0%})")
    print()
    return answer_text


def chat_loop(engine: PolicyEngine) -> None:
    print("Type a question, 'quit' to exit, or 'reset' to clear memory.\n")
    history: list[dict] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            return
        if question.lower() == "reset":
            history.clear()
            print("[memory cleared]\n")
            continue

        try:
            answer = ask(engine, question, history)
        except Exception as exc:
            print(f"\n[error] {type(exc).__name__}: {exc}\n")
            continue

        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acme policy assistant (terminal)")
    parser.add_argument("--demo", action="store_true", help="run the sample questions first")
    parser.add_argument(
        "--mode",
        choices=[MODE_AGENT, MODE_DIRECT],
        default=MODE_AGENT,
        help="agent (model decides) or direct (always retrieve)",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="chunks to retrieve")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    load_environment()
    if not groq_api_key():
        print("Missing GROQ_API_KEY. Add it to .env (see .env.example).")
        return 1

    args = parse_args()
    config = RuntimeConfig(mode=args.mode, top_k=args.k, temperature=args.temperature)
    engine = build_engine(config)
    print(
        f"\nReady — model {config.model}, mode {config.mode}, "
        f"k={config.top_k}, {engine.stats['chunks']} chunks indexed.\n"
    )

    if args.demo:
        history: list[dict] = []
        for question in DEMO_QUESTIONS:
            answer = ask(engine, question, history)
            history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
        print()

    try:
        chat_loop(engine)
    except MissingAPIKey as exc:
        print(f"\n{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
