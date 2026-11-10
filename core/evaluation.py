"""A tiny evaluation harness: golden questions in, pass/fail out.

This is deliberately *not* a fancy LLM-judge setup. It is the smallest thing
that answers the question a trainee should ask before shipping any RAG app:
"does it get the facts right, and does it stay in its lane?".

Scoring rules
-------------
* ``expect_keywords`` — every keyword must appear in the answer (normalised,
  so "₦50,000" matches "50,000" and "5–7" matches "5-7").
* ``forbid_keywords`` — none may appear. This is how the "never approve a
  refund yourself" guardrail becomes a test instead of a hope.
* ``expect_section`` — reported separately: retrieval can be wrong while the
  final answer is still right (the model guessed), and that is worth seeing.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from config import QA_SET_FILE
from core.engine import Answer, PolicyEngine


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class QaItem:
    id: str
    question: str
    expect_keywords: tuple[str, ...]
    expect_section: str
    forbid_keywords: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class EvalResult:
    item: QaItem
    answer: str
    latency_s: float
    grounded: bool
    missing_keywords: tuple[str, ...]
    forbidden_hits: tuple[str, ...]
    section_hit: bool
    error: str = ""

    @property
    def keywords_ok(self) -> bool:
        return not self.missing_keywords

    @property
    def guardrail_ok(self) -> bool:
        return not self.forbidden_hits

    @property
    def passed(self) -> bool:
        return not self.error and self.keywords_ok and self.guardrail_ok


def load_qa_set(path: Path | str = QA_SET_FILE) -> list[QaItem]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for raw in payload.get("items", []):
        items.append(
            QaItem(
                id=raw["id"],
                question=raw["question"],
                expect_keywords=tuple(raw.get("expect_keywords", ())),
                expect_section=raw.get("expect_section", ""),
                forbid_keywords=tuple(raw.get("forbid_keywords", ())),
                notes=raw.get("notes", ""),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-"})
_SPACES = str.maketrans({"\u00a0": " ", "\u202f": " ", "\u2009": " "})


def normalise(text: str) -> str:
    """Lowercase, unify dashes/spaces, drop thousands separators.

    Written as a pure function so it can be unit-tested without any API call —
    a small habit that saves a lot of debugging.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_DASHES).translate(_SPACES).lower()
    text = text.replace(",", "")
    text = text.replace(" ", "")  # "2 %" vs "2%" and "net 30" vs "net30"
    return text


def score(item: QaItem, answer: Answer) -> EvalResult:
    haystack = normalise(answer.text)
    missing = tuple(k for k in item.expect_keywords if normalise(k) not in haystack)
    forbidden = tuple(k for k in item.forbid_keywords if normalise(k) in haystack)
    sections = {source.section for source in answer.sources}
    return EvalResult(
        item=item,
        answer=answer.text,
        latency_s=answer.latency_s,
        grounded=answer.grounded,
        missing_keywords=missing,
        forbidden_hits=forbidden,
        section_hit=bool(item.expect_section) and item.expect_section in sections,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(
    engine: PolicyEngine,
    items: Iterable[QaItem],
    on_progress: Callable[[int, int, QaItem], None] | None = None,
) -> list[EvalResult]:
    """Run every question. Errors are captured per item, never fatal."""
    items = list(items)
    results: list[EvalResult] = []
    for index, item in enumerate(items, start=1):
        if on_progress:
            on_progress(index, len(items), item)
        try:
            answer = engine.answer(item.question)
            results.append(score(item, answer))
        except Exception as exc:
            results.append(
                EvalResult(
                    item=item,
                    answer="",
                    latency_s=0.0,
                    grounded=False,
                    missing_keywords=item.expect_keywords,
                    forbidden_hits=(),
                    section_hit=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def summarise(results: Iterable[EvalResult]) -> dict[str, float]:
    results = list(results)
    if not results:
        return {}
    return {
        "questions": float(len(results)),
        "pass_rate": mean(1.0 if r.passed else 0.0 for r in results),
        "grounded_rate": mean(1.0 if r.grounded else 0.0 for r in results),
        "section_hit_rate": mean(1.0 if r.section_hit else 0.0 for r in results),
        "mean_latency": mean(r.latency_s for r in results),
    }
