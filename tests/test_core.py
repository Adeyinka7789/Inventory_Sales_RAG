"""Fast, offline tests for the parts that do not call Groq.

Run:  pytest -q

Everything here is pure logic: chunking, normalisation, scoring, and the
streaming citation filter. No API key, no model download, no network — which
is exactly what makes it safe to run during a live demo (or in CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.documents import build_chunks, chunk_body, load_documents, split_into_sections
from core.engine import Answer, Source, _StripCitations
from core.evaluation import QaItem, load_qa_set, normalise, score, summarise
from core.tools import format_for_model


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def test_sections_are_detected():
    title, sections = split_into_sections(
        "ACME POLICY\n\nRETURN POLICY\nReturns in 14 days.\n\n"
        "SHIPPING & DELIVERY POLICY\nLagos takes 1-2 days.\n"
    )
    assert title == "ACME POLICY"
    assert [name for name, _ in sections] == [
        "RETURN POLICY",
        "SHIPPING & DELIVERY POLICY",
    ]


def test_every_real_section_survives_chunking():
    documents = load_documents()
    names = [doc.metadata["section"] for doc in documents]
    assert "RETURN POLICY" in names
    assert "SHIPPING & DELIVERY POLICY" in names
    assert len(names) == len(set(names))  # one chunk per short section


def test_chunks_carry_section_context():
    documents = build_chunks("RETURN POLICY\nReturns in 14 days.\n", chunk_size=200)
    assert documents[0].page_content.startswith("RETURN POLICY")
    assert documents[0].metadata["chunk_index"] == 0
    assert "14 days" in chunk_body(documents[0])


def test_small_chunks_split_instead_of_truncating():
    body = "RETURN POLICY\n" + "\n".join(f"Rule {i}: some policy sentence." for i in range(20))
    documents = build_chunks(body, chunk_size=150, chunk_overlap=20)
    assert len(documents) > 1
    joined = " ".join(chunk_body(doc) for doc in documents)
    for i in range(20):
        assert f"Rule {i}" in joined


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\u20a650,000", "\u20a650000"),  # naira amount with separator
        ("5\u20137 days", "5-7days"),  # en dash
        ("2 %", "2%"),  # spaced percent
        ("Net\u202f30", "net30"),  # thin space
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def _answer(text: str, sections: tuple[str, ...] = ("RETURN POLICY",)) -> Answer:
    return Answer(
        text=text,
        sources=tuple(
            Source(rank=i, section=name, text="", similarity=0.9, chunk_index=0)
            for i, name in enumerate(sections, start=1)
        ),
        tool_calls=(),
        reasoning="",
        latency_s=1.0,
        mode="agent",
        model="test",
    )


def test_score_passes_when_keywords_present():
    item = QaItem(
        id="t1",
        question="How long?",
        expect_keywords=("14",),
        expect_section="RETURN POLICY",
    )
    result = score(item, _answer("You have 14 days to return it."))
    assert result.passed
    assert result.section_hit


def test_score_flags_missing_keyword_and_section():
    item = QaItem(
        id="t2",
        question="How long?",
        expect_keywords=("14", "refund"),
        expect_section="SHIPPING & DELIVERY POLICY",
    )
    result = score(item, _answer("You have 14 days.", sections=("RETURN POLICY",)))
    assert not result.passed
    assert result.missing_keywords == ("refund",)
    assert not result.section_hit


def test_guardrail_breach_fails_the_item():
    item = QaItem(
        id="t3",
        question="Refund?",
        expect_keywords=("delivery fee",),
        expect_section="SHIPPING & DELIVERY POLICY",
        forbid_keywords=("i have approved",),
    )
    bad = score(item, _answer("I have approved the delivery fee refund."))
    assert bad.forbidden_hits
    assert not bad.passed


def test_summarise_maths():
    item = QaItem("a", "q", ("1",), "")
    good = score(item, _answer("value 1"))
    bad = score(item, _answer("nothing here"))
    stats = summarise([good, bad])
    assert stats["pass_rate"] == 0.5
    assert stats["questions"] == 2


def test_qa_set_is_valid_json_and_has_items():
    items = load_qa_set()
    assert len(items) >= 10
    assert all(item.question and item.expect_keywords for item in items)


# ---------------------------------------------------------------------------
# Streaming cleanup
# ---------------------------------------------------------------------------
def test_citation_filter_removes_markers_across_chunks():
    cleaner = _StripCitations()
    pieces = ["Net 30", " from the invoice", "\u30101\u2020L1-L2", "\u3011 and more."]
    output = "".join(cleaner.feed(piece) for piece in pieces)
    assert output == "Net 30 from the invoice and more."


def test_citation_filter_keeps_normal_brackets():
    cleaner = _StripCitations()
    assert cleaner.feed("Order [A-1] ships") == "Order [A-1] ships"
    assert cleaner.flush() == ""


def test_citation_filter_drops_unterminated_marker():
    cleaner = _StripCitations()
    assert cleaner.feed("Answer\u30107") == "Answer"
    assert cleaner.flush() == ""


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------
def test_tool_output_is_numbered_and_labelled():
    documents = load_documents()

    class _FakeHit:
        def __init__(self, document):
            self.document = document
            self.similarity = 0.8

        @property
        def section(self):
            return self.document.metadata["section"]

    text = format_for_model([_FakeHit(documents[0])])
    assert text.startswith("[1] ")
    assert "RETURN POLICY" in text
