"""Stage 1 + 2 of the pipeline: load the policy file and split it into chunks.

The original demo loaded the whole .txt and split it blindly. That works, but
chunks straddle section boundaries and the model gets context like "Orders of
500 units or more..." with no idea which policy it belongs to.

Here we split *section first, chunk second*: every chunk keeps a ``section``
metadata field, which powers the citations in the UI and the ``section_hit``
score in the evaluation tab.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_FILE, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

# "RETURN POLICY", "SHIPPING & DELIVERY POLICY", "SUPPLIER PAYMENT TERMS"
_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 &/,.'\-()]{3,}$")


def is_section_heading(line: str) -> bool:
    """True for the ALL-CAPS section titles used in policies.txt."""
    line = line.strip()
    if not line or not _HEADING_RE.match(line):
        return False
    # Reject lines that are mostly digits or punctuation.
    return sum(char.isalpha() for char in line) >= 4


def split_into_sections(raw_text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split the document into (title, [(section, body), ...]).

    The first non-empty line is treated as the document title *only* when the
    next non-empty line is a section heading — otherwise we would eat the
    first real section.
    """
    lines = raw_text.splitlines()
    non_empty = [index for index, line in enumerate(lines) if line.strip()]

    title = ""
    start = 0
    if len(non_empty) >= 2:
        first, second = non_empty[0], non_empty[1]
        gap = lines[first + 1 : second]
        # A title is the first line, followed by a blank line, followed by a
        # section heading. Requiring the gap stops us eating a real section in
        # files that begin directly with "RETURN POLICY".
        if any(not line.strip() for line in gap) and is_section_heading(lines[second]):
            title = lines[first].strip()
            start = first + 1

    sections: list[tuple[str, list[str]]] = []
    current_title = "GENERAL"
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_title, buffer.copy()))

    for line in lines[start:]:
        if is_section_heading(line):
            flush()
            buffer = []
            current_title = line.strip()
        else:
            buffer.append(line)
    flush()

    return title, [(name, "\n".join(body).strip()) for name, body in sections]


def build_chunks(
    raw_text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    source: str = "policies.txt",
) -> list[Document]:
    """Turn raw policy text into citation-ready LangChain documents."""
    title, sections = split_into_sections(raw_text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Policy documents are line-oriented: prefer breaking on newlines.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for section, body in sections:
        for index, piece in enumerate(splitter.split_text(body)):
            # Prepending the section name means the embedding also captures
            # *which policy* this is, not just the sentence text.
            content = f"{section}\n\n{piece.strip()}"
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "section": section,
                        "chunk_index": index,
                        "doc_title": title,
                        "source": source,
                    },
                )
            )
    return documents


def load_documents(
    path: Path | str = DATA_FILE,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Read the policy file from disk and chunk it."""
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    return build_chunks(
        raw_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source=path.name,
    )


def chunk_body(document: Document) -> str:
    """The chunk text without the section header we prepended.

    Keeps the UI from printing "RETURN POLICY" twice, once as a badge and
    once as the first line of the body.
    """
    section = document.metadata.get("section", "")
    prefix = f"{section}\n\n"
    if section and document.page_content.startswith(prefix):
        return document.page_content[len(prefix) :]
    return document.page_content
