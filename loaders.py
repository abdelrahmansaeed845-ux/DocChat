from pathlib import Path
from langchain_core.documents import Document
from pypdf import PdfReader
from typing import List

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

def load_pdf(path: Path)-> List[Document]:
    """One Document per page, so we keep page numbers for citations."""
    reader = PdfReader(path)
    doc: List[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or '').strip()
        if text:
            doc.append(
                Document(
                    page_content= text, 
                    metadata= {
                        'name': path.name,
                        'page_number': page_number
                    }
                )
            )
    return doc

def load_text(path: Path)-> List[Document]:
    """TXT and Markdown: the whole file is one Document."""

    text = path.read_text(encoding='utf-8', errors='replace').strip()
    doc: List[Document] = []
    if text:
        doc.append(
            Document(
                page_content= text, 
                metadata= {
                    'name': path.name,
                }
            )
        )
    return doc

def load_file(path:str | Path) -> List[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Use: {SUPPORTED_EXTENSIONS}"
        )
    doc = load_pdf(path) if suffix == '.pdf' else load_text(path)
    if not doc:
        raise ValueError(f"No extractable text in {path.name} (scanned PDF?)")
    return doc

def load_files(paths: list[str | Path]) -> list[Document]:
    all_docs: list[Document] = []
    for p in paths:
        all_docs.extend(load_file(p))
    return all_docs

import io


def load_bytes(filename: str, data: bytes) -> List[Document]:
    """Load an uploaded file (name + raw bytes), for Streamlit."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Use: {SUPPORTED_EXTENSIONS}"
        )

    docs: List[Document] = []

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"name": filename, "page_number": page_number},
                    )
                )
    else:
        text = data.decode("utf-8", errors="replace").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"name": filename}))

    if not docs:
        raise ValueError(f"No extractable text in {filename} (scanned PDF?)")
    return docs