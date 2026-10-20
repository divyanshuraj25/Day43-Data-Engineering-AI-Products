
"""
Day 43 — Data Engineering for AI Products
Gemini Embeddings + FAISS
Run: python main.py
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "faiss_index"
REPORT_DIR = ROOT / "reports"
MANIFEST = ROOT / "manifest.json"

MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "models/gemini-embedding-001"
)

PRICE_PER_MILLION_TOKENS = float(
    os.getenv("EMBEDDING_PRICE_PER_MILLION_TOKENS", "0.15")
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "850"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

SUPPORTED = {".md", ".txt", ".html", ".htm", ".pdf"}


# --------------------------------------------------
# 1. TEXT PREPROCESSING
# --------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def extract_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".md", ".txt"}:
        raw = path.read_text(
            encoding="utf-8",
            errors="replace"
        )

    elif suffix in {".html", ".htm"}:
        soup = BeautifulSoup(
            path.read_text(
                encoding="utf-8",
                errors="replace"
            ),
            "html.parser"
        )

        for node in soup(
            ["script", "style", "nav", "footer", "header"]
        ):
            node.decompose()

        raw = soup.get_text(separator="\n")

    elif suffix == ".pdf":
        import fitz

        pages = []

        with fitz.open(path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                page_text = page.get_text("text")

                if page_text.strip():
                    pages.append(
                        f"[Page {page_num}]\n{page_text}"
                    )

        raw = "\n\n".join(pages)

    else:
        raise ValueError(
            f"Unsupported file type: {suffix}"
        )

    return normalize_text(raw)


# --------------------------------------------------
# 2. DUPLICATE DETECTION
# --------------------------------------------------

def content_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_documents(
    paths: Iterable[Path]
) -> tuple[list[Document], dict]:

    docs = []
    errors = []
    seen_hashes = set()

    for path in paths:
        if path.suffix.lower() not in SUPPORTED:
            continue

        try:
            text = extract_file(path)

            if len(text) < 20:
                errors.append({
                    "file": path.name,
                    "error": "Too little extracted text"
                })
                continue

            digest = content_hash(text)

            if digest in seen_hashes:
                continue

            seen_hashes.add(digest)

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "source_id": digest[:16],
                        "content_hash": digest,
                        "file_type": path.suffix.lower(),
                        "ingested_at":
                            datetime.now(timezone.utc).isoformat()
                    }
                )
            )

        except Exception as exc:
            errors.append({
                "file": path.name,
                "error": f"{type(exc).__name__}: {exc}"
            })

    return docs, {
        "errors": errors,
        "unique_documents": len(docs)
    }


# --------------------------------------------------
# 3. DOCUMENT CHUNKING
# --------------------------------------------------

def split_documents(
    docs: list[Document]
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n## ",
            "\n# ",
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
        add_start_index=True
    )

    chunks = splitter.split_documents(docs)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_hash"] = content_hash(
            chunk.page_content
        )

    return chunks


def approx_tokens(text: str) -> int:
    # Rough estimate only; not the provider's exact token count.
    return max(1, len(text) // 4)


# --------------------------------------------------
# 4. MANIFEST MANAGEMENT
# --------------------------------------------------

def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(
            MANIFEST.read_text(encoding="utf-8")
        )

    return {
        "model": MODEL,
        "documents": {},
        "updated_at": None
    }


def save_manifest(manifest: dict) -> None:
    manifest["model"] = MODEL
    manifest["updated_at"] = (
        datetime.now(timezone.utc).isoformat()
    )

    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )


# --------------------------------------------------
# 5. GEMINI EMBEDDINGS
# --------------------------------------------------

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY missing. Add it to your .env file."
        )

    return GoogleGenerativeAIEmbeddings(
        model=MODEL,
        google_api_key=api_key
    )


# --------------------------------------------------
# 6. INGESTION AND FAISS INDEX
# --------------------------------------------------

def build_or_update_index() -> None:
    start = time.perf_counter()

    if not DATA_DIR.exists():
        raise RuntimeError(
            f"Data directory not found: {DATA_DIR}"
        )

    source_paths = sorted(
        p for p in DATA_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED
    )

    documents, load_report = load_documents(source_paths)
    chunks = split_documents(documents)

    if not chunks:
        raise RuntimeError(
            "No readable chunks found in data/."
        )

    manifest = load_manifest()

    if manifest.get("model") not in (None, MODEL):
        raise RuntimeError(
            "Embedding model changed. Back up and remove "
            "the old faiss_index/ and manifest.json, then rebuild."
        )

    if (
        (INDEX_DIR / "index.faiss").exists()
        and not MANIFEST.exists()
    ):
        raise RuntimeError(
            "Existing index has no manifest. Back it up and "
            "rebuild a fresh Gemini index."
        )

    old_hashes = {
        item["content_hash"]
        for item in manifest.get("documents", {}).values()
    }

    new_docs = [
        doc for doc in documents
        if doc.metadata["content_hash"] not in old_hashes
    ]

    new_chunks = split_documents(new_docs)

    embedding_start = time.perf_counter()
    embeddings = get_embeddings()

    if (INDEX_DIR / "index.faiss").exists():
        store = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

        if new_chunks:
            store.add_documents(new_chunks)

    else:
        store = FAISS.from_documents(
            chunks,
            embeddings
        )
        new_chunks = chunks

    embedding_seconds = (
        time.perf_counter() - embedding_start
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))

    for doc in documents:
        manifest["documents"][
            doc.metadata["source"]
        ] = {
            "content_hash": doc.metadata["content_hash"],
            "source_id": doc.metadata["source_id"],
            "char_count": len(doc.page_content)
        }

    save_manifest(manifest)

    total_seconds = time.perf_counter() - start
    total_chars = sum(
        len(chunk.page_content) for chunk in chunks
    )

    new_token_estimate = sum(
        approx_tokens(chunk.page_content)
        for chunk in new_chunks
    )

    metrics = {
        "timestamp_utc":
            datetime.now(timezone.utc).isoformat(),
        "embedding_provider": "Google Gemini",
        "embedding_model": MODEL,
        "document_files_seen": len(source_paths),
        "unique_documents_discovered": len(documents),
        "new_documents_embedded_this_run": len(new_docs),
        "chunks_in_current_source_set": len(chunks),
        "chunks_added_this_run": len(new_chunks),
        "average_chunk_size_characters": round(
            total_chars / len(chunks), 2
        ),
        "total_chunk_characters": total_chars,
        "estimated_tokens_for_new_chunks": new_token_estimate,
        "estimated_embedding_cost_usd_this_run": round(
            new_token_estimate / 1_000_000
            * PRICE_PER_MILLION_TOKENS,
            8
        ),
        "embedding_seconds": round(
            embedding_seconds, 3
        ),
        "total_ingestion_seconds": round(
            total_seconds, 3
        ),
        "preprocessing_errors": load_report["errors"],
        "pricing_note": (
            "Configured estimate only. Verify current Gemini "
            "embedding pricing and actual API usage."
        )
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    (REPORT_DIR / "ingestion_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("\n===== INGESTION METRICS =====")
    print(json.dumps(metrics, indent=2))
    print(f"\nFAISS index saved: {INDEX_DIR}")
    print("Next: python evaluate.py")
    print("Then: python cost_projection.py")


# --------------------------------------------------
# 7. INCREMENTAL DOCUMENT UPDATES
# --------------------------------------------------

def incremental_update(
    new_docs: list[str | Path]
) -> dict:
    """Append only previously unseen document content."""

    if not (INDEX_DIR / "index.faiss").exists():
        raise RuntimeError(
            "No FAISS index found. Run python main.py first."
        )

    if not MANIFEST.exists():
        raise RuntimeError(
            "manifest.json missing. Rebuild a trusted index first."
        )

    manifest = load_manifest()

    if manifest.get("model") != MODEL:
        raise RuntimeError(
            "Embedding model differs from stored manifest."
        )

    existing_hashes = {
        item["content_hash"]
        for item in manifest["documents"].values()
    }

    docs = []
    errors = []

    for item in new_docs:
        path = Path(item)

        if not path.is_absolute():
            path = (ROOT / path).resolve()

        try:
            if path.suffix.lower() not in SUPPORTED:
                raise ValueError(
                    f"Unsupported file type: {path.suffix}"
                )

            text = extract_file(path)

            if len(text) < 20:
                raise ValueError(
                    "Too little extracted text"
                )

            digest = content_hash(text)

            if digest in existing_hashes:
                continue

            existing_hashes.add(digest)

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "source_id": digest[:16],
                        "content_hash": digest,
                        "file_type": path.suffix.lower(),
                        "ingested_at":
                            datetime.now(timezone.utc).isoformat()
                    }
                )
            )

        except Exception as exc:
            errors.append({
                "file": str(path),
                "error": str(exc)
            })

    chunks = split_documents(docs)

    if chunks:
        store = FAISS.load_local(
            str(INDEX_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True
        )

        store.add_documents(chunks)
        store.save_local(str(INDEX_DIR))

        for doc in docs:
            manifest["documents"][
                doc.metadata["source"]
            ] = {
                "content_hash": doc.metadata["content_hash"],
                "source_id": doc.metadata["source_id"],
                "char_count": len(doc.page_content)
            }

        save_manifest(manifest)

    result = {
        "new_documents_added": len(docs),
        "chunks_added": len(chunks),
        "errors": errors
    }

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    build_or_update_index()
