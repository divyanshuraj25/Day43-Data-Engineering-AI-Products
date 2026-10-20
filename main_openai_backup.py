"""
Day 43 — Data Engineering for AI Products
Run: pip install -r requirements.txt
Set OPENAI_API_KEY in your environment, then: python main.py
"""
from __future__ import annotations
import os, re, json, time, hashlib, csv, html
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "faiss_index"
REPORT_DIR = ROOT / "reports"
MANIFEST = ROOT / "manifest.json"
MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
# Pricing is a configurable estimate in USD per 1M input tokens; verify current pricing before submission.
PRICE_PER_MILLION_TOKENS = float(os.getenv("EMBEDDING_PRICE_PER_MILLION_TOKENS", "0.02"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "850"))  # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
SUPPORTED = {".md", ".txt", ".html", ".htm", ".pdf"}

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    return text.strip()

def extract_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
    elif suffix in {".html", ".htm"}:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        for node in soup(["script", "style", "nav", "footer", "header"]):
            node.decompose()
        raw = soup.get_text(separator="\n")
    elif suffix == ".pdf":
        import fitz
        pages = []
        with fitz.open(path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append(f"[Page {page_num}]\n{text}")
        raw = "\n\n".join(pages)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return normalize_text(raw)

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_documents(paths: Iterable[Path]) -> tuple[list[Document], dict]:
    docs, errors, seen_hashes = [], [], set()
    for path in paths:
        if path.suffix.lower() not in SUPPORTED:
            continue
        try:
            text = extract_file(path)
            if len(text) < 20:
                errors.append({"file": path.name, "error": "Too little extracted text; possibly scanned/empty"})
                continue
            digest = content_hash(text)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            docs.append(Document(page_content=text, metadata={
                "source": path.name, "source_id": digest[:16],
                "content_hash": digest, "file_type": path.suffix.lower(),
                "ingested_at": datetime.now(timezone.utc).isoformat()
            }))
        except Exception as exc:
            errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    return docs, {"errors": errors, "unique_documents": len(docs)}

def split_documents(docs: list[Document]) -> list[Document]:
    # Heading boundaries are preserved by markdown text, then long sections are split with overlap.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_hash"] = content_hash(chunk.page_content)
    return chunks

def approx_tokens(text: str) -> int:
    # Rough estimate only; actual token counts depend on tokenizer and content.
    return max(1, int(len(text) / 4))

def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"model": MODEL, "documents": {}, "updated_at": None}

def save_manifest(manifest: dict) -> None:
    manifest["model"] = MODEL
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def get_embeddings() -> OpenAIEmbeddings:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")
    return OpenAIEmbeddings(model=MODEL)

def build_or_update_index() -> None:
    start = time.perf_counter()
    source_paths = sorted(p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    documents, load_report = load_documents(source_paths)
    chunks = split_documents(documents)
    if not chunks:
        raise RuntimeError("No chunks extracted. Add readable source documents to data/.")
    manifest = load_manifest()
    if manifest.get("model") not in (None, MODEL):
        raise RuntimeError("Embedding model changed. Rebuild index from scratch; do not mix vector spaces.")
    old_hashes = {v["content_hash"] for v in manifest.get("documents", {}).values()}
    new_docs = [d for d in documents if d.metadata["content_hash"] not in old_hashes]
    new_chunks = split_documents(new_docs)
    embedding_start = time.perf_counter()
    if INDEX_DIR.exists() and (INDEX_DIR / "index.faiss").exists():
        store = FAISS.load_local(str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True)
        if new_chunks:
            store.add_documents(new_chunks)
    else:
        store = FAISS.from_documents(chunks, get_embeddings())
        new_chunks = chunks
    embedding_seconds = time.perf_counter() - embedding_start
    store.save_local(str(INDEX_DIR))
    for doc in documents:
        manifest["documents"][doc.metadata["source"]] = {
            "content_hash": doc.metadata["content_hash"],
            "source_id": doc.metadata["source_id"],
            "char_count": len(doc.page_content)
        }
    save_manifest(manifest)
    total_seconds = time.perf_counter() - start
    total_chars = sum(len(c.page_content) for c in chunks)
    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": MODEL,
        "unique_documents_discovered": len(documents),
        "document_files_seen": len(source_paths),
        "new_documents_embedded_this_run": len(new_docs),
        "chunks_in_current_source_set": len(chunks),
        "chunks_added_this_run": len(new_chunks),
        "average_chunk_size_characters": round(total_chars / len(chunks), 2),
        "total_chunk_characters_in_current_source_set": total_chars,
        "estimated_tokens_for_new_chunks": sum(approx_tokens(c.page_content) for c in new_chunks),
        "estimated_embedding_cost_usd_this_run": round(sum(approx_tokens(c.page_content) for c in new_chunks) / 1_000_000 * PRICE_PER_MILLION_TOKENS, 8),
        "embedding_seconds": round(embedding_seconds, 3),
        "total_ingestion_seconds": round(total_seconds, 3),
        "preprocessing_errors": load_report["errors"],
        "note": "Token counts and costs are estimates; confirm with API usage and current pricing."
    }
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "ingestion_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nFAISS index saved to: {INDEX_DIR}")
    print("Next: run `python evaluate.py` and manually review reports/retrieval_review.csv.")

def incremental_update(new_docs: list[str | Path]) -> dict:
    """Embed and append only previously unseen documents. Paths must point to readable files."""
    if not (INDEX_DIR / "index.faiss").exists():
        raise RuntimeError("No index exists yet. Run python main.py first.")
    manifest = load_manifest()
    if manifest.get("model") != MODEL:
        raise RuntimeError("Embedding model differs from the stored manifest.")
    existing = {v["content_hash"] for v in manifest.get("documents", {}).values()}
    docs, errors = [], []
    for item in new_docs:
        path = Path(item)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        try:
            text = extract_file(path)
            digest = content_hash(text)
            if digest in existing:
                continue
            docs.append(Document(page_content=text, metadata={
                "source": path.name, "source_id": digest[:16], "content_hash": digest,
                "file_type": path.suffix.lower(), "ingested_at": datetime.now(timezone.utc).isoformat()
            }))
            existing.add(digest)
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)})
    chunks = split_documents(docs)
    if chunks:
        store = FAISS.load_local(str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True)
        store.add_documents(chunks)
        store.save_local(str(INDEX_DIR))
        for doc in docs:
            manifest["documents"][doc.metadata["source"]] = {
                "content_hash": doc.metadata["content_hash"], "source_id": doc.metadata["source_id"],
                "char_count": len(doc.page_content)
            }
        save_manifest(manifest)
    return {"new_documents_added": len(docs), "chunks_added": len(chunks), "errors": errors}

if __name__ == "__main__":
    build_or_update_index()
