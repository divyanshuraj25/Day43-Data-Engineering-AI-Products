# Day 43 — Data Engineering for AI Products

A complete starter knowledge-base ingestion and retrieval evaluation pipeline for an AI Study Assistant.

## What's included
- 50 Markdown knowledge documents in `data/` (demo corpus tailored to an AI study assistant)
- Format-aware preprocessing for Markdown, TXT, HTML, and PDF
- Normalization, duplicate detection, metadata, and section-aware recursive chunking
- OpenAI embeddings and a persisted LangChain FAISS index
- Logged document/chunk counts, average chunk size, timing, and approximate embedding cost
- Ten-query retrieval review CSV with expected source labels and manual-review columns
- `incremental_update(new_docs)` that adds new unique documents without re-embedding existing documents
- Configurable cost projections for initial ingestion, monthly full re-embedding, and 10x corpus size

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```
Add your own OpenAI API key to `.env`. Never commit `.env`.

## Run
```powershell
python main.py
python evaluate.py
python cost_projection.py
```
Open `reports/retrieval_review.csv`. Review each of the ten queries by comparing the top-1 source and passage against the expected source. Replace `REVIEW` in `manual_correct_top1` and `flag_failure` with `YES` or `NO`, and write notes for failures. The script deliberately does not pretend that a machine-generated score is a human review.

## Incremental update example
Place a new supported file in `data/`, then run the pipeline again. Or call the function from Python:
```python
from main import incremental_update
print(incremental_update(["data/new_topic.md"]))
```
The function appends only unseen content hashes to an existing index. For production, implement changed-file deletion/replacement and index compaction, plus persistent source IDs across renames.

## Cost notes
The default `0.02 USD / 1M tokens` is a configurable example assumption, not a guaranteed current price. Check the current official OpenAI pricing before submitting. The script estimates tokens as characters / 4, so costs are approximate; actual billing depends on provider token counts and retries. For an exact report, record API usage from the embedding responses and cite the price/date used.

## Important limitations
- The included 50-document corpus is a reproducible demo corpus. If your Day 41 project has real source documents and its ten test questions, replace the demo corpus and `TESTS` in `evaluate.py` with those actual inputs before final submission.
- Scanned PDFs need OCR; this pipeline logs empty/short extraction instead of silently treating it as valid content.
- FAISS local loading uses pickle-based metadata deserialization. Only load indexes you created and trust.
- This pipeline creates an embedding/retrieval index; it does not itself generate final LLM answers.
