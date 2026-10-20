
"""
Gemini embedding cost projection helper.

Estimates are illustrative. Verify the current official Gemini pricing
and use provider-reported token usage for billing accuracy.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"
METRICS_PATH = REPORT_DIR / "ingestion_metrics.json"
OUTPUT_PATH = REPORT_DIR / "cost_projection.json"

PRICE_PER_MILLION = float(
    os.getenv("EMBEDDING_PRICE_PER_MILLION_TOKENS", "0.15")
)

if not METRICS_PATH.exists():
    print("Ingestion metrics not found. Run python main.py first.")
    raise SystemExit(1)

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

# Use the metrics produced by the current Gemini ingestion pipeline.
new_tokens = int(metrics.get("estimated_tokens_for_new_chunks", 0))
total_characters = int(metrics.get("total_chunk_characters", 0))

if total_characters <= 0:
    print("No chunk character count found in ingestion metrics.")
    raise SystemExit(1)

# Approximate corpus token count: characters divided by four.
# This is a rough estimate, not an exact tokenizer measurement.
full_corpus_tokens = total_characters // 4

result = {
    "embedding_provider": metrics.get(
        "embedding_provider", "Google Gemini"
    ),
    "embedding_model": metrics.get(
        "embedding_model",
        os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "models/gemini-embedding-001"
        )
    ),
    "assumed_price_usd_per_1m_input_tokens": PRICE_PER_MILLION,
    "new_tokens_estimated_this_run": new_tokens,
    "full_corpus_tokens_approx": full_corpus_tokens,
    "initial_ingestion_estimate_usd": round(
        new_tokens / 1_000_000 * PRICE_PER_MILLION, 8
    ),
    "monthly_full_reembedding_estimate_usd": round(
        full_corpus_tokens / 1_000_000 * PRICE_PER_MILLION, 8
    ),
    "projected_10x_corpus_embedding_cost_usd": round(
        full_corpus_tokens * 10 / 1_000_000 * PRICE_PER_MILLION, 8
    ),
    "estimated_token_basis": (
        "New tokens use the ingestion pipeline's estimate; full corpus "
        "tokens approximate total characters divided by four."
    ),
    "pricing_note": (
        "Illustrative only. Verify official Gemini pricing and actual "
        "provider usage before treating estimates as billing amounts."
    )
}

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

print("\n===== GEMINI COST PROJECTION =====")
print(json.dumps(result, indent=2))
print(f"\nReport saved to: {OUTPUT_PATH}")
