
import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "faiss_index"
REPORT_DIR = ROOT / "reports"

MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "models/gemini-embedding-001"
)

# Ten representative retrieval evaluation questions.
TESTS = [
    (
        "What should the assistant do when the knowledge base lacks an answer?",
        "no_answer.md"
    ),
    (
        "How are scripts and navigation removed from HTML?",
        "html_cleaning.md"
    ),
    (
        "Why is chunk overlap useful, and what is its trade-off?",
        "chunk_overlap.md"
    ),
    (
        "What information should be stored in chunk metadata?",
        "chunk_metadata.md"
    ),
    (
        "How is embedding cost estimated from token volume?",
        "token_cost.md"
    ),
    (
        "What is the recommended approach when adding new documents to an existing index?",
        "incremental_updates.md"
    ),
    (
        "Why should the index be rebuilt if the embedding model changes?",
        "index_versioning.md"
    ),
    (
        "What should be checked when extracting text from scanned PDFs?",
        "ocr_limits.md"
    ),
    (
        "Which metrics should the ingestion pipeline report?",
        "latency_metrics.md"
    ),
    (
        "How can unsupported answers be reduced and source attribution retained?",
        "grounded_answers.md"
    ),
]


def main():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY missing. Check your .env file."
        )

    if not (INDEX_DIR / "index.faiss").exists():
        raise RuntimeError("Run python main.py first.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # Use the same embedding model that created the index.
    store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "retrieval_review.csv"

    fieldnames = [
        "query",
        "expected_source",
        "top1_source",
        "top1_chunk_preview",
        "retrieval_score",
        "manual_correct_top1",
        "flag_failure",
        "review_notes"
    ]

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for number, (query, expected) in enumerate(TESTS, start=1):
            print(f"\n{'=' * 65}")
            print(f"TEST {number}/{len(TESTS)}")
            print("QUERY:", query)
            print("EXPECTED SOURCE:", expected)

            try:
                results = store.similarity_search_with_score(
                    query,
                    k=3
                )

                if not results:
                    print("No matching documents found.")

                    writer.writerow({
                        "query": query,
                        "expected_source": expected,
                        "top1_source": "NO_RESULT",
                        "top1_chunk_preview": "",
                        "retrieval_score": "",
                        "manual_correct_top1": "NO",
                        "flag_failure": "YES",
                        "review_notes": "No documents returned."
                    })
                    continue

                doc, score = results[0]
                source = doc.metadata.get("source", "unknown")
                preview = doc.page_content[:240].replace("\n", " ")

                print("TOP-1 SOURCE:", source)
                print("FAISS DISTANCE:", round(float(score), 4))
                print("PREVIEW:", preview)

                writer.writerow({
                    "query": query,
                    "expected_source": expected,
                    "top1_source": source,
                    "top1_chunk_preview": preview,
                    "retrieval_score": round(float(score), 4),
                    "manual_correct_top1": "REVIEW",
                    "flag_failure": "REVIEW",
                    "review_notes": (
                        "Compare top-1 source and content with "
                        "expected source; replace REVIEW manually."
                    )
                })

            except Exception as exc:
                print("Evaluation error:", str(exc))

                writer.writerow({
                    "query": query,
                    "expected_source": expected,
                    "top1_source": "ERROR",
                    "top1_chunk_preview": "",
                    "retrieval_score": "",
                    "manual_correct_top1": "NO",
                    "flag_failure": "YES",
                    "review_notes": str(exc)[:300]
                })

    print(f"\n{'=' * 65}")
    print("Evaluation completed.")
    print(f"CSV report saved to: {report_path}")
    print(
        "Review retrieval_review.csv and replace REVIEW with "
        "YES/NO after checking each result."
    )


if __name__ == "__main__":
    main()
