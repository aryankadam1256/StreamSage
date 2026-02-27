"""
Oracle RAG Evaluation Runner
==============================

Runs the full evaluation suite against the ingested ChromaDB collection:

  1. Retrieval quality (Precision@5, Recall@5, MRR@5, Timestamp Precision@5)
  2. Quality metrics (Context Faithfulness, Answer Relevance)
     — Only when Ollama is connected (skipped in retrieval-only mode)
  3. Latency profiling (per-stage breakdown)

Usage::

    # Full evaluation (requires oracle service dependencies)
    cd services/oracle-rag-service
    python -m evaluation.run_evaluation

    # Retrieval-only (no Ollama required)
    python -m evaluation.run_evaluation --retrieval-only

    # Save results to JSON
    python -m evaluation.run_evaluation --output results.json

Output summary::

    +==================================================╗
    ║         ORACLE RAG — EVALUATION REPORT          ║
    +==================================================+
    ║ Test Set:  14 cases  (5 easy / 5 medium / 4 hard)║
    ║ Ollama:    NOT connected (retrieval-only mode)   ║
    +==================================================+
    ║ RETRIEVAL                                        ║
    ║  Precision@5     : 0.71                         ║
    ║  Recall@5        : 0.82                         ║
    ║  MRR@5           : 0.75                         ║
    ║  Ts-Precision@5  : 0.64                         ║
    ╚==================================================+
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

# Ensure the service root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.test_set import TEST_CASES, summary as test_summary
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    timestamp_precision_at_k,
    context_faithfulness,
    answer_relevance,
    aggregate_scores,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================================
# Core Evaluation Logic
# ============================================================================

def evaluate_retrieval(
    collection,
    embedder,
    test_cases: list[dict],
    top_k: int = 5,
    verbose: bool = True,
) -> list[dict]:
    """
    Run retrieval evaluation for all test cases.

    For each test case:
    1. Embed the query
    2. Query ChromaDB for top-K chunks
    3. Compute Precision@K, Recall@K, MRR@K, Timestamp Precision@K

    Args:
        collection: ChromaDB collection.
        embedder: Initialized SubtitleEmbedder.
        test_cases: List of test case dicts from test_set.py.
        top_k: Number of chunks to retrieve.
        verbose: Print per-case results.

    Returns:
        List of per-case result dicts.
    """
    results = []

    if verbose:
        print(f"\n  {'ID':<4} {'Difficulty':<8} {'P@5':>5} {'R@5':>5} {'MRR':>5} {'TSP@5':>6}  Query")
        print("  " + "-" * 75)

    for tc in test_cases:
        # Embed and retrieve
        query_emb = embedder.embed_texts([tc["query"]])[0].tolist()
        try:
            raw = collection.query(
                query_embeddings=[query_emb],
                where={"movie_id": tc["movie_id"]},
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            chunks = []
            if raw["ids"][0]:
                for doc, meta, dist in zip(
                    raw["documents"][0],
                    raw["metadatas"][0],
                    raw["distances"][0],
                ):
                    chunks.append({
                        "content": doc,
                        "movie_id": meta.get("movie_id", ""),
                        "timestamp_start": meta.get("timestamp_start", 0),
                        "timestamp_end": meta.get("timestamp_end", 0),
                        "relevance_score": max(0.0, 1.0 - dist),
                    })
        except Exception as e:
            logger.error(f"ChromaDB error for {tc['id']}: {e}")
            chunks = []

        # Compute metrics
        p5 = precision_at_k(chunks, tc["expected_keywords"], top_k)
        r5 = recall_at_k(chunks, tc["expected_keywords"], top_k)
        mrr = reciprocal_rank(chunks, tc["expected_keywords"])
        tsp5 = timestamp_precision_at_k(chunks, tc["key_timestamp_range"], top_k)

        result = {
            "id": tc["id"],
            "query": tc["query"],
            "movie_id": tc["movie_id"],
            "difficulty": tc["difficulty"],
            "intent": tc["intent"],
            "precision@5": round(p5, 3),
            "recall@5": round(r5, 3),
            "mrr": round(mrr, 3),
            "ts_precision@5": round(tsp5, 3),
            "n_chunks_retrieved": len(chunks),
            "top_chunk_relevance": round(chunks[0]["relevance_score"], 3) if chunks else 0.0,
        }
        results.append(result)

        if verbose:
            query_short = tc["query"][:42]
            print(
                f"  {tc['id']:<4} {tc['difficulty']:<8} "
                f"{p5:>5.2f} {r5:>5.2f} {mrr:>5.2f} {tsp5:>6.2f}  {query_short}"
            )

    return results


def evaluate_quality(
    collection,
    embedder,
    ollama_client,
    test_cases: list[dict],
    top_k: int = 5,
    verbose: bool = True,
) -> list[dict]:
    """
    Run quality evaluation (faithfulness + answer relevance).

    Requires Ollama to generate answers; skipped in retrieval-only mode.

    Args:
        collection: ChromaDB collection.
        embedder: Initialized SubtitleEmbedder.
        ollama_client: OllamaClient instance.
        test_cases: List of test case dicts.
        top_k: Number of chunks to retrieve.
        verbose: Print per-case results.

    Returns:
        List of per-case quality result dicts.
    """
    from main import build_rag_prompt, classify_intent

    results = []

    if verbose:
        print(f"\n  {'ID':<4} {'Difficulty':<8} {'Faith':>6} {'AnsRel':>7}  Query")
        print("  " + "-" * 65)

    for tc in test_cases:
        # Retrieve context
        query_emb = embedder.embed_texts([tc["query"]])[0].tolist()
        try:
            raw = collection.query(
                query_embeddings=[query_emb],
                where={"movie_id": tc["movie_id"]},
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            chunks = []
            if raw["ids"][0]:
                for doc, meta, dist in zip(
                    raw["documents"][0],
                    raw["metadatas"][0],
                    raw["distances"][0],
                ):
                    chunks.append({
                        "content": doc,
                        "relevance_score": max(0.0, 1.0 - dist),
                        "timestamp_start": meta.get("timestamp_start", 0),
                        "timestamp_end": meta.get("timestamp_end", 0),
                    })
        except Exception:
            chunks = []

        # Build prompt and generate
        intent = classify_intent(tc["query"])
        prompt = build_rag_prompt(tc["query"], chunks, intent)
        answer = ollama_client.generate(prompt)

        # Quality metrics
        faith = context_faithfulness(answer, chunks)
        rel = answer_relevance(tc["query"], answer)

        result = {
            "id": tc["id"],
            "difficulty": tc["difficulty"],
            "faithfulness": round(faith, 3),
            "answer_relevance": round(rel, 3),
            "answer_preview": answer[:120].replace("\n", " "),
        }
        results.append(result)

        if verbose:
            query_short = tc["query"][:42]
            print(
                f"  {tc['id']:<4} {tc['difficulty']:<8} "
                f"{faith:>6.3f} {rel:>7.3f}  {query_short}"
            )

    return results


# ============================================================================
# Report Rendering
# ============================================================================

def print_report(
    retrieval_results: list[dict],
    quality_results: Optional[list[dict]],
    agg: dict,
    ollama_connected: bool,
    latency_agg: Optional[dict] = None,
):
    """Render a formatted evaluation report to stdout."""
    ts = test_summary()

    print("\n" + "=" * 60)
    print("  ORACLE RAG -- PHASE 5 EVALUATION REPORT")
    print("=" * 60)
    print(f"  Test Set : {ts['total']} cases", end="")
    diff = ts["by_difficulty"]
    print(f"  ({diff.get('easy',0)} easy / {diff.get('medium',0)} medium / {diff.get('hard',0)} hard)")
    ollama_str = "connected" if ollama_connected else "NOT connected (retrieval-only)"
    print(f"  Ollama   : {ollama_str}")
    print("-" * 60)

    print("\n  RETRIEVAL METRICS (top-5 chunks)")
    r = agg["retrieval"]
    print(f"    Precision@5       : {r['mean_precision@5']:.3f}  (fraction of top-5 chunks on-topic)")
    print(f"    Recall@5          : {r['mean_recall@5']:.3f}  (fraction of expected keywords found)")
    print(f"    MRR@5             : {r['mean_mrr']:.3f}  (1/rank of first relevant chunk)")
    print(f"    Ts-Precision@5    : {r['mean_ts_precision@5']:.3f}  (timestamp range overlap)")

    if "quality" in agg:
        print("\n  QUALITY METRICS")
        q = agg["quality"]
        print(f"    Context Faithful  : {q['mean_faithfulness']:.3f}  (answer grounded in sources)")
        print(f"    Answer Relevance  : {q['mean_answer_relevance']:.3f}  (TF-IDF query↔answer sim)")

    print("\n  BY DIFFICULTY")
    for diff, stats in sorted(agg.get("by_difficulty", {}).items()):
        print(f"    {diff:<8}  P@5={stats['mean_precision@5']:.3f}  MRR={stats['mean_mrr']:.3f}  (n={stats['n']})")

    if latency_agg:
        total_mean = latency_agg.get("total_ms", {}).get("mean", 0)
        print("\n  LATENCY BREAKDOWN")
        stage_labels = {
            "embedding_ms": "Query Embedding",
            "vector_search_ms": "Vector Search",
            "prompt_construction_ms": "Prompt Construction",
            "llm_generation_ms": "LLM Generation",
            "total_ms": "TOTAL",
        }
        for k, label in stage_labels.items():
            if k in latency_agg:
                mean_ms = latency_agg[k]["mean"]
                pct = mean_ms / total_mean * 100 if total_mean else 0
                print(f"    {label:<22} {mean_ms:>8.1f}ms  ({pct:.1f}%)")

    # Interpretation
    print("\n  INTERPRETATION")
    p5 = r["mean_precision@5"]
    mrr = r["mean_mrr"]
    if p5 >= 0.7 and mrr >= 0.7:
        print("    Retrieval is performing well. Most queries find relevant chunks.")
    elif p5 >= 0.5:
        print("    Retrieval is acceptable. Hard queries may need chunking refinement.")
    else:
        print("    Retrieval needs improvement. Consider reducing chunk size or overlap.")

    print()
    print("=" * 60)


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Oracle RAG retrieval and quality metrics."
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Skip quality eval (no Ollama needed)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file (e.g., results.json)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Chunks to retrieve per query (default: 5)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only easy test cases for a fast check"
    )
    args = parser.parse_args()

    print("  Loading Oracle RAG resources...")

    from data_pipeline.embedder import SubtitleEmbedder
    from data_pipeline.config import CHROMADB
    import chromadb
    from main import OllamaClient, OLLAMA_BASE_URL, LLM_MODEL

    embedder = SubtitleEmbedder()
    chroma_client = chromadb.PersistentClient(path=CHROMADB["persist_dir"])
    try:
        collection = chroma_client.get_collection(CHROMADB["collection_name"])
    except Exception:
        print("  [ERROR] Collection not found. Run the ingest pipeline first.")
        sys.exit(1)

    ollama_client = OllamaClient(OLLAMA_BASE_URL, LLM_MODEL)
    ollama_connected = ollama_client.connect()

    test_cases = TEST_CASES
    if args.quick:
        test_cases = [tc for tc in TEST_CASES if tc["difficulty"] == "easy"]
        print(f"  Quick mode: running {len(test_cases)} easy cases only.")

    print(f"  Collection: {collection.count()} chunks | {len(test_cases)} test cases\n")

    # -- Retrieval Evaluation -------------------------------------------------
    print("  RETRIEVAL EVALUATION")
    retrieval_results = evaluate_retrieval(
        collection, embedder, test_cases, top_k=args.top_k
    )

    # -- Quality Evaluation (skipped if no Ollama) -----------------------------
    quality_results = None
    if not args.retrieval_only and ollama_connected:
        print("\n  QUALITY EVALUATION (Ollama connected)")
        quality_results = evaluate_quality(
            collection, embedder, ollama_client, test_cases, top_k=args.top_k
        )
    elif not args.retrieval_only:
        print("\n  Quality evaluation skipped (Ollama not connected).")
        print("  Start Ollama with 'ollama pull llama3:8b' for full evaluation.")

    # -- Latency Profiling -----------------------------------------------------
    print("\n  LATENCY PROFILING (5 queries × 3 runs)")
    from evaluation.latency_profiler import run_profiling
    latency_agg = run_profiling(
        movie_id="the_signal", n_runs=3, top_k=args.top_k, verbose=True
    )

    # -- Aggregate -------------------------------------------------------------
    agg = aggregate_scores(retrieval_results, quality_results)

    # -- Report ----------------------------------------------------------------
    print_report(retrieval_results, quality_results, agg, ollama_connected, latency_agg)

    # -- Save to JSON ----------------------------------------------------------
    if args.output:
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ollama_connected": ollama_connected,
            "top_k": args.top_k,
            "test_set_summary": test_summary(),
            "retrieval_results": retrieval_results,
            "quality_results": quality_results or [],
            "aggregate": agg,
            "latency": latency_agg,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
