"""
Latency Profiler for Oracle RAG Service
=========================================

Measures wall-clock time for each stage of the RAG pipeline independently:

    Stage 1: Query Embedding      (sentence-transformers forward pass)
    Stage 2: Vector Search        (ChromaDB HNSW query)
    Stage 3: Prompt Construction  (string formatting, negligible)
    Stage 4: LLM Generation       (Ollama first-token + total latency)
    -------------------------------------------------------------
    Total:   End-to-end latency

Why profile each stage?
- Embedding: model load (once) vs. per-query warm inference (~15ms)
- ChromaDB: usually <5ms for <100K docs; degrades with collection size
- LLM generation: dominates at 2-30s; first-token latency matters for UX
- If latency is high, profiling tells you WHERE to optimise

Usage::

    python -m evaluation.latency_profiler --movie the_signal --runs 5

Output format::

    +---------------------+-----------+-----------+-----------+------------+
    | Stage               | Mean (ms) | Min (ms)  | Max (ms)  | % of Total |
    +---------------------+-----------+-----------+-----------+------------+
    | Query Embedding     |      14.2 |      12.1 |      17.3 |       1.2% |
    | Vector Search       |       2.8 |       2.1 |       4.5 |       0.2% |
    | Prompt Construction |       0.1 |       0.1 |       0.2 |       0.0% |
    | LLM Generation      |    1140.5 |     980.0 |    1350.0 |      98.6% |
    | TOTAL               |    1157.6 |    994.3  |    1372.0 |     100.0% |
    +---------------------+-----------+-----------+-----------+------------+
"""

import sys
import time
import statistics
import argparse
import logging
from typing import Optional

sys.path.insert(0, ".")

logger = logging.getLogger(__name__)


# ============================================================================
# Stage Profiler
# ============================================================================

class StageTimer:
    """Simple context manager for timing a named stage."""

    def __init__(self, name: str):
        self.name = name
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def profile_pipeline(
    query: str,
    movie_id: str,
    embedder,
    collection,
    ollama_client,
    top_k: int = 5,
) -> dict:
    """
    Run the full RAG pipeline and return per-stage latencies.

    This replicates the exact steps of the /ask endpoint, with timing
    instrumentation around each stage.

    Args:
        query: The question to ask.
        movie_id: Which movie to retrieve from.
        embedder: Initialized SubtitleEmbedder.
        collection: ChromaDB collection.
        ollama_client: Initialized OllamaClient (may be in fallback mode).
        top_k: Number of chunks to retrieve.

    Returns:
        Dict with latencies:
        {
            "embedding_ms": float,
            "vector_search_ms": float,
            "prompt_construction_ms": float,
            "llm_generation_ms": float,
            "total_ms": float,
            "n_chunks": int,
            "answer_length": int,
        }
    """
    import numpy as np
    total_start = time.perf_counter()

    # -- Stage 1: Query Embedding ---------------------------------------------
    with StageTimer("embedding") as t_embed:
        query_embedding = embedder.embed_texts([query])[0].tolist()
    embedding_ms = t_embed.elapsed_ms

    # -- Stage 2: Vector Search ------------------------------------------------
    with StageTimer("search") as t_search:
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                where={"movie_id": movie_id},
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            chunks = []
            if results["ids"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    chunks.append({
                        "content": doc,
                        "movie_id": meta["movie_id"],
                        "timestamp_start": meta["timestamp_start"],
                        "timestamp_end": meta["timestamp_end"],
                        "relevance_score": max(0.0, 1.0 - dist),
                    })
        except Exception as e:
            logger.error(f"ChromaDB error: {e}")
            chunks = []
    search_ms = t_search.elapsed_ms

    # -- Stage 3: Prompt Construction -----------------------------------------
    with StageTimer("prompt") as t_prompt:
        from data_pipeline.srt_parser import format_timestamp
        context_lines = [
            f"[{i}] ({format_timestamp(c['timestamp_start'])} - {format_timestamp(c['timestamp_end'])})\n{c['content']}"
            for i, c in enumerate(chunks, 1)
        ]
        context = "\n\n".join(context_lines)
        prompt = f"You are The Oracle. Context:\n{context}\n\nQuestion: {query}"
    prompt_ms = t_prompt.elapsed_ms

    # -- Stage 4: LLM Generation -----------------------------------------------
    with StageTimer("llm") as t_llm:
        answer = ollama_client.generate(prompt) if ollama_client else "[no llm]"
    llm_ms = t_llm.elapsed_ms

    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "embedding_ms": round(embedding_ms, 2),
        "vector_search_ms": round(search_ms, 2),
        "prompt_construction_ms": round(prompt_ms, 2),
        "llm_generation_ms": round(llm_ms, 2),
        "total_ms": round(total_ms, 2),
        "n_chunks": len(chunks),
        "answer_length": len(answer),
        "ollama_connected": bool(ollama_client and ollama_client.connected),
    }


# ============================================================================
# Multi-run Aggregation
# ============================================================================

def run_profiling(
    movie_id: str,
    queries: Optional[list[str]] = None,
    n_runs: int = 5,
    top_k: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Run the profiler over multiple queries and compute statistics.

    Args:
        movie_id: Movie to query.
        queries: List of test queries. Defaults to 5 varied queries.
        n_runs: How many times to repeat each query (for stable averages).
        top_k: Chunks to retrieve per query.
        verbose: Print stage table to stdout.

    Returns:
        Dict with mean, min, max and stddev per stage.
    """
    from data_pipeline.embedder import SubtitleEmbedder
    from data_pipeline.config import CHROMADB
    import chromadb

    if queries is None:
        queries = [
            "What matches the Fibonacci sequence?",
            "What is Code Omega?",
            "What happened to the quantum array?",
            "Who is the narrator?",
            "Where did the signal appear?",
        ]

    if verbose:
        print("\n" + "=" * 60)
        print("  Oracle RAG ? Latency Profiler")
        print("=" * 60)
        print(f"  Movie  : {movie_id}")
        print(f"  Queries: {len(queries)}")
        print(f"  Runs   : {n_runs} per query")
        print("=" * 60)
        print("  Loading resources...")

    # Init resources
    embedder = SubtitleEmbedder()
    chroma_client = chromadb.PersistentClient(path=CHROMADB["persist_dir"])
    try:
        collection = chroma_client.get_collection(CHROMADB["collection_name"])
    except Exception:
        print("  [ERROR] Collection not found. Run ingest first.")
        return {}

    # Lazy import of OllamaClient to avoid circular import
    import importlib, sys
    main_mod = importlib.import_module("main")
    ollama_client = main_mod.OllamaClient(
        main_mod.OLLAMA_BASE_URL,
        main_mod.LLM_MODEL,
    )
    ollama_connected = ollama_client.connect()
    if verbose:
        llm_status = "connected" if ollama_connected else "NOT available (retrieval-only mode)"
        print(f"  Ollama : {llm_status}\n")

    # Collect results
    all_results = []
    for query in queries:
        for run in range(n_runs):
            result = profile_pipeline(
                query, movie_id, embedder, collection, ollama_client, top_k
            )
            result["query"] = query
            result["run"] = run
            all_results.append(result)
            if verbose:
                is_last_run = run == n_runs - 1
                if is_last_run:
                    print(
                        f"  [{query[:40]:<40}] "
                        f"embed={result['embedding_ms']:5.1f}ms "
                        f"search={result['vector_search_ms']:4.1f}ms "
                        f"llm={result['llm_generation_ms']:7.1f}ms "
                        f"total={result['total_ms']:7.1f}ms"
                    )

    stages = [
        "embedding_ms",
        "vector_search_ms",
        "prompt_construction_ms",
        "llm_generation_ms",
        "total_ms",
    ]
    stage_labels = {
        "embedding_ms": "Query Embedding",
        "vector_search_ms": "Vector Search (ChromaDB)",
        "prompt_construction_ms": "Prompt Construction",
        "llm_generation_ms": "LLM Generation",
        "total_ms": "TOTAL",
    }

    agg = {}
    for stage in stages:
        vals = [r[stage] for r in all_results]
        agg[stage] = {
            "mean": round(statistics.mean(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
        }

    if verbose:
        total_mean = agg["total_ms"]["mean"]
        print("\n" + "-" * 76)
        print(f"  {'Stage':<26} {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'% Total':>9}")
        print("-" * 76)
        for stage in stages:
            s = agg[stage]
            pct = (s["mean"] / total_mean * 100) if total_mean > 0 else 0
            sep = "-" * 76 if stage == "total_ms" else ""
            if sep:
                print(sep)
            label = stage_labels[stage]
            print(f"  {label:<26} {s['mean']:>7.1f}ms {s['min']:>7.1f}ms {s['max']:>7.1f}ms {pct:>8.1f}%")
        print("-" * 76)
        n_docs = collection.count()
        print(f"\n  Collection size : {n_docs} chunks")
        print(f"  top_k           : {top_k}")
        print(f"  Total runs      : {len(all_results)}")
        llm_pct = agg["llm_generation_ms"]["mean"] / total_mean * 100 if total_mean else 0
        if not ollama_connected:
            print("\n  Note: LLM stage is 0ms ? Ollama not connected (retrieval-only).")
            print("        Re-run with Ollama for realistic end-to-end timing.")
        elif llm_pct > 95:
            print(f"\n  Insight: LLM dominates at {llm_pct:.1f}% of total latency.")
            print("           Consider streaming (/ask/stream) to hide generation latency.")
        print()

    return agg


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile Oracle RAG pipeline latency per stage."
    )
    parser.add_argument(
        "--movie", default="the_signal",
        help="Movie ID to query (default: the_signal)"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of runs per query (default: 5)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Chunks to retrieve per query (default: 5)"
    )
    args = parser.parse_args()

    run_profiling(
        movie_id=args.movie,
        n_runs=args.runs,
        top_k=args.top_k,
    )
