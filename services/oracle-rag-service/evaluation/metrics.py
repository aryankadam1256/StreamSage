"""
Evaluation Metrics for Oracle RAG Service
==========================================

Implements a lightweight evaluation suite inspired by RAGAS
(Retrieval-Augmented Generation Assessment).

Reference:
    "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
    (Es et al., 2023 — https://arxiv.org/abs/2309.15217)

Metrics implemented (no external LLM required):
┌─────────────────────────────┬──────────────────────────────────────────────────┐
│ Metric                      │ Definition                                       │
├─────────────────────────────┼──────────────────────────────────────────────────┤
│ Retrieval Precision@K       │ Fraction of top-K chunks that are keyword-       │
│                             │ relevant to the expected answer                  │
│ Retrieval Recall@K          │ Fraction of expected keywords found in ANY       │
│                             │ of the top-K chunks                              │
│ MRR@K                       │ 1/rank of the first keyword-relevant chunk,      │
│                             │ averaged over all queries                        │
│ Timestamp Precision@K       │ Fraction of retrieved chunks whose timestamp     │
│                             │ overlaps the ground-truth timestamp range        │
│ Context Faithfulness        │ Fraction of non-stopword answer tokens present   │
│                             │ in at least one retrieved chunk (grounding score)│
│ Answer Relevance            │ TF-IDF cosine similarity between query and answer│
└─────────────────────────────┴──────────────────────────────────────────────────┘

Why these metrics?
- Retrieval Precision/Recall: detect when the vector search misses ground truth
- MRR: penalises relevant chunks ranked low (rank 3 is worse than rank 1)
- Timestamp Precision: validates that the timestamp re-ranking logic helps
- Context Faithfulness: detects hallucination risk (answer uses non-retrieved info)
- Answer Relevance: detects off-topic answers / prompt drift
"""

import re
import math
from typing import Optional


# ============================================================================
# Stopword filtering (English)
# ============================================================================

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "should", "can", "could", "may", "might", "shall", "to", "of",
    "in", "on", "at", "by", "for", "with", "about", "from", "up", "out",
    "as", "into", "that", "this", "it", "its", "i", "you", "he", "she", "we",
    "they", "what", "which", "who", "how", "when", "where", "why", "not",
    "no", "so", "if", "then", "than", "there", "here", "just", "very",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, return non-stopword tokens."""
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


# ============================================================================
# Chunk Relevance Scoring (keyword-based)
# ============================================================================

def chunk_contains_keywords(chunk: dict, keywords: list[str]) -> bool:
    """
    Check if a retrieved chunk's content contains any expected keywords.

    A keyword phrase like "first contact" matches if all individual words
    in the phrase appear in the chunk text (proximity not required).

    Args:
        chunk: Retrieved chunk dict with "content" key.
        keywords: List of keyword strings (can be phrases like "fibonacci sequence").

    Returns:
        True if at least one keyword is found in the chunk.
    """
    content_lower = chunk["content"].lower()
    for kw in keywords:
        kw_words = kw.lower().split()
        if all(w in content_lower for w in kw_words):
            return True
    return False


def timestamp_overlaps(chunk: dict, ts_range: tuple[float, float]) -> bool:
    """
    Check if a retrieved chunk's timestamp range overlaps the ground-truth range.

    Overlap condition: chunk_start < gt_end AND chunk_end > gt_start.

    Args:
        chunk: Retrieved chunk dict with timestamp_start and timestamp_end.
        ts_range: (start, end) in seconds from the test case.

    Returns:
        True if there is any temporal overlap.
    """
    c_start = chunk.get("timestamp_start", 0)
    c_end = chunk.get("timestamp_end", 0)
    gt_start, gt_end = ts_range
    return c_start < gt_end and c_end > gt_start


# ============================================================================
# Retrieval Metrics
# ============================================================================

def precision_at_k(chunks: list[dict], keywords: list[str], k: int = 5) -> float:
    """
    Precision@K: Fraction of top-K retrieved chunks that contain
    at least one expected keyword.

    Range: [0.0, 1.0]  Higher is better.
    Perfect score = 1.0 (all retrieved chunks are relevant).

    Args:
        chunks: Top-K retrieved chunks (in rank order).
        keywords: Expected keyword phrases.
        k: How many top chunks to evaluate.

    Returns:
        Precision@K score.
    """
    evaluated = chunks[:k]
    if not evaluated:
        return 0.0
    relevant = sum(1 for c in evaluated if chunk_contains_keywords(c, keywords))
    return relevant / len(evaluated)


def recall_at_k(chunks: list[dict], keywords: list[str], k: int = 5) -> float:
    """
    Recall@K: Fraction of expected keywords that appear in ANY
    of the top-K retrieved chunks.

    Measures whether the retrieval system found ALL the information
    needed to answer, not just some of it.

    Range: [0.0, 1.0]  Higher is better.

    Args:
        chunks: Top-K retrieved chunks (in rank order).
        keywords: Expected keyword phrases.
        k: How many top chunks to evaluate.

    Returns:
        Recall@K score.
    """
    evaluated = chunks[:k]
    if not keywords or not evaluated:
        return 0.0

    combined_content = " ".join(c["content"].lower() for c in evaluated)
    found = sum(
        1 for kw in keywords
        if all(w in combined_content for w in kw.lower().split())
    )
    return found / len(keywords)


def reciprocal_rank(chunks: list[dict], keywords: list[str]) -> float:
    """
    Reciprocal Rank: 1/rank of the FIRST keyword-relevant chunk.

    RR = 1.0 → relevant chunk at rank 1 (best)
    RR = 0.5 → relevant chunk at rank 2
    RR = 0.2 → relevant chunk at rank 5
    RR = 0.0 → no relevant chunk found

    Args:
        chunks: Retrieved chunks in rank order.
        keywords: Expected keyword phrases.

    Returns:
        Reciprocal rank score.
    """
    for rank, chunk in enumerate(chunks, start=1):
        if chunk_contains_keywords(chunk, keywords):
            return 1.0 / rank
    return 0.0


def timestamp_precision_at_k(
    chunks: list[dict],
    ts_range: tuple[float, float],
    k: int = 5,
) -> float:
    """
    Timestamp Precision@K: Fraction of top-K chunks whose timestamp
    range overlaps the ground-truth containing the answer.

    Evaluates whether the timestamp re-ranking logic correctly promotes
    chunks from the relevant time window.

    Args:
        chunks: Top-K retrieved chunks.
        ts_range: (start, end) seconds from the test case.
        k: How many top chunks to evaluate.

    Returns:
        Timestamp precision score.
    """
    evaluated = chunks[:k]
    if not evaluated:
        return 0.0
    overlapping = sum(1 for c in evaluated if timestamp_overlaps(c, ts_range))
    return overlapping / len(evaluated)


# ============================================================================
# Quality Metrics (RAGAS-lite, no external LLM needed)
# ============================================================================

def context_faithfulness(answer: str, chunks: list[dict]) -> float:
    """
    Context Faithfulness (RAGAS-lite): Fraction of non-trivial answer
    tokens that appear in at least one retrieved chunk.

    Full RAGAS faithfulness uses an NLI model to check if each claim in
    the answer is entailed by the context. This approximates it with
    token overlap, which is fast and LLM-free.

    High score: the answer is grounded in retrieved text (good)
    Low score:  the answer contains information not in the context (hallucination risk)

    Args:
        answer: The generated answer string.
        chunks: Retrieved chunks used as context.

    Returns:
        Faithfulness score [0.0, 1.0].
    """
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 1.0  # empty answer is trivially faithful

    context_text = " ".join(c["content"] for c in chunks).lower()
    context_tokens = set(tokenize(context_text))

    grounded = answer_tokens & context_tokens
    return len(grounded) / len(answer_tokens)


def answer_relevance(query: str, answer: str) -> float:
    """
    Answer Relevance (RAGAS-lite): TF-IDF cosine similarity between
    the query and the answer.

    Full RAGAS answer relevance generates n artificial questions from
    the answer and measures similarity to the original query. This
    approximates with direct TF-IDF cosine similarity.

    High score: the answer addresses the query topic
    Low score:  the answer is off-topic (prompt drift / wrong context)

    Args:
        query: The original user question.
        answer: The generated answer string.

    Returns:
        Answer relevance score [0.0, 1.0].
    """
    query_tokens = tokenize(query)
    answer_tokens = tokenize(answer)

    if not query_tokens or not answer_tokens:
        return 0.0

    # Build TF vectors
    all_terms = set(query_tokens) | set(answer_tokens)

    def tf_vector(tokens: list[str]) -> dict[str, float]:
        count = {}
        for t in tokens:
            count[t] = count.get(t, 0) + 1
        total = len(tokens)
        return {t: c / total for t, c in count.items()}

    q_vec = tf_vector(query_tokens)
    a_vec = tf_vector(answer_tokens)

    # Cosine similarity
    dot = sum(q_vec.get(t, 0) * a_vec.get(t, 0) for t in all_terms)
    q_mag = math.sqrt(sum(v**2 for v in q_vec.values()))
    a_mag = math.sqrt(sum(v**2 for v in a_vec.values()))

    if q_mag == 0 or a_mag == 0:
        return 0.0

    return dot / (q_mag * a_mag)


# ============================================================================
# Aggregate Score
# ============================================================================

def aggregate_scores(
    retrieval_results: list[dict],
    quality_results: Optional[list[dict]] = None,
) -> dict:
    """
    Compute aggregate metrics across all evaluated test cases.

    Args:
        retrieval_results: List of per-case retrieval metric dicts.
        quality_results: Optional list of per-case quality metric dicts.

    Returns:
        Dict with mean scores and per-difficulty breakdown.
    """
    def mean(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    agg = {
        "n_cases": len(retrieval_results),
        "retrieval": {
            "mean_precision@5": mean([r["precision@5"] for r in retrieval_results]),
            "mean_recall@5": mean([r["recall@5"] for r in retrieval_results]),
            "mean_mrr": mean([r["mrr"] for r in retrieval_results]),
            "mean_ts_precision@5": mean([r["ts_precision@5"] for r in retrieval_results]),
        },
    }

    if quality_results:
        agg["quality"] = {
            "mean_faithfulness": mean([q["faithfulness"] for q in quality_results]),
            "mean_answer_relevance": mean([q["answer_relevance"] for q in quality_results]),
        }

    # Per-difficulty breakdown
    difficulties = sorted({r["difficulty"] for r in retrieval_results})
    agg["by_difficulty"] = {}
    for diff in difficulties:
        subset = [r for r in retrieval_results if r["difficulty"] == diff]
        agg["by_difficulty"][diff] = {
            "n": len(subset),
            "mean_precision@5": mean([r["precision@5"] for r in subset]),
            "mean_mrr": mean([r["mrr"] for r in subset]),
        }

    return agg
