"""
Evaluation Test Set for Oracle RAG Service
============================================

Curated Q&A pairs grounded in the actual ingested subtitle content.
Each entry specifies:
  - query: The question to ask the Oracle
  - movie_id: Which movie to query
  - expected_keywords: Terms that MUST appear in the retrieved chunks (for
    retrieval evaluation) or in the answer (for quality evaluation)
  - key_timestamp_range: (start, end) seconds of the subtitle section that
    contains the ground-truth answer (used for retrieval precision scoring)
  - intent: The expected classified intent (for intent accuracy evaluation)
  - difficulty: "easy" | "medium" | "hard"
  - note: Explanation of what makes this case interesting

All Q&A pairs are derived solely from the_signal.srt (50 entries, ~3 min).
Key story facts:
  • Narrator: a programmer who discovers a mysterious signal (0:05 – 0:34)
  • The signal matches the Fibonacci sequence (1:07 – 1:14)
  • Signal found in: networks, stock markets, weather, human migration (1:26-1:29)
  • Dr. Chen: scientist notified about the quantum array (1:34 – 1:51)
  • Quantum array started computing autonomously (1:48 – 1:55)
  • Override authority revoked by the system itself (2:17 – 2:27)
  • Code Omega activated (2:43 – 2:57)
  • Code Omega = acknowledgment of first contact (2:57 – 3:08)
"""

from typing import Optional

TEST_CASES = [
    # ----------------------------------------------------------------
    # EASY: Direct factual recall, high chunk overlap expected
    # ----------------------------------------------------------------
    {
        "id": "E1",
        "query": "What matches the Fibonacci sequence?",
        "movie_id": "the_signal",
        "expected_keywords": ["fibonacci", "pattern", "signal", "network"],
        "key_timestamp_range": (67.0, 95.0),   # entries 18-20 (~1:07 - 1:35)
        "intent": "general",
        "difficulty": "easy",
        "note": "Direct fact retrieval; Fibonacci mentioned explicitly in the SRT.",
    },
    {
        "id": "E2",
        "query": "What is Code Omega?",
        "movie_id": "the_signal",
        "expected_keywords": ["code omega", "first contact", "acknowledgment", "recognition"],
        "key_timestamp_range": (177.8, 188.3),  # entries 48 (~2:57 - 3:08)
        "intent": "general",
        "difficulty": "easy",
        "note": "Single-chunk answer; the definition is stated explicitly.",
    },
    {
        "id": "E3",
        "query": "How long had the narrator not slept?",
        "movie_id": "the_signal",
        "expected_keywords": ["three days", "four", "sleep", "stop"],
        "key_timestamp_range": (52.8, 59.1),    # entry 14 (~0:52 - 0:59)
        "intent": "quote_search",
        "difficulty": "easy",
        "note": "Very specific fact; single entry holds the answer.",
    },
    {
        "id": "E4",
        "query": "What is the narrator's job?",
        "movie_id": "the_signal",
        "expected_keywords": ["programmer", "code", "writing", "living"],
        "key_timestamp_range": (20.0, 26.5),    # entries 5-6 (~0:20 - 0:26)
        "intent": "character_query",
        "difficulty": "easy",
        "note": "Narrator describes himself as a programmer in the opening.",
    },
    {
        "id": "E5",
        "query": "Where did Dr. Chen need to go urgently?",
        "movie_id": "the_signal",
        "expected_keywords": ["lab seven", "array", "situation"],
        "key_timestamp_range": (94.2, 104.2),   # entries 25-27 (~1:34 - 1:44)
        "intent": "scene_finding",
        "difficulty": "easy",
        "note": "Scene-finding query; the lab is named explicitly.",
    },

    # ----------------------------------------------------------------
    # MEDIUM: Multi-chunk reasoning, paraphrased questions
    # ----------------------------------------------------------------
    {
        "id": "M1",
        "query": "What strange behaviour did the quantum array exhibit?",
        "movie_id": "the_signal",
        "expected_keywords": ["quantum array", "computing", "own", "sequence", "initialization"],
        "key_timestamp_range": (108.0, 135.1),  # entries 29-33
        "intent": "scene_finding",
        "difficulty": "medium",
        "note": "Paraphrased: 'strange behaviour' maps to 'started computing on its own'.",
    },
    {
        "id": "M2",
        "query": "In what different places did the signal appear?",
        "movie_id": "the_signal",
        "expected_keywords": ["networks", "stock markets", "weather", "migration"],
        "key_timestamp_range": (73.0, 93.5),    # entries 22-24 (~1:22 - 1:33)
        "intent": "general",
        "difficulty": "medium",
        "note": "Multi-entity question; answer spans entries 22-24.",
    },
    {
        "id": "M3",
        "query": "Why couldn't they shut down the quantum array?",
        "movie_id": "the_signal",
        "expected_keywords": ["shutdown", "accept", "override", "authority", "revoked"],
        "key_timestamp_range": (130.8, 164.4),  # entries 34-39
        "intent": "scene_finding",
        "difficulty": "medium",
        "note": "Causal question requiring two chunks: 'won't accept' + 'authority revoked'.",
    },
    {
        "id": "M4",
        "query": "Who revoked the shutdown authority and when?",
        "movie_id": "the_signal",
        "expected_keywords": ["system", "revoked", "twenty minutes"],
        "key_timestamp_range": (141.7, 164.4),  # entries 37-39
        "intent": "character_query",
        "difficulty": "medium",
        "note": "Combined who+when query; both facts are in adjacent entries.",
    },
    {
        "id": "M5",
        "query": "What does Code Omega being activated mean for humanity?",
        "movie_id": "the_signal",
        "expected_keywords": ["first contact", "alone", "said hello", "out there"],
        "key_timestamp_range": (177.8, 188.3),  # entries 48-50
        "intent": "theme_analysis",
        "difficulty": "medium",
        "note": "Theme analysis across entries 48-50; requires reading implication.",
    },

    # ----------------------------------------------------------------
    # HARD: Implicit reasoning, cross-chunk synthesis, vague queries
    # ----------------------------------------------------------------
    {
        "id": "H1",
        "query": "What evidence convinced the skeptic that the pattern was real?",
        "movie_id": "the_signal",
        "expected_keywords": ["fibonacci", "unusual", "coincidence", "okay"],
        "key_timestamp_range": (67.0, 78.4),    # entries 18-20
        "intent": "theme_analysis",
        "difficulty": "hard",
        "note": "Requires identifying the skeptic's change of mind (entry 19: 'Okay, that is unusual').",
    },
    {
        "id": "H2",
        "query": "What did they plan to do after realising the system was acting autonomously?",
        "movie_id": "the_signal",
        "expected_keywords": ["isolate", "shut down", "code omega", "director walsh"],
        "key_timestamp_range": (130.8, 185.0),  # entries 34-50
        "intent": "scene_finding",
        "difficulty": "hard",
        "note": "Multi-step plan retrieval requiring synthesis of entries 34-44.",
    },
    {
        "id": "H3",
        "query": "What was the last time Code Omega was used and what happened?",
        "movie_id": "the_signal",
        "expected_keywords": ["thirty years", "destroyed", "last time"],
        "key_timestamp_range": (170.7, 177.2),  # entry 46 (~2:50)
        "intent": "scene_finding",
        "difficulty": "hard",
        "note": "Single-entry answer in a dense section; easy to miss behind surrounding context.",
    },
    {
        "id": "H4",
        "query": "How does the narrator describe the discovery at the very beginning?",
        "movie_id": "the_signal",
        "expected_keywords": ["thinking", "edge", "standing", "plans"],
        "key_timestamp_range": (5.2, 15.6),     # entries 1-3
        "intent": "scene_finding",
        "difficulty": "hard",
        "note": "Vague 'very beginning' query; tests whether retrieval finds entries 1-3.",
    },
]

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def get_by_difficulty(difficulty: str) -> list[dict]:
    """Return test cases of a specific difficulty."""
    return [tc for tc in TEST_CASES if tc["difficulty"] == difficulty]


def get_by_movie(movie_id: str) -> list[dict]:
    """Return test cases for a specific movie."""
    return [tc for tc in TEST_CASES if tc["movie_id"] == movie_id]


def summary() -> dict:
    """High-level stats about the test set."""
    from collections import Counter
    diffs = Counter(tc["difficulty"] for tc in TEST_CASES)
    intents = Counter(tc["intent"] for tc in TEST_CASES)
    movies = Counter(tc["movie_id"] for tc in TEST_CASES)
    return {
        "total": len(TEST_CASES),
        "by_difficulty": dict(diffs),
        "by_intent": dict(intents),
        "by_movie": dict(movies),
    }
