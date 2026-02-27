"""
Oracle RAG Service - Sliding Window Chunker
=============================================
Groups parsed subtitle entries into overlapping chunks suitable for
embedding and retrieval.

Pipeline Stage: 2 of 4 (Parse → Chunk → Embed → Store)

Why Sliding Window?
-------------------
Individual subtitle lines are too short for meaningful retrieval
(e.g., "Yes." or "Run!"). Full movie scripts are too long for
embedding models. The sliding window groups ~10 consecutive lines
(~30-60 seconds of dialogue) into chunks that capture complete
conversational exchanges.

The overlap of 3 lines ensures that dialogue spanning a chunk
boundary isn't split and lost. This is critical for questions
like "What did they discuss about X?" where the answer may
straddle two adjacent windows.

Chunk Structure:
    {
        "chunk_id": "inception_chunk_0",
        "movie_id": "inception",
        "chunk_index": 0,
        "text": "What is the most resilient parasite? An idea...",
        "timestamp_start": 75.2,
        "timestamp_end": 112.8,
        "num_entries": 10,
        "start_entry_index": 1,
        "end_entry_index": 10,
    }
"""

import logging
from dataclasses import dataclass, asdict

from .srt_parser import SubtitleEntry, format_timestamp
from .config import CHUNKING

logger = logging.getLogger(__name__)


# =============================================================================
# Data Model
# =============================================================================

@dataclass
class SubtitleChunk:
    """
    A chunk of consecutive subtitle entries, ready for embedding.

    Attributes:
        chunk_id: Unique identifier (movie_id + chunk_index)
        movie_id: Identifier for the source movie
        chunk_index: Sequential index of this chunk within the movie
        text: Combined text of all subtitle entries in this chunk
        timestamp_start: Start time of the first entry (seconds)
        timestamp_end: End time of the last entry (seconds)
        num_entries: Number of subtitle entries in this chunk
        start_entry_index: Index of first subtitle entry
        end_entry_index: Index of last subtitle entry
    """
    chunk_id: str
    movie_id: str
    chunk_index: int
    text: str
    timestamp_start: float
    timestamp_end: float
    num_entries: int
    start_entry_index: int
    end_entry_index: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def duration(self) -> float:
        """Duration of this chunk in seconds."""
        return self.timestamp_end - self.timestamp_start

    @property
    def timestamp_range(self) -> str:
        """Human-readable timestamp range (e.g., '1:15 - 2:03')."""
        return f"{format_timestamp(self.timestamp_start)} - {format_timestamp(self.timestamp_end)}"


# =============================================================================
# Sliding Window Chunker
# =============================================================================

def chunk_subtitles(
    entries: list[SubtitleEntry],
    movie_id: str,
    window_size: int | None = None,
    overlap: int | None = None,
) -> list[SubtitleChunk]:
    """
    Group subtitle entries into overlapping chunks using a sliding window.

    Algorithm:
        Given entries [E0, E1, E2, ..., EN] with window=4, overlap=1:

        Chunk 0: [E0, E1, E2, E3]           ← window_size entries
        Chunk 1:          [E3, E4, E5, E6]   ← slides by (window - overlap)
        Chunk 2:                   [E6, E7, E8, E9]
        ...

        The step size = window_size - overlap.
        Overlap entries appear in both adjacent chunks, preserving context.

    Args:
        entries: List of parsed SubtitleEntry objects (from srt_parser).
        movie_id: Unique identifier for the movie (used in chunk_id).
        window_size: Number of entries per chunk. Defaults to config value.
        overlap: Number of overlapping entries. Defaults to config value.

    Returns:
        List of SubtitleChunk objects ready for embedding.

    Example:
        >>> entries = parse_srt("inception.srt")  # 1,247 entries
        >>> chunks = chunk_subtitles(entries, "inception")
        >>> len(chunks)
        178
        >>> chunks[0].text
        "What is the most resilient parasite? An idea..."
        >>> chunks[0].timestamp_range
        "1:15 - 2:03"
    """
    window_size = window_size or CHUNKING["window_size"]
    overlap = overlap or CHUNKING["overlap"]
    min_chunk = CHUNKING["min_chunk_size"]
    separator = CHUNKING["line_separator"]
    max_chars = CHUNKING["max_chunk_chars"]

    if overlap >= window_size:
        raise ValueError(
            f"Overlap ({overlap}) must be less than window_size ({window_size})"
        )

    if not entries:
        logger.warning(f"No entries to chunk for movie '{movie_id}'")
        return []

    step = window_size - overlap
    chunks = []
    chunk_index = 0

    for start_idx in range(0, len(entries), step):
        end_idx = min(start_idx + window_size, len(entries))
        window_entries = entries[start_idx:end_idx]

        # Skip if too few entries to form a meaningful chunk
        if len(window_entries) < min_chunk:
            continue

        # Combine text from all entries in the window
        combined_text = separator.join(entry.text for entry in window_entries)

        # Truncate if exceeding max chars (safety for embedding model)
        if len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars].rsplit(" ", 1)[0]

        chunk = SubtitleChunk(
            chunk_id=f"{movie_id}_chunk_{chunk_index}",
            movie_id=movie_id,
            chunk_index=chunk_index,
            text=combined_text,
            timestamp_start=window_entries[0].start_time,
            timestamp_end=window_entries[-1].end_time,
            num_entries=len(window_entries),
            start_entry_index=window_entries[0].index,
            end_entry_index=window_entries[-1].index,
        )
        chunks.append(chunk)
        chunk_index += 1

        # If we've reached the end of entries, stop
        if end_idx >= len(entries):
            break

    logger.info(
        f"Chunked '{movie_id}': {len(entries)} entries → {len(chunks)} chunks "
        f"(window={window_size}, overlap={overlap}, step={step})"
    )

    return chunks


def get_chunk_stats(chunks: list[SubtitleChunk]) -> dict:
    """
    Compute statistics about the generated chunks for logging/debugging.

    Returns:
        Dict with chunk count, avg/min/max text length, avg/min/max duration,
        and total movie duration covered.
    """
    if not chunks:
        return {"chunk_count": 0}

    text_lengths = [len(c.text) for c in chunks]
    durations = [c.duration for c in chunks]

    return {
        "chunk_count": len(chunks),
        "avg_text_length": sum(text_lengths) / len(text_lengths),
        "min_text_length": min(text_lengths),
        "max_text_length": max(text_lengths),
        "avg_duration_seconds": sum(durations) / len(durations),
        "min_duration_seconds": min(durations),
        "max_duration_seconds": max(durations),
        "total_duration_seconds": chunks[-1].timestamp_end - chunks[0].timestamp_start,
        "movie_id": chunks[0].movie_id,
    }
