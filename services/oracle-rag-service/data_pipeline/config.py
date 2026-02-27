"""
Oracle RAG Service - Pipeline Configuration
=============================================
Central configuration for the subtitle ingestion pipeline.
Controls chunking parameters, embedding model, ChromaDB settings,
and file paths.

Design Decisions:
- Chunk window of 10 lines (~30-60s of dialogue) balances semantic
  completeness with retrieval precision.
- Overlap of 3 lines prevents context loss at chunk boundaries.
- all-MiniLM-L6-v2 matches the embedding model used in movie-assistant-service
  for consistency across the platform.
- Single ChromaDB collection with movie_id metadata filter scales better
  than per-movie collections for cross-movie queries and maintenance.
"""

import os
from pathlib import Path

# =============================================================================
# Base Paths
# =============================================================================
SERVICE_ROOT = Path(__file__).parent.parent
DATA_DIR = SERVICE_ROOT / "data"

# =============================================================================
# Subtitle Parsing Configuration
# =============================================================================
SRT_PARSER = {
    # Tags to strip from subtitle text (common in .srt files)
    "strip_html_tags": True,
    # Patterns to remove (sound effects, music cues, etc.)
    "remove_patterns": [
        r"\[.*?\]",         # [music playing], [gunshot], etc.
        r"\(.*?\)",         # (sighs), (laughing), etc.
        r"♪.*?♪",          # Music notes
        r"<.*?>",           # HTML tags like <i>, <b>, </i>
        r"^\s*-\s*",        # Leading dashes (speaker indicators in some SRTs)
    ],
    # Minimum text length after cleaning to keep a subtitle entry
    "min_text_length": 2,
    # Supported encodings to try when reading .srt files
    "encodings": ["utf-8", "latin-1", "cp1252", "iso-8859-1"],
}

# =============================================================================
# Chunking Configuration
# =============================================================================
CHUNKING = {
    # Number of subtitle entries per chunk
    # ~10 lines = 30-60 seconds of dialogue (a complete exchange)
    "window_size": 10,
    # Number of overlapping entries between consecutive chunks
    # Prevents context loss at chunk boundaries
    "overlap": 3,
    # Minimum number of entries to form a valid chunk
    # (handles end-of-file edge case)
    "min_chunk_size": 3,
    # Separator between subtitle lines within a chunk
    "line_separator": " ",
    # Maximum characters per chunk (safety limit for embedding models)
    # all-MiniLM-L6-v2 has 256 word-piece token limit (~1200 chars)
    "max_chunk_chars": 1500,
}

# =============================================================================
# Embedding Configuration
# =============================================================================
EMBEDDING = {
    # Model name (same as movie-assistant-service for consistency)
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    # Embedding dimension (fixed by model architecture)
    "dimension": 384,
    # Batch size for embedding generation
    "batch_size": 64,
    # Device: "cuda" for GPU, "cpu" for CPU, "auto" for automatic detection
    "device": "auto",
    # Normalize embeddings for cosine similarity
    "normalize": True,
}

# =============================================================================
# ChromaDB Configuration
# =============================================================================
CHROMADB = {
    # Persistence directory for ChromaDB data
    "persist_dir": str(DATA_DIR / "chromadb"),
    # Collection name for all subtitle chunks
    "collection_name": "oracle_subtitles",
    # Distance metric: cosine similarity (matches normalized embeddings)
    "distance_metric": "cosine",
    # HNSW index parameters (tune for speed vs accuracy tradeoff)
    "hnsw_space": "cosine",
    "hnsw_construction_ef": 200,    # Higher = better index quality, slower build
    "hnsw_search_ef": 100,          # Higher = more accurate search, slower query
    "hnsw_M": 16,                   # Number of connections per node
}

# =============================================================================
# File Paths
# =============================================================================
DATA_PATHS = {
    "subtitles_dir": str(DATA_DIR / "subtitles"),
    "processed_dir": str(DATA_DIR / "processed"),
    "chromadb_dir": str(DATA_DIR / "chromadb"),
    "ingestion_log": str(DATA_DIR / "processed" / "ingestion_log.json"),
}

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
