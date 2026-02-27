# Phase 1: Data Pipeline & Infrastructure

## Overview

Phase 1 establishes the subtitle ingestion pipeline - the foundation that feeds the Oracle's knowledge base. This pipeline transforms raw `.srt` subtitle files into searchable vector embeddings stored in ChromaDB.

```
.srt file → Parse → Chunk → Embed → ChromaDB
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌───────────┐
│  SRT Parser │────►│  Chunker    │────►│  Embedder    │────►│ ChromaDB  │
│             │     │  (sliding   │     │  (MiniLM     │     │ Ingest    │
│ srt_parser  │     │   window)   │     │   L6-v2)     │     │           │
│ .py         │     │ chunker.py  │     │ embedder.py  │     │ ingest.py │
└─────────────┘     └─────────────┘     └──────────────┘     └───────────┘
     │                    │                    │                    │
     ▼                    ▼                    ▼                    ▼
 SubtitleEntry      SubtitleChunk         numpy array         ChromaDB
 {index, start,     {chunk_id,           (N, 384)            Collection
  end, text}         movie_id,                               with metadata
                     timestamp_start,
                     timestamp_end,
                     text}
```

## Pipeline Stages

### Stage 1: SRT Parser (`srt_parser.py`)

**What it does:** Reads `.srt` (SubRip) subtitle files and outputs structured data.

**SRT Format:**
```
1
00:01:15,200 --> 00:01:18,500
What do you mean by that?

2
00:01:19,000 --> 00:01:22,300
I mean exactly what I said.
```

**Implementation:** State machine parser with three states:
- `INDEX` → expecting numeric index
- `TIMING` → expecting timestamp line
- `TEXT` → collecting text until blank line

**Text Cleaning Pipeline:**
1. Strip HTML tags (`<i>`, `<b>`, `</font>`, etc.)
2. Remove bracketed annotations (`[music playing]`, `[gunshot]`)
3. Remove parenthetical annotations (`(sighs)`, `(laughing)`)
4. Remove music markers (`♪...♪`)
5. Remove leading dashes (speaker indicators)
6. Normalize whitespace

**Encoding Handling:** Tries UTF-8, Latin-1, CP1252, ISO-8859-1 in order (SRT files from different sources use different encodings).

**Output:** `list[SubtitleEntry]` sorted by start_time.

**Key Design Decision:** We built a custom parser instead of using `pysrt` library for more control over the cleaning pipeline and to handle malformed SRT files that `pysrt` chokes on.

### Stage 2: Sliding Window Chunker (`chunker.py`)

**What it does:** Groups subtitle entries into overlapping chunks for semantic retrieval.

**Why chunking is necessary:**
- A single subtitle line (e.g., "Yes.") is too short for meaningful embedding
- A full movie script is too long for embedding models (256 token limit)
- 10 consecutive lines (~30-60 seconds) captures a complete conversational exchange

**Algorithm:**
```
Given entries [E0, E1, E2, ..., EN] with window=10, overlap=3:

Chunk 0: [E0, E1, E2, E3, E4, E5, E6, E7, E8, E9]     ← 10 entries
Chunk 1:                [E7, E8, E9, E10, E11, ... E16]  ← slides by 7 (10-3)
Chunk 2:                              [E14, E15, ... E23]
...

Step size = window_size - overlap = 10 - 3 = 7
```

**Why overlap matters:** If a user asks "What did they discuss about the Fibonacci pattern?", the relevant dialogue might span entries E8-E12. Without overlap, this would be split across chunk boundaries. The 3-entry overlap ensures it appears fully in at least one chunk.

**Parameters (from config.py):**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `window_size` | 10 | ~30-60s of dialogue, one conversational exchange |
| `overlap` | 3 | Prevents context loss at boundaries |
| `min_chunk_size` | 3 | Avoids tiny end-of-file chunks |
| `max_chunk_chars` | 1500 | Safety limit for embedding model (256 tokens ≈ 1200 chars) |

**Output:** `list[SubtitleChunk]` with metadata (movie_id, timestamps, chunk_index).

### Stage 3: Embedding Generator (`embedder.py`)

**What it does:** Converts text chunks into 384-dimensional dense vectors for similarity search.

**Model: `sentence-transformers/all-MiniLM-L6-v2`**
- 384-dimensional embeddings (compact, fast search)
- Trained on 1B+ sentence pairs for semantic similarity
- 5x faster than larger models with ~95% quality
- Same model used in movie-assistant-service (cross-platform consistency)
- 80MB model size, fits easily in memory

**Process:**
1. Load model once at initialization (stays in memory)
2. Batch encode chunk texts (batch_size=64)
3. L2-normalize embeddings (for cosine similarity)
4. Return as numpy array `(num_chunks, 384)`

**Why normalized embeddings?** ChromaDB uses cosine distance. Pre-normalizing vectors means cosine distance = dot-product distance, which is faster to compute.

### Stage 4: ChromaDB Ingestion (`ingest.py`)

**What it does:** Orchestrates the full pipeline and stores results in ChromaDB.

**Storage Strategy: Single Collection + Metadata Filter**

We chose a single `oracle_subtitles` collection for all movies rather than per-movie collections because:
- Simpler maintenance (one collection to manage)
- Scales to cross-movie queries in the future
- ChromaDB metadata filtering (`where={"movie_id": "inception"}`) narrows search space efficiently
- HNSW index handles mixed-movie data well with metadata pre-filtering

**ChromaDB Document Schema:**
```python
{
    "id": "inception_chunk_0",           # Unique chunk ID
    "document": "What is the most...",   # Chunk text (for keyword fallback)
    "embedding": [0.07, -0.03, ...],     # 384-dim vector
    "metadata": {
        "movie_id": "inception",         # Movie identifier (filter key)
        "chunk_index": 0,                # Sequential position
        "timestamp_start": 75.2,         # Start time (seconds)
        "timestamp_end": 112.8,          # End time (seconds)
        "num_entries": 10,               # Subtitle entries in chunk
        "start_entry_index": 1,          # First subtitle index
        "end_entry_index": 10,           # Last subtitle index
    }
}
```

**HNSW Index Parameters:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `space` | cosine | Matches normalized embeddings |
| `construction_ef` | 200 | Higher = better index quality at build time |
| `search_ef` | 100 | Higher = more accurate search at query time |
| `M` | 16 | Graph connectivity (higher = more memory, better recall) |

## File Structure

```
services/oracle-rag-service/
├── data_pipeline/
│   ├── __init__.py          # Package marker
│   ├── config.py            # All configuration parameters
│   ├── srt_parser.py        # Stage 1: SRT parsing + text cleaning
│   ├── chunker.py           # Stage 2: Sliding window chunking
│   ├── embedder.py          # Stage 3: Embedding generation
│   └── ingest.py            # Stage 4: ChromaDB ingestion orchestrator
├── data/
│   ├── subtitles/           # Raw .srt files (place files here)
│   │   └── the_signal.srt   # Sample test subtitle
│   ├── processed/           # Ingestion logs
│   └── chromadb/            # ChromaDB persistent storage
├── docs/
│   └── PHASE1_DATA_PIPELINE.md  # This document
└── requirements.txt         # Python dependencies
```

## Usage

### Ingest a Single Movie
```bash
cd services/oracle-rag-service
python -m data_pipeline.ingest --file data/subtitles/inception.srt --movie-id inception
```

### Ingest All Movies in Directory
```bash
python -m data_pipeline.ingest --dir data/subtitles/
```

### Force Re-ingestion
```bash
python -m data_pipeline.ingest --file data/subtitles/inception.srt --movie-id inception --force
```

### View Collection Stats
```bash
python -m data_pipeline.ingest --stats
```

## Test Results

Pipeline tested end-to-end with sample subtitle (`the_signal.srt`, 50 entries, ~3 minutes):

```
Stage 1 (Parse):  49 valid entries from 50 (1 music-only entry filtered)
Stage 2 (Chunk):  49 entries → 7 chunks (window=10, overlap=3, step=7)
Stage 3 (Embed):  7 embeddings generated, shape (7, 384), ~1096 texts/sec
Stage 4 (Store):  7 documents stored in ChromaDB, 0.09s total

Query test: "What is the Fibonacci pattern in the signal?"
  → Top result (relevance=0.552): Chunk about Fibonacci sequence discussion
  → Correct chunks surfaced with proper timestamp metadata
```

## Configuration Reference

All parameters are in `data_pipeline/config.py`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNKING.window_size` | 10 | Subtitle entries per chunk |
| `CHUNKING.overlap` | 3 | Overlapping entries between chunks |
| `EMBEDDING.model_name` | all-MiniLM-L6-v2 | Sentence transformer model |
| `EMBEDDING.batch_size` | 64 | Embedding batch size |
| `CHROMADB.collection_name` | oracle_subtitles | ChromaDB collection |
| `CHROMADB.hnsw_search_ef` | 100 | Search accuracy parameter |

## Adding New Movies

1. Place `.srt` file in `data/subtitles/` (filename becomes movie_id)
2. Run: `python -m data_pipeline.ingest --dir data/subtitles/`
3. Existing movies are skipped (use `--force` to re-ingest)

## Next Phase

**Phase 2: RAG Service Core** will build the FastAPI endpoints that query this ChromaDB collection, integrate with Ollama for LLM generation, and implement the prompt engineering for grounded answers with timestamp citations.
