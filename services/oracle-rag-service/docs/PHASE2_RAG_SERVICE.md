# Phase 2: RAG Service Core

## Overview

Phase 2 builds the FastAPI-based RAG service that queries the ChromaDB vector store (populated in Phase 1) and generates grounded answers using Ollama's Llama 3:8B model. This is the **inference pipeline** - the brain behind The Oracle.

```
User Query → Query Understanding → Retrieval → Context Construction → Generation → Response
```

## Architecture

```
                    POST /ask
                       │
                       ▼
            ┌─────────────────────┐
            │  Query Understanding │
            │  - classify_intent() │ → quote_search, scene_finding,
            │  - extract_timestamp │   theme_analysis, character_query,
            │    _hint()           │   timestamp_query, general
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Retrieval Engine    │
            │  - Embed query       │ → sentence-transformers (384-dim)
            │  - ChromaDB search   │ → movie_id metadata filter
            │  - Timestamp rerank  │ → proximity-weighted re-ranking
            └──────────┬──────────┘
                       │ top-K chunks
                       ▼
            ┌─────────────────────┐
            │  Prompt Construction │
            │  - System prompt     │ → The Oracle persona + grounding rules
            │  - Context formatting│ → Numbered excerpts with timestamps
            │  - Intent suffix     │ → Task-specific instruction
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  LLM Generation      │
            │  - OllamaClient      │ → Llama 3:8B local inference
            │  - Fallback mode     │ → Retrieval-only if Ollama down
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Response            │
            │  {answer, sources[], │
            │   model_used,        │
            │   query_time_ms}     │
            └─────────────────────┘
```

## Components

### 1. Query Understanding

Two functions work together to understand user intent before retrieval.

**`classify_intent(query)`** - Regex-based intent classifier

Scans the query for keyword patterns and returns the highest-scoring intent:

| Intent | Trigger Words | Example Query |
|--------|--------------|---------------|
| `quote_search` | say, said, quote, words, line, dialogue | "What did he say about love?" |
| `scene_finding` | scene, happen, when, moment, part where | "What happens in the opening scene?" |
| `theme_analysis` | theme, about, mean, symbol, message | "What themes does the movie explore?" |
| `character_query` | character, who, relationship, motivation | "Who is Dr. Chen?" |
| `timestamp_query` | minute, hour, beginning, end, mark | "What happens at the 45 minute mark?" |
| `general` | (default fallback) | "Tell me about the plot" |

**Why classify intent?** The intent drives two things:
1. Retrieval strategy (timestamp queries trigger proximity weighting)
2. Prompt suffix (tells the LLM to focus on quotes vs themes vs scenes)

**`extract_timestamp_hint(query)`** - Natural language timestamp parser

Extracts timestamps from queries without explicit `timestamp` field input:
- "at the 45 minute mark" → 2700s
- "around 1 hour 20 minutes" → 4800s
- "in the beginning" → 300s
- "near the end" → None (unknown length)

### 2. Retrieval Engine

**`retrieve_chunks(collection, embedder, query, movie_id, top_k, timestamp_hint)`**

Pipeline:
1. **Embed query** using the same `all-MiniLM-L6-v2` model used during ingestion (critical for consistent vector space)
2. **ChromaDB search** with `where={"movie_id": movie_id}` metadata filter (narrows to one movie)
3. **Timestamp re-ranking** if a timestamp hint exists

**Timestamp Proximity Re-ranking**

When a user says "What happened around the 2 minute mark?", purely semantic search might return the most semantically similar chunks from anywhere in the movie. The re-ranker combines semantic relevance with temporal proximity:

```
combined_score = 0.7 * semantic_relevance + 0.3 * temporal_proximity

where temporal_proximity = 1 / (1 + |chunk_midpoint - timestamp_hint| / 60)
```

This decays smoothly: chunks exactly at the target get 1.0, chunks 1 minute away get 0.5, 2 minutes away get 0.33, etc.

To support this, the retriever fetches `3 * top_k` candidates before re-ranking, then returns only `top_k` after scoring.

### 3. Prompt Engineering

The prompt has four components:

**System Prompt (The Oracle persona):**
```
You are The Oracle, a movie dialogue expert. You answer questions about
movies using ONLY the subtitle excerpts provided below.

RULES:
1. ONLY use information from the provided subtitle excerpts.
2. When referencing dialogue, cite the timestamp range.
3. If the answer isn't in the excerpts, say so honestly.
4. Keep answers concise but thorough.
5. If asked about a quote, provide the closest matching dialogue.
```

**Context Section:**
```
[Excerpt 1] (0:56 - 1:33) [relevance: 55%]
You need help. This obsession isn't healthy. It's not an obsession...

[Excerpt 2] (1:48 - 2:24) [relevance: 45%]
It started computing on its own. No one initiated the sequence...
```

**User Question:** Direct pass-through from request

**Intent-Specific Suffix:** Steers the LLM based on classified intent:
- `quote_search` → "Focus on finding the most relevant quote. Cite exact words and timestamp."
- `scene_finding` → "Describe what's happening based on the dialogue. Reference the timestamp."
- `theme_analysis` → "Analyze what themes or ideas emerge from the excerpts."

### 4. Ollama LLM Client

**`OllamaClient`** wraps the `ollama` Python package with graceful degradation:

- **Connected mode**: Sends prompt to Ollama, receives generated answer
- **Retrieval-only mode**: If Ollama is down, returns the raw source chunks with a message explaining the LLM is unavailable

**Generation parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.2 | Low for factual grounding (not creative generation) |
| `top_p` | 0.9 | Nucleus sampling for diversity within confidence |
| `num_predict` | 512 | Max output tokens (sufficient for dialogue answers) |
| `repeat_penalty` | 1.1 | Slight penalty to reduce repetitive output |

### 5. API Endpoints

**`POST /ask`** - Main RAG endpoint
```json
// Request
{
    "query": "What did they discuss about the Fibonacci pattern?",
    "movie_id": "the_signal",
    "timestamp": null,
    "top_k": 5
}

// Response (200 OK)
{
    "answer": "Based on the dialogue, the Fibonacci pattern is discussed...",
    "sources": [
        {
            "content": "Random? Then explain why it matches the Fibonacci...",
            "movie_id": "the_signal",
            "timestamp_start": 56.2,
            "timestamp_end": 93.5,
            "relevance_score": 0.552
        }
    ],
    "model_used": "llama3:8b",
    "query_time_ms": 1234.5
}
```

**`GET /health`** - Health check
```json
{
    "status": "healthy",           // or "degraded" if Ollama down
    "ollama_connected": true,
    "chroma_documents": 7,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "llama3:8b"
}
```

**`GET /collections`** - List ingested movies
```json
{
    "movies": [
        {"movie_id": "the_signal", "chunks": 7},
        {"movie_id": "inception", "chunks": 178}
    ],
    "total_documents": 185
}
```

## Application Lifecycle

The FastAPI app uses `lifespan` context manager for startup/shutdown:

**Startup (in order):**
1. Load `SubtitleEmbedder` (sentence-transformers model into GPU/CPU memory)
2. Connect to ChromaDB (open persistent client, get collection)
3. Test Ollama connection (if fails, service runs in degraded/retrieval-only mode)

**Runtime:** Handles concurrent requests. Embedder and ChromaDB client are thread-safe singletons.

**Shutdown:** Clean resource release (logged).

## Test Results

All endpoints tested with the `the_signal` sample movie (7 chunks):

```
Health:       200 OK | status=degraded (Ollama not running locally, expected)
Collections:  200 OK | 1 movie, 7 chunks
Ask:          200 OK | 3 sources retrieved, 123ms latency (no LLM)
Ask+timestamp:200 OK | 3 sources, re-ranked by proximity, 11ms
Ask bad movie:200 OK | Graceful "movie not ingested" message
```

**Query latency breakdown (without LLM generation):**
| Stage | Time |
|-------|------|
| Query embedding | ~10ms |
| ChromaDB search | ~1ms |
| Total (retrieval only) | ~12ms |
| + Ollama generation | ~1-3s (estimated, depends on model/GPU) |

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `main.py` | **Rewritten** | Full RAG service with query understanding, retrieval, prompt engineering, Ollama client |
| `ingest.py` | Superseded | Old root-level ingest script; replaced by `data_pipeline/ingest.py` from Phase 1 |

## Running the Service

```bash
# Start the service (requires Phase 1 data to be ingested first)
cd services/oracle-rag-service
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# API docs available at http://localhost:8000/docs
```

## Degraded Mode

The service is designed to work without Ollama:
- **With Ollama**: Full RAG (retrieve + generate). Best user experience.
- **Without Ollama**: Retrieval-only mode. Returns source chunks without AI summary. Still useful for quote search and scene finding.

This graceful degradation means the service never fails completely - retrieval always works.

## Next Phase

**Phase 3: Integration & Gateway** will connect this service to the existing gateway routes, set up Docker, and wire the frontend's OracleChat component to the live service.
