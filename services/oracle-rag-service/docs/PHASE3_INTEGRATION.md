# Phase 3: Integration & Gateway

## Overview

Phase 3 wires the Oracle RAG service into StreamSage's existing microservices architecture. This connects the backend service (Phase 2) to the API gateway and updates the frontend to communicate with it.

```
Frontend (OracleChat.jsx)
    │  POST /api/v1/oracle/ask
    ▼
Gateway (:8000)
    │  Forwards raw JSON body
    ▼
Oracle RAG Service (:8001 external / :8000 internal)
    │  Full RAG pipeline
    ▼
Response: {answer, sources[], model_used, query_time_ms}
```

## Integration Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ DOCKER COMPOSE                                               │
│                                                              │
│  ┌──────────┐     ┌─────────┐     ┌───────────────────┐    │
│  │ Frontend │────►│ Gateway │────►│ Oracle RAG Service │    │
│  │ :3000    │     │ :8000   │     │ :8001→:8000        │    │
│  └──────────┘     │         │     │                    │    │
│                   │         │     │  ┌─────────┐       │    │
│                   │         │     │  │ ChromaDB │       │    │
│                   │         │     │  └─────────┘       │    │
│                   │         │     │        │           │    │
│                   └─────────┘     └────────┼───────────┘    │
│                                            │                │
│                                   ┌────────┴───────────┐    │
│                                   │ Ollama (Llama 3:8B)│    │
│                                   │ :11434             │    │
│                                   └────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## Changes Made

### 1. Dockerfile (Updated)

`services/oracle-rag-service/Dockerfile`

- Multi-stage build (Python 3.11-slim) matching movie-assistant pattern
- Installs dependencies in builder stage, copies to production stage
- Creates data directories: `/app/data/chromadb`, `/app/data/subtitles`, `/app/data/processed`
- Health check with 30s start period (allows time for embedding model loading)
- Runs on port 8000 internally (mapped to 8001 externally)

### 2. docker-compose.yml (Updated)

Updated the `oracle-rag-service` block:

| Before | After |
|--------|-------|
| `CHROMA_PERSIST_DIR` env var | Removed (config.py handles path) |
| `./data/chromadb` volume | `./services/oracle-rag-service/data/chromadb` |
| `./data/subtitles` volume | `./services/oracle-rag-service/data/subtitles` |
| N/A | Added `LLM_MODEL=llama3:8b` env var |

Volume mounts now point to the actual data directories inside the service, consistent with how the data pipeline stores ingested data.

### 3. Gateway (No Changes Needed)

The existing gateway routes at `gateway/main.py:164-206` are already compatible:

| Gateway Route | Forwards To | Status |
|--------------|-------------|--------|
| `POST /api/v1/oracle/ask` | `ORACLE_SERVICE_URL/ask` | Compatible |
| `GET /api/v1/oracle/collections` | `ORACLE_SERVICE_URL/collections` | Compatible |

The gateway is a pure pass-through (reads raw JSON body, forwards as-is). No schema transformation needed.

### 4. Frontend api.js (Updated)

`frontend/src/api.js` - `askOracle()` function:

**Change:** `timestamp_start` field renamed to `timestamp` to match the new service's `QueryRequest` schema.

```javascript
// Before
timestamp_start: timestamp ? parseFloat(timestamp) : undefined

// After
timestamp: timestamp ? parseFloat(timestamp) : undefined
```

### 5. OracleChat.jsx (Enhanced)

`frontend/src/components/OracleChat.jsx` - Major UI overhaul:

| Feature | Before | After |
|---------|--------|-------|
| Movie ID | Manual text input | Auto-derived from movie title via `titleToMovieId()` |
| Chat style | Single Q&A (replaces previous answer) | Conversational history (accumulates) |
| Input layout | Textarea for question | Inline input + send button (chat-style) |
| Example queries | None | 4 clickable suggestion chips |
| Source display | Full content in scrollable list | Compact with `line-clamp-2` and relevance % |
| Time formatting | Minutes:Seconds | Hours:Minutes:Seconds (for long movies) |
| Error display | Block error with title | Inline error message in chat flow |

**`titleToMovieId(title)` function:**

Converts display titles to ingested movie IDs:
- "Inception" → `inception`
- "The Dark Knight" → `the_dark_knight`
- "The Signal" → `the_signal`

This matches the ingestion convention where `movie_id = filename.stem.lower().replace(" ", "_")`.

## Request/Response Flow

### Request Path

```
1. User types question in OracleChat.jsx
2. handleAsk() calls askOracle(query, movieId, timestamp) from api.js
3. api.js sends POST to /api/v1/oracle/ask with:
   {query, movie_id, timestamp?, top_k: 5}
4. Gateway receives at POST /api/v1/oracle/ask
5. Gateway forwards raw body to ORACLE_SERVICE_URL/ask
6. Oracle service validates with QueryRequest pydantic model
7. RAG pipeline executes (query understand → retrieve → prompt → generate)
8. Response returns as QueryResponse
```

### Response Path

```
1. Oracle service returns:
   {answer, sources: [{content, movie_id, timestamp_start, timestamp_end, relevance_score}],
    model_used, query_time_ms}
2. Gateway passes response through unchanged
3. api.js returns res.data to OracleChat
4. OracleChat appends to chatHistory state
5. React renders answer + source chunks with timestamps
```

## Test Results

All integration tests passed:

```
Request Schema Compatibility:  3/3 payloads valid
Response Schema Compatibility: All fields present (answer, sources, model_used, query_time_ms)
Gateway Route Mapping:         Both routes match service endpoints
Docker-Compose Config:         Correct ports, volumes, env vars, service URL
Movie Title→ID Mapping:        3/3 correct conversions
Frontend Build:                Compiled successfully (448 modules, 744ms)
```

## Running the Full Stack

```bash
# 1. Start Ollama first (for LLM generation)
docker-compose up -d ollama
docker exec -it streamsage-ollama ollama pull llama3:8b

# 2. Ingest subtitles (run locally before Docker)
cd services/oracle-rag-service
python -m data_pipeline.ingest --dir data/subtitles/

# 3. Start all services
docker-compose up -d

# 4. Access the app
# Frontend: http://localhost:3000
# Gateway API: http://localhost:8000/api/v1
# Oracle direct: http://localhost:8001/docs
```

## Next Phase

**Phase 4: Frontend Enhancement** will add streaming response display, conversation memory, and per-movie suggested questions. **Phase 5: Optimization & Evaluation** will tune chunking parameters, improve prompt quality, and add evaluation metrics.
