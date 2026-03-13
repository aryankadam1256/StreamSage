# StreamSage Architecture

This document provides a deep dive into the system architecture, design decisions, and data flows within StreamSage.

---

## System Overview

StreamSage follows a **microservices architecture** where each AI/ML capability is encapsulated in an independent service. This enables:

- **Independent scaling:** Each service can be scaled based on its load
- **Technology flexibility:** Each service can use the best tools for its task
- **Fault isolation:** A failure in one service doesn't bring down the entire system
- **Educational clarity:** Each concept can be studied in isolation

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  React + Vite Frontend                       │   │
│  │     LandingPage | MovieDiscover | OracleChat | VibeBar       │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────────────┘
                              │ HTTP/SSE
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         GATEWAY LAYER                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  FastAPI API Gateway                         │   │
│  │         Route Mapping | Health Checks | Error Handling       │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Movie Assistant │ │   Oracle RAG    │ │ Binge Predictor │
│     :8004        │ │     :8001       │ │     :8002       │
│                  │ │                 │ │                 │
│ Fine-tuned LLM   │ │ Subtitle Q&A    │ │ Watch Behavior  │
│ RAG + Reranking  │ │ SSE Streaming   │ │ LSTM Prediction │
└────────┬─────────┘ └────────┬────────┘ └────────┬────────┘
         │                    │                   │
         ▼                    ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    ChromaDB     │ │    ChromaDB     │ │   Keras .h5     │
│   6,147 movies  │ │   Subtitles     │ │   LSTM Model    │
└─────────────────┘ └─────────────────┘ └─────────────────┘

                              │
                              ▼
                    ┌─────────────────┐
                    │Sentiment Engine │
                    │     :8003       │
                    │   DistilBERT    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  PyTorch Model  │
                    │  Transformers   │
                    └─────────────────┘
```

---

## Service Architecture Details

### 1. API Gateway (Port 8000)

**Technology:** FastAPI

**Responsibilities:**
- Request routing to appropriate backend services
- Health aggregation across all services
- Centralized error handling and logging
- CORS configuration for frontend
- SSE proxy for streaming responses

**Route Mapping:**
```
/api/v1/discover    → Movie Assistant :8004/discover
/api/v1/ask         → Oracle RAG :8001/ask
/api/v1/ask/stream  → Oracle RAG :8001/ask/stream (SSE)
/api/v1/predict     → Binge Predictor :8002/predict
/api/v1/analyze     → Sentiment Engine :8003/analyze
/health             → Aggregated health status
```

**Design Decisions:**
- **Why a gateway?** Single entry point simplifies CORS, security, and service discovery
- **Async routing:** Uses httpx for non-blocking proxy requests
- **Health monitoring:** Periodic pings to all services with response time tracking

---

### 2. Movie Assistant Service (Port 8004)

**Technology:** FastAPI, ChromaDB, Sentence-Transformers, Hugging Face Transformers

**This is the most complex service, implementing a multi-stage retrieval and ranking pipeline:**

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING                          │
│  Query expansion | Mood augmentation | Name-phrase detection │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
    ┌─────────────┐                 ┌─────────────┐
    │   ChromaDB  │                 │    BM25     │
    │ Vector Search│                │ Text Search │
    │  (Semantic)  │                │  (Keyword)  │
    └──────┬──────┘                 └──────┬──────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Reciprocal Rank      │
              │   Fusion (RRF)         │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   Cross-Encoder        │
              │   Re-ranking           │
              │  (ms-marco-MiniLM)     │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   Custom Reranking     │
              │  - Genre matching      │
              │  - Title hint bonus    │
              │  - Franchise dedup     │
              │  - Mood alignment      │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   LLM Generation       │
              │  (Fine-tuned Llama 3)  │
              │      (Optional)        │
              └───────────┬────────────┘
                          │
                          ▼
                    Response JSON
```

**Key Components:**

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| Embedding Model | Convert text to vectors | BAAI/bge-large-en-v1.5 (1024 dim) |
| Vector Store | Semantic similarity search | ChromaDB with cosine distance |
| BM25 Index | Keyword matching | rank-bm25 library |
| Cross-Encoder | Pairwise relevance scoring | ms-marco-MiniLM-L-6-v2 |
| LLM | Natural language generation | Fine-tuned Llama 3 8B |

**Design Decisions:**
- **Hybrid retrieval:** Vector search alone misses keyword matches ("Tarantino"); BM25 alone misses semantic similarity ("movies like John Wick")
- **Two-stage ranking:** Cross-encoder is slow (requires forward pass for each query-doc pair) so we only run it on top candidates from stage 1
- **30/70 blend:** 30% cross-encoder score + 70% vector score balances semantic understanding with retrieval confidence

---

### 3. Oracle RAG Service (Port 8001)

**Technology:** FastAPI, ChromaDB, Ollama (Llama 3:8B)

**Answers questions about specific movies using subtitle transcripts.**

```
.srt File
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
│  Parse timestamps | Chunk by scene | Extract metadata        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  ChromaDB   │
                    │  (per movie)│
                    └──────┬──────┘
                           │
User Question              │
    │                      │
    ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      RETRIEVAL                               │
│  Embed question | Search ChromaDB | Filter by timestamp      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   PROMPT CONSTRUCTION                        │
│  System prompt | Retrieved context | User question           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      GENERATION                              │
│  Ollama Llama 3:8B | Token-by-token streaming via SSE        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                    Streamed Response
```

**Features:**
- **Spoiler protection:** User can specify a timestamp; chunks after that time are excluded from retrieval
- **Multi-turn conversation:** Maintains conversation history for follow-up questions
- **Suggested questions:** Generates contextual questions based on ingested content
- **SSE streaming:** Delivers LLM output token-by-token for responsive UX

---

### 4. Binge Predictor Service (Port 8002)

**Technology:** FastAPI, TensorFlow/Keras

**Predicts viewing continuation probability from watch history.**

```
Watch History JSON
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                        │
│  Genre sequences | Rating trends | Watch velocity | Time     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      LSTM NETWORK                            │
│  Input (sequence) → LSTM layers → Dense → Sigmoid            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                        OUTPUT                                │
│  Continuation probability | Risk level | Recommendations     │
└─────────────────────────────────────────────────────────────┘
```

**LSTM Architecture:**
```python
Model Architecture:
├── Input Layer: (sequence_length, features)
├── LSTM Layer 1: 64 units, return_sequences=True
├── Dropout: 0.2
├── LSTM Layer 2: 32 units
├── Dropout: 0.2
├── Dense Layer: 16 units, ReLU
└── Output Layer: 1 unit, Sigmoid
```

**Design Decisions:**
- **Why LSTM over simple RNN?** LSTM handles long-term dependencies (user's viewing patterns over time) better than vanilla RNN
- **Heuristic fallback:** When `.h5` model is unavailable, uses rule-based prediction based on recent activity patterns

---

### 5. Sentiment Engine (Port 8003)

**Technology:** FastAPI, PyTorch, Hugging Face Transformers

**Classifies sentiment of movie review text.**

```
Review Text
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                     TOKENIZATION                             │
│  DistilBERT tokenizer | WordPiece | Attention mask           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    DistilBERT MODEL                          │
│  6 transformer layers | 768 hidden dim | 66M parameters      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   CLASSIFICATION HEAD                        │
│  [CLS] token → Linear → Softmax → {positive, negative}       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                { label, confidence }
```

**Design Decisions:**
- **Why DistilBERT over BERT?** 40% smaller, 60% faster, retains 97% of BERT's accuracy
- **Context awareness:** Handles negation ("not bad" → positive) and slang ("sick" → positive) through bidirectional attention
- **Batch endpoint:** Processes up to 100 texts in a single request for efficiency

---

## Data Flow Patterns

### Synchronous Request-Response
Used by: Sentiment Engine, Binge Predictor, Movie Assistant (retrieval-only mode)

```
Client → Gateway → Service → Gateway → Client
         (proxy)             (proxy)
```

### Server-Sent Events (SSE)
Used by: Oracle RAG, Movie Assistant (LLM mode)

```
Client ←────────────────────────────
       │ event: token               │
       │ data: {"text": "The"}     │
       │                            │
       │ event: token               │
       │ data: {"text": " movie"}  │
Gateway ←── SSE ── Service          │
       │                            │
       │ event: done                │
       │ data: {}                   │
       ←────────────────────────────┘
```

---

## Database Architecture

### ChromaDB (Vector Store)

**Movie Assistant Collection:**
```
Collection: movies
Documents: 6,147
Embedding Dimension: 1024 (bge-large-en-v1.5)

Metadata per document:
├── title: string
├── year: int
├── genres: string (comma-separated)
├── director: string
├── cast: string (top 5)
├── rating: float
├── overview: string
└── popularity: float
```

**Oracle RAG Collection:**
```
Collection: {movie_id}_subtitles
Documents: Variable (depends on subtitle length)
Embedding Dimension: 384 (all-MiniLM-L6-v2)

Metadata per document:
├── start_time: float (seconds)
├── end_time: float (seconds)
├── text: string (dialogue chunk)
└── scene_index: int
```

---

## Error Handling Strategy

| Layer | Strategy |
|-------|----------|
| Frontend | Try-catch with user-friendly error messages; retry on network failure |
| Gateway | Catch service errors, return standardized error JSON with status codes |
| Services | Graceful degradation (e.g., retrieval-only when LLM unavailable) |
| Models | Fallback to heuristics when ML models fail to load |

---

## Scalability Considerations

| Concern | Current Solution | Production Alternative |
|---------|------------------|----------------------|
| Embedding compute | GPU acceleration, caching | GPU cluster, batch processing |
| LLM inference | HuggingFace API fallback | vLLM, TensorRT-LLM |
| Vector search | Single ChromaDB instance | Pinecone, Weaviate, Milvus |
| Service orchestration | Docker Compose | Kubernetes |
| Load balancing | Single gateway | NGINX, Traefik |

---

## Security Architecture

| Layer | Measures |
|-------|----------|
| Secrets | Environment variables, `.env` files excluded from git |
| API Keys | TMDB, HuggingFace tokens stored in env vars |
| Input Validation | Pydantic models for all request/response schemas |
| CORS | Configured in gateway for allowed origins |
| Rate Limiting | Not implemented (educational project) |

---

## Monitoring & Observability

**Health Endpoints:**
```bash
GET /health  # Each service exposes this

# Response includes:
# - Service status (healthy/degraded/unhealthy)
# - Model load status
# - Database connectivity
# - Memory usage (GPU if applicable)
```

**Logging:**
- Python `logging` module with INFO level
- Structured logs for request tracing
- Error logs with stack traces

---

## Testing Strategy

| Type | Tools | Coverage |
|------|-------|----------|
| Unit Tests | pytest | Core functions (ranking, embedding) |
| Integration Tests | pytest + httpx | API endpoint contracts |
| Benchmark Tests | Custom 100-query suite | Retrieval quality metrics |
| Manual Testing | Swagger UI | End-to-end flows |
