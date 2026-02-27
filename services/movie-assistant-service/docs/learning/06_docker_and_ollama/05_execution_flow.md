# Execution Flow: Request Journey Through Docker & Ollama

This document traces a complete request from the user's browser through every Docker container, through Ollama, and back — showing exactly what happens at each step.

---

## Scenario: User Asks Oracle "What happened at the 45-minute mark?"

We'll trace this through two paths:
1. **Docker mode** — all services in containers
2. **Local mode** — services running natively

---

## Part 1: Docker Startup Sequence

### What Happens When You Run `docker-compose up -d`

```
Step 1: Docker reads docker-compose.yml
         │
         ├──► Creates network: streamsage_default
         │    (Virtual bridge network connecting all containers)
         │
         ├──► Creates volume: streamsage-ollama-data
         │    (Persistent storage for Ollama models)
         │
Step 2: Docker resolves dependency graph
         │
         └──► Start order (based on depends_on):
              │
              1. ollama (no dependencies)
              │   └── Pulls ollama/ollama:latest image
              │   └── Starts container "streamsage-ollama"
              │   └── Mounts ollama_data → /root/.ollama
              │   └── Opens port 11434
              │   └── Ollama server starts, loads model into memory
              │
              2. oracle-rag-service (depends_on: ollama)
              │   binge-service (no LLM dependency)
              │   sentiment-service (no LLM dependency)
              │   movie-assistant-service (no LLM dependency)
              │   └── Docker builds images from Dockerfiles (first time)
              │   └── Starts 4 containers in parallel
              │
              3. gateway (depends_on: all 4 services)
              │   └── Starts after all services are running
              │
              4. frontend (depends_on: gateway)
                  └── Starts last
```

### What Each Service Does During Its Lifespan Startup

```
Oracle Service Startup:
┌─────────────────────────────────────────────────────────────┐
│ 1. uvicorn starts FastAPI app                               │
│ 2. lifespan() runs:                                         │
│    a. Load SubtitleEmbedder (sentence-transformers model)   │
│       ~5 seconds, loads into CPU/GPU memory                 │
│    b. Connect to ChromaDB at /app/data/chromadb             │
│       (mounted from host via volume)                        │
│    c. Create OllamaClient(http://ollama:11434, llama3:8b)   │
│    d. ollama_client.connect()                               │
│       → ollama.Client(host="http://ollama:11434")           │
│       → client.list()  ← health check                      │
│       → self.connected = True (or False if Ollama is slow)  │
│ 3. Service ready on port 8000 (internal)                    │
│    → Mapped to port 8001 on host                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 2: Full Request Trace (Streaming SSE)

### Step 1: Browser → Frontend Container

```
User types: "What happened at the 45-minute mark?"
Selects movie: "inception"

Browser:
  URL: http://localhost:3000
  ──► Frontend container (Nginx serving React app)
  ──► React app loaded in browser
  ──► User clicks Send
```

The React app sends a POST request:
```javascript
// frontend/src/components/OracleChat.jsx
fetch('/api/v1/oracle/ask/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "What happened at the 45-minute mark?",
    movie_id: "inception",
    top_k: 5,
    conversation_history: [
      { role: "user", content: "Tell me about the main character" },
      { role: "oracle", content: "Cobb is a skilled extractor who..." }
    ]
  })
})
```

### Step 2: Browser → Gateway Container

```
Browser sends to: http://localhost:8000/api/v1/oracle/ask/stream
                           │
                           ▼
               ┌──────────────────────────┐
               │  Gateway Container (:8000)│
               │                          │
               │  Port mapping:           │
               │  Host 8000 → Container   │
               │  8000                    │
               │                          │
               │  Endpoint matched:       │
               │  /api/v1/oracle/ask/     │
               │  stream                  │
               └──────────────────────────┘
```

**Gateway code executes:** `gateway/main.py:208-247`

```python
# 1. Read the raw request body
body = await request.body()
# body = b'{"query":"What happened...","movie_id":"inception",...}'

# 2. Open streaming connection to Oracle
async with httpx.AsyncClient(timeout=300.0) as client:
    async with client.stream(
        "POST",
        "http://oracle-rag-service:8000/ask/stream",  # Docker DNS!
        content=body,
        headers={"Content-Type": "application/json"},
    ) as response:
        async for chunk in response.aiter_raw():
            yield chunk  # Pass bytes straight through
```

**DNS resolution inside Docker:**
```
"oracle-rag-service" → Docker DNS → 172.18.0.3 (Oracle's internal IP)
```

### Step 3: Gateway → Oracle Container

```
Gateway sends to: http://oracle-rag-service:8000/ask/stream
                           │
                           ▼
               ┌──────────────────────────┐
               │  Oracle Container (:8000) │
               │                          │
               │  Note: internal port is  │
               │  8000, not 8001!         │
               │  8001 is only for host   │
               │  access.                 │
               └──────────────────────────┘
```

**Oracle code executes:** `oracle-rag-service/main.py:911-1043`

#### Step 3a: Query Understanding (Lines 941-946)

```python
intent = classify_intent(request.query)
# "What happened at the 45-minute mark?"
# → Matches patterns: "happen(s|ed)?", "minute", "mark"
# → intent = "timestamp_query" (highest score)

timestamp_hint = request.timestamp or extract_timestamp_hint(request.query)
# → Regex matches "45 minutes" → returns 2700.0 (seconds)
```

#### Step 3b: Embedding + Retrieval (Lines 949-956)

```python
chunks = retrieve_chunks(
    collection=collection,       # ChromaDB at /app/data/chromadb (volume mount)
    embedder=embedder,           # sentence-transformers (loaded in memory)
    query=request.query,
    movie_id="inception",
    top_k=5,
    timestamp_hint=2700.0,       # 45 minutes = 2700 seconds
)
```

Inside `retrieve_chunks`:
```
1. Embed query: embedder.embed_texts(["What happened..."]) → [384-dim vector]
2. ChromaDB search:
   - where_filter: movie_id = "inception" AND timestamp_start <= 2700
   - top 15 results (3x top_k for re-ranking)
3. Re-rank by timestamp proximity (2700 seconds target)
4. Return top 5 chunks
```

#### Step 3c: Build RAG Prompt

```python
prompt = build_rag_prompt(
    query="What happened at the 45-minute mark?",
    chunks=chunks,                    # 5 subtitle excerpts
    intent="timestamp_query",
    conversation_history=[...],       # Last 3 turns
    user_timestamp=2700.0,            # Spoiler boundary
)
```

The resulting prompt looks like:
```
You are The Oracle — a movie watching companion...
The viewer is currently at 0:45:00 in the movie.
ONLY discuss events from this point and earlier.

SUBTITLE EXCERPTS:
[Excerpt 1] (0:42:15 - 0:43:30) [relevance: 87%]
COBB: The dream is collapsing...

[Excerpt 2] (0:44:50 - 0:46:10) [relevance: 82%]
ARTHUR: We need to go deeper...
...

CONVERSATION HISTORY:
User: Tell me about the main character
Oracle: Cobb is a skilled extractor who...

CURRENT QUESTION: What happened at the 45-minute mark?

Describe what's happening at this point. Only reference events up to this moment.
```

### Step 4: Oracle → Ollama Container

```
Oracle sends to: http://ollama:11434/api/generate
                           │
                           ▼
               ┌──────────────────────────┐
               │  Ollama Container         │
               │  (:11434)                │
               │                          │
               │  Model: llama3:8b        │
               │  (loaded from volume)    │
               └──────────────────────────┘
```

**OllamaClient code executes:** `oracle-rag-service/main.py:674-707`

```python
stream = await self._async_client.generate(
    model="llama3:8b",
    prompt=prompt,                    # ~400 tokens
    options={
        "temperature": 0.2,           # Factual, grounded
        "top_p": 0.9,
        "num_predict": 220,           # Max ~80 words
        "repeat_penalty": 1.1,
        "num_gpu": 99,                # All layers on GPU
    },
    stream=True,
)
```

### Step 5: Inside Ollama (LLM Inference)

```
┌─────────────────────────────────────────────────────┐
│                 Ollama Server                       │
│                                                     │
│  1. Receive prompt text                             │
│     "You are The Oracle... SUBTITLE EXCERPTS..."    │
│                                                     │
│  2. Tokenize (BPE)                                  │
│     → [128000, 67375, 14175, ...] (~400 tokens)     │
│                                                     │
│  3. Prefill Phase (process all prompt tokens)       │
│     → Compute KV for all 400 tokens                 │
│     → Store in KV cache                             │
│     → Time: ~500ms (GPU), ~3s (CPU)                 │
│                                                     │
│  4. Decode Phase (generate new tokens one-by-one)   │
│     Token 1: "At"    → stream: {"response":"At"}    │
│     Token 2: " that" → stream: {"response":" that"} │
│     Token 3: " point"→ stream: {"response":" point"}│
│     ...                                              │
│     Token N: "."     → stream: {"response":".","done":true} │
│                                                     │
│  5. Each token: ~25ms (GPU) or ~125ms (CPU)         │
│     Total for 100 tokens: ~2.5s (GPU) or ~12s (CPU) │
└─────────────────────────────────────────────────────┘
```

### Step 6: Ollama → Oracle → Gateway → Browser (Streaming)

Each token flows back through the entire chain:

```
Ollama          Oracle              Gateway           Browser
  │               │                    │                 │
  │ {"response":  │                    │                 │
  │  "At",        │                    │                 │
  │  "done":false}│                    │                 │
  │ ─────────────►│                    │                 │
  │               │                    │                 │
  │               │ data: {"type":     │                 │
  │               │  "token",          │                 │
  │               │  "content":"At"}   │                 │
  │               │ \n\n               │                 │
  │               │ ──────────────────►│                 │
  │               │                    │                 │
  │               │                    │ (raw bytes      │
  │               │                    │  forwarded)     │
  │               │                    │ ───────────────►│
  │               │                    │                 │
  │               │                    │                 │ Append "At"
  │               │                    │                 │ to chat bubble
  │               │                    │                 │
  │ {"response":  │                    │                 │
  │  " that",     │                    │                 │
  │  "done":false}│                    │                 │
  │ ─────────────►│ data: {"type":     │                 │
  │               │  "token",          │                 │
  │               │  "content":" that"}│                 │
  │               │ ──────────────────►│ ───────────────►│ Append " that"
  │               │                    │                 │
  │  ... (repeats for ~100 tokens) ... │                 │
  │               │                    │                 │
  │ {"done":true} │                    │                 │
  │ ─────────────►│ data: {"type":     │                 │
  │               │  "done",           │                 │
  │               │  "query_time_ms":  │                 │
  │               │  3456}             │                 │
  │               │ ──────────────────►│ ───────────────►│ Show complete
```

### The SSE Event Sequence (What the Browser Receives)

```
data: {"type":"sources","sources":[{"content":"COBB: The dream...","timestamp_start":2535.0,"timestamp_end":2610.0,"relevance_score":0.87},...]}

data: {"type":"token","content":"At"}

data: {"type":"token","content":" that"}

data: {"type":"token","content":" point"}

data: {"type":"token","content":" in"}

data: {"type":"token","content":" the"}

data: {"type":"token","content":" movie"}

data: {"type":"token","content":","}

data: {"type":"token","content":" Cobb"}

... (~100 more token events)

data: {"type":"done","model_used":"llama3:8b","query_time_ms":3456.2}
```

---

## Part 3: Timing Breakdown

### Docker Mode (GPU Available to Ollama)

```
Phase                          Time      Cumulative
─────                          ────      ──────────
Request hits Gateway           0ms       0ms
Gateway reads body             1ms       1ms
Gateway opens stream to Oracle 5ms       6ms
Oracle parses request          1ms       7ms
Intent classification          1ms       8ms
Timestamp extraction           1ms       9ms
Query embedding                50ms      59ms
ChromaDB vector search         20ms      79ms
Timestamp re-ranking           1ms       80ms
Sources SSE event sent         2ms       82ms     ← User sees citations
RAG prompt construction        1ms       83ms
Ollama async_generate called   5ms       88ms
Ollama prefill (400 tokens)    500ms     588ms
First token generated          20ms      608ms    ← User sees first word
Remaining ~100 tokens          2,500ms   3,108ms
Done SSE event sent            1ms       3,109ms  ← Complete

Total: ~3.1 seconds
```

### Local Mode (No Docker, GPU)

```
Phase                          Time      Cumulative
─────                          ────      ──────────
Request hits Gateway           0ms       0ms
Gateway connects to Oracle     1ms       1ms       (localhost, no DNS)
Oracle processes               80ms      81ms
Ollama generates               2,500ms   2,581ms

Total: ~2.6 seconds (slightly faster — no Docker network overhead)
```

### Docker Mode (CPU Only — No GPU)

```
Phase                          Time      Cumulative
─────                          ────      ──────────
Same retrieval pipeline        ~80ms     80ms
Ollama prefill (CPU)           3,000ms   3,080ms
Token generation (CPU)         12,000ms  15,080ms

Total: ~15 seconds
```

---

## Part 4: Docker Mode vs Local Mode Comparison

### Network Paths

```
Docker Mode:
Browser ──► localhost:3000 ──► Frontend container (Nginx)
Browser ──► localhost:8000 ──► Gateway container ──(Docker DNS)──►
                                Oracle container ──(Docker DNS)──►
                                Ollama container

Local Mode:
Browser ──► localhost:5173 ──► Vite dev server (hot reload)
Browser ──► localhost:8000 ──► Gateway (uvicorn) ──(localhost)──►
                                Oracle (uvicorn) ──(localhost)──►
                                Ollama (native)
```

### Key Differences

| Aspect | Docker Mode | Local Mode |
| :--- | :--- | :--- |
| **Network** | Docker bridge network, DNS resolution | `localhost` direct connections |
| **Port mapping** | `8001:8000` (host:container) | Services listen on host ports directly |
| **Ollama URL** | `http://ollama:11434` | `http://localhost:11434` |
| **GPU access** | Requires `nvidia-docker` setup | Automatic GPU detection |
| **File access** | Via volume mounts | Direct filesystem access |
| **Process isolation** | Each service isolated | Shared system resources |
| **Startup** | `docker-compose up -d` | 5-7 separate terminal commands |
| **Code changes** | Rebuild container | Restart process (or hot reload) |
| **Latency overhead** | ~5-10ms per hop | ~1ms per hop |

---

## Part 5: Failure Scenarios

### Scenario A: Ollama Container Crashes

```
                        Ollama: CRASHED
                            ✗
Oracle ── connect() → self.connected = False
Oracle ── generate() → _fallback_response()
                         │
                         ▼
User sees: "[Retrieval-Only Mode - Ollama not connected]
            The relevant subtitle excerpts have been retrieved..."

Sources still show: ✓ (ChromaDB retrieval works independently)
LLM answer: ✗ (fallback message instead)
```

### Scenario B: Oracle Container Crashes

```
                        Oracle: CRASHED
                            ✗
Gateway ── POST → httpx.ConnectError
Gateway ── raises HTTPException(503)
                         │
                         ▼
User sees: {"error": "Oracle service unavailable: ..."}

restart: unless-stopped → Docker auto-restarts Oracle
After ~30 seconds: Oracle is back (lifespan startup completes)
```

### Scenario C: ChromaDB Data Missing

```
Oracle starts → connects to ChromaDB at /app/data/chromadb
                Volume mount: ./data/chromadb:/app/data/chromadb
                    │
                    ├── If volume exists: collection = client.get_or_create_collection(...)
                    │   doc_count = 356 (movies are ingested)
                    │
                    └── If volume empty: collection is created but empty
                        doc_count = 0
                        User sees: "I couldn't find any relevant dialogue for 'inception'.
                                    This movie may not have been ingested yet."
```

---

## Summary: The Complete Request Lifecycle

```
1. Browser → Frontend (Nginx)           Static files served
2. Browser → Gateway (FastAPI)           Route matched: /api/v1/oracle/ask/stream
3. Gateway → Oracle (FastAPI)            Docker DNS: oracle-rag-service:8000
4. Oracle: Query Understanding           Intent + timestamp extraction
5. Oracle: Embedding                     sentence-transformers (384-dim)
6. Oracle: ChromaDB Search               Vector similarity + timestamp filter
7. Oracle → SSE: Sources Event           Citations sent immediately
8. Oracle → Ollama (llama.cpp)           Docker DNS: ollama:11434
9. Ollama: Tokenize + Prefill            BPE + KV cache computation
10. Ollama → Oracle: Token Stream        One JSON per token
11. Oracle → Gateway: SSE Stream         data: {"type":"token",...}
12. Gateway → Browser: Raw Proxy         Bytes forwarded as-is
13. Browser: Append to Chat Bubble       Progressive text display
14. Ollama: Done Signal                  {"done": true}
15. Oracle → SSE: Done Event             {"type":"done","query_time_ms":...}
16. Browser: Show Complete State         Typing indicator removed
```
