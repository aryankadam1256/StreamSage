# Feature 1: The Oracle — Complete End-to-End Journey

The Oracle is StreamSage's real-time movie watching companion. You're watching Inception, something confusing happens at the 45-minute mark, you type a question, and within 3 seconds you get a context-aware answer that only references events you've already seen — streamed token-by-token directly from a local LLM running on your machine.

This document traces every single step from raw subtitle file to the answer appearing in your browser.

---

## Stage 1: Data Preparation (Offline — Done Once)

### 1.1 Collecting Subtitle Files

Subtitle files (`.srt`) are the foundation of Oracle. Without them, the model has no "book" to look up answers from.

**What an SRT file looks like:**
```
1
00:42:15,000 --> 00:42:18,500
COBB: The dream is collapsing.

2
00:42:18,600 --> 00:42:22,000
ARTHUR: We need to go deeper.
```

Each subtitle entry has:
- **Sequence number**: 1, 2, 3...
- **Timestamps**: start → end (hours:minutes:seconds,milliseconds)
- **Dialogue**: The actual spoken text

**Where these come from:** The Oracle's `data_pipeline/` scripts download subtitle files from public sources and store them in `services/oracle-rag-service/data/subtitles/<movie_id>.srt`.

### 1.2 Parsing: SRT → Timestamped Chunks

Raw SRT lines are tokenized into **chunks** — groups of lines that form a coherent segment:

```
Raw SRT:                            Parsed Chunk:
─────────────────────               ──────────────────────────────────────
"00:42:15 → 00:42:18               {
 COBB: The dream is                   "movie_id": "inception",
 collapsing."                         "content": "COBB: The dream is
                                        collapsing. ARTHUR: We need to
"00:42:18 → 00:42:22                    go deeper. COBB: Arthur...",
 ARTHUR: We need to                   "timestamp_start": 2535.0,  (seconds)
 go deeper."                          "timestamp_end": 2610.0,
                                      "chunk_index": 147
"00:42:22 → 00:42:25               }
 COBB: Arthur..."
```

**Chunking strategy:** Instead of one line per chunk (too granular), the code groups 8-12 lines per chunk. This gives enough context for meaningful retrieval — an 8-line chunk represents about 30-40 seconds of dialogue.

**Timestamp conversion:** `00:42:15` → `2535.0` seconds. This enables the timestamp spoiler boundary feature: when you're at 45 minutes (2700 seconds), Oracle only retrieves chunks with `timestamp_start <= 2700`.

### 1.3 Embedding: Text → Vectors

Each chunk is converted to a 384-dimensional vector using `sentence-transformers`:

```
Model: BAAI/bge-large-en-v1.5

Input text:  "COBB: The dream is collapsing. ARTHUR: We need..."
     │
     ▼
[Tokenizer]  →  [32 Transformer Layers]  →  [Mean Pooling]
     │
     ▼
Output:  [0.021, -0.134, 0.087, ..., 0.062]  (384 numbers)
```

**Why this model?** BGE (Beijing Academy of AI) models are top-ranked on MTEB (Massive Text Embedding Benchmark). They encode semantic meaning better than older models — "dream collapsing" and "dream shattering" would get similar vectors even though those exact words don't appear together.

### 1.4 Storing in ChromaDB

All chunks and their vectors go into ChromaDB:

```python
collection.add(
    ids=["inception_chunk_147"],
    embeddings=[[0.021, -0.134, 0.087, ...]],   # 384 floats
    documents=["COBB: The dream is collapsing..."],
    metadatas=[{
        "movie_id": "inception",
        "timestamp_start": 2535.0,
        "timestamp_end": 2610.0,
        "chunk_index": 147
    }]
)
```

**What ChromaDB stores on disk:**
```
services/oracle-rag-service/data/chromadb/
├── chroma.sqlite3          # Metadata, chunk text, IDs
└── <collection_uuid>/
    ├── data_level0.bin     # Raw vector data (binary HNSW index)
    └── length.bin          # Index structure metadata
```

The HNSW (Hierarchical Navigable Small World) index enables approximate nearest-neighbor search in milliseconds, even across millions of vectors.

**After this stage:** The database contains all subtitle chunks for all ingested movies, each with a 384-dim embedding vector and timestamp metadata. This database is persisted via Docker volume — it only needs to be built once.

---

## Stage 2: Service Startup (When Docker Starts)

When `docker-compose up -d` runs, the Oracle container starts and `lifespan()` executes:

```
Oracle Container Startup:

1. Load SubtitleEmbedder
   ─────────────────────
   model = HuggingFaceEmbeddings("BAAI/bge-large-en-v1.5")
   → Downloads ~370MB model to container memory (or loads from cache)
   → Time: ~5 seconds

2. Connect to ChromaDB
   ───────────────────
   client = chromadb.PersistentClient("/app/data/chromadb")
   collection = client.get_or_create_collection("subtitles")
   → Reads chroma.sqlite3 and loads HNSW index into memory
   → Time: ~2 seconds

3. Create OllamaClient
   ───────────────────
   ollama_client = OllamaClient("http://ollama:11434", "llama3:8b")
   ollama_client.connect()
   → client.list() → tests Ollama API
   → If Ollama is ready: self.connected = True
   → If Ollama is still loading model: self.connected = False (degraded mode)
   → Time: ~1 second

Service is now ready to accept requests.
```

Meanwhile in the Ollama container:
```
Ollama Startup:

1. Server starts on :11434
2. Reads /root/.ollama/models/ (Docker volume)
   → Finds llama3:8b GGUF file (~4.7GB)
3. Memory-maps the GGUF file
   → Loads model weights into GPU/CPU RAM (~5-6GB)
4. LLM is ready to generate
   → Time: ~30-60 seconds (first time GPU loads)
```

---

## Stage 3: A Live Request — "What happened at the 45-minute mark?"

### 3.1 User Interaction in the Browser

The user is watching Inception. At the 45-minute mark, they open the Oracle panel and type their question. They select the movie from the dropdown.

**What the React component does** (`OracleChat.jsx`):
1. Captures `query = "What happened at the 45-minute mark?"`
2. Captures `movie_id = "inception"`
3. Captures `conversation_history = [...]` (last few turns)
4. Creates an `EventSource` or sends a `fetch` with streaming

```javascript
const response = await fetch('/api/v1/oracle/ask/stream', {
  method: 'POST',
  body: JSON.stringify({
    query: "What happened at the 45-minute mark?",
    movie_id: "inception",
    top_k: 5,
    stream: true,
    conversation_history: [...]
  })
});
```

### 3.2 Gateway Routing

```
POST /api/v1/oracle/ask/stream
         │
         ▼
Gateway (gateway/main.py:208)
  ├── Route matches: /api/v1/oracle/ask/stream
  ├── Reads raw body bytes
  ├── Opens streaming HTTP connection to:
  │     http://oracle-rag-service:8000/ask/stream   (Docker DNS)
  └── Proxies all bytes through without modification
```

### 3.3 Oracle: Step-by-Step Processing

**Step A — Intent Classification** (1ms)

```python
intent = classify_intent("What happened at the 45-minute mark?")
```

The function runs regex patterns against the query:
```python
patterns = {
    "timestamp_query": [r"\d+[\s-]?min(ute)?", r"mark", r"happen(s|ed)?"],
    "character_query": [r"who is", r"character", r"actor"],
    "plot_query":      [r"why did", r"how did", r"explain"],
    ...
}
```
→ `intent = "timestamp_query"` (highest match score)

**Step B — Timestamp Extraction** (1ms)

```python
timestamp_hint = extract_timestamp_hint("What happened at the 45-minute mark?")
# Regex: r"(\d+)\s*min(ute)?s?"
# Finds "45" → 45 * 60 = 2700.0 seconds
```

**Step C — Query Embedding** (50ms)

```python
query_with_prefix = f"Represent this sentence for searching relevant passages: {query}"
query_vector = embedder.embed_texts([query_with_prefix])
# → [384-dim float array]
```

**Why the prefix?** BGE models require a query instruction prefix for retrieval tasks. Documents are stored without the prefix; queries use it. This asymmetric instruction-following significantly improves retrieval accuracy.

**Step D — ChromaDB Vector Search** (20ms)

```python
results = collection.query(
    query_embeddings=[query_vector],
    n_results=15,                           # Retrieve 3x top_k for re-ranking
    where={
        "$and": [
            {"movie_id": {"$eq": "inception"}},
            {"timestamp_start": {"$lte": 2700.0}}  # Spoiler boundary!
        ]
    },
    include=["documents", "metadatas", "distances"]
)
```

ChromaDB uses the HNSW index to find the 15 subtitle chunks whose 384-dim vectors are closest (cosine similarity) to the query vector, *filtered* to only include chunks before the 45-minute mark.

**Step E — Re-Ranking by Timestamp Proximity** (1ms)

From the 15 retrieved chunks, the top 5 are selected by combining:
- **Semantic score**: cosine similarity from ChromaDB (higher = more relevant)
- **Timestamp proximity**: chunks near the 45-minute mark get a bonus

```python
combined_score = 0.7 * semantic_score + 0.3 * timestamp_proximity_score
```

This ensures chunks exactly at the 45-minute mark are prioritized over semantically similar chunks from completely different parts of the film.

**Step F — SSE Event 1: Sources** (sent immediately)

Before calling Ollama, Oracle sends the retrieved chunks to the browser:

```
data: {
  "type": "sources",
  "sources": [
    {
      "content": "COBB: The dream is collapsing. ARTHUR: We need to go deeper.",
      "timestamp_start": 2535.0,
      "timestamp_end": 2610.0,
      "relevance_score": 0.87
    },
    { ... 4 more chunks ... }
  ]
}

```

The user sees the source citations rendered in the UI **immediately** — before a single token of the answer arrives. This improves perceived responsiveness significantly.

**Step G — Build RAG Prompt**

The 5 chunks are formatted into a structured prompt:

```
SYSTEM PROMPT:
You are The Oracle — a movie watching companion with encyclopedic knowledge
of cinema dialogue and subtext. The viewer is currently at 0:45:00 in the movie.
ONLY discuss events from this point and earlier. Maximum 80 words.

SUBTITLE EXCERPTS (most relevant to the question):
─────────────────────────────────────────────────
[Excerpt 1] (0:42:15 - 0:43:30) [relevance: 87%]
COBB: The dream is collapsing. ARTHUR: We need to go deeper...

[Excerpt 2] (0:44:50 - 0:46:10) [relevance: 82%]
ARTHUR: What's your totem? COBB: A spinning top...
─────────────────────────────────────────────────

CONVERSATION HISTORY (last 2 turns):
User: Who is Cobb?
Oracle: Cobb is a skilled extractor who uses shared dreaming to steal secrets...

CURRENT QUESTION: What happened at the 45-minute mark?
ANSWER (be specific, 80 words max, only reference what you've seen):
```

**Step H — Ollama Async Generation + Token Streaming**

```python
async for chunk in ollama_client.async_generate_stream(prompt, temperature=0.2):
    token = chunk.get("response", "")
    done = chunk.get("done", False)

    if token:
        sse_event = f'data: {{"type":"token","content":"{token}"}}\n\n'
        yield sse_event.encode()

    if done:
        yield f'data: {{"type":"done","query_time_ms":{elapsed}}}\n\n'.encode()
```

Inside Ollama, for each token:
1. The KV cache is looked up (the previously computed attention keys/values)
2. One forward pass of the 32-layer transformer
3. Softmax over 128,000 vocabulary
4. Sample next token (temperature=0.2 → very confident picks)
5. Stream back: `{"response": "At", "done": false}`

### 3.4 Token-by-Token Browser Update

The browser's `EventSource` or fetch stream reader processes each SSE event:

```javascript
// OracleChat.jsx
reader.read().then(function process({done, value}) {
  if (done) return;

  const text = decoder.decode(value);
  const events = text.split('\n\n').filter(Boolean);

  events.forEach(event => {
    const data = JSON.parse(event.replace('data: ', ''));

    if (data.type === 'sources')  renderSources(data.sources);
    if (data.type === 'token')    appendToken(data.content);
    if (data.type === 'done')     finalizeResponse(data.query_time_ms);
  });

  reader.read().then(process);
});
```

The chat bubble grows word by word as each token arrives. The final result looks like:

> **"At the 45-minute mark**, Cobb and Arthur are inside a collapsing dream. Arthur questions Cobb about his totem — a spinning top that tells him whether he's dreaming. The dream destabilizes because Cobb is thinking of Mal, his deceased wife. They're about to go deeper into a second dream level."

---

## End-to-End Summary Diagram

```
OFFLINE PHASE (done once):
─────────────────────────────────────────────────────────────────────────

  SRT Files                ChromaDB (persisted)        Ollama Model
  (subtitle text)          (subtitle embeddings)       (llama3:8b GGUF)
       │                          │                          │
  Parse + chunk            Each chunk = 384-dim       ~4.7GB on disk
  ~30 sec segments         vector + timestamp          in Docker volume
       │                          │
  Embed with BGE           ┌──────▼──────────────────┐
  sentence-transformers    │  HNSW Vector Index       │
       └────────────────►  │  (fast cosine search)    │
                           └─────────────────────────-┘


LIVE REQUEST PHASE:
─────────────────────────────────────────────────────────────────────────

Browser → Gateway             Oracle Service                  Ollama
──────────────────────────   ────────────────────────────    ─────────────
                              1. Classify intent (1ms)
POST /ask/stream          →   2. Extract timestamp (1ms)
                              3. Embed query (50ms)
                              4. ChromaDB search (20ms)
                              5. Re-rank chunks (1ms)
                              6. ──► SEND SOURCES SSE (75ms)   (no Ollama needed)
                              7. Build RAG prompt (1ms)
                              8. Call Ollama async ──────────► Prefill 400 tokens
                                                              Generate token 1 (600ms)
                              9. Wrap in SSE ◄──────────────── "At"
                             10. Proxy to browser             Generate token 2
                              + wrap in SSE ◄─────────────── " the"
                             11. ...repeats 100x...            ...
Browser: show "At the..."    12. Done SSE ◄──────────────── {"done": true}
Browser: show full answer         (at ~3 seconds)
```
