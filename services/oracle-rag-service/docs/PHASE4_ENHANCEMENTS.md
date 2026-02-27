# Phase 4: Frontend Enhancement

## Overview

Phase 4 elevates the Oracle Chat from a basic request-response interface to a real-time, conversational experience. The three pillars are:

1. **SSE Streaming** — Token-by-token answer display (like ChatGPT)
2. **Multi-Turn Context** — Conversation history included in LLM prompt
3. **Dynamic Suggestions** — Questions grounded in the actual movie's subtitle data

```
User types question
        │
        ▼
frontend/src/api.js  askOracleStream()
   POST /api/v1/oracle/ask/stream
        │
        ▼
gateway/main.py  oracle_ask_stream()
   [transparent SSE proxy]
        │
        ▼
oracle-rag-service/main.py  ask_oracle_stream()
   ├── retrieve_chunks()        → ChromaDB
   ├── emit: {type:"sources"}  → SSE event 1 (instant)
   ├── generate_stream()        → Ollama token loop
   │     emit: {type:"token"}  → SSE event per token
   └── emit: {type:"done"}     → SSE event (final metadata)
        │
        ▼
OracleChat.jsx  onSources / onToken / onDone callbacks
  ├── Sources appear immediately (before LLM starts)
  ├── Tokens accumulate in streamTokens state (live typing)
  └── onDone commits full answer to chatHistory
```

## Architecture

### SSE Protocol

The streaming endpoint emits three event types over `text/event-stream`:

| Event | When | Payload |
|-------|------|---------|
| `sources` | Immediately after retrieval | `{type, sources[], intent}` |
| `token` | Once per Ollama token | `{type, content: "word "}` |
| `done` | After last token | `{type, model_used, query_time_ms}` |

**Why sources first?** The vector search completes in ~50ms while LLM generation takes 5-30s. Emitting sources immediately lets the UI render timestamp citations the moment retrieval finishes, before any LLM output arrives.

**Why `fetch()` instead of `EventSource`?** The browser's native `EventSource` API only supports GET requests. The Oracle needs a POST body (query, movie_id, top_k, history). So `api.js` uses `fetch()` with `ReadableStream` and manual line-by-line SSE parsing.

### Multi-Turn Conversation

```
User: "What did they discuss about the signal?"
Oracle: "The characters debated whether the signal was natural..."

User: "What about Fibonacci?"   ← ambiguous without context
```

**Without history:** The LLM has no idea what "Fibonacci" and "that" refer to.

**With history:** The prompt includes:
```
CONVERSATION HISTORY (for context):
User: What did they discuss about the signal?
Oracle: The characters debated whether the signal was natural...

CURRENT QUESTION: What about Fibonacci?
```

**Implementation:**
- Frontend accumulates `chatHistory` state (array of `{query, response}` objects)
- Before each new request, `buildHistory()` flattens it to `[{role, content}, ...]`
- Sent as `conversation_history` in the request body
- Backend `build_rag_prompt()` appends last 3 turns (6 role turns) to the prompt
- Each prior answer is truncated to 400 chars to keep the prompt manageable

### Dynamic Suggested Questions

Instead of hardcoded questions ("What did the characters discuss about reality?"), the `/suggestions/{movie_id}` endpoint samples the movie's actual ChromaDB content and generates grounded questions:

```
GET /suggestions/the_signal
→ {
    "suggestions": [
        "What's happening around 0:00 in the movie?",
        "What do the characters say about signal unusual?",
        "Who is speaking around 0:42 and what are they discussing?",
        "What's happening around 1:43 in the movie?"
    ]
}
```

**Strategy:**
1. Fetch up to 20 chunks for the movie from ChromaDB
2. Sample 4 chunks at positions [0%, 33%, 66%, 90%] of the total
3. For each chunk, use a rotating question type:
   - Type 0: Timestamp-based ("What's happening around X?")
   - Type 1: Content-based ("What do characters say about [key words]?")
   - Type 2: Speaker-based ("Who is speaking around X?")
4. Pad with generic fallbacks if <4 are generated

## Changes Made

### 1. oracle-rag-service/main.py (Updated)

**New models:**

```python
class ConversationTurn(BaseModel):
    role: str  # "user" | "oracle"
    content: str

# QueryRequest gains:
conversation_history: List[ConversationTurn] = []
```

**Updated `build_rag_prompt()`:**
- New `conversation_history` parameter
- Injects last 3 turns as `CONVERSATION HISTORY (for context):` block
- Truncates each prior turn to 400 chars
- Changes `QUESTION:` → `CURRENT QUESTION:` for clarity

**New `OllamaClient.generate_stream()`:**
- Calls `ollama.Client.generate(..., stream=True)`
- Yields raw chunk dicts `{"response": "token", "done": bool}`
- Fallback: yields single done=True chunk when Ollama is unavailable

**New `POST /ask/stream` endpoint:**
- Same retrieval pipeline as `/ask`
- Returns `StreamingResponse(event_stream(), media_type="text/event-stream")`
- `X-Accel-Buffering: no` header disables nginx buffering
- Sources event emitted before LLM call (instant feedback)
- Tokens streamed from `ollama_client.generate_stream()`

**New `GET /suggestions/{movie_id}` endpoint:**
- Samples movie's ChromaDB chunks
- Generates 4 grounded questions using distributed sampling
- Falls back to generic questions if movie not ingested

**New `_generic_suggestions()` helper:**
- Returns 4 fallback questions (same as prior hardcoded list)

**Version:** Updated to `4.0.0`

### 2. gateway/main.py (Updated)

| New Route | Forwards To | Method |
|-----------|-------------|--------|
| `POST /api/v1/oracle/ask/stream` | `ORACLE_SERVICE_URL/ask/stream` | SSE proxy |
| `GET /api/v1/oracle/suggestions/{movie_id}` | `ORACLE_SERVICE_URL/suggestions/{movie_id}` | JSON |

**SSE proxy implementation:**
```python
async with client.stream("POST", f"{ORACLE_SERVICE_URL}/ask/stream", ...) as response:
    async for chunk in response.aiter_raw():
        yield chunk
```
Uses httpx `client.stream()` with 120s timeout (LLM generation can take 30-120s). Raw bytes forwarded without deserializing.

**New import:** `StreamingResponse` added to `fastapi.responses` imports.

### 3. frontend/src/api.js (Updated)

**`askOracle()` updated:**
- New `conversationHistory = []` parameter
- Passes `conversation_history` in POST body

**New `askOracleStream()`:**
```javascript
export function askOracleStream(query, movieId, timestamp, conversationHistory, callbacks)
```
- Uses `fetch()` + `ReadableStream` (not EventSource — POST required)
- Parses SSE line-by-line: `data: {...}\n\n`
- Three callbacks: `onSources(sources, intent)`, `onToken(content)`, `onDone(meta)`
- Returns an `AbortController` so the caller can cancel mid-stream

**New `getOracleSuggestions(movieId)`:**
- `GET /api/v1/oracle/suggestions/{movieId}`
- Returns `[]` on any error (never throws)

### 4. frontend/src/components/OracleChat.jsx (Rewritten)

Complete rewrite with the following additions:

| Feature | Implementation |
|---------|---------------|
| Token streaming | `streamTokens` state accumulated via `onToken`; shows blinking cursor |
| Stop button | Replaces "Ask" during streaming; calls `AbortController.abort()` |
| Partial commit | Stopped streams are saved as `[stopped]` in history |
| Copy button | `navigator.clipboard.writeText(answer)` with 2s "Copied!" feedback |
| Collapsible sources | Per-answer "N sources" toggle button (AnimatePresence slide) |
| Expandable source chunks | Per-chunk "Show more / Show less" toggle |
| Clear button | Appears when history exists; resets `chatHistory` |
| Dynamic suggestions | `getOracleSuggestions(movieId)` on mount, falls back to hardcoded |
| Auto-scroll | `bottomRef.scrollIntoView()` on `chatHistory` and `streamTokens` changes |
| Loading indicator | 3-dot bounce animation while sources are being retrieved |
| Multi-turn context | `buildHistory()` flattens chatHistory to `[{role, content}]` |

**Closure fix for streaming state:**
React closures capture stale state values. The `onDone` callback (created when streaming starts) can't read the accumulated `streamTokens` or `streamSources` from during streaming. Fix: mirror the live state into refs:
```javascript
const streamTokensRef = useRef('')
useEffect(() => { streamTokensRef.current = streamTokens }, [streamTokens])
```
`onDone` reads `streamTokensRef.current` (always current).

## Request/Response Flow (Streaming)

```
1. User types question → handleAsk()
2. setStreaming(true), setStreamTokens(''), setStreamSources([])
3. buildHistory() → [{role:"user", content:...}, {role:"oracle", content:...}, ...]
4. askOracleStream(q, movieId, ts, history, {onSources, onToken, onDone})
5.   → fetch POST /api/v1/oracle/ask/stream
6.   → gateway proxies to oracle-rag-service:8000/ask/stream
7.
8. Oracle service:
9.   → classify_intent(query)
10.  → retrieve_chunks() from ChromaDB
11.  → yield SSE: {type:"sources", sources:[...]}  ← instant
12.  → build_rag_prompt(query, chunks, intent, history)
13.  → for chunk in ollama.generate(stream=True):
14.       yield SSE: {type:"token", content:"token "}  ← per word
15.  → yield SSE: {type:"done", model_used, query_time_ms}
16.
17. api.js callbacks fire:
18.  → onSources → setStreamSources(sources)
19.  → onToken  → setStreamTokens(prev => prev + content)  [many times]
20.  → onDone   → setChatHistory([...prev, {query, response:{
21.                   answer: streamTokensRef.current,
22.                   sources: streamSourcesRef.current,
23.                   ...meta
24.               }}])
25.              → setStreaming(false), setQuery('')
```

## Test Results

All validations passed:

```
PHASE 4 INTEGRATION TESTS
===================================================
✓ ConversationTurn + QueryRequest models:   PASS
✓ build_rag_prompt with history:            PASS
✓ History turn truncation:                  PASS
✓ SSE event format:                         PASS
✓ _generic_suggestions:                     PASS
✓ OllamaClient.generate_stream fallback:    PASS
✓ Gateway routes:                           PASS
✓ api.js new functions:                     PASS
✓ OracleChat.jsx streaming features:        PASS
✓ Frontend build:                           0 errors (727ms)

10/10 checks passed
```

## Running Phase 4

```bash
# Start the full stack
docker-compose up -d

# Access
# Frontend: http://localhost:3000
# Gateway API: http://localhost:8000/docs
# Oracle direct: http://localhost:8001/docs

# Test streaming manually
curl -X POST http://localhost:8001/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"What happens in the opening?","movie_id":"the_signal"}' \
  --no-buffer

# Test suggestions
curl http://localhost:8001/suggestions/the_signal

# Test with conversation history
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What about the signal?",
    "movie_id": "the_signal",
    "conversation_history": [
      {"role": "user", "content": "What happens at the start?"},
      {"role": "oracle", "content": "The researchers detect an unusual signal."}
    ]
  }'
```

## Known Limitations

- **Streaming + nginx:** Ensure `proxy_buffering off` in nginx config if deployed behind a reverse proxy. The `X-Accel-Buffering: no` header handles this for nginx automatically.
- **Concurrent streams:** Each stream holds a connection open for the LLM generation duration (30-120s). Docker's default limits apply.
- **History window:** Only last 3 turns are included to avoid exceeding Llama 3:8B's 8K context window. Full history is preserved in the frontend.

## Next Phase

**Phase 5: Optimization & Evaluation** will:
- Tune chunking parameters (window size, overlap) using RAGAS metrics
- Evaluate answer faithfulness and source relevance
- Profile end-to-end latency (embedding → retrieval → LLM → stream)
- Expand the subtitle database with more movies
- Improve the Oracle system prompt based on failure analysis
