# Ollama Deep Dive: Local LLM Serving Explained

This document explains everything about how Ollama works in StreamSage — from model format to API protocol to streaming, and how our code integrates with it.

---

## Part 1: What Ollama Actually Is

Ollama is a **local LLM inference server**. It:
1. Downloads quantized model weights
2. Loads them into GPU/CPU memory
3. Exposes a REST API for text generation
4. Handles the complex inference loop (tokenize, KV cache, decoding, sampling)

### Why Not Just Use `transformers` or `vllm` Directly?

| Approach | Setup Complexity | Memory | Speed | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Ollama** | `ollama pull llama3:8b` (one command) | Optimized (GGUF) | Good | Development, prototyping |
| **transformers** | Write ~100 lines of inference code | High (FP16/BF16) | Moderate | Training, research |
| **vllm** | Install + configure server | Optimized (PagedAttention) | Best | Production at scale |
| **llama.cpp** | Compile from source | Lowest (GGUF) | Good (CPU optimized) | Edge deployment |

Ollama wraps `llama.cpp` under the hood — it's the user-friendly layer on top of the raw C++ inference engine.

---

## Part 2: GGUF Model Format

### What Is GGUF?

GGUF (GPT-Generated Unified Format) is a file format specifically designed for local LLM inference. It's the successor to GGML.

```
Standard HuggingFace Model:          Ollama GGUF Model:
┌─────────────────────────┐          ┌─────────────────────────┐
│ model.safetensors       │          │ llama3-8b-Q4_0.gguf     │
│ (FP16 weights: ~16GB)  │          │ (4-bit weights: ~4.7GB) │
│                         │          │                         │
│ tokenizer.json          │          │ + Tokenizer embedded    │
│ config.json             │          │ + Config embedded       │
│ special_tokens_map.json │          │ + Metadata embedded     │
│                         │          │                         │
│ Total: ~16GB + files    │          │ Total: ~4.7GB (1 file)  │
└─────────────────────────┘          └─────────────────────────┘
```

**Advantages of GGUF:**
1. **Single file**: Everything (weights + tokenizer + config) is in one file
2. **Quantized**: 4-bit, 5-bit, or 8-bit weights reduce memory 2-4x
3. **Memory-mapped**: Can be loaded directly from disk using `mmap` — the OS manages memory swapping automatically
4. **Metadata**: Stores architecture info, training params, and tokenizer in the file header

### Quantization Levels in Ollama

When you `ollama pull llama3:8b`, you get the Q4_0 quantized version by default:

| Quantization | Bits/Weight | Model Size | RAM Needed | Quality Loss |
| :--- | :--- | :--- | :--- | :--- |
| FP16 | 16 | ~16 GB | ~18 GB | None (baseline) |
| Q8_0 | 8 | ~8.5 GB | ~10 GB | Minimal (~1%) |
| **Q4_0** | **4** | **~4.7 GB** | **~6 GB** | **Small (~3-5%)** |
| Q4_K_M | 4 (mixed) | ~4.9 GB | ~6.5 GB | Very small (~2%) |
| Q2_K | 2 | ~3.0 GB | ~4 GB | Noticeable (~10%) |

**Q4_0** is the sweet spot for development — the model fits in ~6GB RAM with acceptable quality. For our Oracle movie Q&A use case, the quality difference vs FP16 is barely noticeable.

---

## Part 3: Ollama's REST API

Ollama exposes a simple HTTP API at port 11434.

### Key Endpoints

```
POST /api/generate      - Generate text (our primary endpoint)
POST /api/chat          - Chat completion (with message history)
GET  /api/tags          - List downloaded models (used in our health check)
POST /api/pull          - Download a model
POST /api/embeddings    - Generate embeddings (not used — we use sentence-transformers)
```

### The Generate API (What StreamSage Uses)

**Request:**
```json
POST http://localhost:11434/api/generate
{
    "model": "llama3:8b",
    "prompt": "You are The Oracle... SUBTITLE EXCERPTS: ...\n\nQUESTION: What happened at the 45 minute mark?",
    "options": {
        "temperature": 0.2,
        "top_p": 0.9,
        "num_predict": 220,
        "repeat_penalty": 1.1,
        "num_gpu": 99
    },
    "stream": false
}
```

**Response (non-streaming):**
```json
{
    "model": "llama3:8b",
    "response": "At that point in the movie, the characters are...",
    "done": true,
    "total_duration": 5424367000,
    "load_duration": 102334000,
    "prompt_eval_count": 428,
    "eval_count": 112,
    "eval_duration": 4835000000
}
```

**Response (streaming — `stream: true`):**
```
{"response":"At","done":false}
{"response":" that","done":false}
{"response":" point","done":false}
... (one JSON per token)
{"response":".","done":true,"total_duration":5424367000, ...}
```

### Generation Parameters Explained

Our code uses these specific parameters:

```python
options={
    "temperature": 0.2,      # Low = more deterministic, factual
    "top_p": 0.9,            # Nucleus sampling — consider top 90% probability mass
    "num_predict": 220,      # Maximum tokens to generate (~80 words)
    "repeat_penalty": 1.1,   # Penalize repeating tokens (prevents loops)
    "num_gpu": 99,           # Use ALL GPU layers (0 = CPU only)
}
```

| Parameter | Value | Effect |
| :--- | :--- | :--- |
| **temperature** | 0.2 | Low temperature = focused, factual answers. Good for Q&A where we want grounded responses, not creative writing. |
| **top_p** | 0.9 | Combined with low temperature, this filters out very unlikely tokens while keeping some diversity. |
| **num_predict** | 220 | Caps output at ~220 tokens (~80 words). Matches our system prompt's "80 words maximum" rule. |
| **repeat_penalty** | 1.1 | Slight penalty for already-used tokens. Prevents the model from saying "the movie the movie the movie..." |
| **num_gpu** | 99 | Tells Ollama to offload as many layers as possible to GPU. 99 > 32 (actual layers), so all layers go to GPU. |

### Temperature: The Most Important Parameter

```
temperature = 0.0                    temperature = 0.5                    temperature = 1.0
(greedy, deterministic)              (balanced)                           (creative, random)

Token probs:                         Token probs:                         Token probs:
"movie"  0.95 ████████████████       "movie"  0.60 ████████████           "movie"  0.35 ███████
"film"   0.04 █                      "film"   0.25 █████                  "film"   0.25 █████
"show"   0.01                        "show"   0.10 ██                     "show"   0.20 ████
                                     "story"  0.05 █                      "story"  0.15 ███
                                                                          "tale"   0.05 █

Always picks "movie"                 Usually picks "movie"                Varied, sometimes surprising
```

We use 0.2 because Oracle answers should be **factually grounded** in the subtitle excerpts, not creative.

---

## Part 4: How Our Code Integrates with Ollama

### The OllamaClient Class

**File:** `services/oracle-rag-service/main.py` (lines 552-707)

The `OllamaClient` class wraps the `ollama` Python package and adds three critical features our service needs:

```
┌─────────────────────────────────────────┐
│           OllamaClient                  │
│                                         │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │  connect()  │  │  generate()     │   │
│  │             │  │  (sync, full    │   │
│  │  Tests if   │  │   response)     │   │
│  │  Ollama is  │  │                 │   │
│  │  reachable  │  │  Used by /ask   │   │
│  └─────────────┘  └─────────────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  generate_stream()               │   │
│  │  (sync streaming — yields tokens)│   │
│  │  Generator pattern               │   │
│  └──────────────────────────────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  async_generate_stream()         │   │
│  │  (async streaming — non-blocking)│   │
│  │  Used by /ask/stream SSE         │   │
│  └──────────────────────────────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  _fallback_response()            │   │
│  │  (graceful degradation)          │   │
│  │  Returns retrieval-only message  │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 4.1 Connection Phase

```python
def connect(self) -> bool:
    import ollama
    self._client = ollama.Client(host=self.base_url)
    self._async_client = ollama.AsyncClient(host=self.base_url)
    self._client.list()  # Test call — lists downloaded models
    self.connected = True
```

Two clients are created:
- `ollama.Client` — for synchronous `/ask` endpoint (blocks until full response)
- `ollama.AsyncClient` — for async `/ask/stream` endpoint (non-blocking, needed for SSE)

The `list()` call is a cheap health check — if Ollama isn't running, it throws an exception and the service enters **retrieval-only mode**.

### 4.2 Sync Generation (`/ask` Endpoint)

```python
response = self._client.generate(
    model=self.model,
    prompt=prompt,          # Full RAG prompt: system + excerpts + question
    options={...},
)
return response["response"]  # Full text response
```

This blocks until Ollama finishes generating. Suitable for the non-streaming `/ask` endpoint where the caller waits for the complete response.

### 4.3 Async Streaming (`/ask/stream` Endpoint)

```python
async def async_generate_stream(self, prompt, temperature=0.2):
    stream = await self._async_client.generate(
        model=self.model,
        prompt=prompt,
        options={...},
        stream=True,        # Enable token-by-token streaming
    )
    async for chunk in stream:
        yield chunk         # {"response": "token", "done": false}
```

**Why async matters here:**

```
Sync streaming (BAD for SSE):        Async streaming (GOOD for SSE):
┌────────────────────────────┐       ┌────────────────────────────┐
│ for chunk in sync_stream:  │       │ async for chunk in stream: │
│   yield token              │       │   yield token              │
│                            │       │                            │
│ BLOCKS the event loop      │       │ Event loop stays free      │
│ Other requests wait!       │       │ Other requests process!    │
│ SSE events queue up        │       │ SSE events flush immediately│
└────────────────────────────┘       └────────────────────────────┘
```

FastAPI runs on an async event loop (uvicorn). If you use synchronous I/O inside an async endpoint, the entire server blocks — no other requests can be processed. The async client from `ollama.AsyncClient` gives up control to the event loop between tokens, allowing concurrent handling.

### 4.4 Graceful Degradation

```python
def _fallback_response(self, prompt):
    return (
        "[Retrieval-Only Mode - Ollama not connected]\n\n"
        "The relevant subtitle excerpts have been retrieved..."
    )
```

If Ollama crashes or is unavailable:
1. The `connect()` sets `self.connected = False`
2. Every `generate*` method checks this flag
3. Instead of failing, it returns a message telling the user to check Ollama

The service still provides the retrieved subtitle chunks — the RAG retrieval works without Ollama. Only the LLM generation step is skipped.

```
Normal mode:                        Degraded mode (no Ollama):
Query → Embed → Retrieve → LLM     Query → Embed → Retrieve → Fallback
                   ↓                                    ↓
            "At that point,          "[Retrieval-Only Mode]
             the characters          Excerpts shown in sources..."
             discuss..."
```

---

## Part 5: Ollama in Docker vs Local

### Docker Mode

```yaml
# docker-compose.yml
ollama:
  image: ollama/ollama:latest
  volumes:
    - ollama_data:/root/.ollama     # Persist downloaded models
  deploy:
    resources:
      reservations:
        memory: 4G                  # Minimum memory allocation
```

```yaml
oracle-rag-service:
  environment:
    - OLLAMA_BASE_URL=http://ollama:11434   # Docker DNS name
  depends_on:
    - ollama
```

**How models persist:** The `ollama_data` named volume maps to `/root/.ollama` inside the container. When you pull a model, the GGUF file is stored in this volume. Even if you restart or rebuild the Ollama container, the model persists because the volume exists independently.

**First-time setup in Docker:**
```bash
docker-compose up -d ollama                           # Start Ollama container
docker exec -it streamsage-ollama ollama pull llama3:8b  # Download model into volume
docker-compose up -d                                   # Start everything
```

### Local Mode

```bash
# Install Ollama on your machine (Windows/Mac/Linux)
# Then:
ollama serve                # Start the server (port 11434)
ollama pull llama3:8b       # Download model to ~/.ollama/models/
```

The Oracle service connects the same way — the only difference is the URL:
- Docker: `http://ollama:11434` (container DNS)
- Local: `http://localhost:11434` (default in code)

### GPU Access

**Local:** Ollama automatically detects your GPU (CUDA, Metal, ROCm) and uses it.

**Docker:** Requires `nvidia-docker` or Docker's native GPU support:
```yaml
# Would need to add to docker-compose.yml for GPU:
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Our current `docker-compose.yml` does NOT include GPU passthrough, so Ollama runs on CPU in Docker mode. This is slower but works everywhere. For development with GPU, run Ollama natively.

---

## Part 6: The SSE Streaming Pipeline (End-to-End)

The streaming endpoint is the most complex integration between our code, FastAPI, and Ollama. Here's the complete token journey:

```
1. User types question in Oracle Chat
     │
2. Frontend sends POST /api/v1/oracle/ask/stream
     │
3. Gateway receives request
     │   Reads body, opens streaming connection to Oracle
     │
4. Oracle receives request
     │   a) Classifies intent (regex patterns)
     │   b) Embeds query (sentence-transformers)
     │   c) Searches ChromaDB (vector similarity)
     │   d) Builds RAG prompt (system + excerpts + question)
     │
5. Oracle sends SSE Event 1: {"type": "sources", ...}
     │   Sources sent FIRST so UI renders citations immediately
     │
6. Oracle calls ollama.AsyncClient.generate(stream=True)
     │
7. Ollama processes prompt:
     │   a) Tokenize (BPE → token IDs)
     │   b) Compute KV for all prompt tokens (prefill)
     │   c) Generate first token (decode)
     │   d) Stream: {"response": "At", "done": false}
     │
8. Oracle wraps in SSE: data: {"type":"token","content":"At"}
     │
9. Gateway proxies raw bytes through
     │
10. Frontend reads chunk, appends "At" to chat bubble
      │
11. Steps 7d-10 repeat for each token (~80-220 tokens)
      │
12. Ollama sends: {"response": ".", "done": true}
      │
13. Oracle sends SSE: data: {"type":"done","query_time_ms":3456}
      │
14. Frontend shows complete, renders final state
```

### Timing Breakdown (Typical)

```
Query received:                    0 ms
Intent classification:             1 ms
Query embedding:                  50 ms    (sentence-transformers)
ChromaDB search:                  20 ms    (vector similarity)
Sources SSE sent:                 75 ms    ← User sees citations immediately

Ollama prefill (prompt tokens):  500 ms    (processes ~400 tokens)
First token generated:           600 ms    ← User sees first word appear

Tokens 2-220:                 3,000 ms    (GPU: ~40 tokens/sec)
                             15,000 ms    (CPU: ~8 tokens/sec)

Total (GPU):                  ~3.5 sec
Total (CPU):                  ~15 sec
```

---

## Part 7: Ollama vs Other LLM Serving Options

### Why Not OpenAI API?

| Factor | Ollama (Local) | OpenAI API |
| :--- | :--- | :--- |
| **Cost** | Free | $0.005-0.06 per 1K tokens |
| **Privacy** | Data stays on your machine | Sent to OpenAI servers |
| **Latency** | Network: 0ms + inference | Network: 50-200ms + inference |
| **Rate limits** | None | Tier-based limits |
| **Internet required** | No | Yes |
| **Model selection** | Open-source models | OpenAI models only |
| **Quality** | Good (LLaMA 3 8B) | Excellent (GPT-4) |
| **Setup** | Install Ollama | Get API key |

For StreamSage's Oracle feature, Ollama makes sense because:
1. Answers are grounded in retrieved subtitle excerpts (so model quality matters less)
2. Users may not want their movie watching data sent to third parties
3. No API costs during development and testing

### Why Doesn't the Movie Assistant Use Ollama?

The Movie Assistant uses **HuggingFace Inference API** (or local fine-tuned model) instead of Ollama because:

1. **Fine-tuned model**: The Movie Assistant uses our custom fine-tuned LLaMA 3 8B, which is in HuggingFace format (not GGUF). Converting to GGUF would lose the LoRA adapter precision.
2. **Different model**: It uses Mistral 7B via HF API, chosen for its instruction-following quality for recommendations.
3. **Deployment flexibility**: HF API works without any local GPU, making it accessible for Docker deployments on CPU-only machines.

```
Oracle Service:        Movie Assistant:
┌──────────┐           ┌──────────────────┐
│  Ollama  │           │  HuggingFace API │
│  Local   │           │  (Remote)        │
│          │           │  OR              │
│ llama3:8b│           │  Local Model     │
│ (GGUF)   │           │  (HF format)     │
└──────────┘           └──────────────────┘
     │                         │
     ▼                         ▼
 Real-time Q&A             Movie Discovery
 (streaming, fast)         (recommendation)
```

---

## Summary

| Concept | Details |
| :--- | :--- |
| **What Ollama is** | Local LLM inference server wrapping `llama.cpp` |
| **Model format** | GGUF (quantized, single file, memory-mapped) |
| **Default quantization** | Q4_0 (~4.7GB for LLaMA 3 8B) |
| **API** | REST at port 11434 (`/api/generate`, `/api/tags`) |
| **Our integration** | `OllamaClient` class with sync + async + fallback |
| **Key params** | `temperature=0.2, num_predict=220, num_gpu=99` |
| **Streaming** | Token-by-token SSE via `async_generate_stream()` |
| **Degradation** | Retrieval-only mode when Ollama is unavailable |
| **Docker networking** | `http://ollama:11434` (container DNS) |
| **Data persistence** | Named volume `ollama_data` for downloaded models |
