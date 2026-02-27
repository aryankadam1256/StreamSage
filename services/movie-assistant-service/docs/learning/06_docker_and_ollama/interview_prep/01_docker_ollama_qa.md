# Interview Prep: Docker, Containers & LLM Serving

Questions you'll face in ML/DevOps/Backend interviews. Each answer is structured to demonstrate both conceptual understanding and practical experience from the StreamSage project.

---

## Docker Fundamentals

### Q1: What is Docker and how does it differ from a Virtual Machine?

**Answer:**
Docker provides **OS-level virtualization** using containers. A container shares the host kernel and isolates only the application and its dependencies. A VM virtualizes the entire hardware and runs a full guest OS.

**Key differences:**

| Aspect | Docker Container | Virtual Machine |
| :--- | :--- | :--- |
| Startup time | Seconds | Minutes |
| Memory overhead | ~50MB | ~1-2GB |
| Isolation | Process-level (namespace/cgroup) | Hardware-level (hypervisor) |
| Disk size | ~100MB-1GB per image | ~5-40GB per VM |
| Performance | Near-native | ~5-10% overhead |

**Containers are better when:** You need fast startup, high density (many services on one host), and reproducible deployments. This is why our microservices use Docker — 7 containers on one machine vs 7 VMs.

**VMs are better when:** You need strong security isolation between tenants (multi-tenant cloud), or you need a different OS (Linux containers on Windows need WSL).

*In StreamSage, we run 7 containers (Frontend, Gateway, Oracle, Binge, Sentiment, Movie Assistant, Ollama) on a single machine. With VMs, this would require ~14GB just for OS overhead.*

---

### Q2: Explain multi-stage Docker builds. Why are they important?

**Answer:**
A multi-stage build uses multiple `FROM` statements in one Dockerfile. Each stage can copy artifacts from previous stages but discards everything else.

**Why it matters:**

```
Without multi-stage:           With multi-stage:
┌──────────────────────┐       ┌──────────────────────┐
│ python:3.11-slim     │       │ python:3.11-slim     │
│ + gcc (build-essential)│     │ + compiled packages  │
│ + pip cache          │       │ + app source code    │
│ + compiled packages  │       │                      │
│ + app source code    │       │ Size: ~500MB         │
│                      │       └──────────────────────┘
│ Size: ~2GB           │
└──────────────────────┘
```

The builder stage compiles Python packages (needs gcc), then the production stage copies only the compiled artifacts into a clean base. The C compiler, pip cache, and build headers are not in the final image.

*In our Oracle Dockerfile, this reduces image size from ~2GB to ~500MB — a 75% reduction. This means faster image pulls, less disk usage, and a smaller attack surface (fewer binaries in production).*

---

### Q3: What is Docker layer caching and how do you optimize for it?

**Answer:**
Each Dockerfile instruction creates a **layer** in the image. Docker caches layers and only rebuilds from the first changed layer downward.

**Optimization rule:** Put things that change rarely (dependencies) before things that change often (source code).

```dockerfile
# GOOD: requirements.txt changes rarely, code changes often
COPY requirements.txt .          # Layer 1: cached most of the time
RUN pip install -r requirements.txt  # Layer 2: cached (expensive — 2-5 min)
COPY . .                         # Layer 3: rebuilt on code changes

# BAD: Any code change invalidates the pip install layer
COPY . .                         # Layer 1: invalidated on ANY change
RUN pip install -r requirements.txt  # Layer 2: reruns every time!
```

The optimized approach saves 2-5 minutes per rebuild because pip install is cached unless requirements actually change.

*In StreamSage, all our Dockerfiles follow this pattern. The frontend does the same with `COPY package*.json` before `npm install`.*

---

### Q4: Explain Docker Compose networking. How do services discover each other?

**Answer:**
Docker Compose creates a **bridge network** for all services in the compose file. Each service gets:
1. An internal IP address (assigned by Docker)
2. A DNS name matching its service name

Services communicate using DNS names instead of IP addresses:
```yaml
# docker-compose.yml
oracle-rag-service:
  environment:
    - OLLAMA_BASE_URL=http://ollama:11434    # "ollama" is the DNS name
```

Docker's embedded DNS server resolves `ollama` → `172.18.0.2` (the Ollama container's IP).

**Port mapping vs internal ports:** `ports: "8001:8000"` means port 8001 on the host maps to port 8000 in the container. But *other containers* access it on port 8000 (the internal port), not 8001.

*In StreamSage, the Gateway connects to `http://oracle-rag-service:8000` (internal port), while from the host machine you'd hit `http://localhost:8001`.*

---

### Q5: What are Docker volumes and why do we need them?

**Answer:**
Containers are **ephemeral** — when you stop or remove a container, all data inside it is lost. Volumes provide persistent storage that survives container lifecycle events.

Two types:

**Bind mounts** (map a host directory):
```yaml
volumes:
  - ./services/oracle-rag-service/data/chromadb:/app/data/chromadb
```
The host directory `./data/chromadb` is mounted at `/app/data/chromadb` inside the container. Files are visible from both sides. Good for development (edit files on host, container sees changes).

**Named volumes** (Docker-managed):
```yaml
volumes:
  - ollama_data:/root/.ollama
```
Docker manages the storage location. More portable, better performance on Mac/Windows. Good for databases and model files.

*In StreamSage, we use bind mounts for ChromaDB (so you can inspect the database from the host) and a named volume for Ollama models (so they persist across container rebuilds).*

---

### Q6: What happens if a container crashes with `restart: unless-stopped`?

**Answer:**
The restart policy controls Docker's behavior when a container exits:

| Policy | On crash (exit code != 0) | On manual stop | On Docker daemon restart |
| :--- | :--- | :--- | :--- |
| `no` | Stay stopped | Stay stopped | Stay stopped |
| `always` | Restart | Restart | Restart |
| `unless-stopped` | Restart | Stay stopped | Stay stopped |
| `on-failure` | Restart | Stay stopped | Stay stopped |

`unless-stopped` is the sweet spot for development — services auto-recover from crashes (e.g., OOM, unhandled exception) but respect `docker-compose stop`.

*Our Oracle service uses this policy. If Ollama disconnects and causes an unhandled error, Docker restarts the container automatically. The `lifespan()` restart reconnects to Ollama and ChromaDB.*

---

## Ollama & LLM Serving

### Q7: What is Ollama and why use it instead of running transformers directly?

**Answer:**
Ollama is a local LLM inference server that wraps `llama.cpp`. It handles model download, quantization, memory management, and exposes a REST API.

**Why Ollama over raw `transformers`:**

| Aspect | Ollama | Transformers Library |
| :--- | :--- | :--- |
| Setup | `ollama pull llama3:8b` | ~100 lines of inference code |
| Format | GGUF (optimized for inference) | SafeTensors/PyTorch (training-optimized) |
| Memory | 4-bit GGUF: ~5GB | FP16: ~16GB |
| CPU inference | Optimized (llama.cpp) | Slow (PyTorch CPU) |
| Streaming | Built-in | Must implement yourself |
| Multi-model | Swap models via API | Load/unload manually |

**When to use transformers instead:** Training, fine-tuning, or when you need full control over the inference pipeline (custom decoding, attention manipulation).

*In StreamSage, the Oracle uses Ollama for serving (simple, optimized), while the Movie Assistant uses transformers for local inference when the fine-tuned model is loaded — because the fine-tuned model is in HuggingFace format, not GGUF.*

---

### Q8: Explain the GGUF model format. How does quantization work?

**Answer:**
GGUF (GPT-Generated Unified Format) stores model weights, tokenizer, and metadata in a single file. The key feature is **quantization** — reducing weight precision from 16-bit float to 4-bit integer.

**How Q4_0 quantization works:**
1. Group weights into blocks of 32
2. For each block, compute scale = max(abs(values)) / 7 (for 4-bit range -8 to 7)
3. Quantize each weight: `q = round(w / scale)`
4. Store: 32 weights × 4 bits + 1 scale factor = 16 bytes + 2 bytes = 18 bytes
5. vs FP16: 32 weights × 2 bytes = 64 bytes (3.5x reduction)

**At inference time:** Weights are dequantized on-the-fly before each matrix multiplication. The quantization error is small (~3-5% quality loss) because the scale factor preserves the relative magnitudes.

*Our Ollama deployment uses the default Q4_0 quantization (~4.7GB for LLaMA 3 8B). For the Oracle's use case — answering questions grounded in subtitle excerpts — the quality difference from FP16 is negligible.*

---

### Q9: How does SSE (Server-Sent Events) streaming work in our architecture?

**Answer:**
SSE is a protocol for one-way server-to-client streaming over HTTP. Unlike WebSockets (bidirectional), SSE uses a standard HTTP response with `Content-Type: text/event-stream`.

**Format:**
```
data: {"type":"token","content":"Hello"}\n\n
data: {"type":"token","content":" world"}\n\n
data: {"type":"done"}\n\n
```

Each event is `data: <json>\n\n` (double newline separates events).

**Why SSE over WebSockets for LLM streaming:**
1. **Simpler protocol**: Standard HTTP, works with all proxies
2. **One direction**: LLM tokens only flow server→client
3. **Auto-reconnect**: EventSource API handles disconnections
4. **Cacheable**: Standard HTTP caching rules apply

**Our 3-layer streaming pipeline:**
```
Ollama → Oracle (async generator) → Gateway (raw proxy) → Browser
```

The Oracle wraps Ollama tokens in SSE format. The Gateway proxies raw bytes without parsing. This avoids JSON parse/serialize overhead at the gateway layer.

*The design choice to send sources FIRST (before tokens) means the frontend renders timestamp citations immediately while the LLM answer streams in.*

---

### Q10: Explain the difference between sync and async Ollama clients. Why does it matter?

**Answer:**
FastAPI runs on an async event loop (uvicorn). The event loop processes all requests concurrently.

**Sync client (blocks the event loop):**
```python
response = ollama.Client().generate(...)  # Blocks for 3-15 seconds!
# During this time, NO other requests are processed
```

**Async client (cooperates with the event loop):**
```python
async for chunk in ollama.AsyncClient().generate(..., stream=True):
    yield chunk  # Yields control between tokens
# Other requests process normally between token generations
```

**Why this matters:** If user A asks a question that takes 10 seconds to generate, and user B asks a question 1 second later:
- **Sync:** User B waits 9 seconds before the server even starts processing their request
- **Async:** User B's request starts immediately. Both streams interleave on the event loop.

*In StreamSage, the `/ask` endpoint uses the sync client (acceptable because it returns a full response), while `/ask/stream` uses the async client (required for concurrent SSE streams).*

---

### Q11: How does the Oracle service handle Ollama being unavailable?

**Answer:**
The Oracle implements **graceful degradation** — it continues serving useful results even when its LLM dependency is down.

**The pattern:**
1. On startup: `OllamaClient.connect()` tests the connection. If it fails, `self.connected = False`.
2. On every generate call: Check `self.connected` before calling Ollama.
3. If disconnected: Return a fallback message instead of throwing an error.

```
Full mode (Ollama up):    Degraded mode (Ollama down):
Query → Retrieve → LLM   Query → Retrieve → Fallback
         ↓                         ↓
   AI-generated answer      "[Retrieval-Only Mode]"
   + Source citations        + Source citations still shown
```

The retrieval pipeline (embedding + ChromaDB search) works independently of Ollama. Users still get the relevant subtitle excerpts — they just don't get the AI-generated summary.

*This is a general microservices principle: design services to provide partial functionality when dependencies fail, rather than returning 503 errors.*

---

### Q12: In our project, we use Ollama for one service and HuggingFace API for another. Why not use the same approach for both?

**Answer:**
The two services have different requirements:

**Oracle RAG Service → Ollama:**
- Needs **streaming** (token-by-token for SSE)
- Runs LLaMA 3 8B in GGUF format (quantized for local inference)
- Requires **low latency** (real-time movie watching companion)
- No internet needed (privacy, offline capability)
- Base (non-fine-tuned) model is sufficient because answers are grounded in retrieved subtitles

**Movie Assistant → HuggingFace API:**
- Uses a **fine-tuned** model (QLoRA adapters merged into weights)
- Fine-tuned model is in HuggingFace format (not GGUF)
- Converting to GGUF would require `llama.cpp` tools and may lose adapter precision
- HF Inference API works without local GPU (accessible in Docker on CPU-only machines)
- Alternatively uses local inference via `transformers` + `InferenceManager`

**Could we unify them?** Yes — we could convert the fine-tuned model to GGUF and serve both through Ollama. The trade-off is added complexity in the build pipeline and potential quality loss from the GGUF conversion.

---

## Architecture & Design

### Q13: Why use an API Gateway pattern? Why not let the frontend call services directly?

**Answer:**
Without a gateway, the frontend would need to know about every service's URL and port:
```javascript
// Without gateway (BAD):
fetch('http://localhost:8001/ask')     // Oracle
fetch('http://localhost:8002/predict') // Binge
fetch('http://localhost:8003/analyze') // Sentiment
```

Problems:
1. Frontend is coupled to service topology
2. CORS must be configured on every service
3. No centralized logging or error handling
4. Can't aggregate multiple service calls into one request

With a gateway:
```javascript
// With gateway (GOOD):
fetch('http://localhost:8000/api/v1/oracle/ask')
fetch('http://localhost:8000/api/v1/binge/predict')
```

The gateway provides:
- **Single entry point**: One URL, one CORS config
- **Service abstraction**: Frontend doesn't know internal ports
- **Request aggregation**: The `/composite/movie-analysis` endpoint calls Oracle + Sentiment + Binge in parallel
- **Centralized error handling**: Consistent error format across all services
- **Future extensibility**: Can add auth, rate limiting, caching without changing services

*In StreamSage, the Gateway also handles SSE stream proxying — a non-trivial concern that would be messy to implement in the frontend directly.*

---

### Q14: What is the `depends_on` limitation in Docker Compose? How would you solve it properly?

**Answer:**
`depends_on` only waits for the container to **start**, not for the service to be **ready**. If the Oracle starts before Ollama has loaded the LLM model (~10-30 seconds), the Oracle's `connect()` will fail.

**How our code handles it:** The Oracle's `connect()` method is fault-tolerant — it catches the exception and sets `self.connected = False`, entering retrieval-only mode. It doesn't crash.

**For stricter ordering, you could add health check conditions:**
```yaml
oracle-rag-service:
  depends_on:
    ollama:
      condition: service_healthy
```

This requires a health check on the Ollama service:
```yaml
ollama:
  healthcheck:
    test: ["CMD", "ollama", "list"]
    interval: 10s
    timeout: 5s
    retries: 5
```

**Our approach is arguably better:** Instead of blocking startup until Ollama is ready, the Oracle starts immediately in degraded mode and upgrades to full mode when Ollama becomes available. This is called the **circuit breaker pattern** — the service works with reduced functionality rather than failing to start entirely.

---

### Q15: How would you add GPU support to the Docker setup?

**Answer:**
Currently, our `docker-compose.yml` runs Ollama on CPU inside Docker. For GPU access, you need:

1. **Install NVIDIA Container Toolkit** (`nvidia-docker`)
2. **Add GPU reservation to compose:**
```yaml
ollama:
  image: ollama/ollama:latest
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

3. **Verify:** `docker exec -it streamsage-ollama nvidia-smi`

**Performance impact:**
- CPU inference: ~8 tokens/second (LLaMA 3 8B Q4_0)
- GPU inference (RTX 3060): ~35 tokens/second
- GPU inference (RTX 4080): ~55 tokens/second

For our Oracle use case (~100 tokens per response), this means:
- CPU: ~12 seconds per answer
- GPU: ~3 seconds per answer

*In practice, most developers run Ollama natively (not in Docker) during development to get automatic GPU access, and use Docker only for deployment.*

---

## Rapid-Fire Questions

### Q16: What command starts all StreamSage services?
**Answer:** `docker-compose up -d`. The `-d` flag runs in detached mode (background).

### Q17: What port does the Gateway listen on?
**Answer:** 8000 (both host and container). The frontend calls `http://localhost:8000/api/v1/*`.

### Q18: How does the Oracle find Ollama in Docker?
**Answer:** Via Docker DNS. The environment variable `OLLAMA_BASE_URL=http://ollama:11434` uses the container name `ollama` as a hostname, which Docker's embedded DNS resolver maps to the container's internal IP.

### Q19: What happens to Ollama's downloaded models when you restart the container?
**Answer:** They persist. The `ollama_data` named volume maps to `/root/.ollama` inside the container. Named volumes survive container restarts, recreations, and even image rebuilds.

### Q20: Why does the frontend Dockerfile use Nginx instead of serving from Node.js?
**Answer:** After `npm run build`, React apps are static HTML/CSS/JS files. Nginx is purpose-built for serving static files: faster, lower memory (~5MB vs ~100MB), better caching, and built-in gzip compression. Node.js would be wasteful since there's no server-side rendering.
