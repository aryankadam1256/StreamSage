# Code Walkthrough: Docker & Ollama in the Codebase

This document maps every Docker and Ollama concept to the exact files and line numbers in the StreamSage codebase.

---

## Part 1: Docker Configuration Files

### 1.1 Docker Compose — The Orchestration Hub

**File:** `docker-compose.yml`

This is the single most important infrastructure file. It defines how all 7 services connect.

```
Line   Code                                         What It Does
────   ────                                         ────────────
19     version: '3.8'                               Compose file format version
22     services:                                    Begin service definitions

       ── Ollama (LLM Server) ──
32     ollama:
33       image: ollama/ollama:latest                Pre-built image (no Dockerfile needed)
35       ports: "11434:11434"                       Same port inside and out
37-38    volumes: ollama_data:/root/.ollama          Named volume for model persistence
39       restart: unless-stopped                    Auto-restart on crash
41-43    deploy.resources.reservations.memory: 4G   Minimum 4GB RAM

       ── Oracle RAG Service ──
59     oracle-rag-service:
60-62    build: context + Dockerfile                Build from local Dockerfile
65       ports: "8001:8000"                         External 8001 → Internal 8000
67       OLLAMA_BASE_URL=http://ollama:11434        Docker DNS for Ollama
68       LLM_MODEL=llama3:8b                        Which model to use
71-72    volumes: chromadb, subtitles               Persist vector DB and source files
73-74    depends_on: ollama                         Start after Ollama

       ── Binge Service ──
88     binge-service:
94       ports: "8002:8000"                         External 8002 → Internal 8000
97-99    volumes: ./data/models:/app/models         Shared model directory

       ── Sentiment Service ──
112    sentiment-service:
118      ports: "8003:5000"                         External 8003 → Internal 5000 (Flask)

       ── Movie Assistant ──
138    movie-assistant-service:
144      ports: "8004:8004"                         Same port inside and out
146      HF_API_TOKEN=${HF_API_TOKEN}               Reads from .env file or shell env
147      INFERENCE_BACKEND=api                       Use HuggingFace API (not local GPU)

       ── Gateway ──
166    gateway:
172      ports: "8000:8000"                         Single entry point
174      ORACLE_SERVICE_URL=http://oracle-rag-service:8000      Internal DNS
175      BINGE_SERVICE_URL=http://binge-service:8000            Internal DNS
176      SENTIMENT_SERVICE_URL=http://sentiment-service:5000     Internal DNS
177      MOVIE_ASSISTANT_SERVICE_URL=http://movie-assistant-service:8004
179-183  depends_on: all services                   Start last

       ── Frontend ──
197    frontend:
203      ports: "3000:3000"                         React app
207      depends_on: gateway                        Start after gateway

       ── Volumes ──
213    volumes:
214      ollama_data:                               Named volume for Ollama models
215        name: streamsage-ollama-data
```

### 1.2 Oracle RAG Dockerfile

**File:** `services/oracle-rag-service/Dockerfile`

```
Line   Code                                         What It Does
────   ────                                         ────────────
       ── Stage 1: Builder ──
14     FROM python:3.11-slim as builder             Base image for building
17-20  ENV PYTHON*=1, PIP*=1                        Optimization flags
22     WORKDIR /app                                 Working directory
25-27  apt-get ... build-essential                  C compiler for numpy, torch
30     COPY requirements.txt .                      Layer caching: requirements first
33     pip install --no-cache-dir                   Install Python packages

       ── Stage 2: Production ──
38     FROM python:3.11-slim                        Fresh slim base (no gcc)
46-47  COPY --from=builder site-packages, bin       Only keep compiled packages
50     COPY . .                                     Copy application code
53     mkdir -p /app/data/*                         Create data directories
56     EXPOSE 8000                                  Document exposed port
59-60  HEALTHCHECK ... /health                      Docker health monitoring
63     CMD uvicorn main:app                         Start the service
```

### 1.3 Frontend Dockerfile

**File:** `frontend/Dockerfile`

```
Line   Code                                         What It Does
────   ────                                         ────────────
       ── Stage 1: Build React ──
6      FROM node:20-alpine as builder               Node.js for building
8      WORKDIR /app
11     COPY package*.json ./                        Layer caching: package.json first
14     RUN npm install                              Install node_modules
17     COPY . .                                     Copy source code
20     RUN npm run build                            Vite builds to dist/

       ── Stage 2: Serve with Nginx ──
25     FROM nginx:alpine                            Lightweight web server (~25MB)
28     COPY --from=builder /app/dist ...            Only keep built static files
31     COPY nginx.conf ...                          Custom Nginx configuration
33     EXPOSE 3000
35     CMD nginx -g "daemon off;"                   Run Nginx in foreground
```

### 1.4 Movie Assistant Dockerfile

**File:** `services/movie-assistant-service/Dockerfile`

```
Line   Code                                         What It Does
────   ────                                         ────────────
       ── Same multi-stage pattern as Oracle ──
11     FROM python:3.11-slim as builder
16     COPY requirements.txt .
17     RUN pip install
27     FROM python:3.11-slim
35-36  COPY --from=builder site-packages, bin
39     COPY . .
42     mkdir -p /app/data/chromadb /app/fine_tuning/outputs
45     EXPOSE 8004                                  Different port from Oracle
48-49  HEALTHCHECK ... :8004/health                 Service-specific health check
52     CMD uvicorn ... --port 8004
```

---

## Part 2: Ollama Integration Code

### 2.1 OllamaClient Class

**File:** `services/oracle-rag-service/main.py`

```
Line    Code                                         What It Does
────    ────                                         ────────────
552     class OllamaClient:                          Wraps ollama Python package
560-565 __init__(base_url, model)                    Stores config, both clients = None

567-584 connect() → bool                             Tests Ollama connectivity
571       ollama.Client(host=self.base_url)           Create sync client
572       ollama.AsyncClient(host=self.base_url)      Create async client for SSE
574       self._client.list()                         Health check (lists models)
575       self.connected = True                       Flag for generate methods
581       self.connected = False                      Graceful degradation

586-616 generate(prompt, temperature) → str           Sync full generation
598       if not self.connected: return fallback       Check connection flag
602-612   self._client.generate(model, prompt, opts)   Call Ollama API
613       return response["response"]                  Extract text from response

618-663 generate_stream(prompt, temperature)           Sync streaming generator
641       if not self.connected: yield fallback          Check connection
647-658   self._client.generate(..., stream=True)       Stream mode
659-660   for chunk in stream: yield chunk               Yield each token

665-672 _fallback_response(prompt) → str               Retrieval-only message
667-671   "[Retrieval-Only Mode - Ollama not connected]"

674-707 async_generate_stream(prompt, temperature)     Async streaming (for SSE)
686       if not self.connected or not _async_client    Check async client
691       await self._async_client.generate(...)        Async Ollama call
703       async for chunk in stream: yield chunk        Non-blocking yield
```

### 2.2 Ollama Initialization (App Startup)

**File:** `services/oracle-rag-service/main.py`

```
Line    Code                                         What It Does
────    ────                                         ────────────
69      OLLAMA_BASE_URL = os.getenv(...)             Read URL from env (Docker vs local)
70      LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b")

714-716 Global state variables                       embedder, collection, ollama_client

719-761 async def lifespan(app):                     App startup handler
753       ollama_client = OllamaClient(...)           Create client
755       ollama_client.connect()                     Test connection (non-blocking)
```

### 2.3 Ollama Usage in Endpoints

**File:** `services/oracle-rag-service/main.py`

```
Line    Code                                         What It Does
────    ────                                         ────────────
        ── /ask endpoint (sync) ──
812     @app.post("/ask")
880     answer = ollama_client.generate(prompt)       Call Ollama sync
906     model_used=LLM_MODEL if connected             Report model in response

        ── /ask/stream endpoint (async SSE) ──
911     @app.post("/ask/stream")
970     async def event_stream():                     SSE generator function
980-985   yield sources_event                          Sources sent FIRST
1008      async for chunk in ollama_client             Token-by-token streaming
              .async_generate_stream(prompt):
1012        yield f"data: {token_event}\n\n"           SSE format

1036    return StreamingResponse(
1037      event_stream(),
1038      media_type="text/event-stream",              SSE content type
1041        "X-Accel-Buffering": "no",                  Disable proxy buffering
```

---

## Part 3: Gateway Proxy Code

### 3.1 Service URL Configuration

**File:** `gateway/main.py`

```
Line    Code                                         What It Does
────    ────                                         ────────────
52      ORACLE_SERVICE_URL = os.getenv(              Docker: http://oracle-rag-service:8000
          "ORACLE_SERVICE_URL",                       Local: http://localhost:8001
          "http://localhost:8001")
53      BINGE_SERVICE_URL = os.getenv(...)           Same pattern
54      SENTIMENT_SERVICE_URL = os.getenv(...)
55      MOVIE_ASSISTANT_SERVICE_URL = os.getenv(...)
```

### 3.2 SSE Stream Proxy

**File:** `gateway/main.py`

```
Line    Code                                         What It Does
────    ────                                         ────────────
208     @app.post("/api/v1/oracle/ask/stream")
220     body = await request.body()                   Get raw POST body
222     async def stream_proxy():                     Generator for proxying
224       async with httpx.AsyncClient(timeout=300)   Long timeout for LLM
226         async with client.stream("POST",          Streaming HTTP connection
              f"{ORACLE_SERVICE_URL}/ask/stream",
230           ) as response:
232             async for chunk in response            Pass bytes directly
                  .aiter_raw():
233               yield chunk                          Raw bytes forwarded
240     return StreamingResponse(stream_proxy(),
242       media_type="text/event-stream")
```

### 3.3 Health Check Cascade

**File:** `gateway/main.py`

```
Line    Code                                         What It Does
────    ────                                         ────────────
106     @app.get("/health")
129     async with httpx.AsyncClient(timeout=5.0)    Short timeout for health
130-135   services = [("oracle", URL), ...]           All 4 downstream services
137-148   for service_name, url in services:          Check each service
139         resp = await client.get(url)              Hit their /health endpoint
141         "healthy" if resp.status_code == 200       Report status
151-155   all_healthy = all(...)                      Aggregate status
```

---

## Part 4: Environment Variable Flow

### How Configuration Passes Through Layers

```
docker-compose.yml                    Python Code                         Runtime Value
─────────────────                     ───────────                         ─────────────

OLLAMA_BASE_URL=                      os.getenv("OLLAMA_BASE_URL",       In Docker:
  http://ollama:11434          ──►      "http://localhost:11434")    ──►  "http://ollama:11434"
                                                                         Locally:
                                                                         "http://localhost:11434"

LLM_MODEL=llama3:8b            ──►    os.getenv("LLM_MODEL",       ──►  "llama3:8b"
                                        "llama3:8b")

HF_API_TOKEN=${HF_API_TOKEN}   ──►    os.getenv("HF_API_TOKEN")    ──►  From .env file
                                                                         or shell environment

INFERENCE_BACKEND=api          ──►    os.getenv("INFERENCE_BACKEND",──►  "api" (Docker)
                                        "auto")                          "auto" (local)
```

### The Two-Mode Pattern

Every service is designed to work in both Docker and local modes. The pattern is:

```python
# In Python service code:
SOME_URL = os.getenv("SOME_URL", "http://localhost:PORT")
```

- **Docker:** `docker-compose.yml` sets the env var to use Docker DNS names
- **Local:** The env var is missing, so the default (`localhost`) is used

This is why you don't need to change any code when switching between Docker and local development.

---

## Part 5: Requirements Files (Dependencies)

### Oracle RAG Service Dependencies (Ollama-related)

**File:** `services/oracle-rag-service/requirements.txt`

```
Line    Package                      Purpose
────    ───────                      ───────
19      fastapi==0.115.0             Web framework
20      uvicorn[standard]==0.30.6    ASGI server
30      chromadb==0.5.7              Vector database
33-34   sentence-transformers, torch  Embedding model
37      ollama==0.3.3                Python client for Ollama API
40      httpx==0.27.2                Async HTTP client
```

The `ollama==0.3.3` package provides the `ollama.Client` and `ollama.AsyncClient` classes. It's a thin wrapper around HTTP calls to Ollama's REST API at port 11434.

---

## Quick Reference: File Map

| File | Role | Key Lines |
| :--- | :--- | :--- |
| `docker-compose.yml` | Orchestration, networking, volumes | Entire file |
| `services/oracle-rag-service/Dockerfile` | Oracle container build | Multi-stage, HEALTHCHECK |
| `services/movie-assistant-service/Dockerfile` | Movie Assistant container build | Port 8004 |
| `frontend/Dockerfile` | Frontend build + Nginx serve | Node → Nginx two-stage |
| `services/oracle-rag-service/main.py` | OllamaClient class + SSE streaming | Lines 552-707 (client), 911-1043 (stream) |
| `gateway/main.py` | Service routing + SSE proxy | Lines 208-247 (stream proxy) |
| `services/oracle-rag-service/requirements.txt` | Ollama Python package dependency | Line 37 |
