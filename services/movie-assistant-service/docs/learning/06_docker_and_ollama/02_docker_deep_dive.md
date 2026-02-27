# Docker Deep Dive: Every Component Explained

This document walks through every Docker-related file in StreamSage — what each line does, why multi-stage builds matter, and how Docker Compose orchestrates the entire platform.

---

## Part 1: Dockerfiles — Building Container Images

### What a Dockerfile Does

A Dockerfile is a script that builds an **image** — a frozen filesystem containing your app, its dependencies, and everything needed to run it. Think of it as an automated installation script that produces a portable, reproducible package.

### 1.1 Oracle RAG Service Dockerfile (Multi-Stage Build)

**File:** `services/oracle-rag-service/Dockerfile`

This is the most instructive Dockerfile because it demonstrates a **multi-stage build** — the key Docker optimization pattern.

```dockerfile
# ===== STAGE 1: Builder =====
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system-level build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST (layer caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

**Line-by-line breakdown:**

| Line | What It Does | Why |
| :--- | :--- | :--- |
| `FROM python:3.11-slim as builder` | Starts from a minimal Python 3.11 image. `as builder` names this stage | `slim` variant is ~150MB vs ~1GB for full image |
| `PYTHONDONTWRITEBYTECODE=1` | Don't create `.pyc` files | Saves disk space in container |
| `PYTHONUNBUFFERED=1` | Print output immediately, don't buffer | Ensures `docker logs` shows output in real-time |
| `PIP_NO_CACHE_DIR=1` | Don't cache downloaded packages | Smaller image size |
| `WORKDIR /app` | Set working directory | All subsequent commands run from here |
| `apt-get ... build-essential` | Install C/C++ compiler | Needed to compile Python packages with C extensions (like `numpy`, `torch`) |
| `rm -rf /var/lib/apt/lists/*` | Delete package manager cache | Smaller image (saves ~30MB) |
| `COPY requirements.txt .` | Copy only requirements first | **Layer caching** — if requirements.txt hasn't changed, Docker reuses the cached pip install layer (saves minutes on rebuilds) |
| `pip install --no-cache-dir` | Install Python packages | `--no-cache-dir` prevents pip from storing wheels |

```dockerfile
# ===== STAGE 2: Production =====
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy ONLY the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY . .

# Create data directories
RUN mkdir -p /app/data/chromadb /app/data/subtitles /app/data/processed

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**The multi-stage magic explained:**

```
Stage 1 (Builder):                    Stage 2 (Production):
┌─────────────────────────┐           ┌─────────────────────────┐
│ python:3.11-slim        │           │ python:3.11-slim        │
│ + build-essential (gcc) │           │ (NO gcc, NO apt cache)  │
│ + apt cache             │           │                         │
│ + pip cache             │    ──►    │ + installed packages    │
│ + installed packages    │  COPY     │   (from builder)        │
│                         │  only     │ + application code      │
│  Size: ~2GB             │  these    │                         │
└─────────────────────────┘           │  Size: ~500MB           │
                                      └─────────────────────────┘
```

**Why multi-stage?** The builder stage needs `build-essential` (gcc, make, etc.) to compile Python packages. But the compiled packages don't need gcc at runtime. By copying only the compiled packages into a clean stage, we cut the image size by ~75%.

**HEALTHCHECK explained:**

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
```

| Parameter | Meaning |
| :--- | :--- |
| `--interval=30s` | Check every 30 seconds |
| `--timeout=10s` | Fail if check takes longer than 10 seconds |
| `--start-period=30s` | Wait 30 seconds before first check (startup grace period) |
| `--retries=3` | Mark unhealthy after 3 consecutive failures |
| `CMD ...` | Hit the `/health` endpoint to verify the service is responding |

Docker uses this to know if a container is healthy, degraded, or dead — critical for orchestration.

---

### 1.2 Frontend Dockerfile (Node.js + Nginx)

**File:** `frontend/Dockerfile`

```dockerfile
# Build stage: compile React app
FROM node:20-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Production stage: serve with Nginx
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
```

**Two-stage strategy:**

```
Stage 1 (node:20-alpine):            Stage 2 (nginx:alpine):
┌─────────────────────────┐           ┌─────────────────────────┐
│ Node.js 20              │           │ Nginx                   │
│ + node_modules (~500MB) │           │ + static HTML/JS/CSS    │
│ + source code           │    ──►    │   (from builder)        │
│ + Vite config           │  COPY     │ + nginx.conf            │
│ + built assets (dist/)  │  only     │                         │
│                         │  dist/    │  Size: ~25MB            │
│  Size: ~1GB             │           └─────────────────────────┘
└─────────────────────────┘
```

**Why Nginx instead of Node.js for serving?**
- React apps are static files after building — you don't need a full Node.js runtime to serve HTML/CSS/JS
- Nginx is purpose-built for serving static files: faster, lower memory, better caching
- The final image is ~25MB instead of ~1GB

---

### 1.3 Movie Assistant Dockerfile

**File:** `services/movie-assistant-service/Dockerfile`

Same multi-stage pattern as Oracle, with a key difference in the comments:

```
# In Docker, INFERENCE_BACKEND=api uses HuggingFace Inference API by default.
# For local inference with the trained model, mount the fine_tuning/outputs
# volume and set INFERENCE_BACKEND=local (requires GPU passthrough).
```

This means the Movie Assistant has two modes:
- **Docker mode (`api`)**: Calls HuggingFace's servers for LLM generation (no local GPU needed)
- **Local mode (`local`)**: Loads the fine-tuned model into GPU memory (requires `nvidia-docker`)

This is a practical design choice: not every deployment has a GPU, but you still want the service to work.

---

## Part 2: Docker Compose — Orchestrating Everything

**File:** `docker-compose.yml`

Docker Compose takes individual containers and wires them into a working system.

### 2.1 The Service Definitions

Each service in `docker-compose.yml` corresponds to one container:

```yaml
services:
  ollama:
    image: ollama/ollama:latest       # Use a pre-built image (no Dockerfile)
    container_name: streamsage-ollama
    ports:
      - "11434:11434"                 # host_port:container_port
    volumes:
      - ollama_data:/root/.ollama     # Named volume for model persistence
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          memory: 4G                  # Reserve at least 4GB RAM
```

### 2.2 Key Concepts in Our Compose File

#### Port Mapping (`host:container`)

```yaml
oracle-rag-service:
  ports:
    - "8001:8000"    # External 8001 → Internal 8000
```

The Oracle service listens on port 8000 **inside** its container. Docker maps port 8001 on your machine to port 8000 in the container. This is why:
- From your browser: `http://localhost:8001/health`
- From another container: `http://oracle-rag-service:8000/health` (internal port)

```
Your Machine                    Docker Network
┌──────────────┐               ┌──────────────────────┐
│              │               │                      │
│  Browser     │──:8001──►────►│  Oracle (:8000)      │
│              │               │                      │
│  Postman     │──:8000──►────►│  Gateway (:8000)     │
│              │               │                      │
│              │               │  Gateway ──────►     │
│              │               │    Oracle (:8000)     │
│              │               │    (internal DNS)     │
└──────────────┘               └──────────────────────┘
```

#### Environment Variables

```yaml
oracle-rag-service:
  environment:
    - OLLAMA_BASE_URL=http://ollama:11434    # Docker DNS name
    - LLM_MODEL=llama3:8b
    - LOG_LEVEL=INFO
```

This is how we configure services differently for Docker vs local development:
- **In Docker:** `OLLAMA_BASE_URL=http://ollama:11434` (container DNS)
- **Locally:** Falls back to default `http://localhost:11434` (from Python code)

The Python code reads these at startup:
```python
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
```

#### Volumes (Data Persistence)

```yaml
oracle-rag-service:
  volumes:
    - ./services/oracle-rag-service/data/chromadb:/app/data/chromadb
```

**Without volumes:** If you stop a container, all data inside it is lost. The ChromaDB vector database, subtitle files, model weights — everything disappears.

**With volumes:** The host directory (`./services/oracle-rag-service/data/chromadb`) is mounted inside the container at `/app/data/chromadb`. Changes persist even when the container restarts.

Two types of volumes in our project:

| Type | Example | Purpose |
| :--- | :--- | :--- |
| **Bind mount** | `./data/chromadb:/app/data/chromadb` | Maps a host directory into the container. You can see and edit files from both sides. |
| **Named volume** | `ollama_data:/root/.ollama` | Docker-managed storage. Defined in the `volumes:` section at the bottom. Files persist but are managed by Docker. |

#### Dependencies (`depends_on`)

```yaml
gateway:
  depends_on:
    - oracle-rag-service
    - binge-service
    - sentiment-service
    - movie-assistant-service
```

This tells Docker Compose to start the AI services **before** the gateway. But note: `depends_on` only waits for the container to start, not for the service to be *ready*. That's why we have health checks — the gateway can tolerate downstream services still initializing.

```
Start Order:

1. ollama
2. oracle-rag-service (depends_on: ollama)
   binge-service
   sentiment-service
   movie-assistant-service
3. gateway (depends_on: all services)
4. frontend (depends_on: gateway)
```

#### Restart Policy

```yaml
restart: unless-stopped
```

| Policy | Behavior |
| :--- | :--- |
| `no` | Never restart (default) |
| `always` | Always restart, even after `docker-compose stop` |
| `unless-stopped` | Restart on crash, but stay stopped if you manually stop it |
| `on-failure` | Restart only on non-zero exit code |

We use `unless-stopped` so services auto-recover from crashes but respect manual stops.

---

## Part 3: Layer Caching — Why Build Order Matters

Docker builds images in layers. Each Dockerfile instruction creates a layer. Docker caches layers and only rebuilds from the first changed layer onward.

```dockerfile
COPY requirements.txt .              # Layer 1: hash of requirements.txt
RUN pip install -r requirements.txt  # Layer 2: cached if requirements.txt unchanged
COPY . .                             # Layer 3: hash of all source files
```

**If you change your Python code but NOT requirements.txt:**
- Layer 1: CACHED (requirements.txt hasn't changed)
- Layer 2: CACHED (pip install hasn't changed)
- Layer 3: REBUILT (source code changed)

**If requirements.txt were copied with all source code:**
```dockerfile
COPY . .                             # Layer 1: ANY file change invalidates this
RUN pip install -r requirements.txt  # Layer 2: pip install reruns every time!
```

This is why we copy `requirements.txt` separately — it saves 2-5 minutes per rebuild because pip install is the slowest step.

```
Optimized:                           Unoptimized:
┌─────────────────────┐              ┌─────────────────────┐
│ COPY requirements   │ ◄── cached   │ COPY . .            │ ◄── invalidated
│ RUN pip install     │ ◄── cached   │ RUN pip install     │ ◄── reruns (slow!)
│ COPY . .            │ ◄── rebuilt  │                     │
│                     │              │                     │
│ Rebuild: ~5 seconds │              │ Rebuild: ~5 minutes │
└─────────────────────┘              └─────────────────────┘
```

---

## Part 4: Docker Networking in StreamSage

When you run `docker-compose up`, Docker creates a virtual network connecting all services. Each container gets:
1. An internal IP address (e.g., `172.18.0.3`)
2. A DNS name matching its service name (e.g., `oracle-rag-service`)

### How the Gateway Finds Services

```python
# gateway/main.py
ORACLE_SERVICE_URL = os.getenv("ORACLE_SERVICE_URL", "http://localhost:8001")
```

In Docker, this is overridden to:
```yaml
ORACLE_SERVICE_URL=http://oracle-rag-service:8000
```

The request flow:
```
Gateway container           Docker DNS              Oracle container
     │                          │                        │
     │ GET oracle-rag-service   │                        │
     │ ─────────────────────►   │                        │
     │                          │  resolves to           │
     │                          │  172.18.0.3            │
     │                          │ ──────────────────────►│
     │                          │                        │ processes request
     │ ◄───────────────────────────────────────────────  │
     │         JSON response                             │
```

### SSE Streaming Through Docker

The Oracle's streaming endpoint (`/ask/stream`) sends Server-Sent Events through two layers of proxying:

```
Browser  ──►  Frontend  ──►  Gateway  ──►  Oracle  ──►  Ollama
                              (proxy)      (stream)     (generate)
                                │
                        StreamingResponse
                     (proxies raw SSE bytes)
```

The gateway's stream proxy is key — it passes bytes through without buffering:
```python
async with client.stream("POST", f"{ORACLE_SERVICE_URL}/ask/stream", ...) as response:
    async for chunk in response.aiter_raw():
        yield chunk
```

The `X-Accel-Buffering: no` header tells Nginx (and any reverse proxy) not to buffer the stream.

---

## Part 5: Key Docker Commands for StreamSage

```bash
# Start all services (detached mode)
docker-compose up -d

# Start only Ollama (for first-time model pull)
docker-compose up -d ollama

# Pull the LLM model (first time only)
docker exec -it streamsage-ollama ollama pull llama3:8b

# View logs from all services
docker-compose logs -f

# View logs from a specific service
docker-compose logs -f oracle-rag-service

# Stop all services
docker-compose down

# Stop and remove all data (careful!)
docker-compose down -v

# Rebuild a specific service after code changes
docker-compose build oracle-rag-service
docker-compose up -d oracle-rag-service

# Check container health status
docker ps

# Enter a running container for debugging
docker exec -it streamsage-oracle /bin/bash

# Check resource usage
docker stats
```

---

## Summary

| Docker Concept | How It's Used in StreamSage |
| :--- | :--- |
| **Multi-stage builds** | Keep production images small (~500MB vs ~2GB) |
| **Layer caching** | Fast rebuilds by isolating `requirements.txt` |
| **Docker Compose** | Orchestrates 7 services with one command |
| **Networking** | Services find each other via DNS names |
| **Volumes** | Persist ChromaDB vectors, Ollama models |
| **Health checks** | Docker monitors service availability |
| **Port mapping** | External ports differ from internal ports |
| **Environment variables** | Configure services differently per environment |
| **Restart policies** | Auto-recover from crashes |
