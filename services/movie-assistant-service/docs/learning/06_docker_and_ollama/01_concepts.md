# Docker & Ollama in StreamSage: The Big Picture

This module explains the two infrastructure pillars that make StreamSage's microservices architecture work: **Docker** for containerized deployment and **Ollama** for local LLM inference.

---

## Why Do We Need Them?

StreamSage has **7 moving parts** that need to run together:

```
Frontend (React + Nginx)         → Port 3000
Gateway (FastAPI)                → Port 8000
Oracle RAG Service (FastAPI)     → Port 8001
Binge Predictor (FastAPI)        → Port 8002
Sentiment Analyzer (Flask)       → Port 8003
Movie Assistant (FastAPI)        → Port 8004
Ollama LLM Server                → Port 11434
```

Without Docker: You'd manually install Python 3.11, Node.js 20, Nginx, and all dependencies on your machine, then open 7 terminal windows to start each service — hoping nothing conflicts. On a different machine? Do it all again.

With Docker: One command — `docker-compose up -d` — and everything starts automatically, isolated, reproducible.

Without Ollama: You'd need OpenAI API keys ($$$), deal with rate limits, and send your data to external servers.

With Ollama: LLMs run on your own hardware. No API costs, no rate limits, no data leaving your machine.

---

## Docker: What It Actually Is

### The Problem Docker Solves

Imagine you build StreamSage on your Windows machine with Python 3.11. Your teammate has Python 3.9 on Ubuntu. The deployment server runs CentOS. Each environment has different:
- Python versions
- System libraries (e.g., `libssl`, `libc`)
- File paths (`C:\Users\...` vs `/home/...`)
- Package manager behavior (`pip` may resolve differently)

**"It works on my machine"** is the classic problem Docker eliminates.

### Core Concepts

| Concept | What It Is | Analogy |
| :--- | :--- | :--- |
| **Image** | A frozen snapshot of an application + all its dependencies | A recipe card |
| **Container** | A running instance of an image | A dish being cooked from the recipe |
| **Dockerfile** | Instructions to build an image | The recipe itself |
| **Docker Compose** | Orchestrates multiple containers together | The full dinner menu |
| **Volume** | Persistent storage that survives container restarts | The pantry (ingredients persist) |
| **Network** | Virtual network connecting containers | The kitchen intercom |

### How It Works Under the Hood

Docker uses **OS-level virtualization** (not full virtual machines):

```
┌─────────────────────────────────────────────────┐
│                   Your Machine                  │
│                                                 │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│   │ Oracle   │ │ Gateway  │ │ Frontend │       │
│   │ Container│ │ Container│ │ Container│ ...   │
│   │          │ │          │ │          │       │
│   │ Python   │ │ Python   │ │ Nginx    │       │
│   │ FastAPI  │ │ FastAPI  │ │ React    │       │
│   │ ChromaDB │ │ httpx    │ │ Assets   │       │
│   └──────────┘ └──────────┘ └──────────┘       │
│                                                 │
│   ┌────────────────────────────────────────┐    │
│   │          Docker Engine                 │    │
│   │   (Manages containers, networking,     │    │
│   │    volumes, image building)            │    │
│   └────────────────────────────────────────┘    │
│                                                 │
│   ┌────────────────────────────────────────┐    │
│   │          Host Operating System         │    │
│   └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

Key difference from VMs: Containers share the host kernel. No need to boot a full OS per service. This means:
- **Startup time:** Seconds (vs minutes for VMs)
- **Memory overhead:** ~50MB per container (vs 1-2GB per VM)
- **Disk usage:** Shared base layers across containers

---

## Ollama: What It Actually Is

### The Problem Ollama Solves

Running an LLM like LLaMA 3 8B locally requires:
1. Downloading the model weights (~4.7GB quantized)
2. Loading them into GPU/CPU memory
3. Running an inference server with the right parameters
4. Managing model versions and updates

Without Ollama, you'd need to write your own inference server using `transformers`, `vllm`, or `llama.cpp` — hundreds of lines of boilerplate.

Ollama packages all of this into a single service with a clean REST API.

### Core Concepts

| Concept | What It Is | In StreamSage |
| :--- | :--- | :--- |
| **Model** | A downloaded LLM (e.g., `llama3:8b`) | Powers Oracle's movie Q&A |
| **Server** | HTTP API at port 11434 | `http://ollama:11434` (Docker) or `http://localhost:11434` (local) |
| **Pull** | Download a model | `ollama pull llama3:8b` |
| **Generate** | Send prompt, get response | `POST /api/generate` |
| **Stream** | Token-by-token generation | `stream=True` in API calls |

### How Ollama Serves LLMs

```
┌─────────────────────────────────────────┐
│              Ollama Server              │
│                                         │
│  ┌─────────┐    ┌──────────────────┐    │
│  │  REST   │    │    Model Runner  │    │
│  │  API    │───►│                  │    │
│  │ :11434  │    │  llama3:8b       │    │
│  └─────────┘    │  (4-bit GGUF)    │    │
│                 │                  │    │
│                 │  ┌────────────┐  │    │
│                 │  │ GPU/CPU    │  │    │
│                 │  │ Memory     │  │    │
│                 │  │ (~5GB)     │  │    │
│                 │  └────────────┘  │    │
│                 └──────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  Model Storage                  │    │
│  │  ~/.ollama/models/              │    │
│  │  (persisted via Docker volume)  │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

Ollama uses **GGUF** format (quantized models optimized for local inference). The `llama3:8b` model in GGUF is ~4.7GB (vs ~16GB in full FP16), making it feasible to run on consumer hardware.

---

## How Docker & Ollama Work Together in StreamSage

### The Complete Architecture

```
Browser (localhost:3000)
    │
    ▼
┌─────────────────┐
│    Frontend      │  Container: React + Nginx
│    :3000         │  Serves static files, proxies API calls
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Gateway       │  Container: FastAPI
│    :8000         │  Routes /api/v1/* to correct service
└──┬───┬───┬───┬──┘
   │   │   │   │
   ▼   ▼   ▼   ▼
┌────┐┌────┐┌────┐┌────────────┐
│Orc.││Bin.││Sen.││Movie Asst. │  4 AI service containers
│8001││8002││8003││    8004    │
└──┬─┘└────┘└────┘└────────────┘
   │                    │
   ▼                    ▼
┌────────┐        ┌──────────────┐
│ Ollama │        │ HuggingFace  │
│ :11434 │        │ Inference API│
│ (local)│        │ (remote)     │
└────────┘        └──────────────┘
```

**Key insight:** Only the Oracle RAG Service uses Ollama. The Movie Assistant uses HuggingFace Inference API (or local model). They solve different problems:
- **Ollama** (Oracle): Runs LLaMA 3 8B for real-time movie dialogue Q&A with streaming
- **HuggingFace API** (Movie Assistant): Uses Mistral 7B for movie recommendations

### Docker Networking: How Services Find Each Other

Inside Docker Compose, all services share a virtual network. Instead of `localhost`, services use **container names** as hostnames:

```
# Gateway finds Oracle:
ORACLE_SERVICE_URL=http://oracle-rag-service:8000      (NOT localhost:8001)

# Oracle finds Ollama:
OLLAMA_BASE_URL=http://ollama:11434                    (NOT localhost:11434)

# Gateway finds Movie Assistant:
MOVIE_ASSISTANT_SERVICE_URL=http://movie-assistant-service:8004
```

Docker's built-in DNS resolves `oracle-rag-service` to that container's internal IP address. This is why port mapping (`8001:8000`) only matters for external access — internally, containers talk on their container ports.

---

## What If We DON'T Use Docker?

You absolutely can run StreamSage without Docker. Here's what changes:

### Without Docker (Manual Setup)

| Aspect | With Docker | Without Docker |
| :--- | :--- | :--- |
| **Starting services** | `docker-compose up -d` | Open 7 terminals, `uvicorn main:app` in each |
| **Service discovery** | Container DNS names | All services use `localhost` with different ports |
| **Python versions** | Isolated per container | Shared system Python (conflict risk) |
| **Dependencies** | Isolated per container | Shared `pip` packages (version conflicts) |
| **Ollama** | Runs in container | Install Ollama natively on your OS |
| **Reproducibility** | Same on any machine | "It works on my machine" problems |
| **Cleanup** | `docker-compose down` | Kill processes, clean up manually |
| **New developer setup** | Install Docker, run one command | Install Python, Node, Nginx, Ollama, each service's deps |

### How We Actually Develop (Local Mode)

During development, we often run services **without** Docker for faster iteration:

```bash
# Terminal 1: Start Ollama natively
ollama serve

# Terminal 2: Pull the model (first time only)
ollama pull llama3:8b

# Terminal 3: Start Oracle service
cd services/oracle-rag-service
uvicorn main:app --host 0.0.0.0 --port 8001

# Terminal 4: Start Gateway
cd gateway
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 5: Start Frontend
cd frontend
npm run dev
```

The code handles both modes because service URLs come from environment variables with `localhost` defaults:
```python
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
```

---

## Summary: When to Use What

| Scenario | Use Docker? | Use Ollama? |
| :--- | :--- | :--- |
| **Local development** | Optional (faster without) | Yes (local LLM) |
| **Demo/presentation** | Yes (one-command startup) | Yes (no API keys needed) |
| **Production deployment** | Yes (reproducible, scalable) | Depends (GPU availability) |
| **CI/CD testing** | Yes (consistent environment) | Maybe (can mock Ollama) |
| **Team onboarding** | Yes (eliminates setup issues) | Yes (zero-config LLM) |

---

## What's Next

- **[02_docker_deep_dive.md](02_docker_deep_dive.md)** — How our Dockerfiles work line-by-line, multi-stage builds, layer caching, docker-compose orchestration
- **[03_ollama_deep_dive.md](03_ollama_deep_dive.md)** — GGUF format, model serving, API protocol, streaming, GPU allocation
- **[04_code_walkthrough.md](04_code_walkthrough.md)** — Maps Docker and Ollama concepts to exact code locations
- **[05_execution_flow.md](05_execution_flow.md)** — Traces a request from browser through all containers to Ollama and back
- **[Interview Prep](interview_prep/01_docker_ollama_qa.md)** — Docker, containers, and LLM serving interview questions
