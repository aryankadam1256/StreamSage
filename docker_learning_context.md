# StreamSage Dockerization Journey - Context File

## 🎯 Goal
Dockerize the StreamSage microservices architecture while learning the deep internal concepts of Docker (Network, Volume, File System, Host-to-Container Bridge).

## 🏗️ Architecture Breakdown
StreamSage is divided into 5 distinct services:
1. **Ollama Service**: Pre-built `ollama/ollama:latest` image to run local LLMs.
2. **Oracle RAG Service**: Python/FastAPI container using ChromaDB & sentence-transformers.
3. **Movie Assistant Service**: Heavy Python container for PyTorch, LSTM, and BERT.
4. **API Gateway**: Lightweight Python/FastAPI routing container.
5. **Frontend**: Node.js/Vite React container.

## 📚 Progress So Far

### **Phase 1: Core Concepts & Architecture (Completed)**
* **Container vs VM**: Containers use Namespaces and Cgroups to isolate processes within the host OS (Linux kernel) instead of emulating a full hardware stack.
* **The Bridge (Windows to WSL2)**: Docker Desktop on Windows runs a hidden WSL2 Linux VM. Port mapping (e.g., `8000:8000`) punches a hole through the Windows-to-WSL hypervisor tunnel to route requests to the container.
* **Internal DNS**: Inside the Docker Bridge Network, containers use their service names (e.g., `http://movie-assistant:8004`) to communicate, bypassing `localhost` and the host OS entirely.
* **Frontend Rendering**: The Docker container acts as a web server that delivers raw JS/HTML/CSS files. The actual graphical rendering happens in the user's browser in Windows.

### **Phase 2: Crafting the Blueprints (In Progress)**
* **Gateway Dockerfile Analysis**: 
  * `python:3.11-slim`: Lightweight base image.
  * Layer Caching trick: Copying `requirements.txt` *before* the source code so `pip install` results are cached on subsequent code changes.
  * `PYTHONDONTWRITEBYTECODE=1` & `PYTHONUNBUFFERED=1`: Prevents `.pyc` clutter and ensures logs stream instantly to the console.
  * `host 0.0.0.0`: Required inside a container to listen to external network traffic instead of just `127.0.0.1` (container loopback).
* **The Missing Piece**: Discussed the importance of `.dockerignore` to prevent logging files (`gateway_live.log`, `__pycache__`) from bloating the image.

## 🚀 Next Steps (Where to resume)
1. Write/Review the `.dockerignore` files.
2. Review remaining complex `Dockerfile`s (Movie Assistant with PyTorch dependencies, and Frontend for React).
3. **Phase 3**: Dive into Orchestration (`docker-compose.yml`) to define the networks and volumes that connect the containers.
