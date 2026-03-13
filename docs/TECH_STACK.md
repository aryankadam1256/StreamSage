# StreamSage Tech Stack

A comprehensive reference of all technologies, libraries, and tools used in this project.

---

## Overview

| Layer | Primary Technologies |
|-------|---------------------|
| Frontend | React 18, Vite, Tailwind CSS |
| Gateway | FastAPI, httpx |
| ML Services | PyTorch, TensorFlow, Hugging Face |
| Vector Storage | ChromaDB |
| LLM Inference | Ollama, HuggingFace Inference API |
| Containerization | Docker, Docker Compose |

---

## Frontend

### Core Framework
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.x | Component-based UI library |
| Vite | 5.x | Build tool and dev server |
| React Router | 6.x | Client-side routing |

### Styling
| Technology | Purpose |
|------------|---------|
| Tailwind CSS 3.x | Utility-first CSS framework |
| Plus Jakarta Sans | Custom typography |
| CSS Animations | Landing page visual effects |

### HTTP Client
| Library | Purpose |
|---------|---------|
| Axios | HTTP requests to API gateway |

### Key Dependencies
```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "react-router-dom": "^6.x",
  "axios": "^1.x",
  "tailwindcss": "^3.x",
  "vite": "^5.x"
}
```

---

## Backend Services

### API Gateway (Port 8000)

| Technology | Purpose |
|------------|---------|
| FastAPI | Async web framework |
| Uvicorn | ASGI server |
| httpx | Async HTTP client for proxying |
| Pydantic | Request/response validation |

### Movie Assistant Service (Port 8004)

**Embeddings & Search:**
| Technology | Purpose |
|------------|---------|
| sentence-transformers | Embedding models |
| BAAI/bge-large-en-v1.5 | Primary embedding model (1024 dim) |
| ChromaDB | Vector storage and similarity search |
| rank-bm25 | BM25 full-text search |

**Re-ranking:**
| Technology | Purpose |
|------------|---------|
| ms-marco-MiniLM-L-6-v2 | Cross-encoder for re-ranking |

**LLM Integration:**
| Technology | Purpose |
|------------|---------|
| transformers | Hugging Face model loading |
| bitsandbytes | 4-bit quantization |
| peft | LoRA/QLoRA adapters |
| trl | DPO/GRPO training |

### Oracle RAG Service (Port 8001)

| Technology | Purpose |
|------------|---------|
| Ollama | Local LLM inference |
| Llama 3 8B | Base language model |
| sentence-transformers | Subtitle embeddings |
| all-MiniLM-L6-v2 | Embedding model (384 dim) |
| ChromaDB | Subtitle vector storage |
| SSE-Starlette | Server-Sent Events streaming |

### Binge Predictor Service (Port 8002)

| Technology | Purpose |
|------------|---------|
| TensorFlow 2.x | Deep learning framework |
| Keras | High-level neural network API |
| NumPy | Numerical operations |
| Pandas | Data manipulation |

**Model Architecture:**
- LSTM layers for sequence modeling
- Keras `.h5` format for model persistence

### Sentiment Engine (Port 8003)

| Technology | Purpose |
|------------|---------|
| PyTorch | Deep learning framework |
| transformers | Model loading and tokenization |
| DistilBERT | Sentiment classification model |

---

## Machine Learning

### Models Used

| Model | Parameters | Use Case | Service |
|-------|------------|----------|---------|
| Llama 3 8B | 8B | Movie recommendation generation | Movie Assistant |
| Llama 3 8B | 8B | Subtitle Q&A | Oracle RAG |
| BAAI/bge-large-en-v1.5 | 335M | Movie embeddings (1024 dim) | Movie Assistant |
| all-MiniLM-L6-v2 | 22M | Subtitle embeddings (384 dim) | Oracle RAG |
| ms-marco-MiniLM-L-6-v2 | 22M | Cross-encoder re-ranking | Movie Assistant |
| DistilBERT | 66M | Sentiment classification | Sentiment Engine |
| Custom LSTM | ~100K | Binge prediction | Binge Predictor |

### Training Techniques

| Technique | Library | Purpose |
|-----------|---------|---------|
| QLoRA | peft, bitsandbytes | Memory-efficient fine-tuning |
| DPO | trl | Preference alignment |
| GRPO | trl | Group preference optimization |
| Transfer Learning | transformers | DistilBERT fine-tuning |
| Sequence Modeling | Keras | LSTM training |

### Training Infrastructure

| Tool | Purpose |
|------|---------|
| Google Colab | Free GPU access (A100/T4) |
| Weights & Biases | Experiment tracking (optional) |
| Hugging Face Hub | Model hosting and sharing |

---

## Data & Storage

### Databases

| Database | Type | Use Case |
|----------|------|----------|
| ChromaDB | Vector store | Movie embeddings, subtitle chunks |

### Data Sources

| Source | Data Type |
|--------|-----------|
| TMDB API | Movie metadata (6,147 films) |
| .srt Files | Subtitle transcripts |
| Custom Dataset | 2,946 Q&A training pairs |

### Data Formats

| Format | Use |
|--------|-----|
| JSON | Movie features, API responses |
| .srt | Subtitle files |
| .h5 | Keras model weights |
| PyTorch .bin | Transformer weights |
| SQLite | ChromaDB persistence |

---

## DevOps & Infrastructure

### Containerization

| Tool | Purpose |
|------|---------|
| Docker | Container runtime |
| Docker Compose | Multi-container orchestration |

### Container Images

| Service | Base Image |
|---------|------------|
| Frontend | node:20-alpine |
| Gateway | python:3.10-slim |
| Movie Assistant | python:3.10-cuda (NVIDIA) |
| Oracle RAG | python:3.10-slim |
| Binge Predictor | python:3.10-slim |
| Sentiment Engine | python:3.10-slim |
| Ollama | ollama/ollama |

### Development Tools

| Tool | Purpose |
|------|---------|
| Git | Version control |
| VS Code | Primary IDE |
| Postman/curl | API testing |
| Swagger UI | Interactive API docs |

---

## Python Dependencies by Service

### Gateway
```
fastapi>=0.109.0
uvicorn>=0.27.0
httpx>=0.26.0
pydantic>=2.5.0
```

### Movie Assistant
```
fastapi>=0.109.0
uvicorn>=0.27.0
chromadb>=0.4.22
sentence-transformers>=2.2.2
transformers>=4.37.0
torch>=2.1.0
rank-bm25>=0.2.2
peft>=0.7.0
bitsandbytes>=0.42.0
```

### Oracle RAG
```
fastapi>=0.109.0
uvicorn>=0.27.0
chromadb>=0.4.22
sentence-transformers>=2.2.2
sse-starlette>=1.8.0
```

### Binge Predictor
```
fastapi>=0.109.0
uvicorn>=0.27.0
tensorflow>=2.15.0
numpy>=1.24.0
pandas>=2.0.0
```

### Sentiment Engine
```
fastapi>=0.109.0
uvicorn>=0.27.0
torch>=2.1.0
transformers>=4.37.0
```

---

## API Specifications

### REST Endpoints

| Method | Endpoint | Service |
|--------|----------|---------|
| POST | /discover | Movie Assistant |
| POST | /ask | Oracle RAG |
| POST | /ask/stream | Oracle RAG (SSE) |
| GET | /suggestions/{movie_id} | Oracle RAG |
| POST | /predict | Binge Predictor |
| POST | /analyze | Sentiment Engine |
| POST | /batch | Sentiment Engine |
| GET | /health | All services |

### Documentation

| Format | File |
|--------|------|
| OpenAPI 3.0 | api_spec.json |
| Swagger UI | /docs (each service) |
| ReDoc | /redoc (each service) |

---

## Hardware Requirements

### Development (Minimum)
| Component | Requirement |
|-----------|-------------|
| RAM | 16 GB |
| Storage | 20 GB |
| GPU | Optional (CPU fallback) |

### Development (Recommended)
| Component | Requirement |
|-----------|-------------|
| RAM | 32 GB |
| Storage | 50 GB SSD |
| GPU | NVIDIA with 4+ GB VRAM |

### Training (Google Colab)
| Resource | Tier |
|----------|------|
| GPU | A100 40GB (Colab Pro) |
| Runtime | ~3 hours for QLoRA |

---

## Security Considerations

| Aspect | Implementation |
|--------|----------------|
| Secrets | Environment variables |
| API Keys | .env files (gitignored) |
| CORS | Configured in gateway |
| Input Validation | Pydantic models |

---

## Monitoring & Logging

| Tool | Purpose |
|------|---------|
| Python logging | Application logs |
| Docker logs | Container output |
| Health endpoints | Service status |

---

## Version Compatibility Matrix

| Component | Tested Version | Minimum Version |
|-----------|---------------|-----------------|
| Python | 3.10.x | 3.10 |
| Node.js | 20.x | 18 |
| Docker | 24.x | 20 |
| CUDA | 12.1 | 11.8 |
| PyTorch | 2.1.x | 2.0 |
| TensorFlow | 2.15.x | 2.13 |
