# StreamSage Study Guide & Revision Cheat Sheet

Use this document for quick revision before interviews, presentations, or when you need to recall a specific concept fast. Everything is condensed — go to the full module doc for depth.

---

## Module 01: Data Acquisition

### Core Concept: Where does the data come from?

**Movie Metadata:**
- Source: TMDB API (The Movie Database)
- How: Browser-based HTML collector → `fetch(tmdb_api/discover)` → JSON
- What: Title, genres, cast, director, keywords, overview, ratings, popularity
- Volume: ~6,000+ movies

**Subtitle Files:**
- Source: Public subtitle repositories (.srt format)
- How: Automated download scripts → stored as `.srt` files
- What: Timestamped dialogue lines (e.g., `00:42:15 COBB: The dream is collapsing.`)
- Used by: Oracle RAG feature only

**Key Terms:**
- `.srt` = SubRip Text, standard subtitle format
- TMDB = The Movie Database, free public API
- `TMDB_API_KEY` = required env variable for collection scripts

---

## Module 02: Feature Engineering

### Core Concept: Raw data → ML-ready data

**Movie features transformation:**

| Raw TMDB Data | Engineered Feature | Why |
|:---|:---|:---|
| `runtime: 148` | `runtime_category: "long"` | Users say "short movie" not "90 minutes" |
| `release_date: 2010-07-16` | `era_tag: "modern"` | Pattern-based filtering |
| Overview text | `mood_tags: ["dark", "intense"]` | Answers mood queries |
| `vote_average: 8.4, vote_count: 35000, popularity: 98.6` | `popularity_score: 0.657` | Balanced normalized score |
| Crew list | `director: "Christopher Nolan"` | Flatten nested object |

**Categorization rules:**
```
Runtime: <90=short, 90-120=medium, 120-150=long, >150=epic
Era:     <1980=classic, 1980-2010=modern, >2010=contemporary
```

**Popularity score formula:**
```
score = 0.40×(vote_avg/10) + 0.30×(log(vote_count)/log(100000)) + 0.30×(popularity/500)
```

**Subtitle chunking:**
- Group 8-12 SRT lines into one chunk (~30-40 seconds of dialogue)
- Preserve `timestamp_start` and `timestamp_end` in seconds
- Why chunks not lines: gives more context for meaningful embedding

---

## Module 03: RAG Architecture

### Core Concept: Retrieval-Augmented Generation

**The RAG loop:**
```
Query → Embed → Search DB → Build Prompt → LLM → Answer
```

**Without RAG:**
- LLM only knows what was in its training data (knowledge cutoff)
- Can't answer about specific dialogue from a specific movie at minute 45

**With RAG:**
- We hand the LLM the relevant information at query time
- LLM becomes a "reader + summarizer" rather than a "memory"

**Key insight:**
- training.jsonl teaches the **style** (offline)
- ChromaDB provides the **facts** (runtime)

**The two ChromaDB collections:**

| Collection | Data | Service |
|:---|:---|:---|
| `subtitles` | ~50K subtitle chunks, timestamps | Oracle RAG |
| `movie_features` | ~6K movie records, metadata | Movie Assistant |

**Embedding model:** `BAAI/bge-large-en-v1.5`
- Output: 384-dimensional float vector
- Query prefix: `"Represent this sentence for searching relevant passages: "`
- Documents: stored without prefix
- Distance metric: cosine similarity

---

## Module 04: Fine-Tuning

### Core Concept: Teaching a model your style

**Why fine-tune for Movie Assistant (not Oracle):**
- Oracle: base model is fine — context (subtitles) does the work
- Movie Assistant: needs consistent style, format, reasoning → must be trained

**The training pipeline:**
```
train.jsonl (3,000 examples)
    ↓
SFT (Supervised Fine-Tuning)
    → Base LLaMA 3 learns the response style and format
    ↓
DPO (Direct Preference Optimization)
    → Preferred vs. rejected pairs → removes bad behaviors
    ↓
GRPO (Group Relative Policy Optimization)
    → Reward-based fine-tuning for quality maximization
    ↓
LoRA Adapters (~100MB saved to fine_tuning/outputs/)
```

**QLoRA key numbers:**
- Base model: LLaMA 3 8B (8 billion parameters)
- Quantization: 4-bit NF4 (loads into ~6GB VRAM)
- LoRA rank: 32
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- LoRA parameters: ~58M (0.7% of 8B)
- Tool: Unsloth (2× faster, 70% less memory)
- Platform: Google Colab T4 GPU (16GB VRAM)

**What LoRA does:**
```
W_effective = W_frozen + alpha × (A @ B)
   8B params    not trained   262K params each ← only these train
```

---

## Module 05: LLaMA 3 Architecture

### Core Concept: What's inside the model

**LLaMA 3 8B specs:**

| Spec | Value |
|:---|:---|
| Parameters | 8 billion |
| Decoder layers | 32 |
| Hidden dimension | 4096 |
| Attention heads | 32 (query) / 8 (key-value) |
| Vocabulary size | 128,000 tokens |
| Context length | 8,192 tokens |
| FFN intermediate dim | 14,336 |

**Architecture blocks (in order):**
```
Input Text
    ↓ [Tokenizer: BPE]
Token IDs
    ↓ [Embedding lookup: 128K × 4096]
Token Embeddings
    ↓ [×32 Decoder Layers]
    │    ├─ RMSNorm (normalize without mean centering)
    │    ├─ GQA Attention (32 query heads, 8 KV heads → grouped)
    │    │   ├─ RoPE (rotary position encodings on Q and K)
    │    │   └─ KV Cache (reuse computed K,V for prior tokens)
    │    ├─ Residual connection (add input to output)
    │    ├─ RMSNorm
    │    └─ SwiGLU FFN (gate × up × down projections)
    │       Intermediate: 14,336 dim
    │       Swish gate: output = (W_gate × x) × σ(W_gate × x) × (W_up × x)
    ↓ [Final RMSNorm]
    ↓ [Linear head: 4096 → 128K]
    ↓ [Softmax → probability over vocabulary]
Next Token (sample using temperature)
```

**Key concepts simplified:**

| Concept | What it does | Why it matters |
|:---|:---|:---|
| **GQA** | 8 KV heads shared across 32 Q heads | Saves memory (KV cache is smaller) |
| **RoPE** | Encodes position by rotating Q,K vectors | Generalizes to longer sequences than trained on |
| **KV Cache** | Stores computed K,V for past tokens | Enables efficient streaming (no recompute) |
| **RMSNorm** | Normalizes by root-mean-square (not mean) | Simpler and faster than LayerNorm |
| **SwiGLU** | Gate mechanism in FFN: Swish(gate) × up | Better than ReLU activation (smoother gradients) |

**LoRA targets explained:**
- `q_proj, k_proj, v_proj` — attention query/key/value projections
- `o_proj` — attention output projection
- `gate_proj, up_proj, down_proj` — SwiGLU FFN weights

---

## Module 06: Docker & Ollama

### Core Concept: How 7 services run together

**Docker in one sentence:** Packages each service + its dependencies into an isolated container so it runs identically everywhere.

**Docker Compose starts everything with `docker-compose up -d`**

**Service → Port mapping:**
```
Frontend     → host:3000  container:3000  (React + Nginx)
Gateway      → host:8000  container:8000  (FastAPI router)
Oracle       → host:8001  container:8000  (FastAPI + ChromaDB + Ollama)
Binge        → host:8002  container:8000  (FastAPI)
Sentiment    → host:8003  container:5000  (Flask)
MovieAsst    → host:8004  container:8004  (FastAPI + ChromaDB + HF API)
Ollama       → host:11434 container:11434 (LLM server)
```

**Inter-container URLs (Docker DNS):**
```
Oracle → Ollama:          http://ollama:11434
Gateway → Oracle:         http://oracle-rag-service:8000
Gateway → MovieAssistant: http://movie-assistant-service:8004
```

**Multi-stage Dockerfile pattern:**
```
Stage 1 (builder):   python:3.11-slim + gcc + pip install → compiled packages
Stage 2 (production): python:3.11-slim + COPY packages from builder
Result: 2GB → 500MB (75% reduction, no gcc in production)
```

**Layer caching rule:** Copy `requirements.txt` BEFORE `COPY . .` so pip install is cached on code changes.

**Ollama key facts:**
- Serves LLaMA 3 8B in GGUF format (~4.7GB Q4_0 quantized)
- REST API at `:11434` → `POST /api/generate`
- `stream=True` → one JSON per token: `{"response": "word", "done": false}`
- `num_gpu: 99` → all 32 layers offloaded to GPU
- Models persist in named Docker volume `ollama_data`

**OllamaClient in code:**
- `sync client` → for `/ask` (full response)
- `async client` → for `/ask/stream` (SSE streaming, non-blocking)
- `fallback` → retrieval-only mode when Ollama is down

**SSE (Server-Sent Events):**
```
Content-Type: text/event-stream
Format:       data: {"type":"token","content":"word"}\n\n
Events:       sources → tokens → done
```

---

## Quick Answers to Common Interview Questions

**What is RAG?**
> Retrieval-Augmented Generation. Instead of asking the LLM to remember facts from training, we retrieve relevant facts at query time from a database and hand them to the model as context in the prompt. The model then reasons over the provided context rather than its memory.

**What is fine-tuning? How is it different from RAG?**
> Fine-tuning updates the model's weights by training it on new examples. It changes *how* the model behaves and responds — style, format, reasoning. RAG changes *what information* the model has access to at inference time. We use both: RAG provides facts, fine-tuning provides personality.

**What is LoRA?**
> Low-Rank Adaptation. Instead of updating all 8B parameters during fine-tuning, we freeze the original weights and add small adapter matrices A (4096×32) and B (32×4096) to each target layer. During inference: W_eff = W + α(AB). This trains only ~58M parameters (0.7%) instead of 8B, enabling fine-tuning on consumer GPUs.

**What is GQA?**
> Grouped Query Attention. LLaMA 3 8B has 32 query heads but only 8 key-value heads. Groups of 4 query heads share one K-V pair. This reduces the KV cache size by 4x (from 32 K and 32 V matrices to 8 of each), enabling longer contexts and faster streaming.

**Why token-by-token streaming?**
> LLMs generate one token at a time — they can't produce the full response at once. If we wait for the complete response (100 tokens × 25ms = 2.5 seconds), the user sees a loading spinner. With streaming we show the first word in 600ms. This dramatically improves perceived responsiveness, which matters for interactive applications.

**What is quantization?**
> Reducing the numerical precision of model weights. LLaMA 3 8B in FP16 takes ~16GB. Q4_0 (4-bit) takes ~4.7GB. We map 16-bit float weights to 4-bit integers using a scale factor per block of 32 weights. Quality loss is ~3-5% because the scale factor preserves relative magnitudes. This makes local inference practical on consumer hardware.

**What is ChromaDB?**
> An open-source vector database. It stores text, metadata, and 384-dimensional embedding vectors. At query time, it finds the most similar vectors to a given query vector using HNSW (Hierarchical Navigable Small World) approximate nearest-neighbor search — returning results in milliseconds even with millions of vectors.

**What is the API Gateway pattern?**
> A single entry point (port 8000) that routes incoming requests to the appropriate microservice. Instead of the frontend knowing about 4 different service URLs and ports, it talks only to the gateway. The gateway handles routing, CORS, SSE stream proxying, and composite requests that aggregate multiple services.

---

## The 60-Second Project Pitch

> StreamSage is an AI movie platform I built with two features. The **Oracle** watches a movie with you — you can ask "what happened at the 45-minute mark?" and it answers in real-time using actual subtitle dialogue retrieved from ChromaDB, streamed token-by-token from a local LLaMA 3 8B model running via Ollama. The **Movie Discovery Assistant** gives personalized recommendations using a fine-tuned LLaMA 3 8B model — I collected 6,000 movies from TMDB, engineered features, generated 3,000 training examples, and fine-tuned with QLoRA on Google Colab. Both features use RAG: the Oracle retrieves subtitle chunks, the Movie Assistant retrieves movie feature vectors. Everything runs in Docker Compose with 7 containers communicating over a virtual network.

---

## Documentation Structure At-a-Glance

```
docs/learning/
├── 01_data_acquisition/      → TMDB API, SRT files, collection scripts
├── 02_feature_engineering/   → mood tags, embeddings, ChromaDB population
├── 03_rag_architecture/      → RAG flow: retrieve → prompt → generate
├── 04_fine_tuning/           → QLoRA, SFT/DPO/GRPO, train.jsonl, Unsloth
├── 05_llama3_architecture/   → 32 layers, GQA, RoPE, SwiGLU, KV Cache
├── 06_docker_and_ollama/     → Dockerfiles, Compose, Ollama, SSE streaming
└── 07_end_to_end/            → Full system: overview, Oracle e2e, Movie Assistant e2e
    ├── 01_project_overview.md        → System map, service list, numbers
    ├── 02_feature1_oracle_e2e.md     → Subtitle → ChromaDB → Ollama → browser
    ├── 03_feature2_movie_assistant_e2e.md → TMDB → fine-tune → RAG → browser
    ├── 04_how_modules_connect.md     → Cross-module knowledge map
    └── 05_study_guide.md             → This file: revision cheat sheets
```
