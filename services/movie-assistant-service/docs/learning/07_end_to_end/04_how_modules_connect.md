# How All Modules Connect: The Knowledge Map

You've now read 6 documentation modules. This document shows you exactly how they connect to each other and to the live system — so you have a complete mental model of StreamSage.

---

## The Dependency Chain

The modules were designed in build order. Each one depends on the previous:

```
Module 01                Module 02                Module 03
Data Acquisition    ──►  Feature Engineering  ──►  RAG Architecture
─────────────────        ──────────────────────    ─────────────────
Answers:                 Answers:                  Answers:
"Where does the          "How do raw movies         "How does the AI
 data come from?"         become searchable?"        find the right
                                                     movies for any
                                                     query?"
       │                        │
       │                        │
       ▼                        ▼
TMDB API → raw JSON    Feature extraction    Vector DB + embedding
Subtitle .srt files    Movie features JSON   ChromaDB collections
                       Training examples


Module 04                Module 05                Module 06
Fine-Tuning         ──►  LLaMA 3 Architecture ──►  Docker & Ollama
─────────────────        ──────────────────────    ─────────────────
Answers:                 Answers:                  Answers:
"How did we train        "What is actually          "How do all 7
 a custom model?"         inside the model           services run
                          we're training?"           together?"

SFT + DPO + GRPO       32 decoder layers          7 containers
LoRA adapters          GQA, RoPE, SwiGLU          Docker Compose
train.jsonl            RMSNorm, KV Cache          Ollama serves LLM


Module 07 (this module)
End-to-End Understanding
─────────────────────────
Answers:
"How do all the pieces
 connect in a live request?"
```

---

## Module-to-Feature Mapping

Each module directly enables part of one or both features:

```
                        ORACLE FEATURE          MOVIE DISCOVERY
                        (Feature 1)             (Feature 2)
                        ────────────────────    ────────────────────

Module 01               Subtitle .srt files     TMDB metadata
Data Acquisition        ↓                       ↓
                        Raw dialogue text       Raw movie JSON

Module 02               SRT → chunks            JSON → movie_features
Feature Engineering     → timestamp metadata    → mood, era, score tags
                        ↓                       ↓
                        Structured subtitle     Structured movie records

Module 03               Subtitle embeddings     Movie feature embeddings
RAG Architecture        → in ChromaDB           → in ChromaDB
                        → vector search         → vector + BM25 + RRF
                        → timestamp filter      → cross-encoder rerank
                        (RETRIEVAL)             (RETRIEVAL)

Module 04               Not used                train.jsonl training
Fine-Tuning             (base LLM is enough)    → LoRA adapters
                                                → style + format + reasoning
                                                (GENERATION)

Module 05               Inside Ollama           Inside HF API
LLaMA 3 Architecture    → GQA attention         → Same architecture
                        → KV Cache (streaming)  → Fine-tuned with LoRA
                        → SwiGLU FFN            → LoRA targets: q/k/v/o
                        (MODEL INTERNALS)       (MODEL INTERNALS)

Module 06               Ollama in Docker        HF API (no container)
Docker & Ollama         → llama3:8b GGUF        OR local model in container
                        → streaming SSE         → INFERENCE_BACKEND=api
                        → OllamaClient          (DEPLOYMENT)
                        (DEPLOYMENT)
```

---

## The Two ChromaDB Collections

One of the key things to understand is that there are **two separate vector databases**, not one:

```
services/oracle-rag-service/data/chromadb/
└── Collection: "subtitles"
    ├── ~50,000+ chunks (depends on movies ingested)
    ├── Each chunk: 30-40 seconds of dialogue
    ├── Metadata: movie_id, timestamp_start, timestamp_end, chunk_index
    └── Embedding: 384-dim (BAAI/bge-large-en-v1.5)

    Used by: Oracle Service only
    Purpose: Answer "what happened at this moment?" questions

services/movie-assistant-service/data/chroma_db/
└── Collection: "movie_features"
    ├── ~6,000 movies
    ├── Each document: rich text representation of movie metadata
    ├── Metadata: title, year, genres, mood_tags, director, popularity_score
    └── Embedding: 384-dim (BAAI/bge-large-en-v1.5)

    Used by: Movie Assistant Service only
    Purpose: Find movies matching user's taste/mood
```

**Same embedding model, different data, different services.**

---

## The Two LLMs

There are also **two different LLMs** running in the system:

```
LLM 1: Ollama (Oracle Service)
─────────────────────────────────────────────────────
Model:   llama3:8b (base, unmodified)
Format:  GGUF (quantized, local inference)
Where:   Ollama container → localhost:11434 inside Docker
How:     OllamaClient.async_generate_stream() via SSE
Why base: The RAG context (subtitles) does all the heavy lifting.
          The model just needs to summarize. No style training needed.
Use:     Oracle's /ask and /ask/stream endpoints


LLM 2: HuggingFace API / Local (Movie Assistant Service)
─────────────────────────────────────────────────────────
Model:   Mistral 7B Instruct (via HF API in Docker)
      OR LLaMA 3 8B + LoRA adapters (local/fine-tuned mode)
Format:  HuggingFace SafeTensors + LoRA adapter weights
Where:   HuggingFace's servers (API mode) or GPU memory (local mode)
How:     InferenceClient.text_generation() or InferenceManager
Why fine-tuned: Recommendations require consistent style, structured
                format, and explicit reasoning — this needs to be learned
                from curated examples.
Use:     Movie Assistant's /discover endpoint
```

---

## Data Flow Across the Full System

Here is every piece of data and where it comes from and goes to:

```
External World          StreamSage                    User
──────────────          ─────────────                 ────

TMDB API ──────────►    movie_features.json
                        → ChromaDB (movie_features) ──────────────────►
                                                       /discover query
                                                       movie recommendation

Subtitle sources ──►    .srt files
                        → parsed chunks
                        → ChromaDB (subtitles) ────────────────────────►
                                                       /ask/stream query
                                                       movie Q&A answer

HuggingFace ───────►    Base LLaMA 3 8B weights
                        → SFT fine-tuning (train.jsonl)
                        → DPO alignment
                        → GRPO alignment
                        → LoRA adapters ────────────────────────────────►
                                                       Movie recommendations
                                                       in fine-tuned style

Ollama Hub ─────────►   llama3:8b GGUF
                        (stored in Docker volume) ──────────────────────►
                                                       Oracle answers
                                                       (streamed tokens)

TMDB Image URLs ───►    https://image.tmdb.org/... ──────────────────────►
                        (fetched by frontend directly)  Movie posters
```

---

## Module Concepts That Appear in Multiple Places

Some concepts from the documentation modules appear in both features. Knowing where each concept is used helps you explain the project:

### Embedding (sentence-transformers / BGE)

| Appears in | Role |
|:---|:---|
| Module 02 (Feature Engineering) | Converting movie feature text to vectors for ChromaDB |
| Module 03 (RAG Architecture) | Runtime query embedding for similarity search |
| Feature 1 (Oracle) | Query embedding before subtitle chunk retrieval |
| Feature 2 (Movie Assistant) | Query embedding before movie feature retrieval |

Same model (`BAAI/bge-large-en-v1.5`), same output dimension (384), used identically across both features.

### ChromaDB

| Appears in | Role |
|:---|:---|
| Module 02 (Feature Engineering) | Building and populating collections |
| Module 03 (RAG Architecture) | Runtime vector search |
| Feature 1 | `subtitles` collection, timestamp-filtered queries |
| Feature 2 | `movie_features` collection, metadata-filtered queries |
| Module 06 (Docker) | Persisted via volume mount so data survives restarts |

### LLaMA 3 8B (the model itself)

| Appears in | Role |
|:---|:---|
| Module 04 (Fine-Tuning) | The base model that gets fine-tuned with LoRA |
| Module 05 (LLaMA 3 Architecture) | Understanding what's inside the model |
| Feature 1 (Oracle via Ollama) | Base LLaMA 3 8B serves movie Q&A |
| Feature 2 (Movie Assistant) | Fine-tuned LLaMA 3 8B serves recommendations |

### FastAPI + uvicorn

| Appears in | Role |
|:---|:---|
| Module 06 (Docker) | `CMD uvicorn main:app` in every Dockerfile |
| Feature 1 | Oracle service is a FastAPI app |
| Feature 2 | Movie Assistant is a FastAPI app |
| Gateway | Routing layer is also FastAPI |

### Docker Volumes

| Volume | What It Stores | Used By |
|:---|:---|:---|
| `ollama_data` | LLaMA 3 8B GGUF model weights | Ollama container |
| `./oracle-rag-service/data/chromadb` | Subtitle embeddings index | Oracle container |
| `./movie-assistant-service/data/chroma_db` | Movie feature embeddings | Movie Assistant container |
| `./oracle-rag-service/data/subtitles` | Raw .srt subtitle files | Oracle container |

---

## The "Big Picture" Mental Model

If you can explain this paragraph, you understand the whole project:

> StreamSage has two AI features that both use RAG — Retrieval-Augmented Generation — but with different data and different LLMs. The Oracle answers questions about movies you're watching by retrieving relevant subtitle chunks from ChromaDB (built from .srt files) and sending them to a base LLaMA 3 8B model running locally via Ollama, streaming answers token-by-token through SSE. The Movie Discovery Assistant recommends movies by retrieving similar movie feature records from a second ChromaDB collection (built from TMDB metadata) and sending them to a fine-tuned LLaMA 3 8B model — fine-tuned with QLoRA on 3,000 curated recommendation examples to give it a consistent expert personality. Both features are containerized in Docker, share the API Gateway as a single entry point, and communicate through Docker's internal DNS network.

---

## Cross-Module "Why" Questions

These questions test whether you understand the connections:

**Q: Why does Oracle use Ollama but Movie Assistant uses HuggingFace API?**
> Oracle uses the base LLaMA 3 8B (no fine-tuning needed — context is everything). Ollama natively serves base GGUF models locally. The Movie Assistant uses a fine-tuned model stored in HuggingFace format (LoRA adapters + SafeTensors), which isn't in GGUF — so it uses the HuggingFace Inference API in Docker mode or the local transformers library in development mode.

**Q: Why do both features use the same embedding model (BGE)?**
> Consistency. Both ChromaDB collections were built using `BAAI/bge-large-en-v1.5` (384-dim). At query time, you must embed the query with the same model to get vectors in the same space — if you used a different model, the cosine similarity would be meaningless.

**Q: Why does fine-tuning help Movie Recommendations but not Oracle Q&A?**
> Oracle answers are fully constrained by the retrieved subtitle excerpts — the model just needs to summarize text you hand it. A base model does this fine. Movie recommendations require the model to: (1) write enthusiastically, (2) use a specific format, (3) explicitly connect user preference to recommendation reasoning. This consistent behavior pattern needs to be trained — it can't be prompted reliably.

**Q: Why is ChromaDB data in a Docker volume instead of being rebuilt on container start?**
> Building the ChromaDB index requires downloading all movies and subtitle files, then embedding them with sentence-transformers — this takes minutes to hours depending on dataset size. Running this every time a container starts would be impractical. The volume persists the pre-built index so startup is instantaneous.

**Q: Module 05 (LLaMA 3 Architecture) — where does this knowledge actually matter in the project?**
> Two places: (1) Module 04 fine-tuning — choosing LoRA target modules (`q_proj`, `k_proj`, etc.) requires knowing which weight matrices exist in the architecture and which are most important for style adaptation. (2) Module 06 Ollama — understanding why `num_gpu=99` loads all 32 layers to GPU, why KV Cache matters for streaming (reusing computed attention for each new token), and why quantization (Q4_0) reduces memory without destroying quality.
