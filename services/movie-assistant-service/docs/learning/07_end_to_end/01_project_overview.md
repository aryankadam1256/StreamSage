# StreamSage: Complete Project Overview

This is the capstone document that ties every module together. By the time you finish this module, you'll be able to explain what StreamSage is, how it was built, how every part connects, and why each technical decision was made.

---

## What Is StreamSage?

StreamSage is an AI-powered movie platform with **two distinct AI features**:

| Feature | What It Does | Who Builds It |
|:---|:---|:---|
| **The Oracle** | Real-time Q&A companion while you watch a movie. Ask "What just happened?" and it answers using actual subtitle dialogue. Streams token-by-token like ChatGPT. | RAG + Ollama (LLaMA 3 8B) |
| **Movie Discovery Assistant** | Conversational recommendation engine. Tell it your mood, get personalized picks with explanations. Uses a **custom fine-tuned** model. | Fine-tuned LLaMA 3 8B + RAG |

These two features share some infrastructure (Docker, ChromaDB, LLaMA 3) but solve completely different problems using different architectures.

---

## The Complete System Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           StreamSage Platform                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Frontend (React + Nginx)                      │   │
│  │  OracleChat  │  SearchHero  │  MovieGrid  │  MovieCard  │  VibeBar  │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │HTTP / SSE                                     │
│  ┌──────────────────────────▼──────────────────────────────────────────┐   │
│  │                       API Gateway (FastAPI)                          │   │
│  │  /api/v1/oracle/*  │  /api/v1/discover  │  /api/v1/composite/*     │   │
│  └──────┬──────────────────────┬──────────────────────┬───────────────┘   │
│         │                      │                       │                   │
│  ┌──────▼──────┐  ┌────────────▼───┐  ┌──────────────▼──┐               │
│  │Oracle RAG   │  │Movie Assistant  │  │ Binge + Sentiment│               │
│  │Service      │  │Service          │  │ Services         │               │
│  │:8001        │  │:8004            │  │:8002 / :8003     │               │
│  │             │  │                 │  │                  │               │
│  │ ChromaDB    │  │ ChromaDB        │  │ ML Models        │               │
│  │ (subtitles) │  │ (movie features)│  │ (XGBoost, etc.)  │               │
│  └──────┬──────┘  └────────────┬───┘  └──────────────────┘               │
│         │                      │                                           │
│  ┌──────▼──────┐  ┌────────────▼───────────────┐                         │
│  │   Ollama    │  │  HuggingFace Inference API  │                         │
│  │  (LLaMA 3)  │  │  OR Local Fine-Tuned Model  │                         │
│  │  :11434     │  │  (LLaMA 3 8B + LoRA)        │                         │
│  └─────────────┘  └────────────────────────────┘                         │
│                                                                             │
│  [All containers connected via Docker bridge network]                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The 7 Services, Plain English

| Service | Port | Language/Framework | What It Does |
|:---|:---|:---|:---|
| **Frontend** | 3000 | React 18 + Vite + TailwindCSS served by Nginx | The UI. Gets data from Gateway, shows it to the user. |
| **Gateway** | 8000 | FastAPI (Python) | Single entry point. Routes requests to the right service. |
| **Oracle RAG** | 8001 (mapped from 8000) | FastAPI (Python) | Answers movie Q&A using subtitle excerpts + LLaMA 3 via Ollama. |
| **Binge Service** | 8002 (mapped from 8000) | FastAPI (Python) | Predicts "binge score" — how likely you are to keep watching. |
| **Sentiment Service** | 8003 (mapped from 5000) | Flask (Python) | Analyzes mood/tone of a movie from its description. |
| **Movie Assistant** | 8004 | FastAPI (Python) | Personalized recommendations using fine-tuned LLaMA 3 + RAG. |
| **Ollama** | 11434 | Pre-built image | Serves LLaMA 3 8B locally for Oracle's real-time Q&A. |

---

## Two Completely Different AI Pipelines

The most important thing to understand is that the two main features use **fundamentally different approaches** even though both use LLaMA 3 8B.

### Feature 1: Oracle RAG Service

```
Problem: User asks "What happened at minute 45 of Inception?"
Approach: Retrieval-Augmented Generation with base (unmodified) LLM

Pipeline:
  Query → Embed → Search Subtitle DB → Build Prompt → Ollama → Stream Answer

Key insight: The model doesn't need to know the movie.
The subtitle DATABASE knows the movie.
The model just summarizes what we retrieve.
```

### Feature 2: Movie Discovery Assistant

```
Problem: User says "I want a dark sci-fi movie like Inception"
Approach: Fine-tuned LLM + RAG over movie features

Pipeline:
  Query → Embed → Search Movie Feature DB → Build Prompt → Fine-tuned LLM → Answer

Key insight: The model was TRAINED to be a great recommender.
It learned the style, tone, and reasoning from 3,000 curated examples.
The TMDB DATABASE provides the movie facts it talks about.
```

### Why the Difference?

| Aspect | Oracle | Movie Assistant |
|:---|:---|:---|
| **Data source** | Subtitle files (.srt) | TMDB movie metadata |
| **LLM used** | Base LLaMA 3 8B (Ollama) | Fine-tuned LLaMA 3 8B |
| **Why fine-tune?** | Not needed — context is everything | Needed for consistent style & reasoning |
| **Response type** | Factual, grounded in subtitles | Enthusiastic, personalized, structured |
| **Streaming** | Yes (SSE, token-by-token) | Yes (HF API streaming) |
| **ChromaDB** | Subtitle embeddings (384-dim) | Movie feature embeddings (384-dim) |

---

## The Build Journey: From Zero to StreamSage

This is the chronological order in which the project was built. Each phase corresponds to one or more documentation modules.

```
Phase 1: Data Collection (Module 01)
   │
   ├── TMDB API → browser collector → 6,000+ movie records (raw JSON)
   └── Subtitle scraping → SRT files → parsed dialogue + timestamps

Phase 2: Feature Engineering (Module 02)
   │
   ├── Raw TMDB JSON → structured movie_features.json
   │   (mood tags, era tags, runtime category, popularity score)
   └── Raw subtitles → timestamped chunks → stored in ChromaDB

Phase 3: RAG Architecture (Module 03)
   │
   ├── sentence-transformers → 384-dim embeddings for movie features
   ├── sentence-transformers → 384-dim embeddings for subtitle chunks
   └── ChromaDB vector stores → semantic search working

Phase 4: Fine-Tuning (Module 04)
   │
   ├── Dataset generation → 3,000 (query, response) pairs in train.jsonl
   ├── SFT training (Llama 3 8B + QLoRA on Colab)
   ├── DPO alignment (preferred vs rejected responses)
   └── GRPO alignment → final fine-tuned model

Phase 5: LLaMA 3 Architecture (Module 05)
   │
   └── Understanding what's inside the model we fine-tuned:
       32 decoder layers, GQA, RoPE, SwiGLU, RMSNorm

Phase 6: Docker + Ollama (Module 06)
   │
   ├── Dockerfiles → containerize every service
   ├── docker-compose.yml → wire all 7 containers
   └── Ollama → serve LLaMA 3 8B locally for Oracle

Phase 7: Full System Running (This Module)
   └── Everything connected, browser to model
```

---

## Reading Guide: Which Document to Read First

If you want to understand a specific aspect of the project, here's where to look:

| Question | Go To |
|:---|:---|
| "How was the movie data collected?" | Module 01: Data Acquisition |
| "How do raw movies become searchable?" | Module 02: Feature Engineering |
| "How does the AI find relevant movies for a query?" | Module 03: RAG Architecture |
| "How was the fine-tuned model trained?" | Module 04: Fine-Tuning |
| "What's actually inside LLaMA 3?" | Module 05: LLaMA 3 Architecture |
| "How do all 7 services run together?" | Module 06: Docker & Ollama |
| "How does a request flow end-to-end?" | Module 07 (this module) |
| "How do I pass interviews about this project?" | Each module's `interview_prep/` folder |

---

## Key Numbers to Remember

| Metric | Value |
|:---|:---|
| Total movies in database | ~6,000+ |
| Embedding dimensions | 384 (sentence-transformers) |
| LLaMA 3 8B parameters | 8 billion |
| LLaMA 3 8B decoder layers | 32 |
| Fine-tuning training examples | ~3,000 |
| LoRA rank | 32 |
| Ollama model size (Q4_0) | ~4.7 GB |
| ChromaDB collections | 2 (subtitles + movie features) |
| Microservices | 7 |
| Docker containers | 7 |
| Main API port (Gateway) | 8000 |
| Ollama port | 11434 |
