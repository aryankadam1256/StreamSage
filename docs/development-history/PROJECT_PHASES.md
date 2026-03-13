# StreamSage Development Phases

This document chronicles the development of StreamSage through its commit history, organized by project phases. Use this as a reference for interviews to explain the incremental approach to building this AI/ML microservices platform.

---

## Phase 1: Foundation (January 21, 2026)

**Objective:** Establish project structure and development environment

| Commit | Description | Date |
|--------|-------------|------|
| `50d786c` | project file structure and setup | 2026-01-21 |

### What Was Done
- Created the monorepo structure with clear separation of concerns
- Set up directories for `services/`, `frontend/`, `gateway/`, `data/`, and `notebooks/`
- Established the microservices architecture foundation
- Created initial configuration files

### Interview Talking Points
- **Why a monorepo?** Simplifies development for educational projects; all services can be versioned together
- **Service separation:** Each AI service is independent with its own `main.py`, `requirements.txt`, and data folder
- **Scalability consideration:** The structure supports both local development and Docker deployment

---

## Phase 2: Movie Discovery Assistant - Core ML (January 22-27, 2026)

**Objective:** Build the flagship AI feature with fine-tuned LLM

| Commit | Description | Date |
|--------|-------------|------|
| `d89dd4b` | starting the movie discovery assistant feature | 2026-01-22 |
| `e7b74ee` | day 3,4: movie discovery assistant feature dataset created, model trained (llama 3:8B) | 2026-01-27 |

### What Was Done
- Collected movie data from TMDB API (6,147 movies)
- Built ChromaDB vector database with BGE-large embeddings (1024 dimensions)
- Implemented BM25 full-text search for hybrid retrieval
- Created dataset of 2,946 Q&A pairs for fine-tuning
- Fine-tuned Llama 3 8B using QLoRA (4-bit quantization)
- Set up Colab notebooks for training workflow

### Technical Achievements
- **Data Pipeline:** TMDB API → JSON processing → Feature engineering → ChromaDB embeddings
- **Training:** QLoRA (4-bit) fine-tuning on A100 GPU in Google Colab
- **Dataset:** Generated training pairs using prompt engineering + manual curation

### Interview Talking Points
- **Why QLoRA?** Allows fine-tuning 8B parameter model on consumer GPU (4GB VRAM) through 4-bit quantization + Low-Rank Adapters
- **Data collection strategy:** Used TMDB API with rate limiting, enriched with genre/cast/crew metadata
- **Embedding choice:** BAAI/bge-large-en-v1.5 chosen for superior semantic similarity performance on MTEB benchmark

---

## Phase 3: Full Stack Integration (February 27, 2026)

**Objective:** Connect all services, build frontend, deploy infrastructure

| Commit | Description | Date |
|--------|-------------|------|
| `85eed82` | feat(oracle-rag): phase 4 - SSE streaming, multi-turn conversation & suggested questions | 2026-02-27 |
| `0335798` | feat(movie-assistant): inference optimizer, DPO/GRPO alignment & BGE embeddings | 2026-02-27 |
| `16eda12` | feat(gateway): add movie-assistant proxy routes & oracle SSE streaming | 2026-02-27 |
| `b71b61d` | feat(frontend): movie discovery UI - page routing, design system & new components | 2026-02-27 |
| `9fce975` | feat(docker): add movie-assistant-service container & fix oracle-rag volumes | 2026-02-27 |
| `8efd1d9` | chore: add local development startup scripts | 2026-02-27 |
| `fcd2989` | docs: update README, QUICKSTART and add OpenAPI spec | 2026-02-27 |
| `b2cf56a` | fix: minor corrections in binge-service and sentiment-service | 2026-02-27 |

### What Was Done

#### Oracle RAG Service
- Implemented Server-Sent Events (SSE) for streaming responses
- Added multi-turn conversation support with context memory
- Built suggested questions feature based on subtitle content
- Spoiler protection: filters chunks beyond user-specified timestamp

#### Movie Assistant Enhancements
- Added DPO (Direct Preference Optimization) and GRPO alignment training
- Upgraded to BGE-large embeddings for better retrieval
- Implemented inference optimizer for faster response times
- Cross-encoder re-ranking pipeline (ms-marco-MiniLM)

#### API Gateway
- Added proxy routes for all backend services
- Implemented SSE pass-through for Oracle streaming
- Centralized error handling and health monitoring

#### Frontend
- Built React 18 + Vite application with Tailwind CSS
- Created MovieGrid, MovieCard, SearchHero components
- Implemented page routing (Home, Movie Details, Discovery)
- Added LLMAnswerBanner for narrative responses
- VibeBar sentiment visualization component

#### DevOps
- Complete Docker Compose setup with all services
- Volume management for ChromaDB persistence
- Local development startup scripts
- OpenAPI specification (api_spec.json)
- Comprehensive README and QUICKSTART documentation

### Interview Talking Points
- **SSE vs WebSockets:** SSE is simpler for server-to-client streaming; fits LLM token-by-token output pattern
- **DPO alignment:** Replaced RLHF with DPO for simpler, reward-model-free preference learning
- **Gateway pattern:** Single entry point simplifies CORS, authentication, and service discovery
- **Hybrid retrieval:** Vector search (semantic) + BM25 (keyword) catches both "movies like Inception" and "Christopher Nolan films"

---

## Phase 4: Security & DevOps (March 1-3, 2026)

**Objective:** Production hardening and developer experience improvements

| Commit | Description | Date |
|--------|-------------|------|
| `47e9edb` | chore: update .gitignore - exclude ChromaDB sqlite/binary files and temp outputs | 2026-03-01 |
| `edc677d` | security: remove hardcoded secrets & update startup scripts to use env vars | 2026-03-01 |
| `45af8b8` | chore: add SETUP_AND_RUN.bat for one-command app startup | 2026-03-03 |

### What Was Done
- Removed all hardcoded API keys and secrets from codebase
- Updated services to read credentials from environment variables
- Created `.env.example` templates for configuration
- Updated `.gitignore` to exclude:
  - ChromaDB SQLite files and binary data
  - Model checkpoint outputs
  - Temporary debug files
  - Log files
- Added `SETUP_AND_RUN.bat` for Windows one-command startup

### Interview Talking Points
- **Security best practices:** Secrets should never be committed; environment variables are the standard for 12-factor apps
- **Developer experience:** One-command startup reduces onboarding friction
- **Git hygiene:** Proper `.gitignore` prevents accidental commit of large binary files and sensitive data

---

## Phase 5: Landing Page & UX Polish (March 11-12, 2026)

**Objective:** Professional UI/UX for portfolio presentation

| Commit | Description | Date |
|--------|-------------|------|
| `3e6628b` | feat: add Plus Jakarta Sans font for landing page | 2026-03-11 |
| `1f2ee84` | feat: add landing page CSS animations | 2026-03-11 |
| `1279a70` | feat: add premium landing page component | 2026-03-11 |
| `940b904` | feat: integrate landing page as default view | 2026-03-11 |
| `8e56c9d` | fix: rewrite LandingPage with complete default export | 2026-03-12 |
| `610609f` | fix: rebrand landing page to StreamSage | 2026-03-12 |

### What Was Done
- Designed and implemented premium landing page
- Added custom typography (Plus Jakarta Sans)
- Created CSS animations for visual appeal
- Integrated landing page as the default app entry point
- Completed StreamSage branding throughout UI

### Interview Talking Points
- **First impressions matter:** A polished landing page demonstrates full-stack capability beyond just ML
- **Modern CSS:** Used Tailwind CSS utilities with custom animations for maintainable styling
- **Component architecture:** React functional components with clean separation of concerns

---

## Commit Summary by Category

| Category | Count | Key Commits |
|----------|-------|-------------|
| Features | 11 | `e7b74ee`, `85eed82`, `0335798`, `b71b61d`, `1279a70` |
| Infrastructure | 4 | `50d786c`, `9fce975`, `8efd1d9`, `45af8b8` |
| Documentation | 1 | `fcd2989` |
| Security | 1 | `edc677d` |
| Fixes | 3 | `b2cf56a`, `8e56c9d`, `610609f` |

---

## Total Development Timeline

| Phase | Duration | Focus |
|-------|----------|-------|
| Phase 1 | 1 day | Project setup |
| Phase 2 | 6 days | Core ML development |
| Phase 3 | 1 day (intensive) | Full stack integration |
| Phase 4 | 3 days | Security & DevOps |
| Phase 5 | 2 days | UI polish |

**Total:** ~13 days of active development (excluding gaps)

---

## Git Commands Reference

```bash
# View full commit history
git log --oneline --all

# View commits with dates
git log --format="%h %s (%ad)" --date=short

# View changes in a specific commit
git show <commit-hash>

# View diff between phases
git diff 50d786c..e7b74ee  # Phase 1 to Phase 2

# View file history
git log --follow -p -- services/movie-assistant-service/main.py
```
