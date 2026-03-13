# StreamSage Documentation

Welcome to the StreamSage documentation. This folder contains comprehensive guides for understanding, running, and discussing this project.

---

## Quick Navigation

| Document | Description | Best For |
|----------|-------------|----------|
| [Main README](../README.md) | Project overview and setup instructions | Getting started |
| [QUICKSTART](../QUICKSTART.md) | Step-by-step local development guide | Running the project |
| [LEARNING_ROADMAP](LEARNING_ROADMAP.md) | 3-week structured learning path | **Learning everything** |
| [ARCHITECTURE](ARCHITECTURE.md) | System design and data flow diagrams | Understanding how it works |
| [TECH_STACK](TECH_STACK.md) | Complete list of technologies used | Technical reference |
| [INTERVIEW_GUIDE](INTERVIEW_GUIDE.md) | Q&A preparation for technical interviews | Interview prep |
| [Development History](development-history/PROJECT_PHASES.md) | Commit history organized by project phases | Explaining your process |

---

## Documentation Structure

```
docs/
├── README.md                 # This file - documentation index
├── LEARNING_ROADMAP.md       # 3-week structured learning path
├── ARCHITECTURE.md           # System design and architecture details
├── TECH_STACK.md             # Technologies, libraries, and tools
├── INTERVIEW_GUIDE.md        # Interview preparation Q&A
│
└── development-history/
    └── PROJECT_PHASES.md     # Git commits organized by development phases
```

---

## For Different Audiences

### If you're a recruiter or hiring manager:
Start with the [Main README](../README.md) for a project overview, then skim [TECH_STACK](TECH_STACK.md) to see the technologies used.

### If you're a technical interviewer:
Read [INTERVIEW_GUIDE](INTERVIEW_GUIDE.md) for the kinds of questions this project can answer, then dive into [ARCHITECTURE](ARCHITECTURE.md) for system design discussions.

### If you're the project author preparing for interviews:
1. Review [Development History](development-history/PROJECT_PHASES.md) to refresh your memory on what you built and when
2. Study [INTERVIEW_GUIDE](INTERVIEW_GUIDE.md) for talking points
3. Reference [TECH_STACK](TECH_STACK.md) for specific version numbers and library names

### If you're a developer wanting to understand or contribute:
Start with [QUICKSTART](../QUICKSTART.md) to run the project, then read [ARCHITECTURE](ARCHITECTURE.md) to understand the codebase.

---

## Key Highlights

### What Makes This Project Impressive

1. **End-to-End ML Pipeline:** From data collection (TMDB API) to model training (QLoRA) to deployment (FastAPI + Docker)

2. **Multiple AI/ML Paradigms:**
   - RAG (Retrieval-Augmented Generation)
   - LLM Fine-tuning (QLoRA, DPO, GRPO)
   - Transformers (DistilBERT)
   - Sequence Models (LSTM)
   - Hybrid Search (Vector + BM25)

3. **Production Patterns:**
   - Microservices architecture
   - API Gateway
   - Docker Compose deployment
   - Health monitoring
   - Streaming responses (SSE)

4. **Quantifiable Results:**
   - 6,147 movies indexed
   - 70% "Good or Excellent" on 100-query benchmark
   - Sub-second retrieval latency
   - 2,946 training examples generated

---

## Additional Resources

### Service-Specific Documentation

The Movie Assistant service has its own detailed documentation:

```
services/movie-assistant-service/
├── README.md                    # Service overview
├── QUICKREF.md                  # Quick reference for common operations
├── PROGRESS.md                  # Development progress notes
├── DATASET_README.md            # Training data documentation
├── ALTERNATIVE_DATASET.md       # Alternative data sources
│
└── docs/learning/               # Educational deep-dives
    ├── 01_data_acquisition/     # TMDB API integration
    ├── 02_feature_engineering/  # Feature creation process
    ├── 03_rag_architecture/     # RAG pipeline details
    └── 04_fine_tuning/          # QLoRA/DPO/GRPO training
```

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/train_rnn.ipynb` | LSTM binge predictor training |
| `notebooks/train_bert.ipynb` | DistilBERT sentiment fine-tuning |
| `services/movie-assistant-service/fine_tuning/llama_finetune_colab.ipynb` | Llama 3 QLoRA training |
| `services/movie-assistant-service/fine_tuning/dpo_alignment_colab.ipynb` | DPO preference alignment |
| `services/movie-assistant-service/fine_tuning/grpo_alignment_colab.ipynb` | GRPO alignment |

---

## Changelog

| Date | Update |
|------|--------|
| 2026-03-12 | Initial documentation structure created |
| 2026-02-27 | README and QUICKSTART updated with full project details |
| 2026-01-21 | Project initialized |
