# Movie Discovery Assistant Service

The most advanced service in StreamSage. Handles conversational movie discovery using RAG (Retrieval-Augmented Generation) with a fine-tuned Llama 3 8B model.

## Features

- **6,147 movies** indexed with semantic embeddings
- **Hybrid retrieval**: Vector search (BGE) + BM25 full-text search
- **Cross-encoder re-ranking** for precision
- **Fine-tuned Llama 3 8B** using QLoRA, DPO, and GRPO
- **Sub-second responses** in retrieval-only mode

## Port

**http://localhost:8004**

---

## Quick Start

### 1. Install Dependencies

```bash
cd services/movie-assistant-service
pip install -r requirements.txt
```

### 2. Run the Service

```bash
python main.py
```

Wait for this output:
```
============================================================
Starting Movie Discovery Assistant
============================================================
Embedding model loaded: BAAI/bge-large-en-v1.5
Cross-encoder loaded!
ChromaDB loaded with 6147 movies!
BM25 index built with 6147 documents
Movie Discovery Assistant is READY!
```

### 3. Test the API

Open **http://localhost:8004/docs** for Swagger UI, or use curl:

```bash
curl -X POST "http://localhost:8004/discover" \
  -H "Content-Type: application/json" \
  -d '{"query": "movies like Inception"}'
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/discover` | POST | Main movie discovery endpoint |
| `/health` | GET | Service health status |
| `/docs` | GET | Swagger UI |

### POST /discover

**Request:**
```json
{
  "query": "something like John Wick but with a female lead",
  "n_results": 5
}
```

**Response:**
```json
{
  "query": "something like John Wick but with a female lead",
  "answer": "Here are action films with strong female protagonists...",
  "recommended_movies": [
    {
      "title": "Atomic Blonde",
      "year": 2017,
      "rating": 6.7,
      "genres": "Action, Thriller",
      "director": "David Leitch",
      "relevance_score": 0.89
    }
  ],
  "model_used": "retrieval_only",
  "retrieval_count": 5
}
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_API_TOKEN` | Optional | HuggingFace token for LLM inference |
| `INFERENCE_BACKEND` | Optional | Set to `local` for local model inference |
| `LOCAL_MODEL_PATH` | Optional | Path to fine-tuned model weights |

### Modes

**Retrieval-Only (Default):**
- No environment variables needed
- Returns movie recommendations without narrative explanation
- Sub-second response times

**LLM-Enabled:**
```bash
export HF_API_TOKEN=hf_your_token_here
python main.py
```
- Generates natural language explanations
- Response time: 5-15 seconds

---

## Architecture

```
Query → Query Processing → Parallel Retrieval → Fusion → Re-ranking → Response
                              ↓         ↓
                          ChromaDB    BM25
                          (Vector)  (Keyword)
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Embedding | BAAI/bge-large-en-v1.5 | 1024-dim semantic vectors |
| Vector Store | ChromaDB | Similarity search |
| Text Search | BM25 (rank-bm25) | Keyword matching |
| Re-ranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder scoring |
| LLM | Llama 3 8B (fine-tuned) | Natural language generation |

---

## Training

The model was fine-tuned using:

1. **QLoRA** (Quantized Low-Rank Adaptation) - `fine_tuning/llama_finetune_colab.ipynb`
2. **DPO** (Direct Preference Optimization) - `fine_tuning/dpo_alignment_colab.ipynb`
3. **GRPO** (Group Relative Policy Optimization) - `fine_tuning/grpo_alignment_colab.ipynb`

Training data: 2,946 Q&A pairs generated from the movie database.

---

## Benchmark Results (R21)

| Category | Excellent | Good | Weak | Terrible |
|----------|-----------|------|------|----------|
| Director | 10 | 0 | 0 | 0 |
| Actor | 4 | 3 | 1 | 0 |
| Genre | 3 | 6 | 6 | 0 |
| Mood | 3 | 6 | 6 | 0 |
| Similar-To | 3 | 4 | 4 | 1 |
| **Total (100)** | **38** | **32** | **27** | **3** |

---

## Troubleshooting

### "ChromaDB not found"
Ensure `data/chroma_db/` exists with the movie embeddings. Run the data collection scripts if needed.

### Slow first query
Normal - the embedding model loads on first query. Subsequent queries are fast.

### Out of memory
The BGE-large model requires ~2GB RAM. Reduce batch size or use CPU fallback.

---

## Directory Structure

```
movie-assistant-service/
├── main.py                    # RAG pipeline (~4,000 lines)
├── requirements.txt
├── data/
│   ├── chroma_db/             # 6,147 movie embeddings
│   └── processed/
│       └── movie_features.json
├── data_collection/
│   ├── create_vector_db.py
│   └── expand_db.py
├── dataset_generation/
│   ├── dataset_builder.py
│   └── response_generator.py
├── fine_tuning/
│   ├── llama_finetune_colab.ipynb
│   ├── dpo_alignment_colab.ipynb
│   └── grpo_alignment_colab.ipynb
└── docs/learning/             # Educational deep-dives
```

---

## Related Documentation

- [Main README](../../README.md) - Project overview
- [Architecture](../../docs/ARCHITECTURE.md) - System design
- [Interview Guide](../../docs/INTERVIEW_GUIDE.md) - Technical Q&A
