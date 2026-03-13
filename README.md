# StreamSage

**A Microservices-based Movie Intelligence Platform**

StreamSage is an educational project that teaches five AI/ML concepts through building a real-world streaming recommendation system. Each microservice isolates one technique so you can study, modify, and experiment with it independently.

---

## AI Concepts Covered

| Service | AI Concept | What It Does |
|---------|-----------|--------------|
| Movie Assistant | RAG + Fine-tuned Llama 3 | Conversational movie discovery across 6,147 films |
| Oracle | RAG (Retrieval-Augmented Generation) | Answer questions about movie dialogues using subtitles |
| Binge Predictor | LSTM (Sequence Modeling) | Predict if a viewer will keep watching or drop off |
| Sentiment Engine | BERT (Transformers) | Classify movie review sentiment with context awareness |
| API Gateway | Microservices Pattern | Single entry point routing to all backend services |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    FRONTEND  (React + Vite)                   │
│   LandingPage │ MovieDiscover │ OracleChat │ VibeBar │ Binge  │
└──────────────────────────┬───────────────────────────────────┘
                           │  http://localhost:3000
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  API GATEWAY  (FastAPI)                       │
│              Routing │ Error handling │ Health                 │
└────┬──────────┬──────────┬──────────┬────────────────────────┘
     │          │          │          │
     ▼          ▼          ▼          ▼
┌─────────┐ ┌────────┐ ┌────────┐ ┌─────────────────────────┐
│  Oracle │ │ Binge  │ │Sentiment│ │   Movie  Assistant      │
│ RAG+LLM │ │  LSTM  │ │  BERT  │ │  Fine-tuned Llama 3     │
└────┬────┘ └───┬────┘ └───┬────┘ └────────────┬────────────┘
     │          │          │                    │
     ▼          ▼          ▼                    ▼
 ChromaDB   Keras .h5  PyTorch            ChromaDB
 (subtitles) (LSTM)   (DistilBERT)      (6,147 movies)
```

---

## Services

### Movie Assistant Service — Port 8004

The most advanced service. Handles conversational movie discovery using a combination of:

- **ChromaDB** with 6,147 movies, each stored as a 1024-dimensional vector (BAAI/bge-large-en-v1.5)
- **BM25** full-text search index for hybrid retrieval
- **Cross-encoder re-ranking** (ms-marco-MiniLM-L-6-v2) blended 30% CE / 70% vector
- **Custom reranking pipeline** with genre matching, mood augmentation, name-phrase detection, title hint bonuses, and franchise deduplication
- **Fine-tuned Llama 3 8B** trained on 2,946 movie recommendation Q&A pairs using QLoRA (4-bit quantization + LoRA adapters)
- **Further aligned** with DPO and GRPO preference training

**Benchmark (R21):** 38 Excellent / 32 Good / 27 Weak / 3 Terrible out of 100 queries.

Endpoint: `POST /discover`

```json
{ "query": "something like John Wick but with a female lead" }
```

### Oracle RAG Service — Port 8001

Answers questions about specific movie dialogues using subtitle transcripts.

- Parses `.srt` subtitle files into time-stamped chunks
- Embeds chunks with sentence-transformers and stores in ChromaDB
- At query time: retrieves top-K relevant chunks, builds a prompt, calls Ollama (Llama 3:8B)
- Spoiler protection: chunks after a user-specified timestamp are excluded
- Streaming support via Server-Sent Events

Endpoints: `POST /ask`, `POST /ask/stream`, `GET /suggestions/{movie_id}`

### Binge Predictor Service — Port 8002

Predicts whether a user will continue watching or drop off based on their viewing history.

- Feature engineering: genre sequences, rating trends, watch velocity, time-of-day patterns
- LSTM neural network trained in Google Colab (`notebooks/train_rnn.ipynb`)
- Returns continuation probability + risk level + personalized recommendations
- Falls back to a heuristic model when the trained `.h5` file is unavailable

Endpoint: `POST /predict`

### Sentiment Engine — Port 8003

Classifies the sentiment of any movie review text.

- Uses DistilBERT (40% smaller than BERT, 60% faster, 97% of its accuracy)
- Understands negation ("not bad" → positive), sarcasm, and slang ("sick" → positive)
- Batch endpoint for up to 100 texts at once
- Trained in Google Colab (`notebooks/train_bert.ipynb`)

Endpoints: `POST /analyze`, `POST /batch`

### API Gateway — Port 8000

Single entry point for all client requests. Provides:

- URL routing to the correct backend service
- Service health monitoring with response time tracking
- Centralised error handling and logging
- Swagger UI at `http://localhost:8000/docs`

---

## Project Structure

```
StreamSage/
├── docker-compose.yml
├── README.md
├── QUICKSTART.md
├── SETUP_AND_RUN.bat              # One-command startup (Windows)
│
├── docs/                              # Project documentation
│   ├── README.md                      # Documentation index
│   ├── ARCHITECTURE.md                # System design & data flows
│   ├── TECH_STACK.md                  # Technologies reference
│   ├── INTERVIEW_GUIDE.md             # Interview Q&A preparation
│   └── development-history/
│       └── PROJECT_PHASES.md          # Git commits by project phase
│
├── frontend/                          # React 18 + Vite + Tailwind CSS
│   ├── src/
│   │   ├── components/
│   │   │   ├── SearchHero.jsx         # Query input + suggestion chips
│   │   │   ├── MovieGrid.jsx          # Results grid
│   │   │   ├── MovieCard.jsx          # Individual movie card
│   │   │   ├── MovieDiscover.jsx      # Discovery page wrapper
│   │   │   ├── LLMAnswerBanner.jsx    # LLM narrative response
│   │   │   ├── OracleChat.jsx         # Movie Q&A interface
│   │   │   ├── VibeBar.jsx            # Sentiment visualization
│   │   │   ├── BingeGauge.jsx         # Binge probability visualization
│   │   │   └── Header.jsx             # Navigation
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx        # Premium landing page (default view)
│   │   │   ├── HomePage.jsx           # Main app home
│   │   │   └── MovieDetailPage.jsx    # Individual movie details
│   │   └── api.js                     # Axios API client
│   └── package.json
│
├── gateway/
│   └── main.py                        # FastAPI gateway router
│
├── services/
│   ├── movie-assistant-service/
│   │   ├── main.py                    # RAG + reranking pipeline (~4,000 lines)
│   │   ├── data/
│   │   │   ├── chroma_db/             # 6,147 movie embeddings
│   │   │   └── processed/
│   │   │       └── movie_features.json
│   │   ├── data_collection/
│   │   │   ├── create_vector_db.py    # Build ChromaDB from scratch
│   │   │   └── expand_db.py           # Add more movies to DB
│   │   ├── dataset_generation/
│   │   │   ├── dataset_builder.py     # Generate Q&A training pairs
│   │   │   └── response_generator.py
│   │   └── fine_tuning/
│   │       ├── llama_finetune_colab.ipynb   # QLoRA SFT training
│   │       ├── dpo_alignment_colab.ipynb    # DPO preference alignment
│   │       └── grpo_alignment_colab.ipynb   # GRPO alignment
│   │
│   ├── oracle-rag-service/
│   │   ├── main.py
│   │   ├── ingest.py
│   │   └── data_pipeline/
│   │
│   ├── binge-service/
│   │   └── main.py
│   │
│   └── sentiment-service/
│       └── main.py
│
├── notebooks/
│   ├── train_rnn.ipynb                # Train LSTM binge model (Colab)
│   └── train_bert.ipynb               # Train BERT sentiment model (Colab)
│
└── data/
    ├── subtitles/                     # Place .srt files here
    ├── chromadb/                      # Oracle RAG vector DB
    └── models/
        ├── binge_model.h5
        └── sentiment_model/
```

---

## Learning Modules

### Module 1 — Docker and Microservices
Study `docker-compose.yml` and `gateway/main.py`.
- Why containers? Reproducibility across machines.
- Service discovery and inter-service HTTP communication.
- API Gateway pattern: single entry point, centralised concerns.

### Module 2 — RAG Pipeline (Oracle Service)
Study `services/oracle-rag-service/main.py` and `ingest.py`.
- Text chunking strategy for subtitle data.
- Embedding models: why sentence-transformers produce better semantic vectors than TF-IDF.
- ChromaDB: vector storage, cosine similarity search, metadata filters.
- Prompt construction: how retrieved context improves LLM answers.

### Module 3 — Sequence Modeling (Binge Predictor)
Study `services/binge-service/main.py` and `notebooks/train_rnn.ipynb`.
- Why RNNs and LSTMs outperform feedforward nets for sequential data.
- Feature engineering from watch history (velocity, diversity, time-of-day patterns).
- Keras LSTM architecture, training, and `.h5` serialisation.

### Module 4 — Transformers and Fine-tuning (Sentiment + Movie Assistant)
Study `services/sentiment-service/main.py` and `services/movie-assistant-service/fine_tuning/`.
- BERT architecture: bidirectional attention, why context matters ("not bad").
- DistilBERT: knowledge distillation for production efficiency.
- QLoRA: fine-tuning a 8B parameter model with 4-bit quantisation on a single GPU.
- DPO and GRPO: aligning model outputs with human preferences after SFT.

### Module 5 — Production Retrieval (Movie Assistant Reranker)
Study `services/movie-assistant-service/main.py`.
- Hybrid retrieval: combining dense vector search (BGE) with sparse BM25.
- Cross-encoder re-ranking: why a two-stage pipeline beats one-stage.
- Reciprocal Rank Fusion (RRF) for merging ranked lists.
- Real-world challenges: keyword bias, franchise deduplication, genre mismatch penalties.

---

## Running the Project

### Local Development (Without Docker)

Requires Python 3.10+ and Node 20+.

```bash
# Terminal 1 — Movie Assistant
cd services/movie-assistant-service
pip install -r requirements.txt
python main.py
# Runs on http://localhost:8004

# Terminal 2 — Gateway
cd gateway
pip install -r requirements.txt
python main.py
# Runs on http://localhost:8000

# Terminal 3 — Frontend
cd frontend
npm install
npm run dev
# Runs on http://localhost:3000
```

The Oracle, Binge, and Sentiment services are optional for the movie discovery feature. Start them the same way if needed (ports 8001, 8002, 8003).

### Docker (Full Stack)

```bash
# First run
docker-compose up -d
docker exec -it streamsage-ollama ollama pull llama3:8b

# View logs
docker-compose logs -f movie-assistant-service

# Stop
docker-compose down
```

### Service Health Checks

```bash
curl http://localhost:8004/health   # Movie Assistant
curl http://localhost:8000/health   # Gateway (shows all services)
```

---

## Fine-tuning the Movie Assistant

The service runs in retrieval-only mode by default. To enable LLM-generated explanations:

```bash
# Set environment variables before starting the service
export HF_API_TOKEN=your_token_here          # HuggingFace Inference API
# OR train locally and point to the model:
export INFERENCE_BACKEND=local
export LOCAL_MODEL_PATH=fine_tuning/outputs/final_model
python main.py
```

Training notebooks are in `services/movie-assistant-service/fine_tuning/`:
1. `llama_finetune_colab.ipynb` — Supervised fine-tuning with QLoRA on 2,946 Q&A pairs
2. `dpo_alignment_colab.ipynb` — DPO preference alignment
3. `grpo_alignment_colab.ipynb` — GRPO alignment

---

## Adding Movies to the Database

```bash
cd services/movie-assistant-service

# Add movies from TMDB (requires TMDB_API_KEY env var)
python data_collection/expand_db.py

# Rebuild the entire ChromaDB from movie_features.json
python data_collection/rebuild_vector_db.py
```

---

## API Reference

### POST /discover (Movie Assistant)

```json
Request:
{
  "query": "movies like Inception",
  "n_results": 5
}

Response:
{
  "query": "movies like Inception",
  "answer": "Here are films that share Inception's...",
  "recommended_movies": [
    {
      "title": "Interstellar",
      "year": 2014,
      "rating": 8.4,
      "genres": "Adventure, Drama, Science Fiction",
      "director": "Christopher Nolan",
      "relevance_score": 0.87
    }
  ],
  "model_used": "retrieval_only",
  "retrieval_count": 5
}
```

Results are sorted by rating (highest first) within the relevance-filtered top-K.

### POST /ask (Oracle)

```json
Request:
{
  "movie_id": "inception",
  "question": "What did Cobb say about dreams?",
  "timestamp_limit": 3600
}
```

### POST /analyze (Sentiment)

```json
Request:  { "text": "An absolutely stunning film" }
Response: { "label": "positive", "confidence": 0.97 }
```

---

## Benchmark Results

The movie discovery pipeline is evaluated against 100 diverse queries across 10 categories. Current results (R21):

| Category | Excellent | Good | Weak | Terrible |
|----------|-----------|------|------|---------|
| Director (10) | 10 | 0 | 0 | 0 |
| Actor (8) | 4 | 3 | 1 | 0 |
| Franchise (5) | 4 | 1 | 0 | 0 |
| Genre (15) | 3 | 6 | 6 | 0 |
| Mood/Emotion (15) | 3 | 6 | 6 | 0 |
| Similar-To (12) | 3 | 4 | 4 | 1 |
| Decade/Era (8) | 3 | 3 | 2 | 0 |
| Abstract/Concept (12) | 3 | 4 | 4 | 1 |
| Specific Request (10) | 2 | 5 | 2 | 1 |
| Conversational (5) | 3 | 0 | 2 | 0 |
| **Total (100)** | **38** | **32** | **27** | **3** |

The 3 remaining Terrible queries are data gaps — the expected films (Ghibli catalog, Amelie, Parasite, road trip films) are absent from the current 6,147-movie database.

---

## References

- *Building Microservices* — Sam Newman
- *Attention Is All You Need* — Vaswani et al. (2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* — Devlin et al. (2018)
- *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al. (2021)
- *Direct Preference Optimization* — Rafailov et al. (2023)
- *Deep Learning* — Goodfellow, Bengio & Courville

---

## License

MIT License — see LICENSE for details.
