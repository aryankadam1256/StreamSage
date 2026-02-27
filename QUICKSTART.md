# StreamSage — Quick Start Guide

## Prerequisites

- Python 3.10+
- Node.js 20+
- An NVIDIA GPU with 4+ GB VRAM (recommended for embedding model speed)
- Git

---

## Local Development Setup

This is the recommended way to run StreamSage for development. You will start three processes: the Movie Assistant service, the API Gateway, and the React frontend.

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/StreamSage.git
cd StreamSage
```

### Step 2 — Start the Movie Assistant service

```bash
cd services/movie-assistant-service
pip install -r requirements.txt
python main.py
```

Wait for this output before continuing:

```
============================================================
Starting Movie Discovery Assistant
============================================================
✅ Embedding model loaded!
✅ Cross-encoder loaded!
✅ ChromaDB loaded with 6147 movies!
✅ BM25 index built with 6147 documents
✅ Movie Discovery Assistant is READY!
```

The service runs on **http://localhost:8004** in retrieval-only mode by default (no LLM, sub-second responses).

### Step 3 — Start the API Gateway

Open a new terminal:

```bash
cd gateway
pip install -r requirements.txt
python main.py
```

The gateway runs on **http://localhost:8000** and routes frontend requests to the movie assistant.

### Step 4 — Start the frontend

Open a third terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on **http://localhost:3000** with hot reload.

### Step 5 — Open the app

Go to **http://localhost:3000** in your browser.

Try these example queries:
- `Quentin Tarantino movies`
- `best horror movies`
- `movies similar to John Wick`
- `something like Inception`
- `I want something like Breaking Bad but as a movie`
- `my girlfriend likes romance and I like action`
- `adrenaline rush action movies`

---

## Verify Everything Works

### Health checks

```bash
# Movie Assistant — shows embedding model, movie count, GPU memory
curl http://localhost:8004/health

# Gateway — shows status of all connected services
curl http://localhost:8000/health
```

Expected response from Movie Assistant:

```json
{
  "status": "degraded",
  "chroma_loaded": true,
  "total_movies": 6147,
  "embedding_model": "BAAI/bge-large-en-v1.5"
}
```

Status shows "degraded" because the LLM backend is not configured — this is normal for retrieval-only mode. Movie discovery still works fully.

### API docs (Swagger UI)

- Movie Assistant: http://localhost:8004/docs
- Gateway: http://localhost:8000/docs

---

## Enable LLM-Generated Explanations (Optional)

By default the service returns movie recommendations without a narrative explanation. To enable the LLM:

**Option A — HuggingFace Inference API (easiest)**

```bash
# Get a free token at https://huggingface.co/settings/tokens
export HF_API_TOKEN=hf_your_token_here
cd services/movie-assistant-service
python main.py
```

**Option B — Local fine-tuned model**

After training with the Colab notebooks, set:

```bash
export INFERENCE_BACKEND=local
export LOCAL_MODEL_PATH=fine_tuning/outputs/final_model
python main.py
```

Requires 16+ GB VRAM. Responses take 10–20 seconds per query.

---

## Docker Setup (Full Stack)

If you prefer to run everything in containers:

```bash
# Start all services
docker-compose up -d

# First time only — pull the LLM
docker exec -it streamsage-ollama ollama pull llama3:8b

# Check status
docker-compose ps

# View logs
docker-compose logs -f movie-assistant-service

# Stop
docker-compose down
```

Service URLs are the same: frontend at 3000, gateway at 8000, services at 8001–8004.

---

## Running Optional Services

The Oracle, Binge, and Sentiment services are independent. Start any of them to enable their features in the frontend.

```bash
# Oracle RAG (movie dialogue Q&A) — Port 8001
cd services/oracle-rag-service
pip install -r requirements.txt
python main.py

# Binge Predictor (watch probability) — Port 8002
cd services/binge-service
pip install -r requirements.txt
python main.py

# Sentiment Engine (review analysis) — Port 8003
cd services/sentiment-service
pip install -r requirements.txt
python main.py
```

### Loading subtitle data into Oracle

```bash
# Place .srt files in data/subtitles/
# Then ingest:
docker exec -it streamsage-oracle python ingest.py \
  --file /app/data/subtitles/inception.srt \
  --movie-id inception
```

---

## Expanding the Movie Database

The default database has 6,147 movies. To add more:

```bash
cd services/movie-assistant-service

# Fetch additional movies from TMDB
# Requires a free TMDB API key: https://www.themoviedb.org/settings/api
export TMDB_API_KEY=your_key_here
python data_collection/expand_db.py

# Rebuild ChromaDB from scratch (after editing movie_features.json)
python data_collection/rebuild_vector_db.py
```

---

## Troubleshooting

### "Request failed with status code 500"

The service may not have finished starting. Check it's showing the READY message:

```bash
curl http://localhost:8004/health
```

If `chroma_loaded` is `false`, wait another 10–15 seconds and retry.

### Port already in use

Find and kill the process on the conflicting port:

```bash
# Windows
netstat -ano | findstr :8004
taskkill /F /PID <PID>

# macOS/Linux
lsof -i :8004
kill -9 <PID>
```

### Embedding model slow on first run

The first query after startup will take 5–10 seconds as the GPU warms up. Subsequent queries are sub-second.

### Frontend shows "Network Error"

Make sure the gateway is running on port 8000. Check `frontend/src/api.js` to confirm the API base URL matches:

```js
// Should point to the gateway
const API_URL = 'http://localhost:8000/api/v1'
```

### GPU out of memory

The embedding model (BAAI/bge-large-en-v1.5) uses ~1.5 GB VRAM. If you get OOM errors, the service will automatically fall back to CPU — slower (~2s per query) but functional.

---

## Training the Models (Google Colab)

| Notebook | What It Trains | Output |
|----------|---------------|--------|
| `services/movie-assistant-service/fine_tuning/llama_finetune_colab.ipynb` | Llama 3 8B on 2,946 Q&A pairs (QLoRA) | `fine_tuning/outputs/final_model/` |
| `services/movie-assistant-service/fine_tuning/dpo_alignment_colab.ipynb` | DPO preference alignment | Updated adapter weights |
| `services/movie-assistant-service/fine_tuning/grpo_alignment_colab.ipynb` | GRPO alignment | Updated adapter weights |
| `notebooks/train_rnn.ipynb` | LSTM binge predictor | `data/models/binge_model.h5` |
| `notebooks/train_bert.ipynb` | DistilBERT sentiment | `data/models/sentiment_model/` |

Upload the notebooks to Google Colab, run all cells, then download the output files and place them in the paths shown above.

---

## What's Next

- Read the educational comments inside each service's `main.py`
- Check `services/movie-assistant-service/fine_tuning/` to understand QLoRA and DPO
- Modify `services/movie-assistant-service/data_collection/` to add niche movies your DB is missing
- Edit React components in `frontend/src/` — hot reload shows changes instantly
