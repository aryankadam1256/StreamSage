# Feature 2: Movie Discovery Assistant — Complete End-to-End Journey

The Movie Discovery Assistant is StreamSage's personalized recommendation engine. You say "I want a dark sci-fi movie like Inception," and the system retrieves the most relevant movies from its database, then uses a **custom fine-tuned LLaMA 3 8B model** to write an enthusiastic, structured recommendation with exact reasoning — in your movie-loving companion's voice.

This document traces every step, from raw TMDB JSON to the recommendation appearing in your browser.

---

## Stage 1: Data Collection (Offline — Done Once)

### 1.1 Collecting Movie Data from TMDB

**What is TMDB?** The Movie Database — a free, community-maintained API with comprehensive metadata on thousands of movies.

**How we collect it** (`data_collection/` scripts):

```python
# Pseudocode of the TMDB collection pipeline
for page in range(1, 300):                          # 300 pages
    movies = tmdb.get("/discover/movie", page=page) # ~20 movies per page
    for movie in movies:
        details = tmdb.get(f"/movie/{movie.id}")    # Full details
        credits = tmdb.get(f"/movie/{movie.id}/credits")
        keywords = tmdb.get(f"/movie/{movie.id}/keywords")
        similar = tmdb.get(f"/movie/{movie.id}/similar")

        save_to_raw_json(movie, details, credits, keywords, similar)
```

**Raw TMDB format** (deeply nested, not ML-ready):
```json
{
  "id": 27205,
  "title": "Inception",
  "genres": [{"id": 28, "name": "Action"}, {"id": 878, "name": "Science Fiction"}],
  "credits": {
    "crew": [{"id": 6, "name": "Christopher Nolan", "job": "Director"}, ...],
    "cast": [{"id": 6193, "name": "Leonardo DiCaprio", "order": 0}, ...]
  },
  "keywords": {"keywords": [{"id": 803, "name": "dream"}, {"id": 4270, "name": "mind bending"}]},
  "vote_average": 8.4,
  "vote_count": 35000,
  "popularity": 98.6,
  "runtime": 148,
  "release_date": "2010-07-16",
  "overview": "A thief who steals corporate secrets through dream-sharing..."
}
```

~6,000 movies collected this way, stored as JSON files.

---

## Stage 2: Feature Engineering (Offline)

This is where raw TMDB data becomes ML-ready. The `dataset_builder.py` transforms nested JSON into a structured flat format.

### 2.1 The Transformations Applied

**Mood Tag Extraction** (keyword-based NLP):
```python
mood_keywords = {
    "dark": ["shadow", "grim", "dark", "fear", "bleak", "dystop"],
    "uplifting": ["hope", "triumph", "inspir", "overcome", "joy"],
    "intense": ["thriller", "suspense", "tense", "pulse"],
    ...
}
# Scan: overview + keywords + genres
# Inception → overview has "steals", "secrets", mind-bending keywords → "dark", "intense"
```

**Runtime Categorization:**
```python
148 minutes → "epic"     # (> 150 mins... very close)
```

Actually 148 mins → `"long"` (120-150 min bucket):
```
< 90 min:  "short"
90-120:    "medium"
120-150:   "long"
> 150:     "epic"
```

**Era Tagging:**
```python
"2010-07-16" → release_year = 2010 → era = "modern"  # (1980-2010)
```

**Popularity Score (0 to 1.0):**
```python
score = (
    0.40 * normalize(vote_average, 0, 10)          # Quality: 8.4/10 → 0.84
  + 0.30 * normalize(log(vote_count), 0, log(100k)) # Reliability: log(35000) → 0.87
  + 0.30 * normalize(popularity, 0, 500)            # Trending: 98.6 → 0.20
)
# Inception: 0.4*0.84 + 0.3*0.87 + 0.3*0.20 = 0.336 + 0.261 + 0.060 = 0.657
```

**Data Flattening:**
```python
# Before: [{"id": 6, "name": "Christopher Nolan", "job": "Director"}, ...]
# After:  "Christopher Nolan"  (just the director string)

# Before: [{"name": "Leonardo DiCaprio", "order": 0}, {"name": "Joseph Gordon-Levitt", ...}]
# After:  ["Leonardo DiCaprio", "Joseph Gordon-Levitt", ...]  (top 10 cast)
```

### 2.2 The Final movie_features.json Structure

After transformation, each movie looks like:
```json
{
  "id": "inception",
  "title": "Inception",
  "year": 2010,
  "genres": ["Action", "Science Fiction", "Adventure"],
  "director": "Christopher Nolan",
  "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"],
  "mood_tags": ["dark", "intense", "mind-bending"],
  "style_tags": ["surreal", "complex", "non-linear"],
  "runtime_category": "long",
  "era_tag": "modern",
  "popularity_score": 0.657,
  "vote_average": 8.4,
  "overview": "A thief who steals corporate secrets...",
  "similar_movies": ["Tenet", "The Matrix", "Memento"],
  "keywords": ["dream", "mind bending", "parallel universe"]
}
```

This is stored in `services/movie-assistant-service/data/processed/movie_features.json`.

### 2.3 Building the Search Text and Embedding

Each movie gets a rich text representation for embedding:

```python
search_text = f"""
Title: {movie['title']} ({movie['year']})
Director: {movie['director']}
Cast: {', '.join(movie['cast'][:5])}
Genres: {', '.join(movie['genres'])}
Mood: {', '.join(movie['mood_tags'])}
Style: {', '.join(movie['style_tags'])}
Era: {movie['era_tag']} | Runtime: {movie['runtime_category']}
Overview: {movie['overview']}
Similar to: {', '.join(movie['similar_movies'][:3])}
Keywords: {', '.join(movie['keywords'][:10])}
"""
```

This text is embedded using `BAAI/bge-large-en-v1.5` → 384-dim vector → stored in ChromaDB:

```python
collection = client.get_or_create_collection("movie_features")
collection.add(
    ids=["inception"],
    embeddings=[[0.031, -0.052, ...]],    # 384 floats
    documents=[search_text],
    metadatas=[{
        "title": "Inception",
        "year": 2010,
        "genres": "Action, Science Fiction",
        "mood_tags": "dark, intense, mind-bending",
        "director": "Christopher Nolan",
        "popularity_score": 0.657,
        "vote_average": 8.4
    }]
)
```

---

## Stage 3: Dataset Generation and Fine-Tuning (Offline)

This is what makes Feature 2 different from Feature 1. Instead of using a base LLM, we train a custom one.

### 3.1 Why Fine-Tune at All?

A base LLaMA 3 8B might respond to "Recommend a dark sci-fi":
> "Here are some dark sci-fi movies: Arrival. Blade Runner. Ex Machina."

Our fine-tuned model responds:
> "If you're craving something that will mess with your mind long after the credits roll, you **have** to watch **Arrival** (2016).
>
> **Director:** Denis Villeneuve
> **Why it fits:** Dense, cerebral, and emotionally devastating — it reframes time and language in ways that feel genuinely alien. Perfect if you loved the introspective complexity of dark sci-fi.
>
> Also consider: *Blade Runner 2049*, *Ex Machina*, *Annihilation*"

The fine-tuned version has:
- Consistent personality ("mess with your mind", "you **have** to")
- Structured formatting (bold titles, "Why it fits" section)
- Explicit reasoning connecting user preference to recommendation

### 3.2 Dataset Generation (response_generator.py)

The training data was **synthetically generated**:

```
For each of 3,000 query examples:

1. Generate a natural-sounding user query:
   "I'm in the mood for something dark and mind-bending, sci-fi preferred"

2. Retrieve relevant movies from the database:
   → ChromaDB returns: Inception, Dark City, Annihilation, ...

3. Pass to a teacher model (GPT or claude):
   "Write an enthusiastic movie recommendation for these movies given this query.
    Format: Bold title, director, why it fits, 3 similar movies.
    Tone: Knowledgeable film enthusiast."

4. Save as training example:
   {"query": "...", "response": "You have to watch **Dark City**..."}
```

3,000 such examples form `train.jsonl` — the training data.

### 3.3 SFT Training (Supervised Fine-Tuning)

**Platform:** Google Colab (free T4 GPU, 16GB VRAM)
**Tool:** Unsloth (2x faster, 70% less memory than standard HuggingFace training)

```python
# fine_tuning/llama_finetune_colab.ipynb
model = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,          # 4-bit NF4 quantization (saves GPU memory)
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                       # Rank: controls adapter size
    target_modules=[            # Which weight matrices to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # SwiGLU FFN
    ],
    lora_alpha=32,
    lora_dropout=0.05,
    use_rslora=True,            # Scaled learning rates per rank
)

# Train on our 3,000 examples
trainer = SFTTrainer(model, dataset=train_data, ...)
trainer.train()
# Learning: "When the model generates X, it should match Y from train.jsonl"
```

**What LoRA does:**
```
Original weight matrix W (frozen):
┌──────────────────────────────────────┐
│  q_proj: 4096×4096 = 16M parameters │  ← NOT changed
└──────────────────────────────────────┘

LoRA addition:
┌──────────────────────────────────────┐
│  A: 4096×32 = 131K parameters       │  ← trained
│  B: 32×4096 = 131K parameters       │  ← trained
│  W_eff = W + alpha * (A @ B)        │
└──────────────────────────────────────┘

Total LoRA parameters: 262K per layer × 7 target modules × 32 layers
= ~58M parameters (vs 8B base = 0.7% of the model)
```

Training only 0.7% of parameters means:
- 70% less GPU memory (fits in free Colab T4)
- Converges quickly (~3-5 epochs for style adaptation)
- Model retains all original knowledge (it "adds" to base LLM, doesn't overwrite)

### 3.4 DPO and GRPO Alignment

After SFT, two additional alignment stages refine the model:

**DPO (Direct Preference Optimization):**
- Generates pairs: (query, good_response, bad_response)
- Trains model to prefer good over bad
- Fixes issues like excessive repetition, wrong format, unhelpful answers

**GRPO (Group Relative Policy Optimization):**
- Tests model on evaluation queries
- Gives reward signal based on format compliance and recommendation quality
- Further fine-tunes to maximize reward

**Final output:** `fine_tuning/outputs/` — LoRA adapter weights (~100MB)

---

## Stage 4: Service Startup

```
Movie Assistant Container Startup:

1. Load ChromaDB
   ─────────────
   client = chromadb.PersistentClient("/app/data/chromadb")
   collection = client.get_collection("movie_features")
   → HNSW index loaded into memory
   → ~6,000 movie vectors available

2. Load Embedding Model
   ──────────────────
   embedder = HuggingFaceEmbeddings("BAAI/bge-large-en-v1.5")
   → Same model used to build the index

3. Load InferenceManager (decides backend)
   ──────────────────────────────────────
   INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "auto")

   "api"   → Use HuggingFace Inference API (Mistral 7B via HTTP)
   "local" → Load fine-tuned model into GPU memory
   "auto"  → Try local first, fallback to API

   In Docker: INFERENCE_BACKEND=api (from docker-compose.yml)
   Locally:   INFERENCE_BACKEND=auto or local

4. Service ready on :8004
```

---

## Stage 5: A Live Request — "I want a dark sci-fi movie like Inception"

### 5.1 Request Flow Through Services

```
Browser
  │  POST /api/v1/discover
  ▼
Gateway
  │  Routes to Movie Assistant
  ▼
Movie Assistant (:8004)
  │  1. Parse query
  │  2. Embed query (384-dim)
  │  3. ChromaDB search → top 5 movies
  │  4. Build RAG prompt
  │  5. Call LLM (fine-tuned/HF API)
  │  6. Return structured recommendation
  ▼
Gateway → Browser
```

### 5.2 Query Embedding

```python
query = "I want a dark sci-fi movie like Inception"

# BGE requires prefix for queries
prefixed = f"Represent this sentence for searching relevant passages: {query}"
query_vector = embedder.embed(prefixed)
# → [384-dim float vector] that captures: dark, sci-fi, mind-bending, complex
```

### 5.3 ChromaDB Retrieval (Hybrid Search)

The Movie Assistant uses a more sophisticated retrieval than Oracle — **hybrid search** combining vector similarity with BM25 keyword matching and cross-encoder re-ranking:

```
Step 1: Vector Search
  ChromaDB → cosine similarity on 384-dim vectors
  Filters: mood_tags contains "dark", genres contains "Science Fiction"
  Returns: Inception, Dark City, Annihilation, Ex Machina, Blade Runner 2049, ...

Step 2: BM25 Keyword Search
  Keyword terms from query: "dark", "sci-fi", "like Inception"
  BM25 score based on term frequency in movie documents
  Returns: slightly different ranked list

Step 3: Reciprocal Rank Fusion (RRF)
  Combines vector ranks and BM25 ranks:
  rrf_score = 1/(k + rank_vector) + 1/(k + rank_bm25)   where k=60
  → Final merged ranking

Step 4: Cross-Encoder Re-Ranking (optional, if model loaded)
  A cross-encoder model takes (query, movie_text) pairs
  and scores them holistically → better relevance than bi-encoder alone

Top 5 after all steps:
  1. Dark City (1998)          - dark, sci-fi, mind-bending, reality
  2. Annihilation (2018)       - dark, sci-fi, surreal, psychological
  3. Blade Runner 2049 (2017)  - dark, sci-fi, atmospheric, complex
  4. Ex Machina (2014)         - dark, sci-fi, philosophical, intense
  5. Tenet (2020)              - sci-fi, complex, non-linear, Christopher Nolan
```

**Why hybrid search matters:** "Dark sci-fi like Inception" is a perfect semantic query for vectors (they capture meaning), but keywords like "Christopher Nolan" or specific movie names would be missed by pure vector search — BM25 catches exact term matches.

### 5.4 Build the RAG Prompt

The retrieved movies are formatted into the LLM prompt:

```
SYSTEM:
You are an expert movie recommendation assistant — enthusiastic, knowledgeable,
and precise. Recommend movies in structured format with bold titles,
director, why-it-fits reasoning, and similar titles.

MOVIE DATABASE RESULTS:
1. Dark City (1998) | Director: Alex Proyas | Genres: Sci-Fi, Mystery
   Mood: dark, mind-bending, noir | Overview: A man struggles with memories
   in a city where reality is controlled by mysterious beings...
   Similar to: Inception, The Matrix

2. Annihilation (2018) | Director: Alex Garland | Genres: Sci-Fi, Horror, Drama
   Mood: dark, surreal, psychological | Overview: A biologist signs up for
   a dangerous secret expedition...
   Similar to: Ex Machina, Arrival

[... 3 more movies ...]

USER QUERY: "I want a dark sci-fi movie like Inception"

RECOMMENDATION:
```

### 5.5 Fine-Tuned LLM Generation

**In Docker mode (HuggingFace API):**
```python
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2",
                         token=HF_API_TOKEN)
response = client.text_generation(prompt, max_new_tokens=500, temperature=0.7)
```

The request goes to HuggingFace's servers → Mistral 7B generates → returns text.

**In local mode (fine-tuned model):**
```python
inference_manager = InferenceManager(...)
response = inference_manager.generate(prompt, max_tokens=500)
# Loads fine-tuned LLaMA 3: base model + LoRA adapters from fine_tuning/outputs/
```

**The fine-tuned model generates:**
```
If you're craving something that will mess with your mind long after
the credits roll, you **have** to watch **Dark City** (1998):

**Director:** Alex Proyas

**Why it fits your taste for dark, mind-bending sci-fi:**
Dark City is criminally underrated — it predates The Matrix but delivers
the same existential gut-punch. A man wakes up with no memories in a
city that transforms itself every night, controlled by beings who harvest
human consciousness. Like Inception, it questions the nature of reality
itself, layered with noir atmosphere and stunning practical effects.

**Also consider:**
- *Annihilation* (2018) — Biological horror meets mysterious reality, eerie and beautiful
- *Blade Runner 2049* (2017) — A slower burn, but rewards patience with stunning world-building
- *Ex Machina* (2014) — Claustrophobic AI thriller that will haunt you

All four share that "the world is not what it seems" DNA that makes Inception unforgettable.
```

Notice:
- **Enthusiastic voice** — "criminally underrated", "gut-punch"
- **Explicit connection** to the user's stated preference — "Like Inception, it questions reality"
- **Structured format** — bold title, director, why-it-fits, similar suggestions
- **Reference to user query** — "dark, mind-bending sci-fi" echoed back

This style was **learned from 3,000 training examples** — the base LLaMA 3 would not produce this consistently without fine-tuning.

### 5.6 Response Returned

The Movie Assistant returns JSON to the Gateway → Frontend renders as a movie card with:
- Movie title + year + director
- The recommendation text
- A "Why it fits" badge
- Poster image (fetched from TMDB)

---

## End-to-End Summary Diagram

```
OFFLINE PHASE (done once):
─────────────────────────────────────────────────────────────────────────

  TMDB API                Feature Engineering            ChromaDB
  ─────────               ───────────────────            ────────
  Raw JSON  ──────────►   mood_tags, era, score ──────►  384-dim
  6,000                   runtime_category               movie vectors
  movies                  popularity_score               per movie
                          director, cast flat

  Training Data           Fine-Tuning Pipeline           Fine-Tuned Model
  ─────────────           ──────────────────────         ────────────────
  3,000 curated           Base LLaMA 3 8B                LoRA adapters
  (query, response)  ──►  + QLoRA (rank=32)   ──────►   (~100MB)
  examples in             SFT → DPO → GRPO               stored in
  train.jsonl                                            fine_tuning/outputs/


LIVE REQUEST PHASE:
─────────────────────────────────────────────────────────────────────────

User: "dark sci-fi like Inception"
              │
              ▼
         Gateway (/api/v1/discover)
              │
              ▼
    Movie Assistant Service
              │
    1. Embed query → 384-dim vector
    2. ChromaDB → top 15 similar movies
    3. BM25 keyword search → re-rank
    4. Cross-encoder → final top 5
    5. Build RAG prompt with 5 movies
              │
              ▼
    Fine-Tuned LLM (local or HF API)
    "If you're craving something that will mess with your mind..."
              │
              ▼
    Structured JSON response
              │
              ▼
    Gateway → Frontend
    Renders as movie cards with poster images
```

---

## How Feature 2 Differs From Feature 1

| Aspect | Oracle (Feature 1) | Movie Assistant (Feature 2) |
|:---|:---|:---|
| **Data type** | Subtitle dialogue (.srt) | Movie metadata (TMDB JSON) |
| **LLM** | Base LLaMA 3 via Ollama | Fine-tuned LLaMA 3 via HF API or local |
| **Why fine-tuned?** | Not needed — context does the work | Needed for style, format, reasoning |
| **Streaming** | SSE token-by-token (real-time) | Full response (then display) |
| **Query context** | "Timestamp in movie being watched" | "User's general taste" |
| **Search filters** | Spoiler timestamp boundary | Genre, mood, era filters |
| **Retrieval** | Basic cosine + timestamp re-rank | Hybrid: vector + BM25 + RRF + cross-encoder |
| **Response goal** | Factual, grounded in subtitle text | Enthusiastic, stylized, comparative |
