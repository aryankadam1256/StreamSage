# ChromaDB in StreamSage - Complete Flow Diagram

Comprehensive visualization of how ChromaDB powers semantic search across the entire StreamSage project.

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMSAGE ARCHITECTURE                       │
│                                                                      │
│  ┌──────────────┐                                                   │
│  │   Frontend   │                                                   │
│  │  (React 18)  │                                                   │
│  └──

────┬───────┘                                                   │
│         │                                                             │
│         ▼                                                             │
│  ┌──────────────┐                                                   │
│  │  API Gateway │                                                   │
│  │  (FastAPI)   │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                             │
│    ┌────┴────┬──────────┬──────────────┐                           │
│    ▼         ▼          ▼              ▼                            │
│  ┌────┐  ┌────┐    ┌────┐      ┌───────────┐                       │
│  │ MA │  │ OR │    │ BP │      │    SE     │                       │
│  └─┬──┘  └─┬──┘    └────┘      └───────────┘                       │
│    │       │                                                         │
│    ▼       ▼                                                         │
│  ┌──────────────────────────┐                                       │
│  │       ChromaDB           │                                       │
│  │  (Vector Database)       │                                       │
│  │                          │                                       │
│  │  • movies (6,147 docs)   │                                       │
│  │  • subtitles (variable)  │                                       │
│  └──────────────────────────┘                                       │
│                                                                      │
│  MA = Movie Assistant    OR = Oracle RAG                            │
│  BP = Binge Predictor    SE = Sentiment Engine                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Flow 1: Movie Discovery (Movie Assistant Service)

### User Query to Response Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MOVIE DISCOVERY PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────┘

USER ACTION:
  Types: "mind-bending sci-fi movies"
     │
     ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: FRONTEND (React)                                         │
│ File: frontend/src/components/MovieDiscover.jsx                  │
├──────────────────────────────────────────────────────────────────┤
│ axios.post('/api/v1/discover', {query: "mind-bending sci-fi"})  │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: GATEWAY (FastAPI)                                        │
│ File: gateway/main.py                                            │
├──────────────────────────────────────────────────────────────────┤
│ Route: POST /api/v1/discover                                     │
│ Action: Proxy → http://localhost:8004/discover                   │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3: MOVIE ASSISTANT SERVICE                                  │
│ File: services/movie-assistant-service/main.py                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 3.1 QUERY PROCESSING                                     │    │
│ │ Query: "mind-bending sci-fi movies"                      │    │
│ │ ↓                                                        │    │
│ │ Expansion: Add synonyms "thought-provoking", "complex"  │    │
│ │ Mood detection: "mind_bending" mood tag                 │    │
│ └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 3.2 EMBEDDING GENERATION                                │    │
│ │                                                          │    │
│ │ model = SentenceTransformer('BAAI/bge-large-en-v1.5')  │    │
│ │ query_embedding = model.encode(query)                   │    │
│ │                                                          │    │
│ │ Output: [0.23, -0.45, 0.78, ..., 0.12]                 │    │
│ │         └─────────  1024 dimensions  ─────────┘        │    │
│ └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 3.3 CHROMADB VECTOR SEARCH                              │    │
│ │                                                          │    │
│ │ collection.query(                                        │    │
│ │     query_embeddings=[query_embedding],                 │    │
│ │     n_results=100,                                      │    │
│ │     include=["documents", "metadatas", "distances"]     │    │
│ │ )                                                        │    │
│ │                                                          │    │
│ │ ChromaDB Process:                                       │    │
│ │ ┌────────────────────────────────────────────────┐     │    │
│ │ │ a) Load query embedding: [0.23, -0.45, ...]   │     │    │
│ │ │                                                 │     │    │
│ │ │ b) HNSW graph navigation:                      │     │    │
│ │ │    Layer 3 → Layer 2 → Layer 1 → Layer 0      │     │    │
│ │ │    (Navigate shortcuts to find nearest)        │     │    │
│ │ │                                                 │     │    │
│ │ │ c) Compute cosine similarity:                  │     │    │
│ │ │    For each candidate movie embedding:         │     │    │
│ │ │    similarity = dot(query, movie) /            │     │    │
│ │ │                (||query|| * ||movie||)         │     │    │
│ │ │                                                 │     │    │
│ │ │ d) Sort by similarity (highest first)          │     │    │
│ │ │                                                 │     │    │
│ │ │ e) Return top 100 results                      │     │    │
│ │ └────────────────────────────────────────────────┘     │    │
│ │                                                          │    │
│ │ Results:                                                │    │
│ │ {                                                        │    │
│ │   'ids': [['550', '27', '155', ...]],                  │    │
│ │   'distances': [[0.12, 0.18, 0.23, ...]],              │    │
│ │   'metadatas': [[{...}, {...}, {...}]],                │    │
│ │   'documents': [["Title: Inception...", ...]]           │    │
│ │ }                                                        │    │
│ └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 3.4 BM25 KEYWORD SEARCH (Parallel)                      │    │
│ │                                                          │    │
│ │ bm25_index.get_scores(query)                            │    │
│ │ Matches: "sci-fi" keyword in genres                     │    │
│ │ Returns: Top 100 by keyword relevance                   │    │
│ └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 3.5 RECIPROCAL RANK FUSION                              │    │
│ │                                                          │    │
│ │ Merge vector search + BM25 results                      │    │
│ │ RRF score = 1/(k + rank_vector) + 1/(k + rank_bm25)    │    │
│ │ Output: Top 50 candidates                               │    │
│ └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 3.6 CROSS-ENCODER RE-RANKING                            │    │
│ │                                                          │    │
│ │ For each candidate:                                     │    │
│ │   score = cross_encoder.predict([query, movie_desc])   │    │
│ │                                                          │    │
│ │ Final score = 0.3*cross_score + 0.7*vector_score       │    │
│ │ Sort by final score                                     │    │
│ └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 3.7 SORT BY RELEVANCE                                   │    │
│ │                                                          │    │
│ │ movies.sort(key=lambda m: m.relevance_score, reverse=True) │
│ │                                                          │    │
│ │ Result: Top 5 most relevant movies                      │    │
│ │ 1. Inception (87% match)                                │    │
│ │ 2. Interstellar (82% match)                             │    │
│ │ 3. The Matrix (79% match)                               │    │
│ │ 4. Blade Runner 2049 (75% match)                        │    │
│ │ 5. Arrival (72% match)                                  │    │
│ └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 3.8 GENERATE RECOMMENDATION REASONS                      │    │
│ │                                                          │    │
│ │ For each movie:                                         │    │
│ │   _generate_recommendation_reason(query, movie, score)  │    │
│ │                                                          │    │
│ │ Example:                                                │    │
│ │ "Matches your mind-bending preference.                  │    │
│ │  Highly matches your search (87% match)"                │    │
│ └─────────────────────────────────────────────────────────┘    │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4: RESPONSE                                                  │
├──────────────────────────────────────────────────────────────────┤
│ {                                                                 │
│   "query": "mind-bending sci-fi movies",                         │
│   "recommended_movies": [                                         │
│     {                                                             │
│       "title": "Inception",                                       │
│       "year": 2010,                                               │
│       "rating": 8.8,                                              │
│       "relevance_score": 0.87,                                    │
│       "recommendation_reason": "Matches your mind-bending..."     │
│     },                                                            │
│     ...                                                           │
│   ],                                                              │
│   "model_used": "fine_tuning/outputs/final_model (local)"        │
│ }                                                                 │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 5: FRONTEND RENDERS                                         │
│ File: frontend/src/components/MovieDiscover.jsx                  │
├──────────────────────────────────────────────────────────────────┤
│ Movie cards with:                                                 │
│ • Title, year, rating                                             │
│ • Relevance match percentage                                      │
│ • Genres, director, runtime                                       │
│ • "AI's Take" panel with recommendation reason                    │
└──────────────────────────────────────────────────────────────────┘
```

**Performance:** ~500ms total (embedding: 10ms, ChromaDB: 30ms, reranking: 450ms)

---

## Flow 2: Oracle RAG (Subtitle Query)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ORACLE RAG - SUBTITLE Q&A PIPELINE                    │
└─────────────────────────────────────────────────────────────────────────┘

USER ACTION:
  Asks: "What did Neo say about the Matrix?"
  Context: Watching "The Matrix", at timestamp 45:30
     │
     ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: ORACLE SERVICE                                            │
│ File: services/oracle-rag-service/main.py                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 1.1 LOAD QUESTION                                        │    │
│ │                                                          │    │
│ │ QueryRequest:                                            │    │
│ │   query: "What did Neo say about the Matrix?"           │    │
│ │   movie_id: "the_matrix"                                │    │
│ │   timestamp: 2730 (45:30 in seconds)                    │    │
│ │   top_k: 5                                              │    │
│ └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 1.2 EMBED QUESTION                                       │    │
│ │                                                          │    │
│ │ model = SentenceTransformer('all-MiniLM-L6-v2')        │    │
│ │ question_embedding = model.encode(query)                │    │
│ │                                                          │    │
│ │ Output: [0.12, -0.34, 0.56, ..., 0.89]                 │    │
│ │         └──────  384 dimensions  ──────┘               │    │
│ └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 1.3 CHROMADB SUBTITLE SEARCH                            │    │
│ │                                                          │    │
│ │ collection_name = "the_matrix_subtitles"                │    │
│ │                                                          │    │
│ │ collection.query(                                        │    │
│ │     query_embeddings=[question_embedding],              │    │
│ │     n_results=5,                                        │    │
│ │     where={                                             │    │
│ │         "timestamp_start": {"$lte": 2730}  # Spoiler!   │    │
│ │     }                                                    │    │
│ │ )                                                        │    │
│ │                                                          │    │
│ │ ChromaDB Process:                                       │    │
│ │ ┌────────────────────────────────────────────────┐     │    │
│ │ │ a) Filter by timestamp (spoiler protection)    │     │    │
│ │ │    SQL: WHERE timestamp_start <= 2730          │     │    │
│ │ │    Result: Only chunks before 45:30 mark       │     │    │
│ │ │                                                 │     │    │
│ │ │ b) Search filtered set with HNSW               │     │    │
│ │ │    Compare question embedding to subtitle      │     │    │
│ │ │    chunk embeddings                            │     │    │
│ │ │                                                 │     │    │
│ │ │ c) Return top 5 most relevant chunks           │     │    │
│ │ └────────────────────────────────────────────────┘     │    │
│ │                                                          │    │
│ │ Results (subtitle chunks):                              │    │
│ │ [                                                        │    │
│ │   {                                                      │    │
│ │     "content": "Neo: The Matrix is everywhere...",      │    │
│ │     "timestamp_start": 2710.5,                          │    │
│ │     "timestamp_end": 2725.3,                            │    │
│ │     "relevance": 0.89                                   │    │
│ │   },                                                     │    │
│ │   ...                                                    │    │
│ │ ]                                                        │    │
│ └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 1.4 CONSTRUCT PROMPT                                     │    │
│ │                                                          │    │
│ │ prompt = f"""                                           │    │
│ │ You are an assistant that answers questions about       │    │
│ │ movies using dialogue from subtitle transcripts.        │    │
│ │                                                          │    │
│ │ Context (from The Matrix subtitles):                    │    │
│ │ {chunk_1_content}                                       │    │
│ │ {chunk_2_content}                                       │    │
│ │ {chunk_3_content}                                       │    │
│ │                                                          │    │
│ │ Question: {query}                                       │    │
│ │                                                          │    │
│ │ Answer based on the context above:                      │    │
│ │ """                                                      │    │
│ └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ 1.5 OLLAMA LLM GENERATION                               │    │
│ │                                                          │    │
│ │ ollama_client.generate(                                 │    │
│ │     prompt=prompt,                                      │    │
│ │     model="llama3:8b"                                   │    │
│ │ )                                                        │    │
│ │                                                          │    │
│ │ Output: "Based on the dialogue at 45:10, Neo           │    │
│ │          explains that the Matrix is everywhere..."     │    │
│ └─────────────────────────────────────────────────────────┘    │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: RESPONSE TO FRONTEND                                     │
├──────────────────────────────────────────────────────────────────┤
│ {                                                                 │
│   "answer": "Based on the dialogue...",                          │
│   "sources": [                                                    │
│     {                                                             │
│       "content": "Neo: The Matrix is everywhere...",             │
│       "timestamp_start": 2710.5,                                 │
│       "timestamp_end": 2725.3                                    │
│     }                                                             │
│   ],                                                              │
│   "model_used": "llama3:8b"                                      │
│ }                                                                 │
└──────────────────────────────────────────────────────────────────┘
```

**Performance:** ~2s (embedding: 5ms, ChromaDB: 15ms, LLM: 1.98s)

---

## ChromaDB Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION PIPELINE                      │
└────────────────────────────────────────────────────────────────┘

MOVIE DATA:
  movie_features.json (6,147 movies)
     │
     ▼
  create_vector_db.py
     │
     ├─→ create_document_text()
     │   "Title: Inception\nGenres: Sci-Fi\n..."
     │
     ├─→ HuggingFace API
     │   Generate 1024-dim embedding
     │
     └─→ ChromaDB
         collection.upsert(
             ids=["550"],
             embeddings=[[0.23, -0.45, ...]],
             metadatas=[{"title": "Inception", ...}],
             documents=["Title: Inception..."]
         )
         │
         ▼
    ┌──────────────────────────────────────────────┐
    │ CHROMADB STORAGE                             │
    ├──────────────────────────────────────────────┤
    │                                              │
    │ SQLite (chroma.sqlite3):                    │
    │   collections table                          │
    │   embeddings table                           │
    │   embedding_metadata table (title, year, etc)│
    │   embedding_documents table                  │
    │                                              │
    │ Binary Files:                                │
    │   data_level0.bin (HNSW graph)              │
    │   header.bin                                 │
    │   link_lists.bin                             │
    └──────────────────────────────────────────────┘

SUBTITLE DATA:
  movie.srt (subtitle file)
     │
     ▼
  Oracle ingestion
     │
     ├─→ Parse .srt → chunks
     │   "Neo: The Matrix is everywhere..."
     │
     ├─→ Sentence Transformer
     │   Generate 384-dim embedding
     │
     └─→ ChromaDB
         collection.add(
             ids=["chunk_1"],
             embeddings=[[0.12, -0.34, ...]],
             metadatas=[{"timestamp_start": 2710.5, ...}],
             documents=["Neo: The Matrix..."]
         )
```

---

## Performance Comparison

| Operation | Without ChromaDB | With ChromaDB | Speedup |
|-----------|------------------|---------------|---------|
| Search 6,147 movies | Compare all: ~500ms | HNSW graph: ~30ms | 16x faster |
| Filter by year | Linear scan: ~100ms | SQL index: ~5ms | 20x faster |
| Update movie | Rebuild entire DB | Upsert by ID: ~2ms | 1000x faster |
| Scale to 100K | ~8 seconds | ~150ms | 53x faster |

---

## Key Takeaways

1. **ChromaDB is the search engine** for semantic similarity
2. **Two collections** in StreamSage:
   - `movies` (6,147 docs, 1024-dim)
   - `{movie_id}_subtitles` (variable docs, 384-dim)
3. **Hybrid storage**: SQLite for metadata, binary for vectors
4. **HNSW graph** enables sub-100ms search on thousands of documents
5. **Metadata filters** combine SQL power with vector search

---

[← Back to Day 3](README.md) | [View Interview Q&A →](Interview-QA.md)
