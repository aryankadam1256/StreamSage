# ChromaDB Deep Dive

A comprehensive explanation of ChromaDB's internal workings, data structures, and usage in StreamSage.

---

## Table of Contents

1. [What is ChromaDB?](#1-what-is-chromadb)
2. [Data Storage Format](#2-data-storage-format)
3. [Collections Deep Dive](#3-collections-deep-dive)
4. [CRUD Operations](#4-crud-operations)
5. [Query Mechanics](#5-query-mechanics)
6. [HNSW Indexing Explained](#6-hnsw-indexing-explained)
7. [Real Examples from StreamSage](#7-real-examples-from-streamsage)

---

## 1. What is ChromaDB?

ChromaDB is an **embedding database** - a database optimized for storing and searching high-dimensional vectors (embeddings).

**Traditional Database vs Vector Database:**

| Operation | SQL Database | Vector Database |
|-----------|--------------|-----------------|
| Insert | `INSERT INTO movies VALUES (...)` | `collection.add(embeddings=[...])` |
| Exact match | `WHERE title = 'Inception'` | N/A |
| Similarity | ❌ Not supported | `query(query_embeddings=[...])` ✅ |
| Filter | `WHERE year > 2010` ✅ | `where={"year": {"$gt": 2010}}` ✅ |

**ChromaDB combines both:** Similarity search + metadata filtering!

---

## 2. Data Storage Format

### Physical Files

When you create ChromaDB, it generates:

```
services/movie-assistant-service/data/chroma_db/
├── chroma.sqlite3                    # SQLite database
│
└── abc-1234-collection-uuid/         # Collection folder
    ├── data_level0.bin                # HNSW graph data
    ├── header.bin                     # Collection metadata
    ├── length.bin                     # Document lengths
    └── link_lists.bin                 # HNSW graph links
```

### 2.1 SQLite Database (chroma.sqlite3)

**Schema:**

```sql
-- Collections table
CREATE TABLE collections (
    uuid TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    topic TEXT,
    dimension INTEGER,
    get_or_create BOOLEAN,
    metadata TEXT  -- JSON blob
);

-- Example row:
uuid: "abc-1234-5678"
name: "movies"
dimension: 1024
metadata: '{"hnsw:space": "cosine", "hnsw:construction_ef": 200}'

-- Segments table (stores embeddings)
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    segment_uuid TEXT,
    collection_uuid TEXT REFERENCES collections(uuid),
    embedding BLOB,  -- Binary vector data
    created_at TIMESTAMP
);

-- Example row:
id: "550"
collection_uuid: "abc-1234-5678"
embedding: <4KB binary: Float32Array[1024]>

-- Metadata table (for filtering)
CREATE TABLE embedding_metadata (
    id TEXT,
    segment_uuid TEXT,
    key TEXT,
    string_value TEXT,
    int_value INTEGER,
    float_value REAL,
    bool_value INTEGER,
    PRIMARY KEY (id, key)
);

-- Example rows for movie ID=550:
id    | key      | string_value      | int_value | float_value
------|----------|-------------------|-----------|------------
550   | title    | Fight Club        | NULL      | NULL
550   | year     | NULL              | 1999      | NULL
550   | rating   | NULL              | NULL      | 8.8
550   | genres   | Drama             | NULL      | NULL
550   | director | David Fincher     | NULL      | NULL

-- Documents table (original text)
CREATE TABLE embedding_full_text (
    id TEXT PRIMARY KEY,
    full_text TEXT
);

-- Example row:
id: "550"
full_text: "Title: Fight Club (1999)\nGenres: Drama\nDirector: David Fincher\n..."
```

### 2.2 Binary Files

**data_level0.bin:**
```
[Binary format - Float32 array]

Byte offset 0:     [0.23456789]  ← Movie 550, dimension 0
Byte offset 4:     [-0.45678901] ← Movie 550, dimension 1
Byte offset 8:     [0.78901234]  ← Movie 550, dimension 2
...
Byte offset 4092:  [0.12345678]  ← Movie 550, dimension 1023

Byte offset 4096:  [0.34567890]  ← Movie 13, dimension 0
...

Total size: 6,147 movies × 1024 dims × 4 bytes = ~25 MB
```

**Why binary?**
- Text storage: `"0.23456789"` = 10 bytes
- Binary storage: `0.23456789` = 4 bytes
- **2.5x space savings + faster loading**

---

## 3. Collections Deep Dive

### 3.1 What is a Collection?

A collection is like a **table in SQL** or a **collection in MongoDB**.

```python
# Create collection
collection = client.create_collection(
    name="movies",                          # Unique name
    metadata={"hnsw:space": "cosine"},     # Distance metric
    embedding_function=embedding_fn         # Auto-embed
)
```

### 3.2 Collection Metadata

```python
metadata = {
    "hnsw:space": "cosine",           # Distance: cosine, l2, or ip
    "hnsw:construction_ef": 200,      # HNSW build quality (higher = better, slower)
    "hnsw:search_ef": 100,            # HNSW search quality
    "hnsw:M": 16,                     # Max connections per node
}
```

**What these mean:**

| Parameter | What It Controls | Trade-off |
|-----------|------------------|-----------|
| `space` | Distance calculation | cosine (default) best for text |
| `construction_ef` | Index build quality | Higher = better results + slower build |
| `search_ef` | Search accuracy | Higher = better results + slower search |
| `M` | Graph connectivity | Higher = more memory + faster search |

**StreamSage uses defaults:**
- `cosine` space (best for embeddings)
- Default ef and M (balanced quality/speed)

---

## 4. CRUD Operations

### 4.1 Create (Add)

```python
collection.add(
    ids=["550", "13", "680"],
    embeddings=[
        [0.23, -0.45, ...],  # 1024 floats for movie 550
        [0.12, -0.33, ...],  # 1024 floats for movie 13
        [0.45, -0.67, ...],  # 1024 floats for movie 680
    ],
    metadatas=[
        {"title": "Fight Club", "year": 1999, "rating": 8.8},
        {"title": "Forrest Gump", "year": 1994, "rating": 8.8},
        {"title": "Pulp Fiction", "year": 1994, "rating": 8.9},
    ],
    documents=[
        "Title: Fight Club (1999)\nGenres: Drama\n...",
        "Title: Forrest Gump (1994)\nGenres: Comedy, Drama\n...",
        "Title: Pulp Fiction (1994)\nGenres: Crime, Thriller\n...",
    ]
)
```

**What happens internally:**
1. SQLite INSERT into `embeddings` table
2. SQLite INSERT into `embedding_metadata` (one row per metadata key)
3. SQLite INSERT into `embedding_full_text`
4. Binary write to `data_level0.bin` (append embeddings)
5. HNSW graph update (add nodes and links)

---

### 4.2 Read (Query)

```python
results = collection.query(
    query_embeddings=[[0.23, -0.45, ...]],  # Your query vector
    n_results=5,                             # Top 5 results
    where={"year": {"$gte": 2000}},         # Metadata filter
    include=["metadatas", "documents", "distances"]  # What to return
)
```

**What happens internally:**
1. **Filter Phase (SQL):**
   ```sql
   SELECT DISTINCT id
   FROM embedding_metadata
   WHERE key = 'year' AND int_value >= 2000
   ```
   → Returns candidate IDs: [550, 27, 155, ...]

2. **Search Phase (HNSW):**
   - Load query embedding
   - Navigate HNSW graph
   - Only consider IDs from filter
   - Compute cosine similarity
   - Keep top 5

3. **Retrieve Phase (SQL JOIN):**
   ```sql
   SELECT e.id, e.embedding, em.*, ef.full_text
   FROM embeddings e
   JOIN embedding_metadata em ON e.id = em.id
   JOIN embedding_full_text ef ON e.id = ef.id
   WHERE e.id IN (550, 27, 155, 680, 13)
   ```

4. **Format response:**
   ```python
   {
       'ids': [['550', '27', '155', '680', '13']],
       'distances': [[0.12, 0.18, 0.23, 0.25, 0.28]],
       'metadatas': [[{...}, {...}, {...}, {...}, {...}]],
       'documents': [["Title: Fight Club...", ...]]
   }
   ```

---

### 4.3 Update (Upsert)

```python
collection.upsert(
    ids=["550"],
    metadatas=[{"title": "Fight Club", "rating": 9.0}]  # New rating
)
```

**What happens:**
1. Check if ID "550" exists
2. If yes → UPDATE existing row
3. If no → INSERT new row

**Why upsert:**
- Idempotent (safe to run multiple times)
- No need to check existence first
- Prevents duplicate entries

---

### 4.4 Delete

```python
collection.delete(
    ids=["550"]
)
```

**What happens:**
1. Delete from SQLite tables
2. Remove from binary files (mark as deleted)
3. Remove from HNSW graph
4. Space may not be immediately reclaimed (requires vacuum)

---

## 5. Query Mechanics

### 5.1 Query Options

```python
collection.query(
    query_texts=["text to search"],      # Option 1: Auto-embed
    query_embeddings=[[0.23, ...]],      # Option 2: Pre-computed
    n_results=10,                         # How many results
    where={"year": {"$gte": 2000}},      # Metadata filter
    where_document={"$contains": "Neo"},  # Full-text search
    include=["metadatas", "documents", "distances", "embeddings"]
)
```

### 5.2 Filtering Operators

**Comparison:**
```python
where={
    "year": {"$eq": 2010},    # Equal
    "rating": {"$gt": 8.0},   # Greater than
    "rating": {"$gte": 8.0},  # Greater than or equal
    "rating": {"$lt": 5.0},   # Less than
    "rating": {"$lte": 5.0},  # Less than or equal
    "year": {"$ne": 1999},    # Not equal
}
```

**Set operations:**
```python
where={
    "genre": {"$in": ["Action", "Thriller"]},     # In list
    "genre": {"$nin": ["Horror", "Romance"]},     # Not in list
}
```

**Logical operators:**
```python
where={
    "$and": [
        {"year": {"$gte": 2000}},
        {"rating": {"$gte": 8.0}}
    ]
}

where={
    "$or": [
        {"director": "Christopher Nolan"},
        {"director": "Denis Villeneuve"}
    ]
}
```

### 5.3 Result Format

```python
{
    'ids': [['550', '27', '155']],

    'distances': [[0.12, 0.18, 0.23]],
    # Lower = more similar
    # Cosine distance: 1 - cosine_similarity

    'metadatas': [[
        {"title": "Fight Club", "year": 1999, "rating": 8.8},
        {"title": "Inception", "year": 2010, "rating": 8.8},
        {"title": "The Matrix", "year": 1999, "rating": 8.7}
    ]],

    'documents': [[
        "Title: Fight Club (1999)\n...",
        "Title: Inception (2010)\n...",
        "Title: The Matrix (1999)\n..."
    ]],

    'embeddings': None  # Only if include=['embeddings']
}
```

**Converting distance to similarity:**
```python
for distance in results['distances'][0]:
    similarity = 1 - distance
    match_percentage = similarity * 100
    print(f"Match: {match_percentage:.1f}%")
```

---

## 6. HNSW Indexing Explained

### 6.1 What Problem Does HNSW Solve?

**Brute-force search:**
```python
def find_similar(query_embedding, all_embeddings):
    similarities = []
    for embedding in all_embeddings:  # 6,147 iterations
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append(sim)
    return top_k(similarities, k=5)

# Time: O(n) where n = 6,147
# For 1024-dim vectors: ~500ms
```

**With HNSW:**
```python
# Navigate graph structure (skip most documents)
# Only compare ~12 nodes instead of 6,147
# Time: O(log n)
# For 1024-dim vectors: ~30ms
```

**Speedup: 16x faster!**

---

### 6.2 HNSW Graph Structure

**Hierarchical layers:**

```
Layer 3 (Top, sparse):
    [A] ←→ [F] ←→ [M]
    ↓       ↓       ↓

Layer 2 (Medium density):
    [A] ←→ [C] → [F] ←→ [J] → [M]
    ↓    ↓   ↓    ↓    ↓   ↓    ↓

Layer 1 (Dense):
    [A]-[B]-[C]-[D]-[E]-[F]-[G]-[H]-[I]-[J]-[K]-[L]-[M]
    ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓

Layer 0 (Complete):
    [All 6,147 movie embeddings with cross-links]
```

**Properties:**
- Each node = one movie embedding
- Higher layers = sparser (fewer nodes)
- Links connect similar embeddings
- Parameter M controls max links per node

---

### 6.3 Search Algorithm Walkthrough

**Query:** Find movies similar to `[0.5, 0.3, 0.8, ...]`

**Simplified example with 13 movies:**

```
         Layer 2         Layer 1       Layer 0 (with similarities)

         [A]            [A]           [A] 0.45 ← start here
          ↓              ↓             ↓
         [F]  ←Step 1→  [C]           [B] 0.62
          ↓              ↓             [C] 0.71
         [M]            [F]  ←Step 2→ [D] 0.55
                         ↓             [E] 0.83 ← closest neighbor!
                        [M]            [F] 0.48
                                      [G] 0.77
                                      [H] 0.69
                                      [I] 0.72
                                      [J] 0.66
                                      [K] 0.58
                                      [L] 0.51
                                      [M] 0.42
```

**Algorithm:**
1. **Start at Layer 2, node A**
   - Compare query to A's neighbors: F
   - F is closer → move to F

2. **Drop to Layer 1, node F**
   - Compare to F's neighbors: C, M
   - C is closer → move to C

3. **Drop to Layer 0, node C**
   - Compare to C's neighbors: B, D, E
   - E is closest (0.83)!
   - Check E's neighbors: G, I
   - G and I also very close

4. **Return top 5:**
   - E (0.83), G (0.77), I (0.72), C (0.71), H (0.69)

**Comparisons:** Only ~12 nodes checked instead of all 13!

---

### 6.4 HNSW Parameters

```python
metadata = {
    "hnsw:construction_ef": 200,  # Build quality
    "hnsw:search_ef": 100,         # Search quality
    "hnsw:M": 16,                  # Max connections per node
}
```

**construction_ef (Build Quality):**
```
Low (100):    Fast build, lower recall
Default (200): Balanced
High (500):   Slow build, high recall
```

**search_ef (Search Quality):**
```
Low (50):     Fast search (check fewer candidates)
Default (100): Balanced
High (500):   Slow search (check more candidates, better results)
```

**M (Connectivity):**
```
Low (8):      Less memory, slower search
Default (16): Balanced
High (32):    More memory, faster search
```

**Trade-off visualization:**
```
Quality ↑
        |           ╱ Slower
        |         ╱
        |       ╱
        |     ╱
        |   ╱ Sweet spot (defaults)
        | ╱
        ├────────────────────> Speed
       Bad                    Fast
```

---

## 7. Real Examples from StreamSage

### 7.1 Movie Assistant Collection

```python
# From: services/movie-assistant-service/main.py

# Initialization
vectorstore = Chroma(
    collection_name="movies",
    persist_directory="data/chroma_db",
    embedding_function=HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )
)

# Stats
print(f"Total movies: {vectorstore._collection.count()}")
# Output: 6147

# Query
results = vectorstore.similarity_search(
    query="mind-bending sci-fi",
    k=100
)
```

**Data format in collection:**

```python
# Each document:
{
    "id": "550",
    "embedding": [0.23, -0.45, 0.78, ..., 0.12],  # 1024 floats
    "metadata": {
        "title": "Inception",
        "year": 2010,
        "rating": 8.8,
        "genres": "Action, Sci-Fi, Thriller",
        "director": "Christopher Nolan",
        "popularity_score": 84.6,
        "moods": "thought_provoking, mind_bending"
    },
    "document": "Title: Inception (2010)\nGenres: Action, Sci-Fi...\nPlot: ..."
}
```

---

### 7.2 Oracle RAG Collection

```python
# From: services/oracle-rag-service/main.py

# Each movie gets its own collection
collection = chroma_client.get_or_create_collection(
    name=f"{movie_id}_subtitles"  # e.g., "the_matrix_subtitles"
)

# Stats for one movie
print(f"Chunks: {collection.count()}")
# Output: 523 (for a 2-hour movie)
```

**Data format:**

```python
# Each subtitle chunk:
{
    "id": "chunk_145",
    "embedding": [0.12, -0.34, 0.56, ..., 0.89],  # 384 floats
    "metadata": {
        "movie_id": "the_matrix",
        "timestamp_start": 2710.5,  # Seconds from start
        "timestamp_end": 2725.3,
        "scene_index": 145
    },
    "document": """Neo: You ever have that feeling where you're not sure
                   if you're awake or still dreaming?
                   Trinity: All the time."""
}
```

**Spoiler protection query:**
```python
# User has watched up to 45:30 (2730 seconds)
results = collection.query(
    query_embeddings=question_embedding,
    n_results=5,
    where={"timestamp_start": {"$lte": 2730}}  # Only chunks before 45:30
)
```

---

## 8. Performance Benchmarks

### 8.1 Search Times (on RTX 4080 GPU)

| Operation | Time | Notes |
|-----------|------|-------|
| Embed query (BGE-large) | 10ms | GPU-accelerated |
| Embed query (MiniLM) | 5ms | Smaller model |
| (ChromaDB search (6,147 docs) | 30ms | HNSW index |
| BM25 keyword search | 20ms | CPU-bound |
| Cross-encoder rerank (50 docs) | 450ms | Expensive |
| **Total pipeline** | ~500ms | End-to-end |

### 8.2 Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| BGE-large model | 1.3 GB | Loaded once at startup |
| ChromaDB (6,147 movies) | 25 MB | Embeddings only |
| Metadata (SQLite) | 5 MB | Text data |
| HNSW graph | 50 MB | Index structure |
| **Total** | ~1.4 GB | Fits easily in consumer hardware |

---

## Summary: Key Concepts

1. **Hybrid storage:** SQLite (metadata) + Binary (vectors) + HNSW (index)
2. **Collections** are like SQL tables, store embeddings + metadata + documents
3. **HNSW** enables O(log n) search instead of O(n) brute-force
4. **Metadata filters** combine SQL WHERE with vector similarity
5. **Upsert** prevents duplicates, simplifies updates
6. **Two collections** in StreamSage: `movies` (6,147) and `{movie_id}_subtitles` (variable)

---

[← Back to Day 3](README.md)
