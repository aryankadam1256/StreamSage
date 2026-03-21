# ChromaDB Data Storage Format & Structure

Understanding how ChromaDB physically stores and organizes data internally.

---

## 1. Storage Architecture

ChromaDB stores data in **SQLite + binary files**:

```
chroma_db/
├── chroma.sqlite3           # Metadata database (SQL)
├── {collection_id}/
│   ├── data_level0.bin      # HNSW graph (binary)
│   ├── header.bin           # Collection metadata
│   ├── length.bin           # Document lengths
│   └── link_lists.bin       # HNSW links
```

### Why This Hybrid Approach?

| Component | Storage | Why? |
|-----------|---------|------|
| **Metadata** (title, year, genre) | SQLite (relational) | Fast filtering with SQL WHERE |
| **Embeddings** (vectors) | Binary files | Efficient vector operations |
| **HNSW Graph** | Binary files | Fast similarity search |
| **Documents** (text) | SQLite | Easy retrieval for display |

---

## 2. Collection Structure

A **Collection** is like a table in SQL:

```python
collection = chroma_client.create_collection(
    name="movies",
    metadata={"hnsw:space": "cosine"}
)
```

### Internal Representation:

```
Collection: movies
├── ID: "uuid-1234"
├── Embedding Function: sentence-transformers/all-MiniLM-L6-v2
├── Distance Metric: cosine
└── Documents:
    ├── Document 1:
    │   ├── ID: "550" (unique identifier)
    │   ├── Embedding: [0.23, -0.45, 0.78, ..., 0.12] (384 floats)
    │   ├── Metadata: {"title": "Fight Club", "year": 1999, "rating": 8.8}
    │   └── Document: "Title: Fight Club (1999)\nGenres: Drama\n..."
    │
    ├── Document 2:
    │   ├── ID: "13"
    │   ├── Embedding: [0.12, -0.33, 0.65, ..., 0.08] (384 floats)
    │   ├── Metadata: {"title": "Forrest Gump", "year": 1994, "rating": 8.8}
    │   └── Document: "Title: Forrest Gump (1994)\nGenres: Comedy, Drama\n..."
    │
    └── ... (6,147 documents total)
```

---

## 3. Data Types

### 3.1 IDs (Unique Identifiers)
```python
# Format: String
ids = ["550", "13", "680", ...]

# Used for:
# - Deduplication (prevents duplicate inserts)
# - Updates (upsert operation)
# - Retrieval (get specific document)
```

### 3.2 Embeddings (Vectors)
```python
# Format: List[float] or numpy array
embedding = [
    0.23456789,   # Dimension 1
    -0.45678901,  # Dimension 2
    0.78901234,   # Dimension 3
    ...           # ...
    0.12345678    # Dimension 384 (for all-MiniLM-L6-v2)
]

# Storage: Float32 (4 bytes per number)
# Total size per embedding: 384 * 4 = 1,536 bytes = 1.5 KB
# For 6,147 movies: 6,147 * 1.5 KB = ~9 MB
```

### 3.3 Metadata (Structured Data)
```python
# Format: Dict with restricted value types
metadata = {
    "title": "Inception",        # str
    "year": 2010,                 # int
    "rating": 8.8,                # float
    "genres": "Sci-Fi, Thriller", # str (comma-separated)
    "is_favorite": True           # bool
}

# IMPORTANT: Only str, int, float, bool allowed!
# Lists and dicts are NOT supported in metadata
```

**Why the restriction?**
- Metadata is stored in SQLite
- SQLite columns must have fixed types
- Simplifies indexing for fast filtering

### 3.4 Documents (Original Text)
```python
# Format: String
document = """Title: Inception (2010)
Genres: Sci-Fi, Thriller
Director: Christopher Nolan
Cast: Leonardo DiCaprio, Ellen Page, Tom Hardy
Mood: thought_provoking, mind_bending
Keywords: dream, subconscious, heist, reality, layers
Plot: A thief who steals corporate secrets through dream-sharing
technology is given the inverse task of planting an idea..."""

# Used for:
# - Display in search results
# - Re-embedding if model changes
# - Human-readable context
```

---

## 4. The SQLite Schema

When you create a collection, ChromaDB creates these tables:

```sql
-- Collections table
CREATE TABLE collections (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE,
    metadata TEXT,  -- JSON blob
    dimension INTEGER
);

-- Embeddings table
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    collection_id TEXT,
    embedding BLOB,  -- Binary vector data
    FOREIGN KEY (collection_id) REFERENCES collections(id)
);

-- Metadata table
CREATE TABLE embedding_metadata (
    id TEXT PRIMARY KEY,
    key TEXT,
    string_value TEXT,
    int_value INTEGER,
    float_value REAL,
    FOREIGN KEY (id) REFERENCES embeddings(id)
);

-- Documents table
CREATE TABLE embedding_documents (
    id TEXT PRIMARY KEY,
    document TEXT,
    FOREIGN KEY (id) REFERENCES embeddings(id)
);
```

**Example Query:**
```sql
-- Find all Nolan films with rating > 8.0
SELECT e.id, em.string_value as title
FROM embeddings e
JOIN embedding_metadata em ON e.id = em.id
WHERE em.key = 'director' AND em.string_value = 'Christopher Nolan'
  AND e.id IN (
      SELECT id FROM embedding_metadata
      WHERE key = 'rating' AND float_value > 8.0
  );
```

---

## 5. HNSW Index Structure

**HNSW** (Hierarchical Navigable Small World) is a graph-based index for fast similarity search.

### Conceptual Visualization:

```
Layer 2 (Coarse):      A ←→ B ←→ C
                       ↓    ↓    ↓
Layer 1 (Medium):  A ← D → E ← F → G ← C
                   ↓   ↓   ↓   ↓   ↓   ↓
Layer 0 (Fine):  A-D-H-I-E-J-F-K-L-G-M-C
                 ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
               [All 6,147 movie embeddings]
```

**How Search Works:**
1. Start at top layer (coarse)
2. Navigate to nearest neighbor
3. Drop down a layer
4. Repeat until bottom layer
5. Return K nearest neighbors

**Why It's Fast:**
- Without HNSW: Compare to all 6,147 movies = slow (O(n))
- With HNSW: Navigate shortcuts = fast (O(log n))
- Search 6,147 movies in ~50ms instead of ~500ms

---

## 6. Real Data Example from StreamSage

### How a Movie is Stored:

```python
# Input to ChromaDB
collection.add(
    ids=["550"],
    embeddings=[[0.23, -0.45, 0.78, ..., 0.12]],  # 1024-dim for BGE-large
    metadatas=[{
        "title": "Fight Club",
        "year": 1999,
        "rating": 8.8,
        "genres": "Drama",
        "director": "David Fincher",
        "popularity_score": 63.8,
        "moods": "dark, thought_provoking"
    }],
    documents=[
        "Title: Fight Club (1999)\n"
        "Genres: Drama\n"
        "Director: David Fincher\n"
        "Cast: Brad Pitt, Edward Norton, Helena Bonham Carter\n"
        "Mood: dark, thought_provoking\n"
        "Keywords: insomnia, alter ego, underground, rebellion\n"
        "Plot: A depressed office worker and a soap salesman build a global organization..."
    ]
)
```

### Internal Storage:

**SQLite (chroma.sqlite3):**
```
collections table:
  id: "abc-123"
  name: "movies"
  dimension: 1024

embeddings table:
  id: "550"
  collection_id: "abc-123"
  embedding: <BLOB: 4KB binary data>

embedding_metadata table:
  | id  | key        | string_value | int_value | float_value |
  |-----|------------|--------------|-----------|-------------|
  | 550 | title      | Fight Club   | NULL      | NULL        |
  | 550 | year       | NULL         | 1999      | NULL        |
  | 550 | rating     | NULL         | NULL      | 8.8         |
  | 550 | genres     | Drama        | NULL      | NULL        |
  | 550 | director   | David Fincher| NULL      | NULL        |

embedding_documents table:
  id: "550"
  document: "Title: Fight Club (1999)\nGenres: Drama\n..."
```

**Binary Files (abc-123/):**
```
data_level0.bin:  [Contains 1024 float32 values = 4KB per movie]
header.bin:       [Collection metadata]
link_lists.bin:   [HNSW graph connections]
```

---

## 7. Query Execution Flow

### Example Query:
```python
results = collection.query(
    query_texts=["mind-bending thriller"],
    n_results=5,
    where={"year": {"$gte": 2000}}
)
```

### Internal Steps:

```
1. EMBED QUERY
   "mind-bending thriller" → [0.34, -0.56, 0.89, ...] (1024 floats)

2. APPLY METADATA FILTER (SQL)
   SELECT id FROM embedding_metadata
   WHERE key = 'year' AND int_value >= 2000

   Result: [13, 27, 155, 550, 680, ...] (3,200 movies after 2000)

3. VECTOR SEARCH (HNSW on filtered set)
   - Navigate HNSW graph
   - Only consider movies from step 2
   - Find 5 closest embeddings using cosine similarity

   Result IDs: [550, 27, 155, 680, 13]

4. RETRIEVE FULL DATA (SQL JOIN)
   SELECT e.id, em.*, ed.document
   FROM embeddings e
   JOIN embedding_metadata em ON e.id = em.id
   JOIN embedding_documents ed ON e.id = ed.id
   WHERE e.id IN (550, 27, 155, 680, 13)

5. FORMAT RESPONSE
   Return: {
       'ids': [['550', '27', '155', '680', '13']],
       'distances': [[0.12, 0.18, 0.23, 0.25, 0.28]],
       'metadatas': [[{...}, {...}, {...}, {...}, {...}]],
       'documents': [["Title: Fight Club...", "Title: Inception...", ...]]
   }
```

**Performance:**
- Step 1 (Embedding): ~10ms
- Step 2 (SQL Filter): ~5ms
- Step 3 (HNSW Search): ~30ms
- Step 4 (Retrieve Data): ~5ms
- **Total: ~50ms** for searching 3,200 movies

---

## 8. Comparison with Traditional Databases

| Feature | Traditional SQL | ChromaDB |
|---------|-----------------|----------|
| **Primary Key** | Integer ID | String ID |
| **Data Types** | All SQL types | str, int, float, bool only |
| **Indexing** | B-tree (exact match) | HNSW (similarity) |
| **Search** | `WHERE title LIKE '%ince%'` | Semantic similarity |
| **Performance** | Fast for exact match | Fast for "similar to" |
| **Storage** | Row-based | Hybrid (SQL + binary) |

**When to use each:**
- **SQL**: "Find all movies by Christopher Nolan" (exact match)
- **ChromaDB**: "Find movies similar to Inception" (semantic similarity)
- **Best of both**: Use ChromaDB with metadata filters!

---

## Summary

ChromaDB stores data in a **hybrid format**:
- **SQLite** for metadata and documents (SQL query power)
- **Binary files** for embeddings (efficient vector ops)
- **HNSW graph** for fast similarity search

This design enables:
- **Fast semantic search** (~50ms for 6,147 movies)
- **Metadata filtering** (SQL WHERE clauses)
- **Easy updates** (upsert by ID)
- **Persistent storage** (survives restarts)

---

[← Back to Day 3](README.md) | [Next: Complex Code Explained →](Complex-Code-Explained.md)
